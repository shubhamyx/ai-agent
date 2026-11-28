"""Agent endpoints. One LangGraph thread per conversation (thread_id = conversation id).
The messages table keeps the user-visible chat (user text, final answer + sources);
the checkpoint keeps the full agent state (tool calls and results)."""

import logging
import re
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.checkpoint import get_checkpointer
from app.agents.graph import build_agent
from app.agents.tools import CitationRegistry, build_tools
from app.api.conversations import get_owned_conversation, sse
from app.api.deps import get_current_user
from app.api.rate_limit import rate_limit_chat
from app.core.config import settings
from app.db.models import User
from app.db.session import get_db, get_sessionmaker
from app.schemas.chat import AgentMessageCreate, AgentTraceItem
from app.services.chat import DEFAULT_TITLE, add_message, load_history, make_title
from app.services.embeddings import Embedder, get_embedder
from app.services.llm import get_llm
from app.services.vectorstore import get_qdrant

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["agent"])

CITE_RE = re.compile(r"\[(\d+)\]")
TOOL_RESULT_PREVIEW = 500
ANSWER_NODES = {"agent", "finalize"}


def thread_config(conversation_id: uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(conversation_id)}}


def history_to_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for role, content in history:
        if role == "user":
            msgs.append(HumanMessage(content))
        elif role == "assistant":
            msgs.append(AIMessage(content))
    return msgs


def text_of(content) -> str:
    return content if isinstance(content, str) else str(content)


@router.post(
    "/{conversation_id}/agent/stream",
    dependencies=[Depends(rate_limit_chat)],
)
async def agent_stream(
    conversation_id: uuid.UUID,
    data: AgentMessageCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    llm: BaseChatModel = Depends(get_llm),
    embedder: Embedder = Depends(get_embedder),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    checkpointer=Depends(get_checkpointer),
    sessionmaker: async_sessionmaker[AsyncSession] = Depends(get_sessionmaker),
):
    # 404 / 422 / 429 are raised before any bytes are streamed
    conv = await get_owned_conversation(conversation_id, db, user)
    if conv.title == DEFAULT_TITLE:
        conv.title = make_title(data.content)
    await add_message(db, conversation_id, "user", data.content)

    citations = CitationRegistry()
    tools = build_tools(user_id=user.id, embedder=embedder, qdrant=qdrant, citations=citations)
    agent = build_agent(llm, tools, max_steps=settings.agent_max_steps, checkpointer=checkpointer)
    cfg = thread_config(conversation_id)

    # first agent turn in a conversation that already has plain-chat history:
    # seed the thread with it so the agent knows what was said
    state = await agent.aget_state(cfg)
    if state.values.get("messages"):
        new_messages: list[BaseMessage] = [HumanMessage(data.content)]
    else:
        new_messages = history_to_messages(await load_history(db, conversation_id))
    graph_input = {"messages": new_messages, "steps": 0}

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for mode, payload in agent.astream(
                graph_input, cfg, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, meta = payload
                    if (
                        meta.get("langgraph_node") in ANSWER_NODES
                        and isinstance(chunk, AIMessageChunk)
                        and isinstance(chunk.content, str)
                        and chunk.content
                    ):
                        yield sse("token", {"text": chunk.content})
                    continue

                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue  # e.g. __interrupt__
                    for msg in update.get("messages", []):
                        if node == "agent" and getattr(msg, "tool_calls", None):
                            for call in msg.tool_calls:
                                yield sse(
                                    "tool_call",
                                    {"id": call["id"], "name": call["name"], "args": call["args"]},
                                )
                        elif node == "tools" and isinstance(msg, ToolMessage):
                            yield sse(
                                "tool_result",
                                {
                                    "id": msg.tool_call_id,
                                    "name": msg.name,
                                    "content": text_of(msg.content)[:TOOL_RESULT_PREVIEW],
                                    "sources": msg.artifact or [],
                                },
                            )

            final = (await agent.aget_state(cfg)).values["messages"][-1]
            text = text_of(final.content)
            if not text.strip():
                yield sse("error", {"detail": "Agent returned an empty response"})
                return

            cited = {int(n) for n in CITE_RE.findall(text)}
            sources = [s for s in citations.sources if s["index"] in cited] or None
            async with sessionmaker() as s:
                msg = await add_message(s, conversation_id, "assistant", text, sources)
                done = {
                    "id": str(msg.id),
                    "created_at": msg.created_at.isoformat(),
                    "sources": sources,
                }
            yield sse("done", done)
        except Exception:
            # the checkpoint keeps the progress made so far
            logger.exception("Agent run failed for conversation %s", conversation_id)
            yield sse("error", {"detail": "Agent run failed"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{conversation_id}/agent/trace", response_model=list[AgentTraceItem])
async def agent_trace(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    llm: BaseChatModel = Depends(get_llm),
    checkpointer=Depends(get_checkpointer),
):
    """Full agent message history for a conversation, including tool calls/results."""
    await get_owned_conversation(conversation_id, db, user)
    agent = build_agent(llm, [], max_steps=settings.agent_max_steps, checkpointer=checkpointer)
    state = await agent.aget_state(thread_config(conversation_id))

    items: list[AgentTraceItem] = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage):
            items.append(AgentTraceItem(type="user", content=text_of(m.content)))
        elif isinstance(m, AIMessage):
            for call in m.tool_calls:
                items.append(
                    AgentTraceItem(type="tool_call", id=call["id"], name=call["name"], args=call["args"])
                )
            if text_of(m.content).strip():
                items.append(AgentTraceItem(type="assistant", content=text_of(m.content)))
        elif isinstance(m, ToolMessage):
            items.append(
                AgentTraceItem(
                    type="tool_result",
                    id=m.tool_call_id,
                    name=m.name,
                    content=text_of(m.content)[:TOOL_RESULT_PREVIEW],
                    sources=m.artifact or None,
                )
            )
    return items
