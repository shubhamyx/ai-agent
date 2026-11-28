"""Tool-calling agent as a LangGraph StateGraph.

    START -> agent --(tool calls?)--> tools --(budget left?)--> agent ...
               |                        |
               +--(no tool calls)--> END +--(budget spent)--> finalize -> END
"""

from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

AGENT_SYSTEM_PROMPT = """You are a helpful assistant that can use tools.

- search_documents: for anything that may be answered by the user's uploaded files. \
Turn follow-up questions into standalone queries (resolve "it", "that project", etc.).
- web_search (if available): for current events, recent facts, or anything the documents don't cover.
- calculator: for any arithmetic beyond the trivial. Never do multi-step maths in your head.

You may call several tools, or the same tool more than once. Stop calling tools as soon as \
you have enough to answer. Cite sources inline with the exact [n] labels from tool results, \
e.g. [1] or [2][4]. Calculator results are not sources; never cite them. Never invent \
citations. If the tools found nothing useful, say so plainly. \
Be concise."""

FINALIZE_NOTE = (
    "You have used all available tool calls for this question. Answer now using only the "
    "information already gathered, and say clearly what is missing if it is incomplete."
)

# messages of history the model sees per call; the checkpoint keeps everything
HISTORY_MESSAGES = 40


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int  # model calls in the current run; reset to 0 by each new user turn


def build_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    *,
    max_steps: int,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):  
    llm_with_tools = llm.bind_tools(tools)
    # same tool schemas (history contains tool calls) but the model may not call any
    llm_final = llm.bind_tools(tools, tool_choice="none")

    def context(state: AgentState) -> list[AnyMessage]:
        # last N messages, never starting mid-turn on a tool result
        return trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=HISTORY_MESSAGES,
            start_on="human",
            include_system=False,
        )

    async def agent(state: AgentState) -> dict:
        response = await llm_with_tools.ainvoke(
            [SystemMessage(AGENT_SYSTEM_PROMPT), *context(state)]
        )
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    async def finalize(state: AgentState) -> dict:
        response = await llm_final.ainvoke(
            [SystemMessage(f"{AGENT_SYSTEM_PROMPT}\n\n{FINALIZE_NOTE}"), *context(state)]
        )
        return {"messages": [response]}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def after_tools(state: AgentState) -> str:
        return "finalize" if state.get("steps", 0) >= max_steps else "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)