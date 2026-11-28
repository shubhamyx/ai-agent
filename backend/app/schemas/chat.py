import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MessageText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
]


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", max_length=255)


class ConversationUpdate(BaseModel):
    title: str = Field(max_length=255)


class MessageCreate(BaseModel):
    content: MessageText
    use_documents: bool = False
    # restrict retrieval to these documents; None = all of the user's documents
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=20)


class AgentMessageCreate(BaseModel):
    content: MessageText


class SourceOut(BaseModel):
    index: int  # the [n] the model cites
    kind: Literal["document", "web"] = "document"
    snippet: str
    score: float | None = None
    # document sources
    document_id: uuid.UUID | None = None
    filename: str | None = None
    page: int | None = None
    chunk_index: int | None = None
    # web sources
    url: str | None = None
    title: str | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    role: str
    content: str
    sources: list[SourceOut] | None = None
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    title: str
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class AgentTraceItem(BaseModel):
    type: Literal["user", "assistant", "tool_call", "tool_result"]
    content: str | None = None
    id: str | None = None  # tool call id, links tool_call <-> tool_result
    name: str | None = None  # tool name
    args: dict[str, Any] | None = None
    sources: list[SourceOut] | None = None