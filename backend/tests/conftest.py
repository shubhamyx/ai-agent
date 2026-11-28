import math
import re
import zlib
from collections.abc import Iterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.redis import get_redis
from app.db.models import Base
from app.db.session import get_db, get_sessionmaker
from app.main import app
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.vectorstore import ensure_collection, get_qdrant
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.checkpoint import get_checkpointer

TEST_DB_URL = settings.database_url.rsplit("/", 1)[0] + "/agent_test"


class FakeChatModel(BaseChatModel):
    """Deterministic stand-in for ChatGroq. Records every input it receives.
    Set `fail = True` to make the next call raise."""

    reply: str = "Hello from the fake model"
    fail: bool = False
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _check(self, messages: list[BaseMessage]) -> None:
        self.calls.append(list(messages))
        if self.fail:
            raise RuntimeError("fake LLM failure")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._check(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _stream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        self._check(messages)
        # word + trailing whitespace, so "".join(chunks) == reply exactly
        for piece in re.findall(r"\S+\s*", self.reply):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


class FakeEmbedder:
    """Bag-of-words hashing embedder: deterministic, instant, and texts sharing
    words get similar vectors, so retrieval tests are meaningful."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_llm() -> FakeChatModel:
    return FakeChatModel()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest_asyncio.fixture
async def fake_qdrant(fake_embedder):
    q = AsyncQdrantClient(location=":memory:")
    await ensure_collection(q, fake_embedder.dim)
    yield q
    await q.close()


@pytest_asyncio.fixture
async def fake_redis():
    r = FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def client(fake_llm, fake_redis, fake_embedder, fake_qdrant):
    saver = InMemorySaver()
    engine = create_async_engine(TEST_DB_URL, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_sessionmaker] = lambda: Session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_redis] = lambda: fake_redis
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_qdrant] = lambda: fake_qdrant
    app.dependency_overrides[get_checkpointer] = lambda: saver
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


@pytest_asyncio.fixture
def make_user(client):
    async def _make(email="a@example.com", password="password123"):
        await client.post("/auth/register", json={"email": email, "password": password})
        r = await client.post("/auth/login", data={"username": email, "password": password})
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _make
