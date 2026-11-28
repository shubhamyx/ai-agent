import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.checkpoint import open_checkpointer
from app.api import agent, auth, conversations, documents
from app.services.embeddings import get_embedder
from app.services.vectorstore import ensure_collection, get_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = await asyncio.to_thread(get_embedder)  # load model once, off the event loop
    await ensure_collection(get_qdrant(), embedder.dim)
    pool, app.state.checkpointer = await open_checkpointer()
    yield
    await pool.close()
    await get_qdrant().close()


app = FastAPI(title="AI Agent Platform", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(agent.router)


@app.get("/health")
async def health():
    return {"status": "ok"}