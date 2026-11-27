"""LangGraph Postgres checkpointer: one psycopg pool opened at startup, shared by
every agent run. Threads are keyed by conversation id."""

from fastapi import Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings


def checkpoint_dsn() -> str:
    # same database as SQLAlchemy, but psycopg wants a plain postgresql:// URL
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def open_checkpointer() -> tuple[AsyncConnectionPool, AsyncPostgresSaver]:
    pool = AsyncConnectionPool(
        checkpoint_dsn(),
        max_size=10,
        open=False,
        # settings required by AsyncPostgresSaver
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # creates/migrates the checkpoint tables; idempotent
    return pool, saver


def get_checkpointer(request: Request):
    """Dependency; tests override it with an in-memory saver."""
    return request.app.state.checkpointer