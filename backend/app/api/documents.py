import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_current_user
from app.db.models import Document, User
from app.db.session import get_db, get_sessionmaker
from app.schemas.documents import DocumentOut
from app.services.embeddings import Embedder, get_embedder
from app.services.ingest import ALLOWED_TYPES, ingest_document
from app.services.vectorstore import delete_document_chunks, get_qdrant

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


async def get_owned_document(document_id: uuid.UUID, db: AsyncSession, user: User) -> Document:
    doc = await db.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user.id)
    )
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    embedder: Embedder = Depends(get_embedder),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    sessionmaker: async_sessionmaker[AsyncSession] = Depends(get_sessionmaker),
):
    filename = Path(file.filename or "").name[:255]
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File exceeds 10 MB")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")

    doc = Document(
        user_id=user.id,
        filename=filename,
        content_type=ALLOWED_TYPES[ext],
        size_bytes=len(data),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    background.add_task(
        ingest_document,
        sessionmaker=sessionmaker,
        embedder=embedder,
        qdrant=qdrant,
        document_id=doc.id,
        user_id=user.id,
        filename=filename,
        ext=ext,
        data=data,
    )
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    result = await db.scalars(
        select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())
    )
    return result.all()


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await get_owned_document(document_id, db, user)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    doc = await get_owned_document(document_id, db, user)
    await delete_document_chunks(qdrant, doc.id)  # vectors first, so a failure leaves the row to retry
    await db.delete(doc)
    await db.commit()