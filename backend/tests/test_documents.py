import io
import json
 
from pypdf import PdfWriter
from qdrant_client.models import FieldCondition, Filter, MatchValue
 
from app.api import conversations as conversations_api
from app.api import documents as documents_api
from app.core.config import settings
from app.services.rag import NO_CONTEXT
 
ORION = (
    "The Orion project launched in March 2024. Its lead engineer is Priya Nair. "
    "The ingestion service is written in Rust."
)
BREAD = "Banana bread needs flour, sugar, ripe bananas and forty minutes in the oven."
 
 
async def upload(client, headers, name: str, content: bytes, ctype: str = "text/plain"):
    r = await client.post("/documents", headers=headers, files={"file": (name, content, ctype)})
    return r
 
 
async def upload_ready(client, headers, name: str, text: str) -> dict:
    r = await upload(client, headers, name, text.encode())
    assert r.status_code == 202
    # ASGITransport waits for background tasks, so ingestion has finished here
    doc = (await client.get(f"/documents/{r.json()['id']}", headers=headers)).json()
    assert doc["status"] == "ready", doc
    return doc
 
 
async def point_count(qdrant, document_id: str) -> int:
    res = await qdrant.count(
        settings.qdrant_collection,
        count_filter=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
        exact=True,
    )
    return res.count
 
 
def blank_pdf() -> bytes:
    w = PdfWriter()
    w.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()
 
 
async def new_conversation(client, headers) -> str:
    r = await client.post("/conversations", json={}, headers=headers)
    return r.json()["id"]
 
 
def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events
 
 
# ---------- upload & ingestion ----------
 
 
async def test_upload_ingests_into_qdrant(client, make_user, fake_qdrant):
    h = await make_user()
    r = await upload(client, h, "orion.txt", ORION.encode())
 
    assert r.status_code == 202
    assert r.json()["status"] == "processing"
    assert r.json()["content_type"] == "text/plain"
 
    doc = (await client.get(f"/documents/{r.json()['id']}", headers=h)).json()
    assert doc["status"] == "ready"
    assert doc["chunk_count"] >= 1
    assert await point_count(fake_qdrant, doc["id"]) == doc["chunk_count"]
 
 
async def test_upload_rejects_bad_input(client, make_user, monkeypatch):
    h = await make_user()
 
    r = await upload(client, h, "image.png", b"\x89PNG", "image/png")
    assert r.status_code == 415
 
    r = await upload(client, h, "empty.txt", b"")
    assert r.status_code == 400
 
    monkeypatch.setattr(documents_api, "MAX_UPLOAD_BYTES", 10)
    r = await upload(client, h, "big.txt", b"x" * 11)
    assert r.status_code == 413
 
    assert (await client.get("/documents", headers=h)).json() == []
 
 
async def test_pdf_without_text_fails_cleanly(client, make_user, fake_qdrant):
    h = await make_user()
    r = await upload(client, h, "scan.pdf", blank_pdf(), "application/pdf")
    doc = (await client.get(f"/documents/{r.json()['id']}", headers=h)).json()
 
    assert doc["status"] == "failed"
    assert "No extractable text" in doc["error"]
    assert await point_count(fake_qdrant, doc["id"]) == 0
 
 
async def test_corrupt_pdf_fails_cleanly(client, make_user):
    h = await make_user()
    r = await upload(client, h, "broken.pdf", b"definitely not a pdf", "application/pdf")
    doc = (await client.get(f"/documents/{r.json()['id']}", headers=h)).json()
 
    assert doc["status"] == "failed"
    assert doc["error"] == "Could not read the PDF file"
 
 
# ---------- ownership & delete ----------
 
 
async def test_documents_are_private(client, make_user):
    alice = await make_user("alice@example.com")
    bob = await make_user("bob@example.com")
    doc = await upload_ready(client, alice, "orion.txt", ORION)
 
    assert (await client.get("/documents", headers=bob)).json() == []
    assert (await client.get(f"/documents/{doc['id']}", headers=bob)).status_code == 404
    assert (await client.delete(f"/documents/{doc['id']}", headers=bob)).status_code == 404
    assert len((await client.get("/documents", headers=alice)).json()) == 1
 
 
async def test_delete_removes_row_and_vectors(client, make_user, fake_qdrant):
    h = await make_user()
    doc = await upload_ready(client, h, "orion.txt", ORION)
    assert await point_count(fake_qdrant, doc["id"]) > 0
 
    r = await client.delete(f"/documents/{doc['id']}", headers=h)
 
    assert r.status_code == 204
    assert await point_count(fake_qdrant, doc["id"]) == 0
    assert (await client.get(f"/documents/{doc['id']}", headers=h)).status_code == 404
 
 
# ---------- RAG chat ----------
 
 
async def test_rag_answer_cites_relevant_document(client, make_user, fake_llm):
    h = await make_user()
    await upload_ready(client, h, "orion.txt", ORION)
    await upload_ready(client, h, "bread.txt", BREAD)
    cid = await new_conversation(client, h)
 
    r = await client.post(
        f"/conversations/{cid}/messages",
        json={"content": "Who is the Orion lead engineer?", "use_documents": True},
        headers=h,
    )
 
    assert r.status_code == 201
    sources = r.json()["sources"]
    assert sources[0]["filename"] == "orion.txt"  # most relevant first
    assert sources[0]["index"] == 1
    assert "Priya Nair" in sources[0]["snippet"]
 
    system_prompt = fake_llm.calls[-1][0].content
    assert "[1] (orion.txt)" in system_prompt
    assert "Priya Nair" in system_prompt
 
    # citations are persisted with the message
    conv = (await client.get(f"/conversations/{cid}", headers=h)).json()
    assistant = [m for m in conv["messages"] if m["role"] == "assistant"][-1]
    assert assistant["sources"] == sources
 
 
async def test_rag_never_retrieves_other_users_documents(client, make_user, fake_llm):
    alice = await make_user("alice@example.com")
    bob = await make_user("bob@example.com")
    await upload_ready(client, bob, "orion.txt", ORION)
    cid = await new_conversation(client, alice)
 
    r = await client.post(
        f"/conversations/{cid}/messages",
        json={"content": "Who is the Orion lead engineer?", "use_documents": True},
        headers=alice,
    )
 
    assert r.json()["sources"] == []
    system_prompt = fake_llm.calls[-1][0].content
    assert "Priya Nair" not in system_prompt
    assert NO_CONTEXT in system_prompt
 
 
async def test_rag_document_ids_filter(client, make_user):
    h = await make_user()
    await upload_ready(client, h, "orion.txt", ORION)
    bread = await upload_ready(client, h, "bread.txt", BREAD)
    cid = await new_conversation(client, h)
 
    r = await client.post(
        f"/conversations/{cid}/messages",
        json={
            "content": "Who is the Orion lead engineer?",
            "use_documents": True,
            "document_ids": [bread["id"]],
        },
        headers=h,
    )
 
    assert {s["filename"] for s in r.json()["sources"]} == {"bread.txt"}
 
 
async def test_plain_chat_has_no_sources(client, make_user, fake_llm):
    h = await make_user()
    await upload_ready(client, h, "orion.txt", ORION)
    cid = await new_conversation(client, h)
 
    r = await client.post(f"/conversations/{cid}/messages", json={"content": "hi"}, headers=h)
 
    assert r.json()["sources"] is None
    assert "Context:" not in fake_llm.calls[-1][0].content
 
 
async def test_rag_stream_sends_sources_first(client, make_user, fake_llm):
    h = await make_user()
    await upload_ready(client, h, "orion.txt", ORION)
    cid = await new_conversation(client, h)
 
    r = await client.post(
        f"/conversations/{cid}/messages/stream",
        json={"content": "Orion lead engineer?", "use_documents": True},
        headers=h,
    )
 
    events = parse_sse(r.text)
    assert events[0][0] == "sources"
    assert events[0][1]["sources"][0]["filename"] == "orion.txt"
    assert events[-1][0] == "done"
 
    conv = (await client.get(f"/conversations/{cid}", headers=h)).json()
    assistant = [m for m in conv["messages"] if m["role"] == "assistant"][-1]
    key = lambda s: (s["index"], s["document_id"], s["chunk_index"])
    assert [key(s) for s in assistant["sources"]] == [key(s) for s in events[0][1]["sources"]]
 
 
async def test_retrieval_failure_is_503_without_llm_call(client, make_user, fake_llm, monkeypatch):
    async def broken(*args, **kwargs):
        raise ConnectionError("qdrant down")
 
    monkeypatch.setattr(conversations_api, "retrieve", broken)
    h = await make_user()
    cid = await new_conversation(client, h)
 
    r = await client.post(
        f"/conversations/{cid}/messages",
        json={"content": "anything", "use_documents": True},
        headers=h,
    )
 
    assert r.status_code == 503
    assert fake_llm.calls == []
 