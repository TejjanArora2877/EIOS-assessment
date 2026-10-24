import io
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings
from backend.app.database import get_db, init_db
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service
from backend.app.services.storage_reconciler import StorageReconciler
from backend.app.services.retriever import HybridRetriever

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_isolated_env():
    """Ensures test runs in isolated test directory and starts clean."""
    init_db()
    VectorStore.reset()
    BM25Service.reset()
    yield

def test_document_persistence_across_restart():
    """
    Regression Test:
    1. Upload a small synthetic document through the real ingestion API.
    2. Confirm document metadata, source file, and indexed chunks are persisted.
    3. Simulate backend shutdown (clear in-memory singletons and re-run startup lifecycle).
    4. Query document list and verify it is not empty.
    5. Retrieve a relevant passage and verify document identity and source metadata.
    """
    # 1. Upload small synthetic document
    doc_content = (
        "Memory-Based Learning Foundations and NLP Applications.\n\n"
        "Memory-Based Learning stores labeled exemplar feature vectors in memory. "
        "When classifying new cases, it computes similarity distances against stored cases. "
        "Standard NLP tasks include part-of-speech tagging, shallow parsing, and morphological analysis."
    )
    doc_bytes = io.BytesIO(doc_content.encode("utf-8"))

    upload_res = client.post(
        "/documents/upload",
        files={"file": ("synthetic_mbl_notes.txt", doc_bytes, "text/plain")},
        data={"title": "Synthetic MBL Notes", "doc_type": "lecture_notes"}
    )
    assert upload_res.status_code == 201, f"Upload failed: {upload_res.text}"
    uploaded_doc = upload_res.json()
    doc_id = uploaded_doc["id"]
    assert uploaded_doc["status"] == "ready"
    assert uploaded_doc["total_chunks"] >= 1

    # 2. Confirm persistence in SQLite, disk, ChromaDB, and BM25 before shutdown
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, file_path, status, total_chunks FROM documents WHERE id = ?", (doc_id,))
        db_doc = cursor.fetchone()
        assert db_doc is not None, "Document not found in SQLite documents table."
        assert db_doc["status"] == "ready"
        assert os.path.exists(db_doc["file_path"]), f"Source file does not exist at {db_doc['file_path']}"

        cursor.execute("SELECT count(*) as count FROM document_chunks WHERE document_id = ?", (doc_id,))
        chunk_count = cursor.fetchone()["count"]
        assert chunk_count == uploaded_doc["total_chunks"]

    # Confirm ChromaDB and BM25 have chunks
    chroma_col = VectorStore.get_collection()
    assert chroma_col.count() >= uploaded_doc["total_chunks"]
    assert len(BM25Service._corpus_records) >= uploaded_doc["total_chunks"]

    # 3. Simulate backend shutdown and restart
    # Reset all in-memory clients, collections, and state
    VectorStore.reset()
    BM25Service.reset()

    # Re-execute startup sequence (equivalent to application lifespan)
    init_db()
    StorageReconciler.reconcile()

    # 4. Query document list via real API after restart
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs_list = list_res.json()["documents"]

    assert len(docs_list) > 0, "Document list must not be empty after backend restart!"
    restarted_doc = next((d for d in docs_list if d["id"] == doc_id), None)
    assert restarted_doc is not None, f"Uploaded document {doc_id} was lost after restart."
    assert restarted_doc["title"] == "Synthetic MBL Notes"
    assert restarted_doc["status"] == "ready"
    assert restarted_doc["total_chunks"] == uploaded_doc["total_chunks"]

    # 5. Retrieve a relevant passage and verify document identity and source metadata
    citations = HybridRetriever.retrieve_evidence(
        query="similarity distances against stored exemplar cases",
        top_k=2,
        doc_scope=[doc_id]
    )
    assert len(citations) > 0, "Hybrid retriever failed to return citations after restart."
    top_citation = citations[0]
    assert top_citation.document_id == doc_id
    assert top_citation.document_title == "Synthetic MBL Notes"
    assert "exemplar" in top_citation.snippet.lower() or "similarity" in top_citation.snippet.lower()
