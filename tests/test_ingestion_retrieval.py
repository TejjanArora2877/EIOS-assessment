import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db, get_db
from backend.app.services.preprocessor import Preprocessor
from backend.app.services.retriever import HybridRetriever
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_preprocessor_pii_and_chunking():
    raw_text = """
    Contact Dr. Jane Doe at jane.doe@university.edu or call 555-123-4567. Student ID: STU-98765.
    Memory-Based Learning (MBL) is a classifier approach in natural language processing.
    It represents instances as feature vectors.
    """
    redacted = Preprocessor.redact_pii(raw_text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_STUDENT_ID]" in redacted
    assert "jane.doe@university.edu" not in redacted
    assert "555-123-4567" not in redacted

    normalized = Preprocessor.normalize_text(redacted)
    assert "jane.doe" not in normalized

    pages = [{"page_number": 1, "text": raw_text}]
    chunks = Preprocessor.elc_chunk_document(pages, target_chunk_words=10)
    assert len(chunks) >= 2
    assert chunks[0]["page_number"] == 1
    assert any("Memory-Based Learning" in c["text"] for c in chunks)

def test_document_ingestion_retrieval_and_deletion():
    # 1. Upload a synthetic test document
    synthetic_content = """
    Artificial Intelligence in Healthcare.
    Supervised learning algorithms analyze medical imaging data to detect anomalies.
    Convolutional neural networks achieve high diagnostic accuracy for radiographic images.
    Patient confidentiality and HIPAA compliance require strict data anonymization protocols.
    """
    file_bytes = io.BytesIO(synthetic_content.encode("utf-8"))
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("ai_healthcare_test.txt", file_bytes, "text/plain")},
        data={"title": "AI in Healthcare Manual", "doc_type": "reference_material"}
    )
    assert upload_res.status_code == 201, upload_res.text
    doc_data = upload_res.json()
    doc_id = doc_data["id"]
    assert doc_data["status"] == "ready"
    assert doc_data["total_chunks"] >= 1

    # 2. Verify document in list
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()["documents"]
    matched = [d for d in docs if d["id"] == doc_id]
    assert len(matched) == 1
    assert matched[0]["title"] == "AI in Healthcare Manual"

    # 3. Test Hybrid Retrieval
    citations = HybridRetriever.retrieve_evidence(
        query="convolutional neural networks in radiographic imaging",
        top_k=2
    )
    assert len(citations) >= 1
    assert any("radiographic" in c.snippet.lower() or "medical" in c.snippet.lower() for c in citations)
    assert citations[0].document_title == "AI in Healthcare Manual"

    # 4. Test Deletion
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200

    # Verify document is gone from database
    status_res = client.get(f"/documents/{doc_id}/status")
    assert status_res.status_code == 404

    # Verify retrieval no longer returns deleted document
    citations_after = HybridRetriever.retrieve_evidence(
        query="convolutional neural networks in radiographic imaging",
        top_k=2
    )
    assert not any(c.document_id == doc_id for c in citations_after)
