import pytest
import shutil
from pathlib import Path
from backend.app.config import settings

# Direct initial import-time execution to an isolated test environment
TEST_DATA_DIR = settings.DATA_DIR / "test_scratch"
settings.SQLITE_DB_PATH = TEST_DATA_DIR / "test_eios.db"
settings.CHROMA_PERSIST_DIR = TEST_DATA_DIR / "chroma"
settings.BM25_PERSIST_DIR = TEST_DATA_DIR / "bm25"
settings.UPLOADS_DIR = TEST_DATA_DIR / "uploads"

for p in [settings.SQLITE_DB_PATH.parent, settings.CHROMA_PERSIST_DIR, settings.BM25_PERSIST_DIR, settings.UPLOADS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

from backend.app.database import get_db, init_db
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service


@pytest.fixture(scope="session", autouse=True)
def isolate_test_database(tmp_path_factory):
    """Session fixture isolating SQLite database, Chroma, and BM25 in a temporary directory."""
    test_dir = tmp_path_factory.mktemp("eios_test_data")
    test_db = test_dir / "test_eios.db"

    settings.SQLITE_DB_PATH = test_db
    settings.CHROMA_PERSIST_DIR = test_dir / "chroma"
    settings.BM25_PERSIST_DIR = test_dir / "bm25"
    settings.UPLOADS_DIR = test_dir / "uploads"

    for p in [test_db.parent, settings.CHROMA_PERSIST_DIR, settings.BM25_PERSIST_DIR, settings.UPLOADS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    init_db()
    VectorStore.reset()
    BM25Service.reset()

    yield

    VectorStore.reset()
    BM25Service.reset()


def clean_database():
    init_db()
    with get_db() as conn:
        conn.execute("DELETE FROM rubric_criteria")
        conn.execute("DELETE FROM grading_records")
        conn.execute("DELETE FROM assessment_questions")
        conn.execute("DELETE FROM assessments")
        conn.execute("DELETE FROM document_chunks")
        conn.execute("DELETE FROM documents")
    VectorStore.reset()
    BM25Service.reset()


@pytest.fixture(autouse=True)
def clean_state_fixture():
    clean_database()
    yield
    clean_database()
