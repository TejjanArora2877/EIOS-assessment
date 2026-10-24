import sqlite3
from typing import Generator
from contextlib import contextmanager
from backend.app.config import settings

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(settings.SQLITE_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for handling database transactions."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Initializes the database schema if tables do not exist."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Institutional reference documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                total_pages INTEGER,
                total_chunks INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'queued',
                error_message TEXT,
                created_at TEXT NOT NULL,
                indexed_at TEXT
            );
        """)

        # Document chunks table (stores chunk text & page mapping for lexical retrieval & provenance)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                text TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id);")

        # Assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                question TEXT NOT NULL,
                model_answer TEXT NOT NULL,
                total_marks REAL NOT NULL,
                doc_scope TEXT NOT NULL DEFAULT 'ALL',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # Rubric criteria table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rubric_criteria (
                id TEXT PRIMARY KEY,
                assessment_id TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                max_marks REAL NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rubric_assessment ON rubric_criteria(assessment_id);")

        # Audit and provenance records for grading
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grading_records (
                id TEXT PRIMARY KEY,
                assessment_id TEXT NOT NULL,
                submission_type TEXT NOT NULL,
                submission_filename TEXT,
                student_text TEXT NOT NULL,
                total_score REAL NOT NULL,
                max_score REAL NOT NULL,
                percentage REAL NOT NULL,
                requires_teacher_review INTEGER NOT NULL DEFAULT 0,
                review_reasons TEXT,
                criterion_results TEXT NOT NULL,
                evidence_citations TEXT NOT NULL,
                provenance TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE
            );
        """)
