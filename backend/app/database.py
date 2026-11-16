import uuid
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
                question_id TEXT,
                order_index INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                max_marks REAL NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE,
                FOREIGN KEY (question_id) REFERENCES assessment_questions (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rubric_assessment ON rubric_criteria(assessment_id);")

        # Assessment questions table (for multi-question assessments)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_questions (
                id TEXT PRIMARY KEY,
                assessment_id TEXT NOT NULL,
                question_order INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                model_answer TEXT,
                max_marks REAL NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_questions_assessment ON assessment_questions(assessment_id);")

        # Ensure question_id column exists on rubric_criteria if table was already created
        cursor.execute("PRAGMA table_info(rubric_criteria);")
        crit_cols = [row["name"] for row in cursor.fetchall()]
        if "question_id" not in crit_cols:
            cursor.execute("ALTER TABLE rubric_criteria ADD COLUMN question_id TEXT REFERENCES assessment_questions(id) ON DELETE CASCADE;")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rubric_question ON rubric_criteria(question_id);")

        # Auto-migration on startup: For any existing single-question Assessment in SQLite
        # that has no rows in assessment_questions, automatically create a default Question 1 record
        # from assessment.question and assessment.model_answer, and link its existing RubricCriterion rows to it.
        cursor.execute("SELECT id, question, model_answer, total_marks FROM assessments;")
        all_assessments = cursor.fetchall()
        for a in all_assessments:
            aid = a["id"]
            cursor.execute("SELECT COUNT(*) as cnt FROM assessment_questions WHERE assessment_id = ?", (aid,))
            if cursor.fetchone()["cnt"] == 0:
                qid = str(uuid.uuid4())
                q_text = a["question"] or ""
                q_ans = a["model_answer"] or ""
                q_marks = float(a["total_marks"] or 0.0)
                cursor.execute("""
                    INSERT INTO assessment_questions (id, assessment_id, question_order, question_text, model_answer, max_marks)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (qid, aid, q_text, q_ans, q_marks))
                cursor.execute("UPDATE rubric_criteria SET question_id = ? WHERE assessment_id = ? AND question_id IS NULL", (qid, aid))

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
                question_results TEXT,
                overall_feedback TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE
            );
        """)

        # Ensure question_results and overall_feedback exist on grading_records
        cursor.execute("PRAGMA table_info(grading_records);")
        rec_cols = [row["name"] for row in cursor.fetchall()]
        if "question_results" not in rec_cols:
            cursor.execute("ALTER TABLE grading_records ADD COLUMN question_results TEXT;")
        if "overall_feedback" not in rec_cols:
            cursor.execute("ALTER TABLE grading_records ADD COLUMN overall_feedback TEXT;")
