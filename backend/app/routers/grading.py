import os
import time
import uuid
import json
import shutil
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Response

logger = logging.getLogger(__name__)
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import (
    GradeTextRequest,
    GradingResponse,
    BatchJobResponse,
    CriterionGradingResult,
    EvidenceCitation,
    QuestionGradingResult
)
from backend.app.routers.assessments import _fetch_assessment_by_id
from backend.app.services.text_extractor import TextExtractor
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.batch_manager import BatchManager

router = APIRouter(prefix="/grading", tags=["Grading Engine"])

def _save_grading_record(res: GradingResponse, student_text: str, sub_type: str, sub_filename: Optional[str] = None):
    now_iso = datetime.now(timezone.utc).isoformat()
    record_id = res.id or str(uuid.uuid4())
    res.id = record_id
    res.created_at = now_iso
    res.submission_type = sub_type
    res.submission_filename = sub_filename
    res.student_text = student_text
    t0 = time.perf_counter()
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO grading_records (
                    id, assessment_id, submission_type, submission_filename,
                    student_text, total_score, max_score, percentage,
                    requires_teacher_review, review_reasons,
                    criterion_results, evidence_citations, provenance,
                    question_results, overall_feedback, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                res.assessment_id,
                sub_type,
                sub_filename,
                student_text,
                res.total_score,
                res.max_score,
                res.percentage,
                1 if res.requires_teacher_review else 0,
                json.dumps(res.review_reasons),
                json.dumps([c.model_dump() for c in res.criterion_breakdown]),
                json.dumps([cit.model_dump() for cit in res.citations]),
                json.dumps(res.provenance),
                json.dumps([q.model_dump() for q in res.question_results]) if res.question_results else None,
                res.overall_feedback or "",
                now_iso
            ))
        save_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Persistence] Recorded submission '{sub_filename or sub_type}' to DB in {save_ms:.2f}ms")
    except Exception as e:
        logger.warning(f"Grading record logging failed for '{sub_filename or sub_type}': {e}")

@router.get("/records", response_model=List[GradingResponse])
def get_grading_records(assessment_id: Optional[str] = None):
    """
    Returns historical grading records stored in SQLite, optionally filtered by assessment_id.
    Joined with assessments table for assessment_title.
    Ordered by created_at DESC (most recent first).
    """
    query = """
        SELECT
            g.id, g.assessment_id, g.submission_type, g.submission_filename,
            g.student_text, g.total_score, g.max_score, g.percentage,
            g.requires_teacher_review, g.review_reasons,
            g.criterion_results, g.evidence_citations, g.provenance,
            g.question_results, g.overall_feedback, g.created_at,
            a.title as assessment_title
        FROM grading_records g
        LEFT JOIN assessments a ON g.assessment_id = a.id
    """
    params = []
    if assessment_id:
        query += " WHERE g.assessment_id = ?"
        params.append(assessment_id)
    query += " ORDER BY g.created_at DESC, g.rowid DESC"

    records: List[GradingResponse] = []
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        for row in rows:
            try:
                review_reasons = json.loads(row["review_reasons"]) if row["review_reasons"] else []
            except Exception:
                review_reasons = []

            try:
                raw_crits = json.loads(row["criterion_results"]) if row["criterion_results"] else []
                criterion_breakdown = [CriterionGradingResult(**c) for c in raw_crits]
            except Exception:
                criterion_breakdown = []

            try:
                raw_cits = json.loads(row["evidence_citations"]) if row["evidence_citations"] else []
                citations = [EvidenceCitation(**cit) for cit in raw_cits]
            except Exception:
                citations = []

            try:
                provenance = json.loads(row["provenance"]) if row["provenance"] else {}
            except Exception:
                provenance = {}

            question_results = None
            if "question_results" in row.keys() and row["question_results"]:
                try:
                    raw_q = json.loads(row["question_results"])
                    question_results = [QuestionGradingResult(**q) for q in raw_q]
                except Exception:
                    question_results = None

            overall_feedback = ""
            if "overall_feedback" in row.keys() and row["overall_feedback"]:
                overall_feedback = row["overall_feedback"]

            records.append(GradingResponse(
                id=row["id"],
                assessment_id=row["assessment_id"],
                assessment_title=row["assessment_title"] or "Assessment",
                submission_type=row["submission_type"],
                submission_filename=row["submission_filename"],
                student_text=row["student_text"],
                total_score=float(row["total_score"]),
                max_score=float(row["max_score"]),
                percentage=float(row["percentage"]),
                requires_teacher_review=bool(row["requires_teacher_review"]),
                review_reasons=review_reasons,
                criterion_breakdown=criterion_breakdown,
                citations=citations,
                provenance=provenance,
                question_results=question_results,
                overall_feedback=overall_feedback,
                created_at=row["created_at"]
            ))

    return records

@router.delete("/records/{record_id}")
def delete_grading_record(record_id: str):
    """
    Deletes an individual grading record from SQLite by record_id.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM grading_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Grading record '{record_id}' not found."
            )
        cursor.execute("DELETE FROM grading_records WHERE id = ?", (record_id,))
    return {"status": "deleted", "id": record_id}

@router.delete("/records")
def delete_grading_records(assessment_id: Optional[str] = None):
    """
    Deletes grading records from SQLite, optionally filtered by assessment_id.
    If assessment_id is omitted, deletes all grading records.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        if assessment_id:
            cursor.execute("DELETE FROM grading_records WHERE assessment_id = ?", (assessment_id,))
        else:
            cursor.execute("DELETE FROM grading_records")
        deleted_count = cursor.rowcount
    return {"status": "deleted", "deleted_count": deleted_count}

@router.post("/grade-text", response_model=GradingResponse)
def grade_text(payload: GradeTextRequest):
    if not payload.assessment_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="assessment_id is required.")

    # Reject empty or whitespace-only answers
    if not payload.student_text or not payload.student_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student answer text cannot be empty or whitespace only."
        )

    assessment = _fetch_assessment_by_id(payload.assessment_id)
    student_text = payload.student_text.strip()

    grading_response = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        submission_type="text",
        feedback_mode=payload.feedback_mode
    )

    _save_grading_record(grading_response, student_text, sub_type="text")
    return grading_response

@router.post("/grade-file", response_model=GradingResponse)
async def grade_file(
    assessment_id: str = Form(...),
    file: UploadFile = File(...),
    feedback_mode: Optional[str] = Form(None)
):
    if not assessment_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="assessment_id is required.")

    assessment = _fetch_assessment_by_id(assessment_id)

    filename = os.path.basename(file.filename or "student_submission")
    temp_path = str(settings.UPLOADS_DIR / f"temp_sub_{uuid.uuid4()}_{filename}")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        t_ext0 = time.perf_counter()
        extracted = TextExtractor.extract_from_file(temp_path)
        student_text = extracted.get("full_text", "")
        ext_ms = (time.perf_counter() - t_ext0) * 1000
        logger.info(f"[TextExtractor] Extracted {len(student_text)} chars from '{filename}' in {ext_ms:.2f}ms")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to extract text from file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # Reject empty or unreadable uploaded documents
    if not student_text or not student_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or contains no readable text."
        )

    grading_response = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        submission_type="file",
        submission_filename=filename,
        feedback_mode=feedback_mode
    )

    _save_grading_record(grading_response, student_text, sub_type="file", sub_filename=filename)
    return grading_response

@router.post("/batch-zip", response_model=BatchJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def grade_batch_zip(
    assessment_id: str = Form(...),
    file: UploadFile = File(...),
    feedback_mode: Optional[str] = Form(None)
):
    """
    Accepts a ZIP archive containing student answer files (.pdf, .docx, .txt),
    validates archive safety, extracts supported documents, and triggers background batch grading.
    """
    if not assessment_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="assessment_id is required.")

    assessment = _fetch_assessment_by_id(assessment_id)

    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be a .zip archive.")

    temp_zip_id = str(uuid.uuid4())
    temp_zip_path = str(settings.UPLOADS_DIR / f"batch_{temp_zip_id}.zip")
    temp_extract_dir = str(settings.UPLOADS_DIR / f"batch_{temp_zip_id}_extracted")

    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_files = BatchManager.validate_and_extract_zip(temp_zip_path, temp_extract_dir)
    except ValueError as ve:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to process archive: {str(e)}")
    finally:
        if os.path.exists(temp_zip_path):
            try:
                os.remove(temp_zip_path)
            except Exception:
                pass

    batch_job = BatchManager.start_batch_job(
        assessment=assessment,
        extracted_files=extracted_files,
        temp_dir=temp_extract_dir,
        feedback_mode=feedback_mode
    )
    return batch_job

@router.get("/batch/{batch_id}", response_model=BatchJobResponse)
def get_batch_status(batch_id: str):
    """Returns live status, progress counters, and per-file results for a batch grading job."""
    job = BatchManager.get_batch(batch_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch job '{batch_id}' not found.")
    return job

@router.get("/batch/{batch_id}/csv")
def export_batch_csv(batch_id: str):
    """Generates and downloads a CSV summary report of batch grading results."""
    job = BatchManager.get_batch(batch_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch job '{batch_id}' not found.")

    assessment = _fetch_assessment_by_id(job.assessment_id)
    csv_content = BatchManager.generate_csv_report(batch_id, assessment)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=batch_{batch_id}_results.csv"
        }
    )
