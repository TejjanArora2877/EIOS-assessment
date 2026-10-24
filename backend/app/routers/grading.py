import os
import time
import uuid
import json
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Response

logger = logging.getLogger(__name__)
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import (
    GradeTextRequest,
    GradingResponse,
    BatchJobResponse
)
from backend.app.routers.assessments import _fetch_assessment_by_id
from backend.app.services.text_extractor import TextExtractor
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.batch_manager import BatchManager

router = APIRouter(prefix="/grading", tags=["Grading Engine"])

def _save_grading_record(res: GradingResponse, student_text: str, sub_type: str, sub_filename: Optional[str] = None):
    now_iso = datetime.now(timezone.utc).isoformat()
    record_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO grading_records (
                    id, assessment_id, submission_type, submission_filename,
                    student_text, total_score, max_score, percentage,
                    requires_teacher_review, review_reasons,
                    criterion_results, evidence_citations, provenance, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now_iso
            ))
        save_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Persistence] Recorded submission '{sub_filename or sub_type}' to DB in {save_ms:.2f}ms")
    except Exception as e:
        logger.warning(f"Grading record logging failed for '{sub_filename or sub_type}': {e}")

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
