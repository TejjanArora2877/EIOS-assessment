import os
import io
import csv
import time
import uuid
import shutil
import zipfile
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import (
    AssessmentResponse,
    BatchItemResult,
    BatchJobResponse,
    GradingResponse
)
from backend.app.services.text_extractor import TextExtractor
from backend.app.services.grading_engine import GradingEngine

logger = logging.getLogger(__name__)

# Constants for safety validation
MAX_ZIP_FILES = 50
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _save_grading_record(res: GradingResponse, student_text: str, sub_type: str, sub_filename: Optional[str] = None):
    """Persists an individual graded submission to the database for auditing and history."""
    import json
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
        logger.info(f"[Persistence] Recorded batch submission '{sub_filename}' to DB in {save_ms:.2f}ms")
    except Exception as e:
        logger.warning(f"Failed to record batch item to grading_records: {e}")


class BatchManager:
    """
    Manages asynchronous ZIP batch grading jobs with live status tracking,
    strict submission isolation, and CSV report export.
    """
    _lock = threading.Lock()
    _batches: Dict[str, BatchJobResponse] = {}

    @classmethod
    def validate_and_extract_zip(cls, zip_path: str, extract_dir: str) -> List[Tuple[str, str]]:
        """
        Validates the safety of a ZIP archive and extracts supported documents.
        Enforces file count limits, uncompressed size bounds, directory traversal prevention,
        and filters system/hidden files.
        """
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("The uploaded file is not a valid ZIP archive.")

        os.makedirs(extract_dir, exist_ok=True)
        abs_extract_dir = os.path.abspath(extract_dir)

        extracted_files: List[Tuple[str, str]] = []
        total_uncompressed_size = 0
        file_count = 0

        with zipfile.ZipFile(zip_path, 'r') as archive:
            entries = archive.infolist()
            for info in entries:
                # Security: check path traversal on raw entry before any processing
                raw_parts = info.filename.replace('\\', '/').split('/')
                if '..' in raw_parts or os.path.isabs(info.filename):
                    raise ValueError(f"Security violation: path traversal detected for file '{info.filename}'.")

                # Skip directory entries
                if info.is_dir() or info.filename.endswith('/'):
                    continue

                filename = os.path.basename(info.filename)

                # Skip hidden/system files and macOS metadata
                if (
                    filename.startswith('.')
                    or info.filename.startswith('__MACOSX')
                    or '/__MACOSX' in info.filename
                    or filename.lower() in {'thumbs.db', 'desktop.ini', '.ds_store'}
                ):
                    continue

                # Check extension
                ext = os.path.splitext(filename)[1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue

                file_count += 1
                if file_count > MAX_ZIP_FILES:
                    raise ValueError(f"Archive exceeds maximum allowable limit of {MAX_ZIP_FILES} documents.")

                total_uncompressed_size += info.file_size
                if total_uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError(f"Archive uncompressed size exceeds maximum allowable limit of {MAX_UNCOMPRESSED_BYTES // (1024*1024)}MB.")

                # Traversal check
                destination_path = os.path.abspath(os.path.join(abs_extract_dir, filename))
                if not destination_path.startswith(abs_extract_dir):
                    raise ValueError(f"Security violation: path traversal detected for file '{info.filename}'.")

                # Handle duplicate filenames in subdirectories
                base, ext_part = os.path.splitext(filename)
                counter = 1
                while os.path.exists(destination_path):
                    destination_path = os.path.abspath(os.path.join(abs_extract_dir, f"{base}_{counter}{ext_part}"))
                    counter += 1

                with archive.open(info) as src, open(destination_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

                extracted_files.append((os.path.basename(destination_path), destination_path))

        if not extracted_files:
            raise ValueError("No supported answer documents (.pdf, .docx, .txt) found in the ZIP archive.")

        return extracted_files

    @classmethod
    def create_batch(cls, assessment_id: str, assessment_title: str, total_files: int) -> BatchJobResponse:
        """Initializes a new batch job tracking record."""
        batch_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        job = BatchJobResponse(
            batch_id=batch_id,
            assessment_id=assessment_id,
            assessment_title=assessment_title,
            total_files=total_files,
            completed=0,
            failed=0,
            skipped=0,
            status="processing",
            created_at=now_iso,
            results=[]
        )
        with cls._lock:
            cls._batches[batch_id] = job
        return job

    @classmethod
    def get_batch(cls, batch_id: str) -> Optional[BatchJobResponse]:
        """Retrieves current batch status."""
        with cls._lock:
            return cls._batches.get(batch_id)

    @classmethod
    def process_batch_worker(
        cls,
        batch_id: str,
        assessment: AssessmentResponse,
        extracted_files: List[Tuple[str, str]],
        temp_dir: str,
        feedback_mode: Optional[str] = None
    ):
        """
        Background worker that grades extracted files sequentially.
        Ensures complete submission isolation, robust error recovery, and proper cleanup.
        """
        logger.info(f"Starting batch {batch_id} processing with {len(extracted_files)} files.")
        t_batch_start = time.perf_counter()
        try:
            for filename, file_path in extracted_files:
                t_file_start = time.perf_counter()
                student_text = ""
                try:
                    t_ext0 = time.perf_counter()
                    extracted = TextExtractor.extract_from_file(file_path)
                    student_text = extracted.get("full_text", "")
                    ext_ms = (time.perf_counter() - t_ext0) * 1000
                    logger.info(f"[TextExtractor] Batch {batch_id} extracted {len(student_text)} chars from '{filename}' in {ext_ms:.2f}ms")
                except Exception as extract_err:
                    logger.warning(f"Batch {batch_id} failed to extract text from {filename}: {extract_err}")
                    file_dur = round(time.perf_counter() - t_file_start, 3)
                    item_res = BatchItemResult(
                        filename=filename,
                        status="failed",
                        error_message=f"Text extraction failed: {str(extract_err)}",
                        duration_s=file_dur
                    )
                    with cls._lock:
                        if batch_id in cls._batches:
                            cls._batches[batch_id].results.append(item_res)
                            cls._batches[batch_id].failed += 1
                    continue

                # Empty submission check: mark as skipped
                if not student_text or not student_text.strip():
                    file_dur = round(time.perf_counter() - t_file_start, 3)
                    logger.info(f"Batch {batch_id} skipped empty file {filename} in {file_dur*1000:.2f}ms")
                    item_res = BatchItemResult(
                        filename=filename,
                        status="skipped",
                        error_message="Uploaded file is empty or contains no readable text.",
                        duration_s=file_dur,
                        ollama_calls=0
                    )
                    with cls._lock:
                        if batch_id in cls._batches:
                            cls._batches[batch_id].results.append(item_res)
                            cls._batches[batch_id].skipped += 1
                    continue

                # Grade single submission in clean, isolated context
                try:
                    grading_response = GradingEngine.grade_submission(
                        assessment=assessment,
                        student_text=student_text,
                        submission_type="file",
                        submission_filename=filename,
                        feedback_mode=feedback_mode
                    )

                    # Persist record for auditability
                    _save_grading_record(grading_response, student_text, sub_type="file", sub_filename=filename)

                    file_dur = round(time.perf_counter() - t_file_start, 3)
                    calls_count = grading_response.provenance.get("ollama_calls", 0) if isinstance(grading_response.provenance, dict) else 0
                    item_res = BatchItemResult(
                        filename=filename,
                        status="completed",
                        total_score=grading_response.total_score,
                        max_score=grading_response.max_score,
                        percentage=grading_response.percentage,
                        requires_teacher_review=grading_response.requires_teacher_review,
                        criterion_breakdown=grading_response.criterion_breakdown,
                        student_text_preview=student_text[:120].strip(),
                        duration_s=file_dur,
                        ollama_calls=calls_count
                    )
                    with cls._lock:
                        if batch_id in cls._batches:
                            cls._batches[batch_id].results.append(item_res)
                            cls._batches[batch_id].completed += 1
                except Exception as grade_err:
                    file_dur = round(time.perf_counter() - t_file_start, 3)
                    logger.error(f"Batch {batch_id} grading error for {filename}: {grade_err}", exc_info=True)
                    item_res = BatchItemResult(
                        filename=filename,
                        status="failed",
                        error_message=f"Grading failed: {str(grade_err)}",
                        duration_s=file_dur,
                        ollama_calls=0
                    )
                    with cls._lock:
                        if batch_id in cls._batches:
                            cls._batches[batch_id].results.append(item_res)
                            cls._batches[batch_id].failed += 1

            # Finalize batch state
            t_batch_total = round(time.perf_counter() - t_batch_start, 3)
            with cls._lock:
                if batch_id in cls._batches:
                    cls._batches[batch_id].status = "completed"
                    cls._batches[batch_id].completed_at = datetime.now(timezone.utc).isoformat()
                    cls._batches[batch_id].total_duration_s = t_batch_total
            logger.info(f"Batch {batch_id} completed successfully in {t_batch_total:.2f}s.")
        except Exception as e:
            t_batch_total = round(time.perf_counter() - t_batch_start, 3)
            logger.error(f"Fatal batch processing error in batch {batch_id}: {e}", exc_info=True)
            with cls._lock:
                if batch_id in cls._batches:
                    cls._batches[batch_id].status = "failed"
                    cls._batches[batch_id].completed_at = datetime.now(timezone.utc).isoformat()
                    cls._batches[batch_id].total_duration_s = t_batch_total
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def start_batch_job(
        cls,
        assessment: AssessmentResponse,
        extracted_files: List[Tuple[str, str]],
        temp_dir: str,
        feedback_mode: Optional[str] = None
    ) -> BatchJobResponse:
        """Creates batch record and launches worker thread."""
        batch_job = cls.create_batch(
            assessment_id=assessment.id,
            assessment_title=assessment.title,
            total_files=len(extracted_files)
        )
        worker_thread = threading.Thread(
            target=cls.process_batch_worker,
            args=(batch_job.batch_id, assessment, extracted_files, temp_dir, feedback_mode),
            daemon=True
        )
        worker_thread.start()
        return batch_job

    @classmethod
    def generate_csv_report(cls, batch_id: str, assessment: AssessmentResponse) -> str:
        """Generates downloadable CSV content summarizing batch evaluation results."""
        batch = cls.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch job {batch_id} not found.")

        criteria_names = [c.name for c in assessment.criteria]
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        headers = [
            "Filename",
            "Status",
            "Total Score",
            "Max Score",
            "Percentage (%)",
            "Requires Teacher Review"
        ]
        for name in criteria_names:
            headers.append(f"{name} (Awarded)")
            headers.append(f"{name} (Max)")
        headers.append("Error Details")

        writer.writerow(headers)

        # Data rows
        for item in batch.results:
            row = [
                item.filename,
                item.status,
                f"{item.total_score:.2f}" if item.total_score is not None else "",
                f"{item.max_score:.2f}" if item.max_score is not None else "",
                f"{item.percentage:.1f}" if item.percentage is not None else "",
                "Yes" if item.requires_teacher_review else ("No" if item.status == "completed" else "")
            ]

            crit_map = {}
            if item.criterion_breakdown:
                for cb in item.criterion_breakdown:
                    crit_map[cb.criterion_name] = cb

            for name in criteria_names:
                cb = crit_map.get(name)
                if cb:
                    row.append(f"{cb.awarded_marks:.2f}")
                    row.append(f"{cb.max_marks:.2f}")
                else:
                    row.append("")
                    row.append("")

            row.append(item.error_message or "")
            writer.writerow(row)

        return output.getvalue()
