import io
import os
import zipfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import init_db
from backend.app.services.batch_manager import BatchManager, MAX_ZIP_FILES
from backend.app.services.llm_feedback import LLMFeedbackService

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def sample_assessment():
    """Dynamically created assessment for non-MBL subject (Operating Systems)."""
    payload = {
        "title": "Operating Systems - Process Scheduling",
        "question": "Explain Round Robin scheduling, the concept of a time quantum, and its trade-offs.",
        "model_answer": "Round Robin is a preemptive scheduling algorithm where each process is assigned a fixed time slice called a quantum. If the quantum is too large, RR behaves like FCFS. If too small, context switching overhead dominates.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Algorithm Mechanism",
                "description": "Explains preemptive time-slicing and circular queue execution.",
                "max_marks": 4.0
            },
            {
                "name": "Time Quantum",
                "description": "Defines the fixed duration allocated to each process per round.",
                "max_marks": 3.0
            },
            {
                "name": "Trade-offs and Overhead",
                "description": "Discusses context switching overhead versus turnaround time and responsiveness.",
                "max_marks": 3.0
            }
        ]
    }
    res = client.post("/assessments", json=payload)
    assert res.status_code == 201
    return res.json()


def create_zip_in_memory(files_dict: dict) -> bytes:
    """Helper to create a zip archive in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files_dict.items():
            zf.writestr(filename, content)
    buf.seek(0)
    return buf.getvalue()


def test_empty_submission_rejection_single(sample_assessment):
    """Verifies Phase 2: empty submissions are rejected with HTTP 400."""
    aid = sample_assessment["id"]

    # 1. Text mode empty string
    res1 = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ""})
    assert res1.status_code == 400
    assert "cannot be empty" in res1.json()["detail"].lower()

    # 2. Text mode whitespace only
    res2 = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": "   \n\t  "})
    assert res2.status_code == 400
    assert "cannot be empty" in res2.json()["detail"].lower()

    # 3. File mode empty file
    empty_file = io.BytesIO(b"")
    res3 = client.post(
        "/grading/grade-file",
        data={"assessment_id": aid},
        files={"file": ("empty.txt", empty_file, "text/plain")}
    )
    assert res3.status_code == 400
    assert "empty or contains no readable text" in res3.json()["detail"].lower()


def test_batch_zip_security_validations(sample_assessment):
    """Verifies ZIP archive validation limits, traversal attacks, and allowed types."""
    aid = sample_assessment["id"]

    # 1. Reject non-zip file
    res_non_zip = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid},
        files={"file": ("answers.txt", io.BytesIO(b"not a zip"), "text/plain")}
    )
    assert res_non_zip.status_code == 400
    assert "must be a .zip archive" in res_non_zip.json()["detail"].lower()

    # 2. Reject ZIP with path traversal attempt
    traversal_zip_bytes = io.BytesIO()
    with zipfile.ZipFile(traversal_zip_bytes, "w") as zf:
        zf.writestr("../evil.txt", "malicious payload")
    traversal_zip_bytes.seek(0)

    res_traversal = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid},
        files={"file": ("traversal.zip", traversal_zip_bytes, "application/zip")}
    )
    assert res_traversal.status_code == 400
    assert "traversal" in res_traversal.json()["detail"].lower()

    # 3. Reject ZIP exceeding maximum file count (MAX_ZIP_FILES + 1)
    too_many_files = {f"student_{i}.txt": f"Content {i}" for i in range(MAX_ZIP_FILES + 1)}
    large_zip_bytes = create_zip_in_memory(too_many_files)
    res_limit = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid},
        files={"file": ("large_batch.zip", io.BytesIO(large_zip_bytes), "application/zip")}
    )
    assert res_limit.status_code == 400
    assert "exceeds maximum allowable limit" in res_limit.json()["detail"].lower()


def test_batch_zip_execution_and_isolation(sample_assessment, monkeypatch):
    """
    Verifies full multi-file batch execution, empty file skipping,
    and strict isolation between submissions.
    """
    aid = sample_assessment["id"]

    student_1 = (
        "Round Robin is a preemptive scheduling algorithm designed for time-sharing systems. "
        "Each process receives a predetermined time quantum to execute. "
        "A small quantum leads to frequent context switching overhead, whereas a large quantum increases response time."
    )
    student_2 = "Round Robin is used in computers for scheduling."
    student_empty = "   "  # Whitespace only
    macos_metadata = "Macintosh metadata"

    zip_bytes = create_zip_in_memory({
        "student_1.txt": student_1,
        "student_2.txt": student_2,
        "student_empty.txt": student_empty,
        "__MACOSX/._student_1.txt": macos_metadata,
        ".DS_Store": "system junk"
    })

    # Mock LLM feedback across all threads to be deterministic and fast for test execution
    monkeypatch.setattr(
        LLMFeedbackService,
        "generate_criterion_feedback",
        lambda *args, **kwargs: "Objective criterion feedback."
    )

    res = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid},
        files={"file": ("submissions.zip", io.BytesIO(zip_bytes), "application/zip")}
    )
    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "processing"
    assert data["total_files"] == 3  # student_1, student_2, student_empty (system files excluded)
    batch_id = data["batch_id"]

    # Wait for worker thread to complete (with mocked LLM, scoring takes ~2-4s after models load)
    import time
    max_wait = 45
    start_t = time.time()
    final_job = None
    while time.time() - start_t < max_wait:
        poll_res = client.get(f"/grading/batch/{batch_id}")
        assert poll_res.status_code == 200
        final_job = poll_res.json()
        if final_job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert final_job is not None
    assert final_job["status"] == "completed"
    assert final_job["completed"] == 2
    assert final_job["skipped"] == 1
    assert final_job["failed"] == 0

    # Verify results breakdown
    results_by_name = {r["filename"]: r for r in final_job["results"]}

    # Student 1: High quality
    r1 = results_by_name["student_1.txt"]
    assert r1["status"] == "completed"
    assert r1["total_score"] > 6.0
    assert len(r1["criterion_breakdown"]) == 3

    # Student 2: Partial
    r2 = results_by_name["student_2.txt"]
    assert r2["status"] == "completed"
    assert r2["total_score"] < r1["total_score"]

    # Student Empty: Skipped
    r3 = results_by_name["student_empty.txt"]
    assert r3["status"] == "skipped"
    assert "empty or contains no readable text" in r3["error_message"].lower()

    # Check CSV export
    csv_res = client.get(f"/grading/batch/{batch_id}/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    csv_lines = csv_res.text.strip().split("\n")
    assert len(csv_lines) == 4  # Header + 3 files
    header = csv_lines[0]
    assert "Filename" in header
    assert "Algorithm Mechanism (Awarded)" in header
    assert "Time Quantum (Awarded)" in header
    assert "student_1.txt" in csv_res.text
    assert "student_2.txt" in csv_res.text
    assert "student_empty.txt" in csv_res.text


def test_individual_vs_batch_parity_and_isolation(sample_assessment, monkeypatch):
    """
    Verifies Phase 4 & Requirement 4:
    Strict numerical and diagnostic parity between individual file upload (/grade-file)
    and ZIP batch processing (/batch-zip).
    Ensures empty files are skipped without fabricated scores, and submission contexts are completely isolated.
    """
    aid = sample_assessment["id"]

    ans_high = (
        "Round Robin is a preemptive scheduling algorithm designed for time-sharing systems. "
        "Each process receives a predetermined time quantum to execute. "
        "A small quantum leads to frequent context switching overhead, whereas a large quantum increases response time."
    )
    ans_partial = (
        "Round Robin allocates a fixed time quantum to each process in turn. "
        "It uses a circular queue to manage active processes."
    )
    ans_empty = "   \n\t  "

    # Deterministic mock feedback to profile grading parity without external LLM latency
    monkeypatch.setattr(
        LLMFeedbackService,
        "generate_criterion_feedback",
        lambda *args, **kwargs: "Verified criterion feedback."
    )

    # 1. Run individual evaluations
    indiv_high_res = client.post(
        "/grading/grade-file",
        data={"assessment_id": aid},
        files={"file": ("student_high.txt", io.BytesIO(ans_high.encode("utf-8")), "text/plain")}
    )
    assert indiv_high_res.status_code == 200
    indiv_high = indiv_high_res.json()

    indiv_partial_res = client.post(
        "/grading/grade-file",
        data={"assessment_id": aid},
        files={"file": ("student_partial.txt", io.BytesIO(ans_partial.encode("utf-8")), "text/plain")}
    )
    assert indiv_partial_res.status_code == 200
    indiv_partial = indiv_partial_res.json()

    # 2. Run batch evaluation with identical answers and an empty file
    zip_bytes = create_zip_in_memory({
        "student_high.txt": ans_high,
        "student_partial.txt": ans_partial,
        "empty_doc.txt": ans_empty
    })

    batch_post_res = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid},
        files={"file": ("parity_batch.zip", io.BytesIO(zip_bytes), "application/zip")}
    )
    assert batch_post_res.status_code == 202
    batch_id = batch_post_res.json()["batch_id"]

    # Poll until complete
    import time
    start_t = time.time()
    final_job = None
    while time.time() - start_t < 25:
        poll_res = client.get(f"/grading/batch/{batch_id}")
        assert poll_res.status_code == 200
        final_job = poll_res.json()
        if final_job["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert final_job is not None
    assert final_job["status"] == "completed"
    assert final_job["completed"] == 2
    assert final_job["skipped"] == 1
    assert final_job["failed"] == 0

    batch_results = {r["filename"]: r for r in final_job["results"]}

    # Verify strict parity for student_high.txt
    b_high = batch_results["student_high.txt"]
    assert b_high["status"] == "completed"
    assert b_high["total_score"] == indiv_high["total_score"], f"Total score mismatch: {b_high['total_score']} vs {indiv_high['total_score']}"
    assert b_high["percentage"] == indiv_high["percentage"]
    assert b_high["requires_teacher_review"] == indiv_high["requires_teacher_review"]
    assert len(b_high["criterion_breakdown"]) == len(indiv_high["criterion_breakdown"])
    for b_crit, i_crit in zip(b_high["criterion_breakdown"], indiv_high["criterion_breakdown"]):
        assert b_crit["criterion_name"] == i_crit["criterion_name"]
        assert b_crit["awarded_marks"] == i_crit["awarded_marks"]
        assert b_crit["missing_concepts"] == i_crit["missing_concepts"]
        assert b_crit["contradictions_detected"] == i_crit["contradictions_detected"]

    # Verify strict parity for student_partial.txt
    b_part = batch_results["student_partial.txt"]
    assert b_part["status"] == "completed"
    assert b_part["total_score"] == indiv_partial["total_score"], f"Total score mismatch: {b_part['total_score']} vs {indiv_partial['total_score']}"
    assert b_part["percentage"] == indiv_partial["percentage"]
    assert b_part["requires_teacher_review"] == indiv_partial["requires_teacher_review"]
    assert len(b_part["criterion_breakdown"]) == len(indiv_partial["criterion_breakdown"])
    for b_crit, i_crit in zip(b_part["criterion_breakdown"], indiv_partial["criterion_breakdown"]):
        assert b_crit["criterion_name"] == i_crit["criterion_name"]
        assert b_crit["awarded_marks"] == i_crit["awarded_marks"]
        assert b_crit["missing_concepts"] == i_crit["missing_concepts"]
        assert b_crit["contradictions_detected"] == i_crit["contradictions_detected"]

    # Verify empty file was safely skipped without fabricated marks
    b_empty = batch_results["empty_doc.txt"]
    assert b_empty["status"] == "skipped"
    assert b_empty["total_score"] is None
    assert b_empty["max_score"] is None
    assert b_empty["percentage"] is None
    assert "empty or contains no readable text" in b_empty["error_message"].lower()
