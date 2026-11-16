import io
import pytest
from unittest.mock import patch
from docx import Document as DocxDocument
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import init_db
from backend.app.services.table_linearizer import TableLinearizer
from backend.app.services.answer_segmenter import AnswerSegmenter
from backend.app.services.assessment_parser import AssessmentDocumentParser

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_table_linearization_grid_and_markdown():
    """
    Verifies that TableLinearizer converts:
    1. 2D comparison table grids into declarative propositions binding column headers to cell values.
    2. 2-column key-value tables into '<Key>: <Value>.' propositions.
    3. Markdown pipe tables into linearized declarative sentences inline while preserving surrounding prose.
    """
    # 1. 2D standard comparison grid
    headers = ["Method", "Volatility", "Tool"]
    rows = [
        ["Live Response", "High", "netstat"],
        ["Disk Image", "Low", "dd"]
    ]
    grid_res = TableLinearizer.linearize_table(headers, rows)
    assert "For Method Live Response, Volatility is High, and Tool is netstat." in grid_res
    assert "For Method Disk Image, Volatility is Low, and Tool is dd." in grid_res

    # 2. 2D key-value table
    kv_headers = ["Property", "Value"]
    kv_rows = [
        ["Platform", "Linux Operating System"],
        ["Collection Mode", "Live Response"]
    ]
    kv_res = TableLinearizer.linearize_table(kv_headers, kv_rows)
    assert "Platform: Linux Operating System." in kv_res
    assert "Collection Mode: Live Response." in kv_res

    # 3. Markdown pipe table with surrounding prose
    md_text = """Introductory guidance for examiner.

| Platform | Acquisition Tool | Target Resource |
|:---|:---:|---:|
| Windows | winpmem | Physical RAM |
| Unix | LiME | Kernel Space |

Concluding guidance for examiner."""

    md_linearized = TableLinearizer.linearize_markdown_tables(md_text)
    assert "Introductory guidance for examiner." in md_linearized
    assert "For Platform Windows, Acquisition Tool is winpmem, and Target Resource is Physical RAM." in md_linearized
    assert "For Platform Unix, Acquisition Tool is LiME, and Target Resource is Kernel Space." in md_linearized
    assert "Concluding guidance for examiner." in md_linearized
    assert "|" not in md_linearized


def test_docx_interleaved_table_extraction():
    """
    Builds an in-memory .docx containing Question 1, a comparison table, and Question 2.
    Verifies exact reading-order extraction and compatibility with AnswerSegmenter.
    """
    doc = DocxDocument()
    doc.add_paragraph("Question 1:")
    doc.add_paragraph("Explain volatile memory acquisition tools across operating systems.")

    tbl = doc.add_table(rows=3, cols=2)
    tbl.rows[0].cells[0].text = "Platform"
    tbl.rows[0].cells[1].text = "Tool"
    tbl.rows[1].cells[0].text = "Windows"
    tbl.rows[1].cells[1].text = "winpmem"
    tbl.rows[2].cells[0].text = "Unix"
    tbl.rows[2].cells[1].text = "LiME"

    doc.add_paragraph("Question 2:")
    doc.add_paragraph("Describe disk analysis and file carving.")

    bio = io.BytesIO()
    doc.save(bio)
    docx_bytes = bio.getvalue()

    extraction = TableLinearizer.extract_docx_with_tables(docx_bytes)
    full_text = extraction["full_text"]

    # Verify exact interleaving reading order
    q1_idx = full_text.find("Question 1:")
    tbl_idx = full_text.find("For Platform Windows, Tool is winpmem.")
    q2_idx = full_text.find("Question 2:")
    q2_body_idx = full_text.find("Describe disk analysis and file carving.")

    assert q1_idx != -1 and tbl_idx != -1 and q2_idx != -1 and q2_body_idx != -1
    assert q1_idx < tbl_idx < q2_idx < q2_body_idx

    # Verify compatibility with AnswerSegmenter
    questions = [
        {
            "id": "q1",
            "question_order": 1,
            "question_text": "Explain volatile memory acquisition tools.",
            "model_answer": "Windows uses winpmem and Unix uses LiME.",
            "criteria": [{"name": "Tools", "description": "winpmem and LiME", "max_marks": 5.0}]
        },
        {
            "id": "q2",
            "question_order": 2,
            "question_text": "Describe disk analysis and file carving.",
            "model_answer": "Disk analysis carves files from raw blocks.",
            "criteria": [{"name": "Carving", "description": "Carving files", "max_marks": 5.0}]
        }
    ]

    slices = AnswerSegmenter.segment_submission(full_text, questions)
    assert "winpmem" in slices["q1"]
    assert "LiME" in slices["q1"]
    assert "disk analysis" in slices["q2"]
    assert "winpmem" not in slices["q2"]


def test_mixed_table_and_prose_grading():
    """
    Verifies cross-format grading symmetry:
    (a) Student answering prose rubric using a structured table receives full marks (and gets penalized if cells are swapped).
    (b) Student answering linearized table rubric using normal prose paragraphs receives full marks.
    """
    # Part (a): Prose rubric, student submits Markdown table
    payload_a = {
        "title": "Digital Forensics Platform Tools",
        "question": "State the live volatile memory acquisition tools across Windows and Unix platforms.",
        "model_answer": "Windows systems use winpmem for volatile memory acquisition. Unix systems use LiME for volatile memory acquisition.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Live Acquisition Tools",
                "description": "Windows systems use winpmem for volatile memory acquisition. Unix systems use LiME for volatile memory acquisition.",
                "max_marks": 5.0
            }
        ]
    }

    create_res_a = client.post("/assessments", json=payload_a)
    assert create_res_a.status_code == 201
    assessment_a = create_res_a.json()

    # Accurate student table
    student_table_valid = """
| Platform | Volatile Memory Acquisition Tool |
|---|---|
| Windows | winpmem |
| Unix | LiME |
"""
    res_valid = client.post("/grading/grade-text", json={
        "assessment_id": assessment_a["id"],
        "student_text": student_table_valid,
        "feedback_mode": "none"
    })
    assert res_valid.status_code == 200
    valid_data = res_valid.json()
    assert valid_data["total_score"] >= 4.5
    assert len(valid_data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert valid_data["requires_teacher_review"] is False

    # Swapped cells misconception table (Windows -> LiME, Unix -> winpmem)
    student_table_swapped = """
| Platform | Volatile Memory Acquisition Tool |
|---|---|
| Windows | LiME |
| Unix | winpmem |
"""
    res_swapped = client.post("/grading/grade-text", json={
        "assessment_id": assessment_a["id"],
        "student_text": student_table_swapped,
        "feedback_mode": "none"
    })
    assert res_swapped.status_code == 200
    swapped_data = res_swapped.json()
    assert swapped_data["total_score"] <= 2.50

    # Part (b): Linearized table rubric, student submits prose
    payload_b = {
        "title": "Linearized Table Assessment",
        "question": "Explain volatile acquisition tools.",
        "model_answer": "For Platform Windows, Tool is winpmem. For Platform Unix, Tool is LiME.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Linearized Tools Criteria",
                "description": "For Platform Windows, Tool is winpmem. For Platform Unix, Tool is LiME.",
                "max_marks": 5.0
            }
        ]
    }
    create_res_b = client.post("/assessments", json=payload_b)
    assert create_res_b.status_code == 201
    assessment_b = create_res_b.json()

    student_prose = (
        "On Windows systems, examiners deploy winpmem to capture volatile RAM. "
        "On Unix platforms, examiners utilize LiME for volatile memory collection."
    )
    res_prose = client.post("/grading/grade-text", json={
        "assessment_id": assessment_b["id"],
        "student_text": student_prose,
        "feedback_mode": "none"
    })
    assert res_prose.status_code == 200
    prose_data = res_prose.json()
    assert prose_data["total_score"] >= 4.5
    assert len(prose_data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert prose_data["requires_teacher_review"] is False


def test_assessment_parse_document_endpoint():
    """
    Tests POST /api/assessments/parse-document with a .docx containing 2 questions
    and a rubric table, asserting the returned JSON pre-fills title, questions,
    model answers, and criteria accurately.
    """
    doc = DocxDocument()
    doc.add_paragraph("Network Security and Incident Response Exam")

    doc.add_paragraph("Question 1:")
    doc.add_paragraph("Distinguish between passive and active network security attacks.")
    doc.add_paragraph("Model Answer: Passive attacks eavesdrop without altering data. Active attacks modify resources or disrupt service.")

    doc.add_paragraph("Question 2:")
    doc.add_paragraph("Explain the core purpose of symmetric and asymmetric encryption.")
    doc.add_paragraph("Model Answer: Symmetric encryption uses a shared key. Asymmetric encryption uses public and private key pairs.")

    # Rubric Table
    tbl = doc.add_table(rows=3, cols=3)
    tbl.rows[0].cells[0].text = "Criterion Name"
    tbl.rows[0].cells[1].text = "Expected Description"
    tbl.rows[0].cells[2].text = "Max Marks"

    tbl.rows[1].cells[0].text = "Attack Distinction"
    tbl.rows[1].cells[1].text = "Passive eavesdrops without alteration whereas active modifies data or disrupts systems."
    tbl.rows[1].cells[2].text = "5.0"

    tbl.rows[2].cells[0].text = "Cryptographic Keys"
    tbl.rows[2].cells[1].text = "Symmetric encryption uses single secret key while asymmetric uses paired public-private keys."
    tbl.rows[2].cells[2].text = "5.0"

    bio = io.BytesIO()
    doc.save(bio)
    docx_bytes = bio.getvalue()

    # 1. Test /api/assessments/parse-document with deterministic fallback
    with patch.object(AssessmentDocumentParser, "_call_ollama_parser", return_value=None):
        res_api = client.post(
            "/api/assessments/parse-document",
            files={"file": ("Network_Security_Exam.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert res_api.status_code == 200
        data_api = res_api.json()

        assert "title" in data_api
        assert "questions" in data_api
        assert len(data_api["questions"]) == 2
        assert data_api["total_marks"] == 10.0

        q1 = data_api["questions"][0]
        assert q1["question_order"] == 1
        assert "passive" in q1["question_text"].lower()
        assert "criteria" in q1
        assert len(q1["criteria"]) >= 1
        assert q1["criteria"][0]["max_marks"] > 0

        q2 = data_api["questions"][1]
        assert q2["question_order"] == 2
        assert "encryption" in q2["question_text"].lower()
        assert "criteria" in q2
        assert len(q2["criteria"]) >= 1
        assert q2["criteria"][0]["max_marks"] > 0

        # 2. Test /assessments/parse-document route directly
        res_direct = client.post(
            "/assessments/parse-document",
            files={"file": ("Network_Security_Exam.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        assert res_direct.status_code == 200
        data_direct = res_direct.json()
        assert data_direct["title"] == data_api["title"]
        assert len(data_direct["questions"]) == len(data_api["questions"])

    # 3. Test with mock LLM response
    mock_llm_payload = {
        "title": "LLM Parsed Assessment",
        "questions": [
            {
                "question_text": "What is cryptography?",
                "model_answer": "Study of secure communications.",
                "criteria": [
                    {"name": "Definition", "description": "Explains cryptography.", "max_marks": 5.0}
                ]
            }
        ]
    }
    with patch.object(AssessmentDocumentParser, "_call_ollama_parser", return_value=mock_llm_payload):
        res_llm = client.post(
            "/api/assessments/parse-document",
            files={"file": ("Crypto.txt", b"What is cryptography?\nModel Answer: Study of secure communications.", "text/plain")}
        )
        assert res_llm.status_code == 200
        data_llm = res_llm.json()
        assert data_llm["title"] == "LLM Parsed Assessment"
        assert len(data_llm["questions"]) == 1
        assert data_llm["total_marks"] == 5.0


def test_autofill_block_parsing_with_comparison_table_and_rubric_table():
    """
    Verifies that parsing a document containing both comparison tables in Model Answer
    and '| Criterion | Description | Marks |' rubric tables:
    1. Strips any 'Title:' or 'Assessment Title:' prefix from the extracted title.
    2. Preserves opening and closing prose paragraphs in Model Answer.
    3. Linearizes the comparison table into Model Answer without any rubric table contamination.
    4. Extracts all rubric criteria with exact names, descriptions, and marks allocations.
    """
    raw_doc = """Assessment Title: Advanced System Diagnostics

Question 1:
Explain volatile memory collection and comparison tools across platforms.

Model Answer:
Examiners must distinguish between Windows and Unix memory acquisition.

| Platform | Primary Tool | Volatility |
|---|---|---|
| Windows | winpmem | High |
| Unix | LiME | High |

The acquired memory dump provides raw process and kernel state.

| Criterion | Description | Marks |
|---|---|---|
| Memory Acquisition Distinction | Distinguishes between Windows winpmem and Unix LiME live capture | 5.0 |
| State Volatility Analysis | Explains volatile memory process and kernel state importance | 5.0 |

Question 2:
Explain network packet analysis.

Model Answer:
Packet analysis inspects protocol headers and payload data.

| Criterion | Description | Marks |
|---|---|---|
| Packet Inspection | Explains inspection of protocol headers and packet payload | 5.0 |
"""

    res = client.post(
        "/api/assessments/parse-document",
        files={"file": ("Diagnostics_Exam.txt", raw_doc.encode("utf-8"), "text/plain")}
    )
    assert res.status_code == 200
    data = res.json()

    # 1. Clean title without 'Assessment Title:' prefix
    assert data["title"] == "Advanced System Diagnostics"
    assert data["total_marks"] == 15.0
    assert len(data["questions"]) == 2

    # 2. Question 1 validation
    q1 = data["questions"][0]
    assert q1["question_order"] == 1
    assert "volatile memory collection" in q1["question_text"]

    # Model answer preserves opening prose, linearized comparison table, and closing prose
    assert "Examiners must distinguish between Windows and Unix memory acquisition." in q1["model_answer"]
    assert "For Platform Windows, Primary Tool is winpmem, and Volatility is High." in q1["model_answer"]
    assert "For Platform Unix, Primary Tool is LiME, and Volatility is High." in q1["model_answer"]
    assert "The acquired memory dump provides raw process and kernel state." in q1["model_answer"]

    # Model answer must NOT contain rubric table rows or "For Criterion"
    assert "For Criterion" not in q1["model_answer"]
    assert "Memory Acquisition Distinction" not in q1["model_answer"]

    # Rubric criteria must be accurately extracted
    assert len(q1["criteria"]) == 2
    assert q1["criteria"][0]["name"] == "Memory Acquisition Distinction"
    assert "Distinguishes between Windows winpmem" in q1["criteria"][0]["description"]
    assert q1["criteria"][0]["max_marks"] == 5.0
    assert q1["criteria"][1]["name"] == "State Volatility Analysis"
    assert q1["criteria"][1]["max_marks"] == 5.0
    assert q1["max_marks"] == 10.0

    # 3. Question 2 validation
    q2 = data["questions"][1]
    assert q2["question_order"] == 2
    assert "network packet analysis" in q2["question_text"]
    assert "Packet analysis inspects protocol headers and payload data." in q2["model_answer"]
    assert len(q2["criteria"]) == 1
    assert q2["criteria"][0]["name"] == "Packet Inspection"
    assert q2["criteria"][0]["max_marks"] == 5.0
    assert q2["max_marks"] == 5.0


def test_stored_grading_records_history_and_api():
    """
    Verifies that grading submissions persists records into SQLite and GET /api/grading/records
    returns the stored evaluation with its full multi-question/criterion breakdown.
    """
    from backend.app.routers.grading import _save_grading_record
    from backend.app.models import (
        GradingResponse,
        CriterionGradingResult,
        QuestionGradingResult,
        EvidenceCitation
    )

    assessment_payload = {
        "title": "Historical Records Validation Assessment",
        "question": "What is digital evidence?",
        "model_answer": "Digital evidence is information stored or transmitted in binary form.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Binary Form Evidence",
                "description": "Explains information stored or transmitted in binary form.",
                "max_marks": 5.0
            }
        ]
    }
    create_res = client.post("/assessments", json=assessment_payload)
    assert create_res.status_code == 201
    assessment = create_res.json()
    aid = assessment["id"]

    mock_response = GradingResponse(
        assessment_id=aid,
        assessment_title=assessment["title"],
        total_score=4.5,
        max_score=5.0,
        percentage=90.0,
        requires_teacher_review=False,
        review_reasons=[],
        overall_feedback="Excellent explanation of binary digital evidence.",
        criterion_breakdown=[
            CriterionGradingResult(
                criterion_name="Binary Form Evidence",
                max_marks=5.0,
                awarded_marks=4.5,
                awarded_percentage=90.0,
                reason="Accurately defines binary evidence.",
                feedback="All expected concepts were accurately addressed.",
                contradictions_detected=[],
                missing_concepts=[],
                present_concepts=["binary form", "digital evidence"]
            )
        ],
        question_results=[
            QuestionGradingResult(
                question_id="q1",
                question_order=1,
                question_text="What is digital evidence?",
                model_answer="Digital evidence is information stored in binary form.",
                student_answer_slice="Digital evidence comprises data in binary format.",
                awarded_marks=4.5,
                max_marks=5.0,
                percentage=90.0,
                question_feedback="Good answer.",
                criterion_breakdown=[
                    CriterionGradingResult(
                        criterion_name="Binary Form Evidence",
                        max_marks=5.0,
                        awarded_marks=4.5,
                        awarded_percentage=90.0,
                        reason="Accurately defines binary evidence.",
                        feedback="All expected concepts were accurately addressed."
                    )
                ]
            )
        ],
        citations=[
            EvidenceCitation(
                document_id="doc1",
                document_title="Forensics Guide",
                page_number=12,
                snippet="Digital evidence is binary information.",
                similarity_score=0.92
            )
        ],
        provenance={"test": True, "feedback_mode": "hybrid"}
    )

    _save_grading_record(
        res=mock_response,
        student_text="Digital evidence comprises data in binary format.",
        sub_type="file",
        sub_filename="Student_Submission_Record.txt"
    )

    # 1. Fetch via /api/grading/records
    res_api = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_api.status_code == 200
    records = res_api.json()
    assert len(records) >= 1

    rec = next(r for r in records if r["assessment_id"] == aid)
    assert rec["assessment_title"] == "Historical Records Validation Assessment"
    assert rec["submission_filename"] == "Student_Submission_Record.txt"
    assert rec["total_score"] == 4.5
    assert rec["max_score"] == 5.0
    assert rec["percentage"] == 90.0
    assert rec["overall_feedback"] == "Excellent explanation of binary digital evidence."
    assert rec["requires_teacher_review"] is False
    assert len(rec["criterion_breakdown"]) == 1
    assert rec["criterion_breakdown"][0]["criterion_name"] == "Binary Form Evidence"
    assert rec["question_results"] is not None
    assert len(rec["question_results"]) == 1
    assert rec["question_results"][0]["student_answer_slice"] == "Digital evidence comprises data in binary format."
    assert len(rec["citations"]) == 1
    assert rec["citations"][0]["document_title"] == "Forensics Guide"

    # 2. Fetch via /grading/records route alias
    res_direct = client.get(f"/grading/records?assessment_id={aid}")
    assert res_direct.status_code == 200
    records_direct = res_direct.json()
    assert len(records_direct) >= 1


def test_delete_grading_record_endpoints():
    """
    Verifies that DELETE /api/grading/records/{id} deletes individual records (returning 404 for missing IDs)
    and DELETE /api/grading/records deletes all records or filtered records by assessment_id.
    """
    import uuid
    from backend.app.routers.grading import _save_grading_record
    from backend.app.models import (
        GradingResponse,
        CriterionGradingResult
    )

    assessment_payload = {
        "title": "Deletion Test Assessment",
        "question": "Define hash function.",
        "model_answer": "A hash function maps data to a fixed-size string.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Mapping",
                "description": "Maps data to fixed-size string.",
                "max_marks": 5.0
            }
        ]
    }
    create_res = client.post("/assessments", json=assessment_payload)
    assert create_res.status_code == 201
    aid = create_res.json()["id"]

    rec1_id = str(uuid.uuid4())
    rec2_id = str(uuid.uuid4())

    mock1 = GradingResponse(
        id=rec1_id,
        assessment_id=aid,
        assessment_title="Deletion Test Assessment",
        total_score=5.0,
        max_score=5.0,
        percentage=100.0,
        requires_teacher_review=False,
        review_reasons=[],
        overall_feedback="Good",
        criterion_breakdown=[
            CriterionGradingResult(
                criterion_name="Mapping",
                max_marks=5.0,
                awarded_marks=5.0,
                awarded_percentage=100.0,
                reason="Accurate",
                feedback="All addressed."
            )
        ],
        citations=[],
        provenance={"test": True}
    )
    mock2 = GradingResponse(
        id=rec2_id,
        assessment_id=aid,
        assessment_title="Deletion Test Assessment",
        total_score=4.0,
        max_score=5.0,
        percentage=80.0,
        requires_teacher_review=False,
        review_reasons=[],
        overall_feedback="Good",
        criterion_breakdown=[
            CriterionGradingResult(
                criterion_name="Mapping",
                max_marks=5.0,
                awarded_marks=4.0,
                awarded_percentage=80.0,
                reason="Partially accurate",
                feedback="Minor points missed."
            )
        ],
        citations=[],
        provenance={"test": True}
    )

    _save_grading_record(mock1, "Sample answer 1", "text")
    _save_grading_record(mock2, "Sample answer 2", "text")

    # 1. Verify both records exist
    res_list = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_list.status_code == 200
    ids = [r["id"] for r in res_list.json()]
    assert rec1_id in ids and rec2_id in ids

    # 2. Delete individual record
    res_del1 = client.delete(f"/api/grading/records/{rec1_id}")
    assert res_del1.status_code == 200
    assert res_del1.json() == {"status": "deleted", "id": rec1_id}

    # 3. Verify rec1 is gone and rec2 remains
    res_list2 = client.get(f"/api/grading/records?assessment_id={aid}")
    ids2 = [r["id"] for r in res_list2.json()]
    assert rec1_id not in ids2
    assert rec2_id in ids2

    # 4. Deleting already-deleted record returns 404
    res_del_404 = client.delete(f"/api/grading/records/{rec1_id}")
    assert res_del_404.status_code == 404

    # 5. Bulk delete by assessment_id
    res_bulk = client.delete(f"/api/grading/records?assessment_id={aid}")
    assert res_bulk.status_code == 200
    assert res_bulk.json()["status"] == "deleted"
    assert res_bulk.json()["deleted_count"] >= 1

    # 6. Verify assessment records are now empty
    res_empty = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_empty.status_code == 200
    assert len(res_empty.json()) == 0

    # 7. Route alias delete
    res_alias_del = client.delete("/grading/records")
    assert res_alias_del.status_code == 200


def test_linearized_table_different_row_subjects_no_false_contradiction():
    """
    Verifies that multi-row linearized tables and multi-sentence prose answers covering different
    table rows earn full marks (15/15) without Row 1 (Forensic Duplicate) falsely contradicting
    Row 2 (Restored Image) or Row 3 (Mirror Image), while swapped values for the same row subject
    still trigger contradiction penalties.
    """
    payload = {
        "title": "Forensics Image Acquisition Specifications",
        "question": "Detail the destination and verification specifications across forensic duplicate, restored image, and mirror image types.",
        "model_answer": (
            "For Image Type Forensic Duplicate, Destination Drive is Raw Image, and Verification is SHA256 Hash. "
            "For Image Type Restored Image, Destination Drive is Physical Drive, and Verification is Media Verification. "
            "For Image Type Mirror Image, Destination Drive is Bit-Stream Copy, and Verification is Block Match."
        ),
        "total_marks": 15.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Forensic Duplicate Specification",
                "description": "For Image Type Forensic Duplicate, Destination Drive is Raw Image, and Verification is SHA256 Hash.",
                "max_marks": 5.0
            },
            {
                "name": "Restored Image Specification",
                "description": "For Image Type Restored Image, Destination Drive is Physical Drive, and Verification is Media Verification.",
                "max_marks": 5.0
            },
            {
                "name": "Mirror Image Specification",
                "description": "For Image Type Mirror Image, Destination Drive is Bit-Stream Copy, and Verification is Block Match.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    aid = create_res.json()["id"]

    # 1. Student submits valid linearized table rows
    valid_table_text = (
        "For Image Type Forensic Duplicate, Destination Drive is Raw Image, and Verification is SHA256 Hash.\n"
        "For Image Type Restored Image, Destination Drive is Physical Drive, and Verification is Media Verification.\n"
        "For Image Type Mirror Image, Destination Drive is Bit-Stream Copy, and Verification is Block Match."
    )
    res_table = client.post("/grading/grade-text", json={
        "assessment_id": aid,
        "student_text": valid_table_text,
        "feedback_mode": "none"
    })
    assert res_table.status_code == 200
    data_table = res_table.json()
    assert data_table["total_score"] >= 14.5
    assert data_table["requires_teacher_review"] is False
    for cb in data_table["criterion_breakdown"]:
        assert len(cb["contradictions_detected"]) == 0

    # 2. Student submits valid multi-sentence natural prose
    valid_prose = (
        "Forensic duplicate images write directly to raw image files and verify with SHA256 hashes. "
        "Restored images write directly to physical target drives and are validated with media verification. "
        "Mirror images create bit-stream copies verified through block match comparisons."
    )
    res_prose = client.post("/grading/grade-text", json={
        "assessment_id": aid,
        "student_text": valid_prose,
        "feedback_mode": "none"
    })
    assert res_prose.status_code == 200
    data_prose = res_prose.json()
    assert data_prose["total_score"] >= 14.0
    assert data_prose["requires_teacher_review"] is False
    for cb in data_prose["criterion_breakdown"]:
        assert len(cb["contradictions_detected"]) == 0

    # 3. Student submits swapped values misconception for Forensic Duplicate (writing to Physical Drive)
    swapped_text = (
        "For Image Type Forensic Duplicate, Destination Drive is Physical Drive, and Verification is Block Match.\n"
        "For Image Type Restored Image, Destination Drive is Physical Drive, and Verification is Media Verification.\n"
        "For Image Type Mirror Image, Destination Drive is Bit-Stream Copy, and Verification is Block Match."
    )
    res_swapped = client.post("/grading/grade-text", json={
        "assessment_id": aid,
        "student_text": swapped_text,
        "feedback_mode": "none"
    })
    assert res_swapped.status_code == 200
    data_swapped = res_swapped.json()
    crit1 = next(c for c in data_swapped["criterion_breakdown"] if "Forensic Duplicate" in c["criterion_name"])
    assert crit1["awarded_marks"] <= 2.5


