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
from backend.app.services.text_extractor import TextExtractor

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


def test_delete_grading_records_batch_endpoint():
    """
    Verifies that POST /api/grading/records/delete-batch deletes selected records in bulk,
    rejects empty record_ids with 400, and works via the /grading/records/delete-batch route alias.
    """
    import uuid
    from backend.app.routers.grading import _save_grading_record
    from backend.app.models import GradingResponse, CriterionGradingResult

    assessment_payload = {
        "title": "Batch Deletion Test Assessment",
        "question": "What is volatility?",
        "model_answer": "Volatility refers to how quickly data is lost when power is removed.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Volatility Definition",
                "description": "Explains loss of data upon power disconnection.",
                "max_marks": 5.0
            }
        ]
    }
    create_res = client.post("/assessments", json=assessment_payload)
    assert create_res.status_code == 201
    aid = create_res.json()["id"]

    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    id3 = str(uuid.uuid4())

    for rid, score in [(id1, 5.0), (id2, 4.0), (id3, 3.0)]:
        mock_resp = GradingResponse(
            id=rid,
            assessment_id=aid,
            assessment_title="Batch Deletion Test Assessment",
            total_score=score,
            max_score=5.0,
            percentage=score * 20.0,
            requires_teacher_review=False,
            review_reasons=[],
            overall_feedback="Ok",
            criterion_breakdown=[
                CriterionGradingResult(
                    criterion_name="Volatility Definition",
                    max_marks=5.0,
                    awarded_marks=score,
                    awarded_percentage=score * 20.0,
                    reason="Ok",
                    feedback="Ok"
                )
            ],
            citations=[],
            provenance={"test": True}
        )
        _save_grading_record(mock_resp, f"Answer for {rid}", "text")

    # 1. Verify 3 records exist
    res_list = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_list.status_code == 200
    ids = {r["id"] for r in res_list.json()}
    assert {id1, id2, id3}.issubset(ids)

    # 2. Reject empty record_ids with 400
    res_empty_batch = client.post("/api/grading/records/delete-batch", json={"record_ids": []})
    assert res_empty_batch.status_code == 400

    # 3. Batch delete id1 and id2
    res_del_batch = client.post("/api/grading/records/delete-batch", json={"record_ids": [id1, id2]})
    assert res_del_batch.status_code == 200
    data_batch = res_del_batch.json()
    assert data_batch["status"] == "deleted"
    assert data_batch["deleted_count"] == 2

    # 4. Verify id1 and id2 are removed, id3 remains
    res_after = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_after.status_code == 200
    remaining_ids = {r["id"] for r in res_after.json()}
    assert id1 not in remaining_ids
    assert id2 not in remaining_ids
    assert id3 in remaining_ids

    # 5. Delete remaining id3 via route alias /grading/records/delete-batch
    res_alias = client.post("/grading/records/delete-batch", json={"record_ids": [id3]})
    assert res_alias.status_code == 200
    assert res_alias.json()["deleted_count"] == 1

    # 6. Verify empty for assessment
    res_final = client.get(f"/api/grading/records?assessment_id={aid}")
    assert res_final.status_code == 200
    assert len(res_final.json()) == 0


def test_extract_linearized_subject_leading_punctuation():
    """
    Verifies that GradingEngine._extract_linearized_subject correctly handles leading colons,
    hyphens, bullets, and stray whitespace, preventing false-positive contradiction flags.
    Also verifies AnswerSegmenter cleanly strips leading punctuation from sliced answers.
    """
    from backend.app.services.grading_engine import GradingEngine
    from backend.app.services.answer_segmenter import AnswerSegmenter

    # Test GradingEngine._extract_linearized_subject with various leading punctuation
    res1 = GradingEngine._extract_linearized_subject(": For Image Type Forensic Duplicate, Destination Drive is Raw Image.")
    assert res1 is not None
    assert res1 == ("image type", "forensic duplicate")

    res2 = GradingEngine._extract_linearized_subject("- For Platform Unix, Primary Tool is LiME.")
    assert res2 is not None
    assert res2 == ("platform", "unix")

    res3 = GradingEngine._extract_linearized_subject("  :  For Component Kernel, Layer is Ring 0.")
    assert res3 is not None
    assert res3 == ("component", "kernel")

    # With known_headers provided
    res4 = GradingEngine._extract_linearized_subject(
        ": For Image Type Forensic Duplicate, Destination Drive is Raw Image.",
        known_headers={"image type", "platform"}
    )
    assert res4 == ("image type", "forensic duplicate")

    # Test AnswerSegmenter clean slices
    questions = [
        {"id": "q1", "question_order": 1, "question_text": "What is memory volatility?", "model_answer": "Memory is volatile."},
        {"id": "q2", "question_order": 2, "question_text": "Detail image acquisition specifications.", "model_answer": "For Image Type Forensic Duplicate, Destination Drive is Raw Image."}
    ]
    submission_with_colons = (
        "Question 1: Memory is volatile and cleared upon power loss.\n\n"
        "Question 2: For Image Type Forensic Duplicate, Destination Drive is Raw Image."
    )
    slices = AnswerSegmenter.segment_submission(submission_with_colons, questions)
    assert "q1" in slices and "q2" in slices
    # The slice must NOT start with a colon
    assert not slices["q1"].startswith(":")
    assert not slices["q2"].startswith(":")
    assert slices["q2"].startswith("For Image Type Forensic Duplicate")


def test_multipage_pdf_table_stitching_and_continuity():
    """
    Verifies multi-page PDF assessment parsing:
    1. A rubric table split across pages without a repeated header stitches cleanly
       with 0 rubric rows leaked into model_answer or question_text.
    2. A multi-page Model Answer spanning across page breaks remains attached to its parent
       question rather than creating phantom questions.
    3. Wrapped cells and hyphenated line-breaks are cleanly joined.
    """
    import io
    import fitz
    from backend.app.services.table_linearizer import TableLinearizer

    # 1. Direct unit verification of cell cleaning and wrapped row merging
    assert TableLinearizer.clean_cell_text("subword-\nsegmentation") == "subword-segmentation"
    assert TableLinearizer.clean_cell_text("first line\nsecond line") == "first line second line"

    raw_rubric_rows = [
        ["Attention Mechanism", "Computes atten-\ntion weights", "5.0"],
        ["", "and projects value vectors.", ""]
    ]
    merged_rows = TableLinearizer._merge_wrapped_rows(["Criterion", "Description", "Marks"], raw_rubric_rows)
    assert len(merged_rows) == 1
    assert merged_rows[0][0] == "Attention Mechanism"
    assert "atten-tion weights and projects value vectors." in merged_rows[0][1]
    assert merged_rows[0][2] == "5.0"

    # 2. Build multi-page PDF in memory using fitz
    doc = fitz.open()

    # Page 1: Assessment Title, Q1, Model Answer, Comparison Table, Rubric Header + Row 1 + Wrapped Row
    p1 = doc.new_page()
    p1.insert_text(
        (50, 40),
        "Natural Language Processing Comprehensive Assessment\n\n"
        "Question 1: Explain the self-attention mechanism in Transformer models.\n"
        "Model Answer: The self-attention mechanism computes scaled dot-product similarity.\n",
        fontsize=9
    )

    # Comparison Table on Page 1 (Architecture comparison)
    for y in [120, 140, 160, 180]:
        p1.draw_line((50, y), (540, y))
    for x in [50, 170, 340, 540]:
        p1.draw_line((x, 120), (x, 180))

    p1.insert_text((55, 135), "Architecture", fontsize=9)
    p1.insert_text((175, 135), "Parallelization", fontsize=9)
    p1.insert_text((345, 135), "Context Window", fontsize=9)

    p1.insert_text((55, 155), "Transformer", fontsize=9)
    p1.insert_text((175, 155), "Fully Parallel", fontsize=9)
    p1.insert_text((345, 155), "Global Attention", fontsize=9)

    p1.insert_text((55, 175), "RNN", fontsize=9)
    p1.insert_text((175, 175), "Sequential", fontsize=9)
    p1.insert_text((345, 175), "Recurrent Hidden State", fontsize=9)

    p1.insert_text((50, 210), "This architecture eliminates the sequential recurrence bottleneck.\n", fontsize=9)

    # Rubric Table on Page 1: Header + Row 1 + Wrapped Row
    for y in [240, 260, 290, 320]:
        p1.draw_line((50, y), (540, y))
    for x in [50, 170, 480, 540]:
        p1.draw_line((x, 240), (x, 320))

    p1.insert_text((55, 255), "Criterion", fontsize=9)
    p1.insert_text((175, 255), "Description", fontsize=9)
    p1.insert_text((485, 255), "Marks", fontsize=9)

    p1.insert_text((55, 275), "Attention Mechanism", fontsize=9)
    p1.insert_text((175, 275), "Computes scaled dot-product atten-\ntion weights", fontsize=9)
    p1.insert_text((485, 275), "5.0", fontsize=9)

    # Wrapped row inside tall cell (empty first and last columns)
    p1.insert_text((175, 305), "and projects linear value representations.", fontsize=9)

    p1.insert_text((250, 750), "Page 1 of 3", fontsize=9)

    # Page 2: Rubric continuation row (NO repeated header!) + Q2 + start of Q2 Model Answer
    p2 = doc.new_page()
    for y in [50, 80]:
        p2.draw_line((50, y), (540, y))
    for x in [50, 170, 480, 540]:
        p2.draw_line((x, 50), (x, 80))

    p2.insert_text((55, 70), "Recurrent Contrast", fontsize=9)
    p2.insert_text((175, 70), "Contrasts parallel attention with sequential recurrence.", fontsize=9)
    p2.insert_text((485, 70), "5.0", fontsize=9)

    p2.insert_text(
        (50, 120),
        "Question 2: Describe subword tokenization algorithms.\n"
        "Model Answer: Subword tokenization balances vocabulary size and out-of-vocabulary handling.\n"
        "Byte-Pair Encoding iteratively merges the most frequent byte pairs.\n",
        fontsize=9
    )
    p2.insert_text((250, 750), "Page 2 of 3", fontsize=9)

    # Page 3: Q2 Model Answer continuation (cross-page prose paragraph) + Q2 Rubric Table
    p3 = doc.new_page()
    p3.insert_text(
        (50, 60),
        "WordPiece and SentencePiece further optimize subword segmentation using likelihood objectives.\n",
        fontsize=9
    )

    for y in [120, 150, 180]:
        p3.draw_line((50, y), (540, y))
    for x in [50, 170, 480, 540]:
        p3.draw_line((x, 120), (x, 180))

    p3.insert_text((55, 140), "Criterion", fontsize=9)
    p3.insert_text((175, 140), "Description", fontsize=9)
    p3.insert_text((485, 140), "Marks", fontsize=9)

    p3.insert_text((55, 170), "Subword Tokenization", fontsize=9)
    p3.insert_text((175, 170), "Explains frequency pair merging and subword vocabulary control.", fontsize=9)
    p3.insert_text((485, 170), "5.0", fontsize=9)

    p3.insert_text((250, 750), "Page 3 of 3", fontsize=9)

    pdf_bytes = doc.write()

    # 3. Test through POST /api/assessments/parse-document endpoint
    response = client.post(
        "/api/assessments/parse-document",
        files={"file": ("NLP_Comprehensive_Assessment.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    data = response.json()

    # Title & Overall marks
    assert data["title"] == "Natural Language Processing Comprehensive Assessment"
    assert data["total_marks"] == 15.0
    assert len(data["questions"]) == 2  # Exactly 2 questions, 0 phantom questions

    # Question 1 Verification
    q1 = data["questions"][0]
    assert q1["question_order"] == 1
    assert "self-attention mechanism in Transformer models" in q1["question_text"]

    # Model answer has comparison table and prose, but NO rubric content
    assert "scaled dot-product similarity" in q1["model_answer"]
    assert "For Architecture Transformer, Parallelization is Fully Parallel" in q1["model_answer"]
    assert "For Architecture RNN, Parallelization is Sequential" in q1["model_answer"]
    assert "eliminates the sequential recurrence bottleneck" in q1["model_answer"]
    assert "Attention Mechanism" not in q1["model_answer"]
    assert "Recurrent Contrast" not in q1["model_answer"]
    assert "For Criterion" not in q1["model_answer"]

    # Stitched Rubric Table for Question 1
    assert len(q1["criteria"]) == 2
    assert q1["criteria"][0]["name"] == "Attention Mechanism"
    assert "atten-tion weights" in q1["criteria"][0]["description"]
    assert "and projects linear value representations." in q1["criteria"][0]["description"]
    assert q1["criteria"][0]["max_marks"] == 5.0

    assert q1["criteria"][1]["name"] == "Recurrent Contrast"
    assert "Contrasts parallel attention with sequential recurrence" in q1["criteria"][1]["description"]
    assert q1["criteria"][1]["max_marks"] == 5.0
    assert q1["max_marks"] == 10.0

    # Question 2 Verification
    q2 = data["questions"][1]
    assert q2["question_order"] == 2
    assert "subword tokenization algorithms" in q2["question_text"]

    # Cross-page paragraph continuity in Model Answer
    assert "Subword tokenization balances vocabulary size" in q2["model_answer"]
    assert "Byte-Pair Encoding iteratively merges" in q2["model_answer"]
    assert "WordPiece and SentencePiece further optimize subword segmentation" in q2["model_answer"]

    # Rubric Criterion for Question 2
    assert len(q2["criteria"]) == 1
    assert q2["criteria"][0]["name"] == "Subword Tokenization"
    assert "frequency pair merging" in q2["criteria"][0]["description"]
    assert q2["criteria"][0]["max_marks"] == 5.0
    assert q2["max_marks"] == 5.0


def test_student_multipage_pdf_table_linearization():
    """
    Verifies that when a multi-page PDF student submission or reference material
    containing a comparison table split across page breaks without repeated headers
    is processed by TextExtractor.extract_from_file:
    1. Cross-page data tables stitch seamlessly.
    2. All rows from Page 1 and Page 2 are linearized using Page 1's column headers.
    3. Multi-page pagination metadata ('total_pages', 'pages') is accurately preserved.
    """
    import fitz

    doc = fitz.open()

    # Page 1: Student intro and first half of comparison table
    p1 = doc.new_page()
    p1.insert_text((50, 40), "Question 1: Compare disk imaging types in digital forensics.\n", fontsize=9)

    # Table on Page 1: 1 header row, 1 data row
    for y in [80, 110, 140]:
        p1.draw_line((50, y), (540, y))
    for x in [50, 200, 370, 540]:
        p1.draw_line((x, 80), (x, 140))

    p1.insert_text((55, 100), "Image Type", fontsize=9)
    p1.insert_text((205, 100), "Destination Drive", fontsize=9)
    p1.insert_text((375, 100), "Verification", fontsize=9)

    p1.insert_text((55, 130), "Forensic Duplicate", fontsize=9)
    p1.insert_text((205, 130), "Raw Image", fontsize=9)
    p1.insert_text((375, 130), "SHA256 Hash", fontsize=9)

    p1.insert_text((250, 750), "Page 1 of 2", fontsize=9)

    # Page 2: Continuation table rows without repeating column headers
    p2 = doc.new_page()
    for y in [50, 80, 110]:
        p2.draw_line((50, y), (540, y))
    for x in [50, 200, 370, 540]:
        p2.draw_line((x, 50), (x, 110))

    # Row 2 (continuation row on Page 2)
    p2.insert_text((55, 70), "Restored Image", fontsize=9)
    p2.insert_text((205, 70), "Physical Drive", fontsize=9)
    p2.insert_text((375, 70), "Media Verification", fontsize=9)

    # Row 3 (continuation row on Page 2)
    p2.insert_text((55, 100), "Mirror Image", fontsize=9)
    p2.insert_text((205, 100), "Bit-Stream Copy", fontsize=9)
    p2.insert_text((375, 100), "Block Match", fontsize=9)

    p2.insert_text((250, 750), "Page 2 of 2", fontsize=9)

    pdf_bytes = doc.write()

    # 1. Test TextExtractor.extract_from_file with bytes and filename
    result = TextExtractor.extract_from_file(pdf_bytes, filename="student_submission.pdf")

    assert result["total_pages"] == 2
    assert len(result["pages"]) == 2

    full_text = result["full_text"]
    assert "Question 1: Compare disk imaging types in digital forensics." in full_text
    assert "For Image Type Forensic Duplicate, Destination Drive is Raw Image, and Verification is SHA256 Hash." in full_text
    assert "For Image Type Restored Image, Destination Drive is Physical Drive, and Verification is Media Verification." in full_text
    assert "For Image Type Mirror Image, Destination Drive is Bit-Stream Copy, and Verification is Block Match." in full_text

    # Verify tables metadata
    assert len(result["tables"]) == 1
    table_meta = result["tables"][0]
    assert table_meta["headers"] == ["Image Type", "Destination Drive", "Verification"]
    assert len(table_meta["rows"]) == 3
    assert table_meta["rows"][0] == ["Forensic Duplicate", "Raw Image", "SHA256 Hash"]
    assert table_meta["rows"][1] == ["Restored Image", "Physical Drive", "Media Verification"]
    assert table_meta["rows"][2] == ["Mirror Image", "Bit-Stream Copy", "Block Match"]

    # 2. Test TextExtractor.extract_linearized_text convenience helper
    linearized_text = TextExtractor.extract_linearized_text(pdf_bytes, filename="student_submission.pdf")
    assert linearized_text == full_text


def test_multipage_model_answer_no_phantom_questions():
    """
    Verifies that when a question's Model Answer spans across multiple pages in a PDF:
    1. Zero phantom questions are created across page breaks.
    2. All paragraphs and comparison tables across pages are appended to the parent question's Model Answer.
    3. Trailing questions and all their rubric criteria remain correctly grouped without scattering.
    4. Total assessment marks accurately sum all criteria.
    """
    import fitz

    doc = fitz.open()

    # --- Page 1: Assessment Title, Question 1, Model Answer, Rubric Table (2 criteria = 20.0 marks) ---
    p1 = doc.new_page()
    p1.insert_text(
        (50, 40),
        "Assessment Title: Comprehensive Natural Language Processing Exam\n\n"
        "Question 1: Explain volatile memory acquisition techniques in forensic computing.\n"
        "Model Answer: Memory acquisition captures the state of active processes and kernel space.\n",
        fontsize=9
    )
    for y in [120, 145, 170, 195]:
        p1.draw_line((50, y), (540, y))
    for x in [50, 180, 470, 540]:
        p1.draw_line((x, 120), (x, 195))

    p1.insert_text((55, 138), "Criterion", fontsize=9)
    p1.insert_text((185, 138), "Description", fontsize=9)
    p1.insert_text((480, 138), "Marks", fontsize=9)

    p1.insert_text((55, 163), "Volatile Capture", fontsize=9)
    p1.insert_text((185, 163), "Captures active process state and volatile kernel data.", fontsize=9)
    p1.insert_text((480, 163), "10.0", fontsize=9)

    p1.insert_text((55, 188), "Kernel Integrity", fontsize=9)
    p1.insert_text((185, 188), "Validates kernel structures and active network sockets.", fontsize=9)
    p1.insert_text((480, 188), "10.0", fontsize=9)

    p1.insert_text((250, 750), "Page 1 of 5", fontsize=9)

    # --- Page 2: Question 2 & Model Answer (Page 1 of Q2) ---
    p2 = doc.new_page()
    p2.insert_text(
        (50, 40),
        "Question 2: Explain subword tokenization algorithms used in language models.\n\n"
        "Model Answer: Subword tokenization balances vocabulary size and out-of-vocabulary handling by segmenting rare words into frequent subword units.\n\n"
        "Byte-Pair Encoding iteratively merges the most frequent pairs of adjacent characters into single tokens.\n",
        fontsize=9
    )
    p2.insert_text((250, 750), "Page 2 of 5", fontsize=9)

    # --- Page 3: Question 2 Continuation (Page 2 of Q2: Prose + Comparison Table) ---
    p3 = doc.new_page()
    p3.insert_text(
        (50, 40),
        "Question 2 (Continued)\n\n"
        "WordPiece and SentencePiece further optimize subword segmentation using likelihood-based scoring criteria.\n",
        fontsize=9
    )
    for y in [100, 125, 150, 175]:
        p3.draw_line((50, y), (540, y))
    for x in [50, 180, 360, 540]:
        p3.draw_line((x, 100), (x, 175))

    p3.insert_text((55, 118), "Method", fontsize=9)
    p3.insert_text((185, 118), "Vocabulary", fontsize=9)
    p3.insert_text((365, 118), "OOV Strategy", fontsize=9)

    p3.insert_text((55, 143), "Word-Level", fontsize=9)
    p3.insert_text((185, 143), "Unbounded", fontsize=9)
    p3.insert_text((365, 143), "Special Unknown Token", fontsize=9)

    p3.insert_text((55, 168), "Subword-BPE", fontsize=9)
    p3.insert_text((185, 168), "Bounded", fontsize=9)
    p3.insert_text((365, 168), "Character Fallback", fontsize=9)

    p3.insert_text(
        (50, 200),
        "Subword segmentation ensures that vocabulary embeddings remain compact without losing lexical coverage.\n",
        fontsize=9
    )
    p3.insert_text((250, 750), "Page 3 of 5", fontsize=9)

    # --- Page 4: Question 2 Continuation (Page 3 of Q2: Prose + Rubric Table, 2 criteria = 20.0 marks) ---
    p4 = doc.new_page()
    p4.insert_text(
        (50, 40),
        "Question 2 Continued:\n\n"
        "During tokenization, unknown words are broken down into known morphemes to preserve semantic representations.\n",
        fontsize=9
    )
    for y in [100, 125, 150, 175]:
        p4.draw_line((50, y), (540, y))
    for x in [50, 150, 480, 540]:
        p4.draw_line((x, 100), (x, 175))

    p4.insert_text((55, 118), "Criterion", fontsize=9)
    p4.insert_text((155, 118), "Description", fontsize=9)
    p4.insert_text((490, 118), "Marks", fontsize=9)

    p4.insert_text((55, 143), "Subword Tradeoff", fontsize=9)
    p4.insert_text((155, 143), "Explains vocabulary compression versus OOV coverage.", fontsize=8)
    p4.insert_text((490, 143), "10.0", fontsize=9)

    p4.insert_text((55, 168), "Algorithm Comparison", fontsize=9)
    p4.insert_text((155, 168), "Contrasts Word-Level and Subword BPE characteristics.", fontsize=8)
    p4.insert_text((490, 168), "10.0", fontsize=9)

    p4.insert_text((250, 750), "Page 4 of 5", fontsize=9)

    # --- Page 5: Question 3 & Model Answer & Rubric Table (3 criteria = 20.0 marks) ---
    p5 = doc.new_page()
    p5.insert_text(
        (50, 40),
        "Question 3: Explain the self-attention mechanism and multi-head projection in Transformer architectures.\n\n"
        "Model Answer: Scaled dot-product self-attention maps query and key matrices to attention weights and projects linear value representations.\n",
        fontsize=9
    )
    for y in [120, 145, 170, 195, 220]:
        p5.draw_line((50, y), (540, y))
    for x in [50, 180, 470, 540]:
        p5.draw_line((x, 120), (x, 220))

    p5.insert_text((55, 138), "Criterion", fontsize=9)
    p5.insert_text((185, 138), "Description", fontsize=9)
    p5.insert_text((480, 138), "Marks", fontsize=9)

    p5.insert_text((55, 163), "Attention Scoring", fontsize=9)
    p5.insert_text((185, 163), "Computes query-key dot products scaled by square root of dimensionality.", fontsize=9)
    p5.insert_text((480, 163), "10.0", fontsize=9)

    p5.insert_text((55, 188), "Multi-Head Projection", fontsize=9)
    p5.insert_text((185, 188), "Projects linear queries keys and values into distinct subspace heads.", fontsize=9)
    p5.insert_text((480, 188), "5.0", fontsize=9)

    p5.insert_text((55, 213), "Context Aggregation", fontsize=9)
    p5.insert_text((185, 213), "Aggregates head outputs through final feed-forward linear projection.", fontsize=9)
    p5.insert_text((480, 213), "5.0", fontsize=9)

    p5.insert_text((250, 750), "Page 5 of 5", fontsize=9)

    pdf_bytes = doc.write()

    # Test through POST /api/assessments/parse-document
    response = client.post(
        "/api/assessments/parse-document",
        files={"file": ("NLP_MultiPage_Comprehensive_Exam.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 200
    parsed = response.json()

    # 1. Exactly 3 questions (0 phantom questions created on page breaks)
    assert len(parsed["questions"]) == 3
    assert parsed["title"] == "Comprehensive Natural Language Processing Exam"

    # 2. Total assessment marks equal 60.0
    assert parsed["total_marks"] == 60.0

    # 3. Question 1 validation
    q1 = parsed["questions"][0]
    assert q1["question_order"] == 1
    assert "volatile memory acquisition" in q1["question_text"]
    assert "Memory acquisition captures" in q1["model_answer"]
    assert len(q1["criteria"]) == 2
    assert q1["max_marks"] == 20.0

    # 4. Question 2 validation: model answer spans across all 3 pages
    q2 = parsed["questions"][1]
    assert q2["question_order"] == 2
    assert "subword tokenization algorithms" in q2["question_text"]
    assert "Subword tokenization balances vocabulary size" in q2["model_answer"]
    assert "Byte-Pair Encoding iteratively merges" in q2["model_answer"]
    assert "WordPiece and SentencePiece further optimize" in q2["model_answer"]
    assert "For Method Word-Level, Vocabulary is Unbounded, and OOV Strategy is Special Unknown Token." in q2["model_answer"]
    assert "For Method Subword-BPE, Vocabulary is Bounded, and OOV Strategy is Character Fallback." in q2["model_answer"]
    assert "During tokenization, unknown words are broken down into known morphemes" in q2["model_answer"]
    assert len(q2["criteria"]) == 2
    assert q2["max_marks"] == 20.0

    # 5. Question 3 validation: retains all 3 of its rubric criteria (20.0 marks)
    q3 = parsed["questions"][2]
    assert q3["question_order"] == 3
    assert "self-attention mechanism and multi-head projection" in q3["question_text"]
    assert "Scaled dot-product self-attention" in q3["model_answer"]
    assert len(q3["criteria"]) == 3
    assert q3["criteria"][0]["name"] == "Attention Scoring"
    assert q3["criteria"][0]["max_marks"] == 10.0
    assert q3["criteria"][1]["name"] == "Multi-Head Projection"
    assert q3["criteria"][1]["max_marks"] == 5.0
    assert q3["criteria"][2]["name"] == "Context Aggregation"
    assert q3["criteria"][2]["max_marks"] == 5.0
    assert q3["max_marks"] == 20.0


def test_markdown_table_with_math_cardinality_pipes():
    """
    Tests that pipe characters used in mathematical cardinality or norm expressions
    (e.g., |G|, |w|, |V_D|) within markdown table cells are not mistaken for column separators.
    """
    md_text = """| Grammar Class | Complexity |
| --- | --- |
| CFG | O(|G| n^3) |
| LR(k) | O(|V_D| + |w|) |"""

    linearized = TableLinearizer.linearize_markdown_tables(md_text)
    assert "For Grammar Class CFG, Complexity is O(|G| n^3)." in linearized
    assert "For Grammar Class LR(k), Complexity is O(|V_D| + |w|)." in linearized

    blocks = TableLinearizer.extract_document_blocks("test_table.txt", md_text.encode("utf-8"))
    assert len(blocks) == 1
    table_block = blocks[0]
    assert table_block["type"] == "table"
    assert table_block["headers"] == ["Grammar Class", "Complexity"]
    assert len(table_block["rows"]) == 2
    assert table_block["rows"][0] == ["CFG", "O(|G| n^3)"]
    assert table_block["rows"][1] == ["LR(k)", "O(|V_D| + |w|)"]

