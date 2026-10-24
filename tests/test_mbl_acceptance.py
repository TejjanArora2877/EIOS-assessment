import os
import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db

client = TestClient(app)

PDF_TEXTBOOK_PATH = os.getenv("EIOS_TEST_TEXTBOOK", "")

@pytest.fixture(scope="module")
def setup_mbl_environment():
    """Module-level setup: initializes DB and indexes the textbook once for acceptance testing."""
    init_db()

    # Verify textbook exists
    if not PDF_TEXTBOOK_PATH or not os.path.isfile(PDF_TEXTBOOK_PATH):
        pytest.skip(
            "Optional MBL acceptance tests require a local textbook. "
            "Set EIOS_TEST_TEXTBOOK to its path to run them."
        )

    # 1. Ingest textbook once
    with open(PDF_TEXTBOOK_PATH, "rb") as f:
        upload_res = client.post(
            "/documents/upload",
            files={"file": ("NLP_AlexanderClark_Wiley.pdf", f, "application/pdf")},
            data={"title": "The Handbook of Computational Linguistics and NLP", "doc_type": "textbook"}
        )
    assert upload_res.status_code == 201, f"Failed to ingest textbook: {upload_res.text}"
    doc_data = upload_res.json()
    assert doc_data["status"] == "ready"
    assert doc_data["total_pages"] is not None and doc_data["total_pages"] > 700
    doc_id = doc_data["id"]

    # 2. Create MBL Assessment
    mbl_assessment_payload = {
        "title": "MBL in Natural Language Processing Acceptance Exam",
        "question": (
            "Explain Memory-Based Learning (MBL) as a learning approach for Natural Language Processing. "
            "Describe how it classifies new instances and list some NLP tasks in which it is used."
        ),
        "model_answer": (
            "Memory-Based Learning (MBL) is a machine-learning classification approach used in Natural Language Processing. "
            "It compares the feature vectors of stored training examples with those of new instances using a similarity or distance measure. "
            "These comparisons help the system construct classification classes for new events or entities.\n\n"
            "MBL is a versatile and efficient learning framework and an alternative to statistical language modeling methods.\n\n"
            "Its applications in NLP include phonological and morphological analysis, part-of-speech tagging, shallow parsing, "
            "word disambiguation, phrasal chunking, named entity recognition, natural language generation, machine translation, and dialogue-act recognition."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Definition of MBL",
                "description": "Definition of MBL as a machine-learning classification approach and alternative to statistical language modeling.",
                "max_marks": 2.0
            },
            {
                "name": "Classification mechanism",
                "description": "Compares feature vectors of stored training examples with new instances using similarity or distance measures to determine classes.",
                "max_marks": 4.0
            },
            {
                "name": "Characteristics",
                "description": "Versatile, flexible, and efficient learning framework.",
                "max_marks": 1.0
            },
            {
                "name": "NLP applications",
                "description": "Applications in part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, machine translation, disambiguation, or dialogue acts.",
                "max_marks": 3.0
            }
        ]
    }

    create_res = client.post("/assessments", json=mbl_assessment_payload)
    assert create_res.status_code == 201, f"Failed to create assessment: {create_res.text}"
    assessment_data = create_res.json()

    yield {
        "doc_id": doc_id,
        "assessment": assessment_data
    }

    # Cleanup: delete created assessment and document after test module
    client.delete(f"/assessments/{assessment_data['id']}")
    client.delete(f"/documents/{doc_id}")

def test_mbl_six_answers_pasted_text(setup_mbl_environment):
    aid = setup_mbl_environment["assessment"]["id"]

    # A. Correct paraphrase
    ans_a = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    # B. Equivalent paraphrase
    ans_b = (
        "In MBL, a new input is classified by comparing its feature representation with those of examples already present in the training data. "
        "A measure of distance or similarity supports the formation of classes. The approach is considered efficient and versatile. "
        "It can be applied to tasks such as word disambiguation, dialogue-act recognition, language generation, and phonological or morphological analysis."
    )

    # C. Partial answer
    ans_c = (
        "Memory-based learning uses previous examples to help classify new data. "
        "It can be used for part-of-speech tagging and machine translation."
    )

    # D. Contradictory answer
    ans_d = (
        "Memory-based learning does not use stored training examples or feature vectors. "
        "It classifies new inputs only through fixed, manually written grammar rules. "
        "It is inefficient and has no applications in natural language processing."
    )

    # E. Irrelevant answer
    ans_e = (
        "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars. "
        "Chlorophyll absorbs light energy, and oxygen is released as a by-product."
    )

    # F. Empty answer
    ans_f = ""

    # Evaluate A
    res_a = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_a}).json()
    # Evaluate B
    res_b = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_b}).json()
    # Evaluate C
    res_c = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_c}).json()
    # Evaluate D
    res_d = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_d}).json()
    # Evaluate E
    res_e = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_e}).json()
    # Evaluate F
    res_f = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_f}).json()

    print(f"\n--- MBL Acceptance Test Scores (Pasted Text) ---")
    print(f"Answer A (Correct paraphrase):    {res_a['total_score']} / 10.0 ({res_a['percentage']}%)")
    print(f"Answer B (Equivalent paraphrase): {res_b['total_score']} / 10.0 ({res_b['percentage']}%)")
    print(f"Answer C (Partial answer):         {res_c['total_score']} / 10.0 ({res_c['percentage']}%)")
    print(f"Answer D (Contradictory answer):  {res_d['total_score']} / 10.0 ({res_d['percentage']}%)")
    print(f"Answer E (Irrelevant answer):     {res_e['total_score']} / 10.0 ({res_e['percentage']}%)")
    print(f"Answer F (Empty answer):          {res_f['total_score']} / 10.0 ({res_f['percentage']}%)")

    # Score assertions
    assert res_a["total_score"] >= 8.0, f"Expected high score for A, got {res_a['total_score']}"
    assert res_b["total_score"] >= 7.5, f"Expected high score for B, got {res_b['total_score']}"
    assert 2.0 <= res_c["total_score"] < res_a["total_score"], f"Expected proportionate partial score for C, got {res_c['total_score']}"
    assert res_d["total_score"] <= 3.0, f"Expected low score for D (contradictory), got {res_d['total_score']}"
    assert res_e["total_score"] <= 2.0, f"Expected near-zero score for E (irrelevant), got {res_e['total_score']}"
    assert res_f["total_score"] == 0.0, f"Expected 0.0 for empty answer F, got {res_f['total_score']}"

    # Verify Citations from AlexanderClark textbook exist
    assert len(res_a["citations"]) > 0, "Expected institutional citations from textbook"
    top_citation = res_a["citations"][0]
    assert "The Handbook of Computational Linguistics" in top_citation["document_title"]
    assert top_citation["page_number"] is not None
    print(f"Verified Citation: {top_citation['document_title']}, Page {top_citation['page_number']}: '{top_citation['snippet'][:80]}...'")

def test_mbl_file_upload_grading(setup_mbl_environment):
    """Verifies that grading through file upload produces identical, assessment-aware results."""
    aid = setup_mbl_environment["assessment"]["id"]
    ans_a = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    file_bytes = io.BytesIO(ans_a.encode("utf-8"))
    res = client.post(
        "/grading/grade-file",
        data={"assessment_id": aid},
        files={"file": ("student_submission_a.txt", file_bytes, "text/plain")}
    )
    assert res.status_code == 200, res.text
    file_data = res.json()

    # Compare with pasted text evaluation
    pasted_data = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": ans_a}).json()

    assert file_data["total_score"] == pasted_data["total_score"]
    assert file_data["percentage"] == pasted_data["percentage"]
    assert len(file_data["criterion_breakdown"]) == len(pasted_data["criterion_breakdown"])
    print(f"\nVerified File Upload vs Pasted Text equivalence: Both scored {file_data['total_score']} / 10.0")
