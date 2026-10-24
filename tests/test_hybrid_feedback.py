import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.database import init_db
from backend.app.models import AssessmentResponse, RubricCriterionResponse
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.llm_feedback import LLMFeedbackService

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

@pytest.fixture
def sample_assessment():
    payload = {
        "title": "Test Hybrid Assessment",
        "question": "Explain Memory-Based Learning (MBL) as a learning approach for NLP and its applications.",
        "model_answer": (
            "Memory-Based Learning (MBL) is a machine-learning classification approach used in Natural Language Processing. "
            "It compares feature vectors of stored training examples with new instances using a similarity metric. "
            "It is versatile and efficient, and applied to part-of-speech tagging and machine translation."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Definition of MBL",
                "description": (
                    "Defines Memory-Based Learning (MBL) as a machine-learning classification approach used in Natural Language Processing (NLP). "
                    "The answer should explain that MBL uses stored training examples to help classify new instances."
                ),
                "max_marks": 2.0
            },
            {
                "name": "Classification mechanism",
                "description": "Explains how MBL stores training examples as feature vectors and compares them with new instances using a similarity or distance measure.",
                "max_marks": 4.0
            },
            {
                "name": "Characteristics",
                "description": "Identifies MBL as a versatile and efficient learning framework.",
                "max_marks": 1.0
            },
            {
                "name": "NLP applications",
                "description": "Lists valid NLP applications such as part-of-speech tagging, shallow parsing, named entity recognition, and machine translation.",
                "max_marks": 3.0
            }
        ]
    }
    res = client.post("/assessments/", json=payload)
    assert res.status_code == 201
    created_data = res.json()
    aid = created_data["id"]
    from backend.app.routers.assessments import _fetch_assessment_by_id
    assessment = _fetch_assessment_by_id(aid)
    yield assessment
    client.delete(f"/assessments/{aid}")

def test_1_fully_correct_answer(sample_assessment):
    """
    1. Fully correct answer:
    Generates concise, criterion-specific positive feedback deterministically without calling Ollama.
    Verifies feedback text, score invariance, and review flag preservation.
    """
    perfect_text = (
        "Memory-Based Learning is a machine-learning classification approach used in Natural Language Processing that uses stored training examples to help classify new instances. "
        "It stores training examples as feature vectors and compares them with new instances using a similarity or distance measure. "
        "MBL is a versatile and efficient learning framework. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        res_hybrid = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=perfect_text,
            feedback_mode="hybrid"
        )
        res_llm = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=perfect_text,
            feedback_mode="llm"
        )

        # Numerical score invariance
        assert res_hybrid.total_score == 10.0
        assert res_hybrid.total_score == res_llm.total_score
        assert res_hybrid.requires_teacher_review is False
        assert res_llm.requires_teacher_review is False

        # In hybrid mode, 0 Ollama calls made
        assert res_hybrid.provenance.get("ollama_calls") == 0

        # Feedback verification
        for crit in res_hybrid.criterion_breakdown:
            assert crit.awarded_marks == crit.max_marks
            assert "accurately and comprehensively addressed" in crit.feedback
            assert crit.feedback.endswith(".")
            assert not any(w in crit.feedback.lower() for w in ["missing", "omitted", "however", "partial"])

def test_2_irrelevant_answer(sample_assessment):
    """
    2. Irrelevant answer:
    Distinguishes an irrelevant submission (e.g. biology answer on an NLP exam)
    from a relevant one, generating concise deterministic feedback without calling Ollama.
    """
    irrelevant_text = (
        "Photosynthesis is the biological process by which plants convert light energy into chemical energy. "
        "Chlorophyll in the plant leaves absorbs sunlight and generates oxygen."
    )

    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        res_hybrid = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=irrelevant_text,
            feedback_mode="hybrid"
        )
        res_llm = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=irrelevant_text,
            feedback_mode="llm"
        )

        assert res_hybrid.total_score == 0.0
        assert res_hybrid.total_score == res_llm.total_score
        assert res_hybrid.requires_teacher_review is False
        assert res_llm.requires_teacher_review is False
        assert res_hybrid.provenance.get("ollama_calls") == 0

        for crit in res_hybrid.criterion_breakdown:
            assert crit.awarded_marks == 0.0
            assert "does not address the expected concepts" in crit.feedback
            assert crit.feedback.endswith(".")
            assert not any(w in crit.feedback.lower() for w in ["you explained", "partially explained", "correctly"])

def test_3_relevant_but_incomplete_answer_with_zero_mark_criterion(sample_assessment):
    """
    3. Relevant but incomplete answer that receives zero marks for at least one criterion:
    The student's submission IS relevant to the assessment overall (discusses MBL and NLP),
    but omits specific criteria (Classification mechanism and Characteristics).
    The feedback must accurately distinguish this from an irrelevant submission,
    specifically describing the missing concepts.
    """
    incomplete_text = (
        "Memory-Based Learning uses previous examples to help classify new data. "
        "It can be used for part-of-speech tagging and machine translation."
    )

    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "The answer identifies example-based classification, but omits the formal NLP machine learning classification definition.",
            "prompt_eval_count": 400,
            "eval_count": 25
        }
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        res_hybrid = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=incomplete_text,
            feedback_mode="hybrid"
        )
        res_llm = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=incomplete_text,
            feedback_mode="llm"
        )

        # Numerical score invariance between modes
        assert res_hybrid.total_score == res_llm.total_score
        assert res_hybrid.requires_teacher_review == res_llm.requires_teacher_review
        assert res_hybrid.requires_teacher_review is False

        crit_map = {c.criterion_name: c for c in res_hybrid.criterion_breakdown}

        # Classification mechanism received 0.0 marks because student did not mention vectors/similarity
        mech_crit = crit_map["Classification mechanism"]
        assert mech_crit.awarded_marks == 0.0
        # Must NOT claim the entire submission is irrelevant; must state the criterion was not addressed and list missing concepts
        assert "The response does not address 'Classification mechanism'" in mech_crit.feedback
        assert "Expected concepts to address" in mech_crit.feedback
        assert mech_crit.feedback.endswith(".")
        assert not any(w in mech_crit.feedback.lower() for w in ["you explained", "partially explained"])

        # Characteristics received 0.0 marks
        char_crit = crit_map["Characteristics"]
        assert char_crit.awarded_marks == 0.0
        assert "The response does not address 'Characteristics'" in char_crit.feedback
        assert "Expected concepts to address" in char_crit.feedback

        # Partial credit criteria (Definition or Applications) invoked the LLM path
        assert res_hybrid.provenance.get("ollama_calls") > 0

def test_4_contradictory_answer_requiring_teacher_review(sample_assessment):
    """
    4. Contradictory answer requiring teacher review:
    Preserves review warnings, contradiction evidence, and invokes the LLM feedback path.
    """
    contradictory_text = (
        "Memory-Based Learning does not use stored training examples or feature vectors. "
        "It classifies new inputs only through fixed, manually written grammar rules. "
        "It is inefficient and has no applications in natural language processing."
    )

    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "Review recommended: The submission states that MBL does not use training examples, contradicting core principles.",
            "prompt_eval_count": 450,
            "eval_count": 30
        }
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        res_hybrid = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=contradictory_text,
            feedback_mode="hybrid"
        )
        res_llm = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=contradictory_text,
            feedback_mode="llm"
        )

        assert res_hybrid.total_score == 0.0
        assert res_hybrid.total_score == res_llm.total_score
        assert res_hybrid.requires_teacher_review is True
        assert res_llm.requires_teacher_review is True
        assert len(res_hybrid.review_reasons) > 0
        assert res_hybrid.provenance.get("ollama_calls") > 0

def test_5_empty_input_behavior(sample_assessment):
    """
    5. Empty file / text behavior:
    Single-file mode rejects empty files with HTTP 400.
    Direct engine call with empty text produces deterministic empty notice without marks.
    """
    # 5a. HTTP Endpoint rejection
    res_text = client.post(
        "/grading/grade-text",
        json={"assessment_id": sample_assessment.id, "student_text": "   "}
    )
    assert res_text.status_code == 400

    empty_file_bytes = io.BytesIO(b"")
    res_file = client.post(
        "/grading/grade-file",
        data={"assessment_id": sample_assessment.id},
        files={"file": ("empty.txt", empty_file_bytes, "text/plain")}
    )
    assert res_file.status_code == 400

    # 5b. Direct engine call
    res_engine = GradingEngine.grade_submission(
        assessment=sample_assessment,
        student_text="",
        feedback_mode="hybrid"
    )
    assert res_engine.total_score == 0.0
    assert res_engine.provenance.get("ollama_calls") == 0
    for crit in res_engine.criterion_breakdown:
        assert crit.awarded_marks == 0.0
        assert "No answer submitted" in crit.feedback or "No response provided" in crit.feedback

def test_feedback_mode_llm_selectable(sample_assessment):
    """
    Confirms that setting feedback_mode='llm' preserves full-LLM evaluation
    for all criteria as a selectable fallback.
    """
    perfect_text = (
        "Memory-Based Learning is a machine-learning classification approach used in Natural Language Processing that uses stored training examples to help classify new instances. "
        "It stores training examples as feature vectors and compares them with new instances using a similarity or distance measure. "
        "MBL is a versatile and efficient learning framework. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "The submission accurately defines and explains all required aspects of this criterion.",
            "prompt_eval_count": 380,
            "eval_count": 20
        }
        mock_client.post.return_value = mock_resp
        mock_get_client.return_value = mock_client

        response = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=perfect_text,
            feedback_mode="llm"
        )

        assert response.total_score == 10.0
        assert response.provenance.get("ollama_calls") == 4
        assert mock_client.post.call_count == 4
