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

def test_feedback_mode_none_score_only(sample_assessment):
    """
    Confirms feedback_mode='none' (Score-Only Benchmarking):
    - Skips all LLM and deterministic feedback generation.
    - Yields empty strings for criterion feedback and overall feedback.
    - Sets ollama_calls=0 and feedback_generation_ms=0.0.
    - Preserves 100% numerical marks, concepts, contradictions, and review flag parity.
    """
    perfect_text = (
        "Memory-Based Learning is a machine-learning classification approach used in Natural Language Processing that uses stored training examples to help classify new instances. "
        "It stores training examples as feature vectors and compares them with new instances using a similarity or distance measure. "
        "MBL is a versatile and efficient learning framework. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    # 1. Baseline hybrid evaluation
    res_hybrid = GradingEngine.grade_submission(
        assessment=sample_assessment,
        student_text=perfect_text,
        feedback_mode="hybrid"
    )

    # 2. Score-only evaluation (guaranteeing Ollama is never touched)
    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client", side_effect=RuntimeError("Ollama must not be called")):
        res_none = GradingEngine.grade_submission(
            assessment=sample_assessment,
            student_text=perfect_text,
            feedback_mode="none"
        )

    # Parity assertions
    assert res_none.total_score == res_hybrid.total_score
    assert res_none.percentage == res_hybrid.percentage
    assert res_none.requires_teacher_review == res_hybrid.requires_teacher_review
    assert res_none.review_reasons == res_hybrid.review_reasons
    assert len(res_none.criterion_breakdown) == len(res_hybrid.criterion_breakdown)

    for c_none, c_hyb in zip(res_none.criterion_breakdown, res_hybrid.criterion_breakdown):
        assert c_none.criterion_name == c_hyb.criterion_name
        assert c_none.awarded_marks == c_hyb.awarded_marks
        assert c_none.awarded_percentage == c_hyb.awarded_percentage
        assert c_none.contradictions_detected == c_hyb.contradictions_detected
        assert c_none.missing_concepts == c_hyb.missing_concepts
        assert c_none.present_concepts == c_hyb.present_concepts
        assert c_none.feedback == "", f"Expected empty feedback in score-only mode, got: {c_none.feedback}"

    assert res_none.overall_feedback == ""
    assert res_none.provenance["feedback_mode"] == "none"
    assert res_none.provenance["ollama_calls"] == 0
    assert res_none.provenance["stage_durations"]["feedback_generation_ms"] == 0.0

    # 3. Test through HTTP API endpoint
    with patch("backend.app.services.llm_feedback.LLMFeedbackService.get_client", side_effect=RuntimeError("Ollama must not be called")):
        api_res = client.post(
            "/grading/grade-text",
            json={
                "assessment_id": sample_assessment.id,
                "student_text": perfect_text,
                "feedback_mode": "none"
            }
        )
    assert api_res.status_code == 200
    api_data = api_res.json()
    assert api_data["total_score"] == res_hybrid.total_score
    assert api_data["overall_feedback"] == ""
    assert all(c["feedback"] == "" for c in api_data["criterion_breakdown"])
    assert api_data["provenance"]["feedback_mode"] == "none"
    assert api_data["provenance"]["ollama_calls"] == 0
    assert api_data["provenance"]["stage_durations"]["feedback_generation_ms"] == 0.0


def test_clean_feedback_text_strips_preambles():
    """
    Verifies that _clean_feedback_text cleanly strips conversational preambles
    such as 'Here is the feedback:', 'Feedback:', 'Evaluation:', and 'Constructive feedback:'.
    """
    # 1. 'Here is the feedback:'
    raw_1 = "Here is the feedback: The student correctly identified asymmetric encryption."
    assert LLMFeedbackService._clean_feedback_text(raw_1) == "The student correctly identified asymmetric encryption."

    # 2. 'Feedback:'
    raw_2 = "Feedback: The student accurately explained the difference between passive and active attacks."
    assert LLMFeedbackService._clean_feedback_text(raw_2) == "The student accurately explained the difference between passive and active attacks."

    # 3. 'Evaluation:'
    raw_3 = "Evaluation: The answer lacks key distribution details."
    assert LLMFeedbackService._clean_feedback_text(raw_3) == "The answer lacks key distribution details."

    # 4. 'Constructive feedback:'
    raw_4 = "Constructive feedback: Consider clarifying how public keys are verified."
    assert LLMFeedbackService._clean_feedback_text(raw_4) == "Consider clarifying how public keys are verified."

    # 5. Text without preamble is untouched
    raw_5 = "The student demonstrated sound reasoning throughout the answer."
    assert LLMFeedbackService._clean_feedback_text(raw_5) == "The student demonstrated sound reasoning throughout the answer."


def test_clean_feedback_text_strips_echoed_rubric():
    """
    Verifies that when an LLM echoes the rubric description or criterion prompt
    before 'Feedback:' or 'Evaluation:', the echoed prefix is removed and only
    the evaluative feedback is retained.
    """
    raw = (
        "Passive attacks are distinguished from active attacks because they monitor traffic without alteration. "
        "Feedback: The student accurately explained passive attacks but omitted active attack mechanisms."
    )
    cleaned = LLMFeedbackService._clean_feedback_text(raw)
    assert cleaned == "The student accurately explained passive attacks but omitted active attack mechanisms."
    assert not cleaned.startswith("Passive attacks are distinguished")

    raw_eval = (
        "Target Rubric Requirement: Encryption cannot prevent traffic analysis. "
        "Evaluation: The student correctly noted that packet headers and volumes remain observable."
    )
    cleaned_eval = LLMFeedbackService._clean_feedback_text(raw_eval)
    assert cleaned_eval == "The student correctly noted that packet headers and volumes remain observable."


def test_dynamic_enumeration_labels():
    """
    Verifies dynamic enumeration label formatting:
    - If criterion name or description mentions 'application', uses 'Additional applications:'.
    - Otherwise, uses domain-neutral 'Missing elements:' (or 'Expected concepts:').
    """
    items = ["digital signatures", "hash functions", "access control"]

    # 1. Application-focused criterion
    label_app = LLMFeedbackService._format_missing_concepts(
        missing_items=items,
        criterion_name="NLP applications",
        criterion_description="Lists valid applications of NLP models."
    )
    assert label_app.startswith("Additional applications:")
    assert "digital signatures" in label_app

    # 2. Non-application criterion (e.g. security mechanisms)
    label_mech = LLMFeedbackService._format_missing_concepts(
        missing_items=items,
        criterion_name="Security mechanisms",
        criterion_description="Identifies specific security mechanisms such as encryption and hashing."
    )
    assert not label_mech.startswith("Additional applications:")
    assert label_mech.startswith("Missing elements:") or label_mech.startswith("Expected concepts:")
    assert "digital signatures" in label_mech

    # 3. Empty items returns empty string
    assert LLMFeedbackService._format_missing_concepts([]) == ""


def test_deterministic_zero_mark_criterion_feedback():
    """
    Verifies that a zero-mark criterion without contradictions or relevant text
    returns the standardized deterministic template:
    'The response does not address \'{criterion_name}\'. Expected concepts to address: {missing_concepts_str}.'
    """
    fb = LLMFeedbackService.generate_criterion_feedback(
        question="Explain network security mechanisms and services.",
        criterion_name="Security Services",
        criterion_description="Explains confidentiality, integrity, and availability services.",
        awarded_marks=0.0,
        max_marks=5.0,
        student_text="Network security mechanisms include packet filtering firewalls to monitor traffic.",
        contradictions=[],
        missing_concepts=["Confidentiality service", "Integrity service"],
        citations=[],
        present_concepts=[],
        relevant_excerpt="",
        feedback_mode="hybrid"
    )
    assert "The response does not address 'Security Services'" in fb
    assert "Expected concepts to address: Confidentiality service; Integrity service." in fb
    assert fb.endswith(".")
