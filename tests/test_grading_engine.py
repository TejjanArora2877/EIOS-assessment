import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db
from backend.app.models import AssessmentResponse, RubricCriterionResponse, EvidenceCitation
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.models_loader import ModelsLoader
from backend.app.services.retriever import HybridRetriever
from backend.app.services.llm_feedback import LLMFeedbackService

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def sample_assessment():
    payload = {
        "title": "Machine Learning Foundations Test",
        "question": "Explain Supervised Learning, how it uses training labels, and provide common examples.",
        "model_answer": "Supervised learning algorithms learn a mapping function from input features to output labels using labeled training examples. Common examples include regression and classification algorithms such as support vector machines and decision trees.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Definition of Supervised Learning",
                "description": "Learning a mapping function from inputs to outputs using labeled data.",
                "max_marks": 4.0
            },
            {
                "name": "Role of Training Labels",
                "description": "Uses ground-truth target labels to guide parameter optimization.",
                "max_marks": 3.0
            },
            {
                "name": "Common Algorithms and Examples",
                "description": "Classification and regression models such as SVMs, decision trees, or linear models.",
                "max_marks": 3.0
            }
        ]
    }
    res = client.post("/assessments", json=payload)
    assert res.status_code == 201
    return res.json()

def test_grading_empty_answer_rejected(sample_assessment):
    # Empty string
    res = client.post("/grading/grade-text", json={
        "assessment_id": sample_assessment["id"],
        "student_text": ""
    })
    assert res.status_code == 400
    assert "cannot be empty" in res.json()["detail"].lower()

    # Whitespace-only string
    res_ws = client.post("/grading/grade-text", json={
        "assessment_id": sample_assessment["id"],
        "student_text": "   \n\t  "
    })
    assert res_ws.status_code == 400
    assert "cannot be empty" in res_ws.json()["detail"].lower()

def test_grading_paraphrase_vs_partial_vs_contradictory(sample_assessment):
    aid = sample_assessment["id"]

    # 1. High-quality paraphrase
    paraphrase_text = (
        "Supervised learning is an ML approach that infers a function to predict output labels from input data "
        "using pairs of training instances. Ground truth labels provide the target signal for training. "
        "Standard applications include classification and regression tasks like decision trees and support vector machines."
    )
    res_para = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": paraphrase_text})
    assert res_para.status_code == 200
    para_data = res_para.json()

    # 2. Partial answer
    partial_text = "Supervised learning uses examples to learn. One example is decision trees."
    res_part = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": partial_text})
    assert res_part.status_code == 200
    part_data = res_part.json()

    # 3. Contradictory answer
    contradictory_text = (
        "Supervised learning never uses labeled training data or input-output pairs. "
        "It operates strictly without any supervision, labels, or optimization."
    )
    res_contra = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": contradictory_text})
    assert res_contra.status_code == 200
    contra_data = res_contra.json()

    # 4. Irrelevant answer
    irrelevant_text = (
        "Mitochondria are known as the powerhouse of the cell. They generate adenosine triphosphate through cellular respiration."
    )
    res_irrel = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": irrelevant_text})
    assert res_irrel.status_code == 200
    irrel_data = res_irrel.json()

    # Assertions on score hierarchy
    assert para_data["total_score"] > part_data["total_score"], f"Expected paraphrase ({para_data['total_score']}) > partial ({part_data['total_score']})"
    assert part_data["total_score"] > contra_data["total_score"], f"Expected partial ({part_data['total_score']}) > contradictory ({contra_data['total_score']})"
    assert contra_data["total_score"] <= 3.0, f"Expected contradictory score to be low, got {contra_data['total_score']}"
    assert irrel_data["total_score"] <= 2.5, f"Expected irrelevant score to be near zero, got {irrel_data['total_score']}"

    # Verify score ranges
    assert 0.0 <= para_data["total_score"] <= 10.0
    assert 0.0 <= part_data["total_score"] <= 10.0

def test_grading_reproducibility(sample_assessment):
    aid = sample_assessment["id"]
    test_text = "Supervised machine learning trains algorithms using input features and ground truth labels."

    res1 = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": test_text})
    res2 = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": test_text})

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["total_score"] == res2.json()["total_score"]
    assert res1.json()["percentage"] == res2.json()["percentage"]
    for c1, c2 in zip(res1.json()["criterion_breakdown"], res2.json()["criterion_breakdown"]):
        assert c1["awarded_marks"] == c2["awarded_marks"]


@pytest.fixture
def dummy_citations():
    return [
        EvidenceCitation(
            document_id="doc-sec-1",
            document_title="Security Handbook",
            page_number=1,
            snippet="Institutional security policy on cryptography and access controls.",
            similarity_score=0.95
        )
    ]


def test_regression_unrelated_statement_relevance_filter(monkeypatch, dummy_citations):
    """
    Test A — Unrelated statement:
    Expected concept: the limitation of encryption against traffic analysis.
    Student statement: "A Cartesian product may be useful in a carefully defined mathematical operation,
    but it usually indicates a mistake when joining related records."
    Mock the NLI result so it would classify the pair as contradictory if called.
    Verify that the relevance filter prevents an ungrounded contradiction.
    Verify that the sentence receives no credit for the unrelated concept and creates no contradiction penalty or teacher-review flag.
    """
    assessment = AssessmentResponse(
        id="test-assess-unrelated",
        title="Network Security - Traffic Analysis",
        question="Explain the limitations of encryption against traffic analysis attacks.",
        model_answer="Encryption obscures payload content but cannot hide packet timing, sizes, or traffic volume, making traffic analysis possible.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "A Cartesian product may be useful in a carefully defined mathematical operation, "
        "but it usually indicates a mistake when joining related records."
    )

    def mock_encode_texts(texts, **kwargs):
        embs = []
        for t in texts:
            t_low = t.lower()
            if any(w in t_low for w in ["cartesian", "sql", "joining", "records", "mathematical"]):
                embs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif any(w in t_low for w in ["traffic", "encrypt", "packet", "timing", "volume"]):
                embs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            else:
                embs.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return np.array(embs)

    nli_called_pairs = []
    def mock_predict_nli(pairs):
        nli_called_pairs.extend(pairs)
        # Hostile mock: return contradiction for every tested pair if called
        return [{"entailment": 0.0, "contradiction": 0.99, "neutral": 0.01} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Relevance filter prevented ungrounded contradiction
    assert crit_res.contradictions_detected == [], f"Expected no contradiction, got {crit_res.contradictions_detected}"

    # 2. Receives no credit for the unrelated concept and no contradiction penalty
    assert crit_res.awarded_marks == 0.0, f"Expected 0.0 marks for unrelated answer, got {crit_res.awarded_marks}"
    assert grading_res.total_score == 0.0
    assert "contradictory" not in crit_res.reason.lower()

    # 3. Teacher review is not flagged for contradiction
    assert grading_res.requires_teacher_review is False
    assert not any("contradiction" in r.lower() for r in grading_res.review_reasons)


def test_regression_genuine_contradiction_detected(monkeypatch, dummy_citations):
    """
    Test B — Genuine contradiction:
    Expected concept: confidentiality protects information from unauthorized disclosure.
    Student statement: "Confidentiality is not a useful objective because all university records should be visible to everyone."
    Mock the relevant NLI result as contradictory.
    Verify that the genuine contradiction is detected and that the appropriate criterion and review behavior are preserved.
    """
    assessment = AssessmentResponse(
        id="test-assess-contra",
        title="Information Security - CIA Triad",
        question="Explain the confidentiality principle in information security.",
        model_answer="Confidentiality ensures that sensitive information is protected from unauthorized access and disclosure.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Confidentiality Principle",
                description="Confidentiality protects information from unauthorized disclosure.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Confidentiality is not a useful objective because all university records should be visible to everyone."
    )

    def mock_encode_texts(texts, **kwargs):
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            if "not a useful objective" in prem:
                results.append({"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03})
            else:
                results.append({"entailment": 0.1, "contradiction": 0.1, "neutral": 0.8})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Genuine contradiction is detected
    assert len(crit_res.contradictions_detected) == 1
    assert "contradicts expected concept" in crit_res.contradictions_detected[0]

    # 2. Criterion behavior preserved (0 marks, reason cites contradiction)
    assert crit_res.awarded_marks == 0.0
    assert "contradictory" in crit_res.reason.lower()

    # 3. Teacher review flag preserved
    assert grading_res.requires_teacher_review is True
    assert any("contradiction detected" in r.lower() for r in grading_res.review_reasons)


def test_regression_valid_paraphrase_not_contradiction(monkeypatch, dummy_citations):
    """
    Test C — Valid paraphrase containing negation:
    Student's statement contains an explicit negation word ('does not') but is
    semantically compatible with the expected security concept ('Encryption cannot prevent traffic analysis').
    Mocking verifies that:
    1. Forward entailment is below 0.20 to deliberately exercise the contradiction-selection path.
    2. The candidate statement is selected for contradiction evaluation (negation present + conceptually related).
    3. Contradiction inference is actually called on this candidate pair.
    4. Because the statement is compatible, NLI contradiction is low (< 0.70) and it is NOT flagged as a contradiction.
    """
    assessment = AssessmentResponse(
        id="test-assess-paraphrase",
        title="Network Security - Traffic Analysis",
        question="Explain why encryption alone is insufficient against traffic analysis.",
        model_answer="Encryption hides payload content but does not conceal communication metadata such as packet timing and volume.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                description="Encryption cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    # Valid statement containing an explicit negation word ('does not') compatible with expected concept
    student_text = (
        "Network encryption alone does not prevent an adversary from analyzing packet volumes and transmission timing."
    )

    def mock_encode_texts(texts, **kwargs):
        return np.array([[0.0, 0.0, 1.0, 0.0] for _ in texts], dtype=np.float32)

    call_type_records = []

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            if prem == "Encryption cannot prevent traffic analysis.":
                # Reverse entailment: return low entailment so ent_score remains < 0.20
                call_type_records.append(("reverse_entailment", pairs))
                results.append({"entailment": 0.20, "contradiction": 0.05, "neutral": 0.75})
            elif "does not prevent" in prem:
                if not any(t == "forward_entailment" for t, _ in call_type_records):
                    # Forward entailment: return low score (< 0.20) to force
                    # execution to enter the contradiction-candidate selection path
                    call_type_records.append(("forward_entailment", pairs))
                    results.append({"entailment": 0.15, "contradiction": 0.05, "neutral": 0.80})
                else:
                    # Contradiction check: candidate with negation evaluated against proposition
                    call_type_records.append(("contradiction_check", pairs))
                    results.append({"entailment": 0.85, "contradiction": 0.05, "neutral": 0.10})
            else:
                results.append({"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Verify that the contradiction-selection path was actually exercised and candidate was evaluated
    assert any(t == "contradiction_check" for t, _ in call_type_records), "Expected contradiction inference path to be exercised"
    contra_calls = [pairs for t, pairs in call_type_records if t == "contradiction_check"]
    assert any("does not prevent" in p[0] for pairs in contra_calls for p in pairs), (
        f"Expected student candidate with negation to be evaluated in contra_pairs, got {contra_calls}"
    )

    # 2. Verify that despite containing negation, it is NOT flagged as a contradiction
    assert crit_res.contradictions_detected == []
    assert not any("contradiction" in r.lower() for r in grading_res.review_reasons)
    assert grading_res.requires_teacher_review is False


def test_regression_merged_window_unrelated_adjacent_negation(monkeypatch, dummy_citations):
    """
    Test Merged Windows:
    Verify that an unrelated adjacent sentence containing negation does not
    turn a relevant statement into a contradiction.
    """
    assessment = AssessmentResponse(
        id="test-assess-merged",
        title="Network Security",
        question="Explain encryption and its role in network security.",
        model_answer="Encryption protects payload confidentiality.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Payload Confidentiality",
                description="Encryption protects data payload confidentiality across networks.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Encryption protects data payload confidentiality across public networks. "
        "A Cartesian product is not recommended when querying relational databases."
    )

    def mock_encode_texts(texts, **kwargs):
        embs = []
        for t in texts:
            t_low = t.lower()
            if "cartesian" in t_low or "relational" in t_low:
                embs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif "payload" in t_low or "confidentiality" in t_low or "encryption" in t_low:
                embs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            else:
                embs.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return np.array(embs)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            if "not recommended" in prem:
                results.append({"entailment": 0.0, "contradiction": 0.95, "neutral": 0.05})
            else:
                results.append({"entailment": 0.90, "contradiction": 0.01, "neutral": 0.09})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # Verify no contradiction was triggered by the adjacent unrelated negation
    assert crit_res.contradictions_detected == []
    assert grading_res.requires_teacher_review is False
    assert crit_res.awarded_marks == 5.0


def test_regression_unrelated_high_entailment_gets_zero_credit(monkeypatch, dummy_citations):
    """
    Verifies that an unrelated statement (e.g. SQL joins) receiving a hostile/spurious
    high NLI forward entailment score (e.g. 0.92) against an unrelated rubric concept
    is blocked by the conceptual relevance gate, earning 0.0 marks and present_concepts == [].
    """
    assessment = AssessmentResponse(
        id="test-assess-unrelated-entailment",
        title="Network Security - Traffic Analysis",
        question="Explain the limitations of encryption against traffic analysis attacks.",
        model_answer="Encryption obscures payload content but cannot hide packet timing, sizes, or traffic volume, making traffic analysis possible.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "A Cartesian product may be useful in a carefully defined mathematical operation, "
        "but it usually indicates a mistake when joining related records."
    )

    def mock_encode_texts(texts, **kwargs):
        embs = []
        for t in texts:
            t_low = t.lower()
            if any(w in t_low for w in ["cartesian", "sql", "joining", "records", "mathematical"]):
                embs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif any(w in t_low for w in ["traffic", "encrypt", "packet", "timing", "volume"]):
                embs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            else:
                embs.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        return np.array(embs)

    def mock_predict_nli(pairs):
        # Hostile mock: return high entailment for all candidate pairs
        return [{"entailment": 0.92, "contradiction": 0.01, "neutral": 0.07} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Spurious high entailment on unrelated statement gets zero credit
    assert crit_res.awarded_marks == 0.0, f"Expected 0.0 marks for unrelated statement, got {crit_res.awarded_marks}"
    assert grading_res.total_score == 0.0

    # 2. Must NOT be added to present_concepts
    assert crit_res.present_concepts == [], f"Expected present_concepts == [], got {crit_res.present_concepts}"

    # 3. Must be recorded under missing_concepts
    assert len(crit_res.missing_concepts) > 0


def test_regression_valid_paraphrase_high_entailment_retains_credit(monkeypatch, dummy_citations):
    """
    Verifies that a genuine security paraphrase with high entailment passes
    the conceptual relevance gate, earns full criterion credit, and populates present_concepts.
    """
    assessment = AssessmentResponse(
        id="test-assess-valid-para-entailment",
        title="Network Security - Traffic Analysis",
        question="Explain the limitations of encryption against traffic analysis attacks.",
        model_answer="Encryption obscures payload content but cannot hide packet timing, sizes, or traffic volume, making traffic analysis possible.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Network encryption alone does not prevent an adversary from analyzing packet volumes and transmission timing."
    )

    def mock_encode_texts(texts, **kwargs):
        # Genuine paraphrase has high semantic similarity to security proposition
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        # High entailment for valid paraphrase
        return [{"entailment": 0.92, "contradiction": 0.01, "neutral": 0.07} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Valid paraphrase with high entailment passes gate and earns full credit
    assert crit_res.awarded_marks == 5.0, f"Expected 5.0 marks for valid paraphrase, got {crit_res.awarded_marks}"
    assert grading_res.total_score == 5.0

    # 2. Populates present_concepts
    assert len(crit_res.present_concepts) > 0, f"Expected present_concepts to be populated, got {crit_res.present_concepts}"
    assert crit_res.contradictions_detected == []


def test_regression_affirmative_misconception_detected(monkeypatch, dummy_citations):
    """
    Verifies that an affirmative misconception phrased positively without negation keywords
    (e.g., claiming network encryption completely prevents traffic analysis) is correctly
    detected via NLI contradiction scoring against the expected concept, receives 0.0 marks
    (p_mult = 0.0), is recorded in missing_concepts and contradictions_detected, and triggers
    teacher review.
    """
    assessment = AssessmentResponse(
        id="test-assess-affirmative-misconception",
        title="Network Security - Traffic Analysis",
        question="Explain why encryption alone is insufficient against traffic analysis attacks.",
        model_answer="Encryption obscures payload content but cannot hide packet timing, sizes, or traffic volume, making traffic analysis possible.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    # Affirmative misconception: stated positively with no negation keywords ('not', 'never', etc.)
    student_text = (
        "Network encryption completely prevents traffic analysis attacks by securing transmission channels."
    )

    def mock_encode_texts(texts, **kwargs):
        # High conceptual similarity to security topic
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            if "completely prevents" in prem:
                results.append({"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03})
            else:
                results.append({"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Affirmative misconception detected as contradiction
    assert len(crit_res.contradictions_detected) == 1
    assert "contradicts expected concept" in crit_res.contradictions_detected[0]

    # 2. Prevented unearned credit: 0.0 marks awarded (p_mult = 0.0)
    assert crit_res.awarded_marks == 0.0, f"Expected 0.0 marks for contradiction, got {crit_res.awarded_marks}"
    assert grading_res.total_score == 0.0
    assert "contradictory" in crit_res.reason.lower()

    # 3. Concepts tracking: not in present_concepts, recorded in missing_concepts
    assert crit_res.present_concepts == []
    assert len(crit_res.missing_concepts) > 0

    # 4. Teacher review flagged
    assert grading_res.requires_teacher_review is True
    assert any("contradiction detected" in r.lower() for r in grading_res.review_reasons)


def test_regression_long_answer_mixed_valid_and_contradictory_sentences(monkeypatch, dummy_citations):
    """
    Verifies that in a multi-sentence answer with both a valid correct sentence and a
    contradictory sentence:
    1. The valid sentence correctly earns positive credit on its matching criterion (not suppressed).
    2. The contradictory sentence is correctly flagged as a contradiction on its matching criterion (not masked by the valid sentence).
    3. Teacher review is triggered by the contradiction without corrupting the positive criterion.
    """
    assessment = AssessmentResponse(
        id="test-assess-mixed-long-answer",
        title="Network Security - Protocols and Attacks",
        question="Explain payload confidentiality and why encryption is insufficient against traffic analysis.",
        model_answer="Encryption protects payload confidentiality across public networks, but encryption alone cannot prevent traffic analysis.",
        total_marks=10.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Payload Confidentiality",
                description="Encryption protects data payload confidentiality across networks.",
                max_marks=5.0
            ),
            RubricCriterionResponse(
                id="crit-2",
                order_index=2,
                name="Traffic Analysis Limitation",
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    # Multi-sentence student answer containing:
    # Sentence 1: Valid statement addressing Criterion 1 (Payload Confidentiality).
    # Sentence 2: Affirmative misconception contradicting Criterion 2 (Traffic Analysis Limitation).
    student_text = (
        "Encryption protects data payload confidentiality across public networks. "
        "Furthermore, network encryption completely prevents traffic analysis attacks by securing transmission channels."
    )

    def mock_encode_texts(texts, **kwargs):
        embs = []
        for t in texts:
            t_low = t.lower()
            has_payload = any(w in t_low for w in ["payload", "confidentiality"])
            has_traffic = any(w in t_low for w in ["traffic", "analysis"])
            if has_payload and not has_traffic:
                embs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif has_traffic and not has_payload:
                embs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            elif has_payload and has_traffic:
                embs.append(np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32))
            else:
                embs.append(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        return np.array(embs)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            # Valid statement entails payload confidentiality
            if "protects data payload" in prem and "confidentiality" in hyp:
                results.append({"entailment": 0.92, "contradiction": 0.01, "neutral": 0.07})
            # Misconception contradicts traffic analysis limitation
            elif "completely prevents" in prem and "cannot prevent traffic" in hyp:
                results.append({"entailment": 0.02, "contradiction": 0.95, "neutral": 0.03})
            else:
                results.append({"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit1 = grading_res.criterion_breakdown[0]
    crit2 = grading_res.criterion_breakdown[1]

    # 1. Valid part (Criterion 1) is NOT suppressed: earns full positive credit
    assert crit1.awarded_marks == 5.0, f"Expected 5.0 marks for Criterion 1, got {crit1.awarded_marks}"
    assert crit1.contradictions_detected == []
    assert len(crit1.present_concepts) > 0
    assert "contradictory" not in crit1.reason.lower()

    # 2. Contradictory part (Criterion 2) is NOT masked by Sentence 1: correctly flagged as contradiction
    assert crit2.awarded_marks == 0.0, f"Expected 0.0 marks for Criterion 2, got {crit2.awarded_marks}"
    assert len(crit2.contradictions_detected) == 1
    assert "contradicts expected concept" in crit2.contradictions_detected[0]
    assert crit2.present_concepts == []
    assert len(crit2.missing_concepts) > 0
    assert "contradictory" in crit2.reason.lower()

    # 3. Overall submission score reflects positive credit on Criterion 1 and zero on Criterion 2
    assert grading_res.total_score == 5.0
    assert grading_res.percentage == 50.0

    # 4. Teacher review is flagged because of Criterion 2 contradiction
    assert grading_res.requires_teacher_review is True
    assert any("contradiction detected in traffic analysis limitation" in r.lower() for r in grading_res.review_reasons)
    assert not any("contradiction detected in payload confidentiality" in r.lower() for r in grading_res.review_reasons)


def test_regression_model_answer_grounding_boosts_paraphrase(monkeypatch, dummy_citations):
    """
    Verifies that a student using model-answer specific terminology (which is absent
    from the brief rubric criterion description) correctly earns proposition credit
    via model-answer grounding.
    """
    assessment = AssessmentResponse(
        id="test-assess-model-answer-grounding",
        title="Network Security - Traffic Analysis",
        question="Explain why encryption alone is insufficient against traffic analysis attacks.",
        # Detailed teacher model answer with specific mechanisms
        model_answer="Encryption obscures payload content but fails to conceal packet inter-arrival timing and transmission volume patterns.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Traffic Analysis Limitation",
                # Brief rubric description without model answer's specific terminology
                description="Encryption alone cannot prevent traffic analysis.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    # Student uses model-answer specific terminology ('inter-arrival timing', 'transmission volume patterns')
    student_text = (
        "An adversary can easily analyze packet inter-arrival timing and transmission volume patterns across the link."
    )

    def mock_encode_texts(texts, **kwargs):
        # Semantic similarity is high on traffic analysis & timing concepts
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            # High entailment when tested against the aligned model answer proposition
            if "timing" in hyp or "volume" in hyp:
                results.append({"entailment": 0.90, "contradiction": 0.01, "neutral": 0.09})
            # Low/borderline entailment when tested against the brief rubric description alone
            elif "cannot prevent traffic analysis" in hyp:
                results.append({"entailment": 0.25, "contradiction": 0.05, "neutral": 0.70})
            else:
                results.append({"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Earns full credit via model-answer auxiliary grounding (would fail on brief description alone)
    assert crit_res.awarded_marks == 5.0, f"Expected 5.0 marks via model-answer grounding, got {crit_res.awarded_marks}"
    assert grading_res.total_score == 5.0

    # 2. Confirmed in present_concepts
    assert len(crit_res.present_concepts) > 0
    assert crit_res.contradictions_detected == []
    assert grading_res.requires_teacher_review is False


def test_regression_rag_citation_evidence_verification(monkeypatch, dummy_citations):
    """
    Verifies that retrieved textbook citations support and verify
    borderline student concept credit into confirmed marks when semantic matching
    is borderline.
    """
    assessment = AssessmentResponse(
        id="test-assess-rag-verification",
        title="Operating Systems - Virtual Memory",
        question="Explain the working set model in virtual memory management.",
        model_answer="The working set model defines the set of pages actively referenced by a process over a recent parameter window.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Working Set Definition",
                description="Defines working set as the set of pages referenced by a process during a recent time window.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    # Retrieved textbook evidence snippet with specific technical terms
    textbook_citations = [
        EvidenceCitation(
            document_id="os-silberschatz-ch9",
            document_title="Operating System Concepts",
            page_number=385,
            snippet="The working set model uses a parameter delta to define the working-set window, tracking actively referenced pages to stabilize page fault frequency and prevent thrashing.",
            similarity_score=0.92
        )
    ]

    # Student answer has borderline NLI entailment against rubric, but strong lexical/semantic overlap with the textbook citation
    student_text = (
        "A process tracks referenced pages across a parameter delta window to stabilize page fault frequency."
    )

    def mock_encode_texts(texts, **kwargs):
        # Semantic similarity is moderate
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        # Borderline NLI entailment (0.35) against rubric proposition
        return [{"entailment": 0.35, "contradiction": 0.02, "neutral": 0.63} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: textbook_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # 1. Verification boost from textbook citation elevates borderline score to confirmed credit
    # Without citation verification, ent_score=0.35 with low overlap would yield < 2.0 marks
    assert crit_res.awarded_marks >= 4.0, f"Expected verified credit >= 4.0, got {crit_res.awarded_marks}"
    assert len(crit_res.present_concepts) > 0
    assert crit_res.contradictions_detected == []


def test_regression_student_a_paraphrase_not_contradiction(monkeypatch, dummy_citations):
    """
    Verifies that a valid student paraphrase with differing vocabulary (e.g. Student A on MBL)
    is not falsely flagged as a contradiction when NLI contradiction confidence is moderate/borderline
    (e.g. c_contra=0.52 vs c_ent=0.48), earning full/high marks and zero contradictions.
    """
    assessment = AssessmentResponse(
        id="test-assess-mbl-student-a",
        title="Natural Language Processing - Memory-Based Learning",
        question="Explain Memory-Based Learning (MBL) and how it classifies new instances in NLP.",
        model_answer=(
            "Memory-Based Learning (MBL) is a machine-learning classification approach that stores "
            "training examples and compares their feature vectors with new instances using distance metrics."
        ),
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Classification Mechanism",
                description="Memory-based learning stores training examples and compares feature vectors with new cases using distance metrics.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure."
    )

    def mock_encode_texts(texts, **kwargs):
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        # Moderate contradiction (0.51) that would previously trigger false positive under c_contra >= 0.50,
        # but with c_ent >= 0.40 and lack of high contradiction confidence (c_contra < 0.70)
        return [{"entailment": 0.48, "contradiction": 0.51, "neutral": 0.01} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # Must NOT be flagged as contradiction
    assert crit_res.contradictions_detected == [], f"Expected no contradiction, got {crit_res.contradictions_detected}"
    assert crit_res.awarded_marks >= 4.0, f"Expected high marks, got {crit_res.awarded_marks}"
    assert grading_res.requires_teacher_review is False


def test_regression_five_lens_cognitive_evaluation_active(monkeypatch, dummy_citations):
    """
    Verifies that the Five-Lens Cognitive Evaluation block (Understanding, Concept Coverage,
    Evidence Use, Coherence, Expression) and Shannon entropy are actively computed and populated
    in criterion breakdowns (uce_signals) and submission provenance.
    """
    assessment = AssessmentResponse(
        id="test-assess-five-lens",
        title="Computer Networks - Flow Control",
        question="Explain how the sliding window protocol achieves flow control in transport protocols.",
        model_answer="Sliding window protocols manage buffer occupancy by pacing unacknowledged transmissions between sender and receiver.",
        total_marks=10.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Window Mechanism",
                description="Sender maintains a window of allowable sequence numbers bounded by receiver window size.",
                max_marks=5.0
            ),
            RubricCriterionResponse(
                id="crit-2",
                order_index=2,
                name="Flow Control Objective",
                description="Prevents a fast sender from overflowing a slower receiver buffer.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "The sliding window protocol regulates transmission rates across network nodes. "
        "The sender dynamically adjusts the sequence window size according to receiver acknowledgments, "
        "ensuring that buffer capacities are not exceeded by rapid data bursts."
    )

    def mock_encode_texts(texts, **kwargs):
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        return [{"entailment": 0.85, "contradiction": 0.02, "neutral": 0.13} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    # 1. Criterion-level UCE signals are active
    for crit in grading_res.criterion_breakdown:
        assert crit.uce_signals is not None, f"uce_signals missing on {crit.criterion_name}"
        assert 0.0 <= crit.uce_signals.understanding <= 1.0
        assert 0.0 <= crit.uce_signals.clarity <= 1.0
        assert 0.0 <= crit.uce_signals.expression <= 1.0

    # 2. Provenance contains Shannon entropy
    provenance = grading_res.provenance
    assert "shannon_entropy" in provenance
    assert isinstance(provenance["shannon_entropy"], float)
    assert provenance["shannon_entropy"] > 0.0

    # 3. Provenance contains all five cognitive lenses
    assert "five_lens_cognitive_evaluation" in provenance
    five_lens = provenance["five_lens_cognitive_evaluation"]
    for lens in ["understanding", "concept_coverage", "evidence_use", "coherence", "expression"]:
        assert lens in five_lens, f"Lens '{lens}' missing from five_lens_cognitive_evaluation"
        val = five_lens[lens]
        assert isinstance(val, (int, float))
        assert 0.0 <= val <= 1.0


def test_regression_cross_criterion_sentence_not_flagged_as_contradiction(monkeypatch, dummy_citations):
    """
    Verifies that an essay containing sentences on both Privacy (Criterion 1) and
    Attacks (Criterion 2) grades each criterion independently without cross-criterion
    false contradictions wiping out credit.
    """
    assessment = AssessmentResponse(
        id="test-assess-cross-criterion",
        title="Information Security - Triad and Attacks",
        question="Explain privacy and confidentiality, and distinguish passive from active attacks.",
        model_answer=(
            "Privacy concerns how personal information is collected, stored, and disclosed. "
            "Passive attacks observe information without altering system resources, while active attacks alter resources or affect operation."
        ),
        total_marks=10.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Privacy and Confidentiality",
                description="Privacy concerns how personal information is collected, stored, and disclosed.",
                max_marks=5.0
            ),
            RubricCriterionResponse(
                id="crit-2",
                order_index=2,
                name="Passive vs Active Attacks",
                description="Passive attacks observe information without altering system resources, while active attacks alter resources.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Privacy concerns how personal information is collected, stored, and disclosed within the institution. "
        "Passive attacks observe information without altering system resources, while active attacks alter resources."
    )

    def mock_encode_texts(texts, **kwargs):
        embs = []
        for t in texts:
            t_low = t.lower()
            if "privacy" in t_low:
                embs.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            elif "attack" in t_low or "passive" in t_low:
                embs.append(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
            else:
                embs.append(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
        return np.array(embs, dtype=np.float32)

    def mock_predict_nli(pairs):
        results = []
        for prem, hyp in pairs:
            # If comparing Privacy to Privacy or Attacks to Attacks -> High entailment
            if ("privacy" in prem.lower() and "privacy" in hyp.lower()) or ("attack" in prem.lower() and "attack" in hyp.lower()):
                results.append({"entailment": 0.90, "contradiction": 0.02, "neutral": 0.08})
            # Cross-criterion comparison: if candidate about attacks is compared to privacy
            elif "attack" in prem.lower() and "privacy" in hyp.lower():
                results.append({"entailment": 0.05, "contradiction": 0.80, "neutral": 0.15})
            elif "privacy" in prem.lower() and "attack" in hyp.lower():
                results.append({"entailment": 0.05, "contradiction": 0.80, "neutral": 0.15})
            else:
                results.append({"entailment": 0.10, "contradiction": 0.10, "neutral": 0.80})
        return results

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    c1 = grading_res.criterion_breakdown[0]
    c2 = grading_res.criterion_breakdown[1]

    # Neither criterion should have false cross-criterion contradictions
    assert c1.contradictions_detected == [], f"C1 got contradictions: {c1.contradictions_detected}"
    assert c2.contradictions_detected == [], f"C2 got contradictions: {c2.contradictions_detected}"
    assert c1.awarded_marks == 5.0
    assert c2.awarded_marks == 5.0
    assert grading_res.total_score == 10.0
    assert grading_res.requires_teacher_review is False


def test_regression_instructional_rubric_description_cleaned(monkeypatch, dummy_citations):
    """
    Verifies that rubric descriptions phrased as commands ("Explain X and distinguish from Y. Award partial marks...")
    are cleaned into declarative propositions and correctly grade valid paraphrases.
    """
    assessment = AssessmentResponse(
        id="test-assess-instructional-cleaned",
        title="Network Security - Domains",
        question="Explain cybersecurity and network security, and distinguish cybersecurity from network security.",
        model_answer="Cybersecurity protects digital systems and networks, while network security focuses specifically on network communication channels.",
        total_marks=5.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="crit-1",
                order_index=1,
                name="Security Domains",
                description="Explain cybersecurity and network security, and distinguish cybersecurity from network security. Award partial marks if at least two domains are defined.",
                max_marks=5.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    student_text = (
        "Cybersecurity covers protection of information and digital devices in networked environments, "
        "while network security focuses specifically on protecting network communications and services."
    )

    def mock_encode_texts(texts, **kwargs):
        return np.array([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def mock_predict_nli(pairs):
        # When tested against cleaned declarative proposition, NLI sees valid entailment
        return [{"entailment": 0.88, "contradiction": 0.02, "neutral": 0.10} for _ in pairs]

    monkeypatch.setattr(ModelsLoader, "encode_texts", mock_encode_texts)
    monkeypatch.setattr(ModelsLoader, "predict_nli", mock_predict_nli)
    monkeypatch.setattr(HybridRetriever, "retrieve_evidence", lambda **kwargs: dummy_citations)
    monkeypatch.setattr(LLMFeedbackService, "generate_criterion_feedback", lambda **kwargs: "Deterministic feedback.")
    monkeypatch.setattr(LLMFeedbackService, "generate_overall_feedback", lambda **kwargs: "Overall feedback.")

    grading_res = GradingEngine.grade_submission(
        assessment=assessment,
        student_text=student_text,
        feedback_mode="hybrid"
    )

    crit_res = grading_res.criterion_breakdown[0]

    # Cleaned declarative proposition yields full credit and zero contradictions
    assert crit_res.contradictions_detected == []
    assert crit_res.awarded_marks >= 4.5
    assert len(crit_res.present_concepts) > 0
    # Ensure administrative sentence was stripped from target propositions
    for pc in crit_res.present_concepts:
        assert "award partial marks" not in pc.lower()


def test_regression_mbl_student_a_beats_or_equals_student_b(monkeypatch, dummy_citations):
    """
    Verifies using the exact student texts from MBL_Student that Student A achieves high marks (>= 8.0/10.0),
    Student A >= Student B, and Student D and E receive 0.0/10.0 (or <= 2.0/10.0).
    """
    assessment = AssessmentResponse(
        id="test-mbl-acceptance-hierarchy",
        title="MBL in Natural Language Processing Acceptance Exam",
        question=(
            "Explain Memory-Based Learning (MBL) as a learning approach for Natural Language Processing. "
            "Describe how it classifies new instances and list some NLP tasks in which it is used."
        ),
        model_answer=(
            "Memory-Based Learning (MBL) is a machine-learning classification approach used in Natural Language Processing. "
            "It compares the feature vectors of stored training examples with those of new instances using a similarity or distance measure. "
            "These comparisons help the system construct classification classes for new events or entities.\n\n"
            "MBL is a versatile and efficient learning framework and an alternative to statistical language modeling methods.\n\n"
            "Its applications in NLP include phonological and morphological analysis, part-of-speech tagging, shallow parsing, "
            "word disambiguation, phrasal chunking, named entity recognition, natural language generation, machine translation, and dialogue-act recognition."
        ),
        total_marks=10.0,
        doc_scope="ALL",
        criteria=[
            RubricCriterionResponse(
                id="mbl-crit-1",
                order_index=1,
                name="Definition of MBL",
                description="Definition of MBL as a machine-learning classification approach and alternative to statistical language modeling.",
                max_marks=2.0
            ),
            RubricCriterionResponse(
                id="mbl-crit-2",
                order_index=2,
                name="Classification mechanism",
                description="Compares feature vectors of stored training examples with new instances using similarity or distance measures to determine classes.",
                max_marks=4.0
            ),
            RubricCriterionResponse(
                id="mbl-crit-3",
                order_index=3,
                name="Characteristics",
                description="Versatile, flexible, and efficient learning framework.",
                max_marks=1.0
            ),
            RubricCriterionResponse(
                id="mbl-crit-4",
                order_index=4,
                name="NLP applications",
                description="Applications in part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, machine translation, disambiguation, or dialogue acts.",
                max_marks=3.0
            )
        ],
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z"
    )

    ans_a = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    ans_b = (
        "In MBL, a new input is classified by comparing its feature representation with those of examples already present in the training data. "
        "A measure of distance or similarity supports the formation of classes. The approach is considered efficient and versatile. "
        "It can be applied to tasks such as word disambiguation, dialogue-act recognition, language generation, and phonological or morphological analysis."
    )

    ans_d = (
        "Memory-based learning does not use stored training examples or feature vectors. "
        "It classifies new inputs only through fixed, manually written grammar rules. "
        "It is inefficient and has no applications in natural language processing."
    )

    ans_e = (
        "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars. "
        "Chlorophyll absorbs light energy, and oxygen is released as a by-product."
    )

    # We run actual ModelsLoader with feedback_mode="none" for fast execution
    res_a = GradingEngine.grade_submission(assessment=assessment, student_text=ans_a, feedback_mode="none")
    res_b = GradingEngine.grade_submission(assessment=assessment, student_text=ans_b, feedback_mode="none")
    res_d = GradingEngine.grade_submission(assessment=assessment, student_text=ans_d, feedback_mode="none")
    res_e = GradingEngine.grade_submission(assessment=assessment, student_text=ans_e, feedback_mode="none")

    # 1. Student A achieves high marks (>= 8.0/10.0)
    assert res_a.total_score >= 8.0, f"Expected Student A >= 8.0, got {res_a.total_score}"
    # 2. Student A >= Student B
    assert res_a.total_score >= res_b.total_score, f"Expected Student A ({res_a.total_score}) >= Student B ({res_b.total_score})"
    assert res_b.total_score >= 7.5, f"Expected Student B >= 7.5, got {res_b.total_score}"
    # 3. Student D (contradictory) receives low score (<= 2.0)
    assert res_d.total_score <= 2.0, f"Expected Student D <= 2.0, got {res_d.total_score}"
    # 4. Student E (irrelevant) receives low score (<= 1.0)
    assert res_e.total_score <= 1.0, f"Expected Student E <= 1.0, got {res_e.total_score}"
    # 5. Student A has zero contradictions
    assert all(len(c.contradictions_detected) == 0 for c in res_a.criterion_breakdown)
    assert res_a.requires_teacher_review is False
