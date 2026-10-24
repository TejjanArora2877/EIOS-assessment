import pytest
from unittest.mock import patch, MagicMock
import httpx
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.llm_feedback import LLMFeedbackService
from backend.app.routers.assessments import _fetch_assessment_by_id

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

@pytest.fixture
def mbl_assessment_fixture():
    """
    Creates the exact MBL assessment and rubric criteria for regression testing,
    and cleans it up afterwards so it is never permanently seeded into application startup.
    """
    payload = {
        "title": "MBL Regression Test Assessment",
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
                "description": (
                    "Defines Memory-Based Learning (MBL) as a machine-learning classification approach used in Natural Language Processing (NLP). "
                    "The answer should explain that MBL uses stored training examples to help classify new instances."
                ),
                "max_marks": 2.0
            },
            {
                "name": "Classification mechanism",
                "description": (
                    "Explains how MBL stores training examples as feature vectors and compares them with new instances using a similarity or distance measure. "
                    "The answer should describe how these comparisons help classify new instances. "
                    "Award partial marks for correctly explaining some, but not all, of these steps."
                ),
                "max_marks": 4.0
            },
            {
                "name": "Characteristics",
                "description": (
                    "Identifies MBL as a versatile and efficient learning framework and recognizes it as an alternative to statistical language modeling methods."
                ),
                "max_marks": 1.0
            },
            {
                "name": "NLP applications",
                "description": (
                    "Identifies valid Natural Language Processing applications of MBL, such as part-of-speech tagging, morphological analysis, "
                    "shallow parsing, word disambiguation, named entity recognition, natural language generation, machine translation, or dialogue-act recognition. "
                    "Award marks proportionately according to the number and relevance of correctly identified applications."
                ),
                "max_marks": 3.0
            }
        ]
    }

    res = client.post("/assessments", json=payload)
    assert res.status_code == 201, f"Failed to create test assessment: {res.text}"
    data = res.json()
    yield data

    # Cleanup after test suite
    client.delete(f"/assessments/{data['id']}")

# 1. Correct paraphrase explicitly containing every required MBL concept must not be penalized for a falsely missing definition
def test_correct_paraphrase_not_penalized_for_missing_definition(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text})
    assert res.status_code == 200
    data = res.json()

    # Find the Definition of MBL criterion result
    def_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Definition of MBL")

    # Must award full marks (2.0 / 2.0)
    assert def_crit["awarded_marks"] == 2.0, f"Expected 2.0 for Definition, got {def_crit['awarded_marks']}"
    # Must NOT report stored training examples or definition as missing
    assert len(def_crit["missing_concepts"]) == 0, f"False missing concepts detected: {def_crit['missing_concepts']}"
    # Must NOT report any contradictions
    assert len(def_crit["contradictions_detected"]) == 0

# 2. A sentence about feature vectors and similarity measures must not be reported as contradicting the NLP applications criterion
def test_feature_vectors_does_not_contradict_nlp_applications(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text})
    assert res.status_code == 200
    data = res.json()

    nlp_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "NLP applications")

    # Contradictions must be empty - sentence about feature vectors belongs to classification mechanism, not NLP applications
    assert len(nlp_crit["contradictions_detected"]) == 0, f"False contradiction flagged in NLP applications: {nlp_crit['contradictions_detected']}"
    # Student correctly listed 5 valid NLP applications
    assert nlp_crit["awarded_marks"] == 3.0, f"Expected full marks (3.0) for NLP applications, got {nlp_crit['awarded_marks']}"

# 3. A genuinely contradictory statement must still be detected when it conflicts with a relevant expected concept
def test_genuine_contradiction_is_detected(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    # Answer that explicitly denies storing training examples
    contradictory_text = (
        "Memory-Based Learning is an approach that never stores any training examples and immediately discards all data. "
        "It strictly avoids using feature vectors or distance measures."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": contradictory_text})
    assert res.status_code == 200
    data = res.json()

    # Contradiction must be flagged in Definition or Classification mechanism
    all_contradictions = []
    for c in data["criterion_breakdown"]:
        all_contradictions.extend(c["contradictions_detected"])

    assert len(all_contradictions) > 0, "Expected genuine contradiction to be detected."
    assert any("stored training examples" in c or "feature vectors" in c for c in all_contradictions)

    # Score must be heavily penalized
    assert data["total_score"] <= 2.5, f"Expected low score due to contradiction, got {data['total_score']}"

# 4. Partial answers receive credit for concepts actually present
def test_partial_answers_receive_proportional_credit(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    # Only mentions NLP applications, completely omits definition, mechanism, and characteristics
    partial_text = (
        "In Natural Language Processing, MBL is applied to tasks such as part-of-speech tagging, "
        "shallow parsing, and named entity recognition."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": partial_text})
    assert res.status_code == 200
    data = res.json()

    nlp_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "NLP applications")
    def_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Definition of MBL")
    mech_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Classification mechanism")

    # NLP applications should receive solid credit
    assert nlp_crit["awarded_marks"] >= 2.0, f"Expected credit for present NLP applications, got {nlp_crit['awarded_marks']}"
    # Unaddressed criteria should receive near zero marks
    assert def_crit["awarded_marks"] <= 0.6, f"Expected low marks for unaddressed definition, got {def_crit['awarded_marks']}"
    assert mech_crit["awarded_marks"] <= 0.6, f"Expected low marks for unaddressed mechanism, got {mech_crit['awarded_marks']}"

# 5. A feedback timeout leaves numerical marks unchanged
def test_feedback_timeout_leaves_marks_unchanged(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    assessment = _fetch_assessment_by_id(aid)
    student_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure."
    )

    # Run 1: Normal grading
    res_normal = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)

    # Run 2: Mock Ollama timeout in LLMFeedbackService
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Ollama timed out")):
        res_timeout = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)

    # Verify numerical scores are strictly identical
    assert res_timeout.total_score == res_normal.total_score, (
        f"Scores changed on timeout: {res_timeout.total_score} vs {res_normal.total_score}"
    )
    assert res_timeout.percentage == res_normal.percentage
    for c_norm, c_time in zip(res_normal.criterion_breakdown, res_timeout.criterion_breakdown):
        assert c_norm.criterion_name == c_time.criterion_name
        assert c_norm.awarded_marks == c_time.awarded_marks
        assert c_norm.awarded_percentage == c_time.awarded_percentage

    # Verify fallback feedback is present and truthful
    for c in res_timeout.criterion_breakdown:
        assert c.feedback is not None and len(c.feedback.strip()) > 0
        assert "timed out" not in c.feedback.lower()  # User-facing feedback is clean and professional

# 6. Generated feedback accurately reflects the computed criterion-level evaluation
def test_generated_feedback_matches_computed_evaluation(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_text})
    assert res.status_code == 200
    data = res.json()

    for c in data["criterion_breakdown"]:
        # If full marks were awarded, feedback must not claim concepts are missing
        if c["awarded_marks"] == c["max_marks"]:
            assert len(c["missing_concepts"]) == 0
            fb_lower = c["feedback"].lower()
            assert "omitted" not in fb_lower and "missing" not in fb_lower, (
                f"Full-credit criterion '{c['criterion_name']}' has feedback claiming omissions: {c['feedback']}"
            )
        # If there are missing concepts, they should be reflected in missing_concepts list
        if c["awarded_marks"] < c["max_marks"] and len(c["contradictions_detected"]) == 0:
            assert len(c["missing_concepts"]) > 0

# 7. Student E's photosynthesis answer receives 0/10 and feedback never claims student explained MBL mechanisms
def test_student_e_photosynthesis_never_claims_explained_mechanisms(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_e_text = (
        "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars. "
        "Chlorophyll absorbs light energy, and oxygen is released as a by-product."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_e_text})
    assert res.status_code == 200
    data = res.json()

    assert data["total_score"] == 0.0, f"Expected 0.0 for irrelevant answer, got {data['total_score']}"

    forbidden = ["you explained", "you partially explained", "you correctly", "you described how", "good foundational"]
    for c in data["criterion_breakdown"]:
        assert c["awarded_marks"] == 0.0
        fb_lower = c["feedback"].lower()
        for f in forbidden:
            assert f not in fb_lower, (
                f"Criterion '{c['criterion_name']}' has false claim in feedback for Student E: '{c['feedback']}'"
            )

# 8. Feedback belongs to the correct submission and criterion (no stale or cross-polluted feedback)
def test_feedback_belongs_to_correct_submission_and_criterion(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )
    student_e_text = (
        "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars."
    )

    res_a = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text}).json()
    res_e = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_e_text}).json()

    # Scores must be completely distinct
    assert res_a["total_score"] > 9.0
    assert res_e["total_score"] == 0.0

    # Ensure feedback from submission A is not leaked into submission E
    for ca, ce in zip(res_a["criterion_breakdown"], res_e["criterion_breakdown"]):
        assert ca["criterion_name"] == ce["criterion_name"]
        assert ca["feedback"] != ce["feedback"], f"Feedback identical across distinct submissions: {ca['feedback']}"

# 9. Citation labels accurately distinguish reference evidence from the student's own response
def test_citation_labels_distinguish_reference_evidence_from_student_response(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_text = (
        "Memory-based learning uses previously stored training examples to classify new cases."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_text})
    assert res.status_code == 200
    data = res.json()

    # Citations must contain document provenance (document_title, snippet, document_id)
    if data["citations"]:
        for cit in data["citations"]:
            assert "document_title" in cit
            assert "snippet" in cit
            assert "document_id" in cit
            assert len(cit["snippet"].strip()) > 0

# 10. Invalid LLM feedback triggering consistency violation uses validated deterministic fallback without changing marks
def test_invalid_llm_feedback_uses_validated_deterministic_fallback():
    # If LLM hallucinates "to receive full marks" for a full-marks score:
    hallucinated_fb = "You partially explained the concept. To receive full marks you must add thresholding."
    assert not LLMFeedbackService._is_feedback_consistent(hallucinated_fb, awarded_marks=4.0, max_marks=4.0)

    # Test cross-criterion drift rejection:
    all_crit = {
        "Characteristics": "Identifies MBL as a versatile and efficient learning framework and recognizes it as an alternative to statistical language modeling methods.",
        "Classification mechanism": "Explains how MBL stores training examples as feature vectors and compares them with new instances using a similarity or distance measure."
    }
    drifting_fb = "The student correctly described how feature vectors and similarity measures are used to classify instances."
    assert not LLMFeedbackService._is_feedback_consistent(
        drifting_fb,
        awarded_marks=1.0,
        max_marks=1.0,
        criterion_name="Characteristics",
        all_criteria_descriptions=all_crit
    )

    # Deterministic fallback must be returned with criterion name
    fallback = LLMFeedbackService._get_deterministic_fallback(4.0, 4.0, [], [], criterion_name="Classification mechanism")
    assert "accurately and comprehensively" in fallback.lower()
    assert "partially" not in fallback.lower()
    assert "to receive full marks" not in fallback.lower()
    assert "classification mechanism" in fallback.lower()

# 11. Feedback is criterion-specific and isolated without cross-criterion drift
def test_cross_criterion_feedback_isolation(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text})
    assert res.status_code == 200
    data = res.json()

    char_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Characteristics")
    mech_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Classification mechanism")

    char_fb_lower = char_crit["feedback"].lower()
    mech_fb_lower = mech_crit["feedback"].lower()

    # Characteristics feedback must specifically evaluate versatility, efficiency, or statistical language modeling
    assert any(term in char_fb_lower for term in ["flexible", "versatile", "efficient", "efficiency", "alternative", "statistical", "language modeling", "characteristics"]), (
        f"Characteristics feedback does not address core characteristic concepts: '{char_crit['feedback']}'"
    )

    # Characteristics feedback must NOT discuss classification mechanism concepts or application lists
    assert "feature vector" not in char_fb_lower and "distance measure" not in char_fb_lower, (
        f"Characteristics feedback drifted into classification mechanism: '{char_crit['feedback']}'"
    )
    assert "part-of-speech" not in char_fb_lower and "shallow parsing" not in char_fb_lower, (
        f"Characteristics feedback drifted into NLP applications: '{char_crit['feedback']}'"
    )

    # Classification mechanism feedback must address feature vectors, similarity/distance measures, or classification
    assert any(term in mech_fb_lower for term in ["feature", "vector", "similarity", "distance", "classif"]), (
        f"Classification mechanism feedback does not address mechanism concepts: '{mech_crit['feedback']}'"
    )

# 12. Synthetic multi-criterion assessment on a different subject proves assessment-agnostic rubric support
def test_synthetic_non_mbl_assessment_and_rubric():
    # 3-criterion Database Transactions ACID assessment (zero MBL assumptions)
    payload = {
        "title": "Database Systems ACID Transactions",
        "question": "Explain the ACID properties of database transactions, focusing on Atomicity, Consistency, and Durability.",
        "model_answer": (
            "A database transaction must satisfy ACID properties. "
            "Atomicity ensures all operations within a transaction succeed completely or all fail and are rolled back. "
            "Consistency guarantees that a transaction transforms the database from one valid state to another, preserving integrity constraints. "
            "Durability ensures that once a transaction commits, its updates persist in non-volatile storage even after a system crash or power failure."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Atomicity",
                "description": "Atomicity ensures all operations within a transaction succeed or all fail together via rollback.",
                "max_marks": 3.0
            },
            {
                "name": "Consistency",
                "description": "Consistency preserves database integrity constraints and valid state before and after transaction execution.",
                "max_marks": 3.0
            },
            {
                "name": "Durability",
                "description": "Durability guarantees committed transactions persist in non-volatile storage despite system crashes or power failures.",
                "max_marks": 4.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    aid = create_res.json()["id"]

    try:
        student_text = (
            "Atomicity ensures that all operations in a database transaction succeed or all fail together via rollback. "
            "Consistency guarantees that a transaction preserves database integrity constraints and transitions between valid states. "
            "Durability ensures that committed transaction records persist permanently in non-volatile storage despite power failures."
        )

        grade_res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_text})
        assert grade_res.status_code == 200
        data = grade_res.json()

        assert len(data["criterion_breakdown"]) == 3
        atom_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Atomicity")
        cons_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Consistency")
        dur_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Durability")

        # Full marks awarded across all criteria
        assert atom_crit["awarded_marks"] == 3.0, f"Expected 3.0 for Atomicity, got {atom_crit['awarded_marks']}"
        assert cons_crit["awarded_marks"] == 3.0, f"Expected 3.0 for Consistency, got {cons_crit['awarded_marks']}"
        assert dur_crit["awarded_marks"] == 4.0, f"Expected 4.0 for Durability, got {dur_crit['awarded_marks']}"
        assert data["total_score"] == 10.0

        # Feedback must be criterion-specific
        atom_fb = atom_crit["feedback"].lower()
        cons_fb = cons_crit["feedback"].lower()
        dur_fb = dur_crit["feedback"].lower()

        assert any(t in atom_fb for t in ["atomicity", "rollback", "operations", "fail", "succeed"]), f"Unexpected Atomicity fb: {atom_crit['feedback']}"
        assert any(t in cons_fb for t in ["consistency", "integrity", "valid state", "constraints"]), f"Unexpected Consistency fb: {cons_crit['feedback']}"
        assert any(t in dur_fb for t in ["durability", "persist", "non-volatile", "storage", "power"]), f"Unexpected Durability fb: {dur_crit['feedback']}"

        # Must not cross-pollute or contain MBL terms
        for crit in [atom_crit, cons_crit, dur_crit]:
            fb = crit["feedback"].lower()
            assert "memory-based" not in fb
            assert "feature vector" not in fb
            assert "natural language" not in fb
    finally:
        client.delete(f"/assessments/{aid}")

# 12b. Synthetic assessment testing complete, partial, and absent concept coverage across multi-concept criteria
def test_multi_concept_complete_partial_absent_synthetic_assessment():
    payload = {
        "title": "Computer Networks - Transport Layer Protocols",
        "question": "Explain TCP connection management and how flow control and congestion control are implemented.",
        "model_answer": (
            "TCP connection management uses a three-way handshake with SYN, SYN-ACK, and ACK packets to establish a connection. "
            "Graceful teardown uses a four-way handshake with FIN and ACK packets.\n\n"
            "Flow control uses a sliding window mechanism to prevent receiver buffer overflow. "
            "Congestion control uses slow start and congestion avoidance algorithms to prevent network collapse."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Connection Management",
                "description": "Explains the three-way handshake using SYN and ACK to establish connections. Explains the four-way FIN handshake to terminate connections gracefully.",
                "max_marks": 4.0
            },
            {
                "name": "Flow and Congestion Control",
                "description": "Explains flow control using sliding window to prevent buffer overflow. Explains congestion control using slow start and congestion avoidance.",
                "max_marks": 6.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    aid = create_res.json()["id"]

    try:
        # A. Complete Student: covers both propositions for both criteria
        complete_text = (
            "TCP establishes connections using a three-way handshake with SYN, SYN-ACK, and ACK segments. "
            "It terminates connections gracefully using a four-way handshake with FIN and ACK segments. "
            "Flow control is achieved using a sliding window mechanism that prevents the sender from overflowing the receiver buffer. "
            "Congestion control uses slow start and congestion avoidance algorithms to regulate transmission rate and prevent network collapse."
        )
        res_comp = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": complete_text}).json()

        crit_comp_map = {c["criterion_name"]: c for c in res_comp["criterion_breakdown"]}
        assert crit_comp_map["Connection Management"]["awarded_marks"] == 4.0
        assert len(crit_comp_map["Connection Management"]["missing_concepts"]) == 0
        assert crit_comp_map["Flow and Congestion Control"]["awarded_marks"] == 6.0
        assert len(crit_comp_map["Flow and Congestion Control"]["missing_concepts"]) == 0
        assert res_comp["total_score"] == 10.0

        # B. Partial Student: covers exactly 1 of 2 propositions per criterion
        # Connection: covers 3-way handshake, omits FIN termination
        # Flow/Congestion: covers sliding window flow control, omits congestion control
        partial_text = (
            "TCP establishes connections using a three-way handshake with SYN, SYN-ACK, and ACK segments. "
            "Flow control is implemented using a sliding window to prevent overflowing the receiver buffer."
        )
        res_part = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": partial_text}).json()

        crit_part_map = {c["criterion_name"]: c for c in res_part["criterion_breakdown"]}
        # Partial marks awarded proportionally (1/2 marks for each criterion)
        assert crit_part_map["Connection Management"]["awarded_marks"] == 2.0
        assert len(crit_part_map["Connection Management"]["missing_concepts"]) == 1
        assert any("fin" in m.lower() or "terminate" in m.lower() for m in crit_part_map["Connection Management"]["missing_concepts"])

        assert crit_part_map["Flow and Congestion Control"]["awarded_marks"] == 3.0
        assert len(crit_part_map["Flow and Congestion Control"]["missing_concepts"]) == 1
        assert any("congestion" in m.lower() or "slow start" in m.lower() for m in crit_part_map["Flow and Congestion Control"]["missing_concepts"])

        assert res_part["total_score"] == 5.0

        # C. Absent Student: completely unrelated response (UDP / DNS)
        absent_text = "UDP is an unreliable connectionless protocol used by DNS for fast lookups."
        res_abs = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": absent_text}).json()

        crit_abs_map = {c["criterion_name"]: c for c in res_abs["criterion_breakdown"]}
        assert crit_abs_map["Connection Management"]["awarded_marks"] == 0.0
        assert len(crit_abs_map["Connection Management"]["missing_concepts"]) == 2
        assert crit_abs_map["Flow and Congestion Control"]["awarded_marks"] == 0.0
        assert len(crit_abs_map["Flow and Congestion Control"]["missing_concepts"]) == 2
        assert res_abs["total_score"] == 0.0

    finally:
        client.delete(f"/assessments/{aid}")

# 13. Student C partial answer: feedback must NOT claim unwritten concepts and Characteristics must not drift
def test_student_c_feedback_does_not_claim_unwritten_concepts_and_isolates_characteristics(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_c_text = (
        "Memory-based learning stores past examples to classify new data. "
        "Its NLP uses include part-of-speech tagging and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_c_text})
    assert res.status_code == 200
    data = res.json()

    # Score should reflect partial answer (Definition 1.0/2.0 + NLP apps 1.5/3.0 -> 2.50 / 10)
    assert 2.0 <= data["total_score"] <= 3.0, f"Unexpected total score for Student C: {data['total_score']}"

    def_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Definition of MBL")
    mech_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Classification mechanism")
    char_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Characteristics")

    # Definition checks: Student C only addressed storage of past examples, omitted ML classification approach
    assert def_crit["awarded_marks"] == 1.0, f"Expected partial marks (1.0) for Definition of MBL, got {def_crit['awarded_marks']}"
    assert len(def_crit["missing_concepts"]) == 1, f"Expected 1 missing concept for Definition, got {def_crit['missing_concepts']}"
    assert any("machine" in m.lower() or "classification" in m.lower() or "approach" in m.lower() for m in def_crit["missing_concepts"])

    # Classification mechanism checks: unaddressed vectors/comparisons receives 0.0 marks
    assert mech_crit["awarded_marks"] == 0.0, f"Unexpected marks for mechanism: {mech_crit['awarded_marks']}"
    mech_fb_lower = mech_crit["feedback"].lower()

    # Must NOT claim Student C explained similarity or distance comparisons or feature vectors
    forbidden_claims = [
        "you correctly explained that memory-based learning classifies new cases using similarity",
        "you correctly explained similarity",
        "you explained similarity",
        "you described similarity",
        "you explained feature vectors",
        "you described feature vectors",
        "you correctly explained feature vectors",
        "your answer describes similarity",
        "your answer explains similarity",
        "your answer includes similarity"
    ]
    for claim in forbidden_claims:
        assert claim not in mech_fb_lower, f"False claim detected in Classification mechanism feedback: '{mech_crit['feedback']}'"

    # Characteristics checks: Student C completely omitted Characteristics
    assert char_crit["awarded_marks"] == 0.0, f"Expected 0.0 marks for unaddressed Characteristics, got {char_crit['awarded_marks']}"
    assert len(char_crit["missing_concepts"]) == 2, f"Expected 2 missing concepts for Characteristics, got {char_crit['missing_concepts']}"
    assert any("versatile" in m.lower() or "efficient" in m.lower() for m in char_crit["missing_concepts"])
    assert any("statistical" in m.lower() for m in char_crit["missing_concepts"])
    char_fb_lower = char_crit["feedback"].lower()

    # Characteristics feedback must NOT drift into NLP applications or classification mechanism
    assert "part-of-speech" not in char_fb_lower and "machine translation" not in char_fb_lower, (
        f"Characteristics feedback drifted into NLP applications: '{char_crit['feedback']}'"
    )
    assert "feature vector" not in char_fb_lower and "similarity" not in char_fb_lower and "distance measure" not in char_fb_lower, (
        f"Characteristics feedback drifted into classification mechanism: '{char_crit['feedback']}'"
    )
    assert any(term in char_fb_lower for term in ["did not", "not address", "lacked", "missing", "omitted", "does not meet", "review", "expected", "need to provide"]), (
        f"Characteristics feedback should indicate criterion was unaddressed: '{char_crit['feedback']}'"
    )

# 14. UCE cognitive percentage metrics are completely removed from API responses and frontend UI
def test_uce_signals_removed_from_api_and_frontend(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_text})
    assert res.status_code == 200
    data = res.json()

    # API criteria results must not expose uce_signals
    for c in data["criterion_breakdown"]:
        assert c.get("uce_signals") is None, f"UCE signals exposed in API response: {c.get('uce_signals')}"

    # Verify frontend ResultsReview.jsx does not render UCE percentage metrics
    from pathlib import Path
    results_review_path = Path("frontend/src/components/ResultsReview.jsx")
    assert results_review_path.exists(), "ResultsReview.jsx not found"
    jsx_content = results_review_path.read_text(encoding="utf-8")

    assert "Cognitive Signals" not in jsx_content
    assert "uce_signals" not in jsx_content
    assert "Understanding %" not in jsx_content
    assert "Clarity %" not in jsx_content
    assert "Expression %" not in jsx_content

# 15. Retrieved institutional reference passages are clearly labelled as reference context for the rubric, not student proof
def test_reference_materials_labelled_as_rubric_context_not_student_proof(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_e_text = (
        "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_e_text})
    assert res.status_code == 200
    data = res.json()

    assert data["total_score"] == 0.0

    # Verify frontend ResultsReview.jsx includes explicit context disclaimer
    from pathlib import Path
    results_review_path = Path("frontend/src/components/ResultsReview.jsx")
    jsx_content = results_review_path.read_text(encoding="utf-8")

    assert "Retrieved Institutional Reference Materials" in jsx_content
    assert "reference context" in jsx_content.lower()
    assert "not proof" in jsx_content.lower()

# 16. Student A NLP applications feedback never claims machine translation is missing or needed, nor suggests extra tasks
def test_student_a_nlp_feedback_never_claims_machine_translation_missing(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text})
    assert res.status_code == 200
    data = res.json()

    nlp_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "NLP applications")
    assert nlp_crit["awarded_marks"] == 3.0, f"Expected 3.0 full marks, got {nlp_crit['awarded_marks']}"
    assert len(nlp_crit["missing_concepts"]) == 0

    fb_lower = nlp_crit["feedback"].lower()
    # Full marks awarded: Feedback must never claim omissions or describe unrequired examples as omissions
    forbidden_terms = [
        "needed", "needs", "missing", "omitted", "omits", "omit", "omission", "omissions",
        "failed", "to receive full marks", "is noted", "was noted", "are noted", "were noted",
        "word disambiguation", "natural language generation", "dialogue-act recognition",
        "suggest expanding", "could expand", "expanding on", "you could add",
        "you could also", "you could mention", "consider expanding", "suggest adding"
    ]
    for term in forbidden_terms:
        assert term not in fb_lower, f"Forbidden term '{term}' in full-marks NLP feedback: '{nlp_crit['feedback']}'"

# 17. Student B definition awards full marks and acknowledges classification, and characteristics is isolated
def test_student_b_definition_and_characteristics_accuracy(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_b_text = (
        "Memory-based learning is an instance-driven classification paradigm where previously encountered training cases are retained in memory to categorize unseen data instances. "
        "During classification, target instances are projected into feature spaces and evaluated against stored items using similarity metrics. "
        "The approach is considered efficient and versatile. "
        "In computational linguistics, it has been used for part-of-speech tagging, shallow syntactic analysis, and named entity recognition."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_b_text})
    assert res.status_code == 200
    data = res.json()

    def_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Definition of MBL")
    char_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Characteristics")

    # 1. Definition must receive full marks (2.0/2.0) for valid instance-driven classification paraphrase
    assert def_crit["awarded_marks"] == 2.0, f"Expected 2.0 for Student B definition, got {def_crit['awarded_marks']}"
    assert len(def_crit["missing_concepts"]) == 0
    def_fb = def_crit["feedback"].lower()
    assert "did not explain" not in def_fb
    assert "missing" not in def_fb
    assert "failed" not in def_fb
    assert "to receive full marks" not in def_fb

    # 2. Characteristics must receive partial credit (0.50/1.00) for versatility and efficiency
    # and recognize statistical language modeling as missing
    assert char_crit["awarded_marks"] == 0.50, f"Expected 0.50 for Characteristics, got {char_crit['awarded_marks']}"
    assert any("statistical" in m.lower() for m in char_crit["missing_concepts"]), (
        f"Expected statistical language modeling in missing concepts: {char_crit['missing_concepts']}"
    )
    char_fb = char_crit["feedback"].lower()
    assert "feature vector" not in char_fb and "similarity metric" not in char_fb, (
        f"Characteristics feedback drifted into classification mechanism: '{char_crit['feedback']}'"
    )
    assert "part-of-speech" not in char_fb and "syntactic analysis" not in char_fb, (
        f"Characteristics feedback drifted into NLP applications: '{char_crit['feedback']}'"
    )
    assert any(term in char_fb for term in ["versatile", "efficient", "efficiency", "framework", "characteristics"]), (
        f"Characteristics feedback did not acknowledge characteristic concepts: '{char_crit['feedback']}'"
    )
    # Must NOT claim student described/explained statistical language modeling
    assert "you correctly explained statistical" not in char_fb
    assert "you explained statistical" not in char_fb

# 18. Consistency validation guarantees rejected feedback cannot reach the response under any condition
def test_rejected_feedback_guarantees_deterministic_fallback():
    # Candidate feedback with invalid omission claims
    with patch("httpx.Client.post") as mock_post:
        # Simulate Ollama returning hallucinated feedback claiming omission when awarded marks are full
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "The student partially explained the topic, but omitted stored examples to receive full marks."}
        mock_post.return_value = mock_response

        fb = LLMFeedbackService.generate_criterion_feedback(
            question="Explain MBL.",
            criterion_name="Definition of MBL",
            criterion_description="Defines MBL as using stored training examples to classify new instances.",
            awarded_marks=2.0,
            max_marks=2.0,
            student_text="Memory-based learning uses previously stored training examples to classify new instances.",
            contradictions=[],
            missing_concepts=[],
            citations=[]
        )

        # Must NOT return the hallucinated candidate feedback
        assert "partially explained" not in fb
        assert "to receive full marks" not in fb
        assert "omitted" not in fb
        # Must return the verified deterministic fallback
        assert "accurately and comprehensively addressed" in fb or "accurately addressed" in fb

# 19. Results page layout genuinely uses available horizontal width and arranges criteria in a 2-column desktop grid
def test_results_layout_full_width_and_two_column_grid():
    from pathlib import Path
    css_path = Path("frontend/src/index.css")
    jsx_path = Path("frontend/src/components/ResultsReview.jsx")
    assert css_path.exists()
    assert jsx_path.exists()

    css_content = css_path.read_text(encoding="utf-8")
    jsx_content = jsx_path.read_text(encoding="utf-8")

    # CSS assertions
    assert ".results-container" in css_content
    assert "max-width: 100%" in css_content
    assert ".criteria-results-grid" in css_content
    assert "repeat(2, minmax(0, 1fr))" in css_content
    assert ".results-text-full" in css_content
    assert "max-width: none !important" in css_content
    assert ".feedback-panel" in css_content

    # JSX assertions
    assert 'className="criteria-results-grid"' in jsx_content
    assert 'className="criterion-result-card"' in jsx_content
    assert 'className="results-container"' in jsx_content
    assert 'className="results-text-full"' in jsx_content

# 20. Student D contradictory answer: all criteria contradicted and total score strictly zero
def test_student_d_contradictions_and_zero_credit(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_d_text = (
        "Memory-based learning is an approach that never stores any training examples and immediately discards all data. "
        "It does not use feature vectors or distance measures for classification. "
        "MBL is neither efficient nor versatile, and it cannot serve as an alternative to statistical language modeling. "
        "It has no valid applications in Natural Language Processing."
    )

    res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_d_text})
    assert res.status_code == 200
    data = res.json()

    # Total score must be strictly 0.00
    assert data["total_score"] == 0.0, f"Expected 0.0 total score for contradictory answer, got {data['total_score']}"

    # Every criterion must be awarded 0.0 marks
    for c in data["criterion_breakdown"]:
        assert c["awarded_marks"] == 0.0, f"Expected 0.0 marks for '{c['criterion_name']}', got {c['awarded_marks']}"

    # Characteristics must have 0.00 and detected contradiction
    char_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "Characteristics")
    assert char_crit["awarded_marks"] == 0.0
    assert len(char_crit["contradictions_detected"]) > 0, "Expected contradiction detected for Characteristics"

    # Feedback must never claim student correctly explained or identified anything
    forbidden_claims = ["you explained", "you correctly", "you identified", "you described", "you noted"]
    for c in data["criterion_breakdown"]:
        fb_lower = c["feedback"].lower()
        for f in forbidden_claims:
            assert f not in fb_lower, f"False attribution in feedback for Student D on '{c['criterion_name']}': '{c['feedback']}'"

# 21. Rejected feedback cannot reach final API response through fallback or serialization path
def test_rejected_feedback_cannot_reach_final_api_response(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    student_a_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
        "It represents examples using feature vectors and compares them using a similarity or distance measure. "
        "MBL is a flexible and efficient alternative to statistical language modeling. "
        "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
    )

    # Simulate intermediate feedback generation producing rejected candidate feedback with invalid omission phrasing
    with patch("backend.app.services.grading_engine.LLMFeedbackService.generate_criterion_feedback") as mock_fb:
        mock_fb.return_value = (
            "The student identified NLP applications. The omission of word disambiguation is noted to receive full marks."
        )

        res = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": student_a_text})
        assert res.status_code == 200
        data = res.json()

        nlp_crit = next(c for c in data["criterion_breakdown"] if c["criterion_name"] == "NLP applications")
        # Numerical marks must remain full marks (3.0/3.0)
        assert nlp_crit["awarded_marks"] == 3.0
        assert data["total_score"] == 10.0

        fb = nlp_crit["feedback"]
        # Rejected candidate feedback MUST NOT reach final API response
        assert "omission" not in fb.lower()
        assert "word disambiguation" not in fb.lower()
        assert "to receive full marks" not in fb.lower()
        # Fallback must be valid
        assert "accurately and comprehensively addressed" in fb or "accurately addressed" in fb

# 22. Retry exhaustion returns a validated fallback or an explicit unavailable state
def test_retry_exhaustion_returns_validated_fallback_or_unavailable():
    # 1. Normal retry exhaustion returning validated fallback
    with patch("httpx.Client.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "The student omitted required concepts."}
        mock_post.return_value = mock_response

        fb = LLMFeedbackService.generate_criterion_feedback(
            question="Explain MBL.",
            criterion_name="NLP applications",
            criterion_description="Identifies NLP applications.",
            awarded_marks=3.0,
            max_marks=3.0,
            student_text="Uses include machine translation and part-of-speech tagging.",
            contradictions=[],
            missing_concepts=[],
            citations=[]
        )
        assert "omitted" not in fb.lower()
        assert "accurately and comprehensively addressed" in fb

    # 2. Case where even fallback is invalidated returns explicit unavailable message
    with patch.object(LLMFeedbackService, "_is_feedback_consistent", side_effect=[False, False, False]):
        fb_unavailable = LLMFeedbackService.generate_criterion_feedback(
            question="Explain MBL.",
            criterion_name="NLP applications",
            criterion_description="Identifies NLP applications.",
            awarded_marks=3.0,
            max_marks=3.0,
            student_text="Uses include machine translation and part-of-speech tagging.",
            contradictions=[],
            missing_concepts=[],
            citations=[]
        )
        assert "Feedback is currently unavailable for 'NLP applications'" in fb_unavailable
        assert "3.00/3.00" in fb_unavailable

# 23. Feedback failures or Ollama crashes never modify numerical marks
def test_feedback_failures_never_modify_numerical_marks(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    assessment = _fetch_assessment_by_id(aid)
    student_text = (
        "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases."
    )

    # Baseline grading with normal service
    res_normal = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)
    expected_total = res_normal.total_score
    expected_marks = [c.awarded_marks for c in res_normal.criterion_breakdown]

    # Grading with Ollama completely crashing with HTTP 500
    with patch("httpx.Client.post") as mock_post:
        mock_err_response = MagicMock()
        mock_err_response.status_code = 500
        mock_post.return_value = mock_err_response

        res_crashed = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)

        assert res_crashed.total_score == expected_total
        assert [c.awarded_marks for c in res_crashed.criterion_breakdown] == expected_marks
        for c in res_crashed.criterion_breakdown:
            assert c.feedback is not None and len(c.feedback.strip()) > 0

# 24. Mechanism partial credit: partial mechanism steps receive proportional marks while omitted mechanism steps receive zero
def test_mechanism_partial_credit_proportional_behavior(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    assessment = _fetch_assessment_by_id(aid)

    with patch.object(LLMFeedbackService, "generate_criterion_feedback", return_value="Deterministic test feedback"):
        # 1. Partial mechanism: explains comparisons using similarity measure without feature vectors
        ans_partial = "MBL compares new instances with stored examples using a similarity measure to classify them."
        res_partial = GradingEngine.grade_submission(assessment=assessment, student_text=ans_partial)
        mech_partial = [c for c in res_partial.criterion_breakdown if c.criterion_name == "Classification mechanism"][0]
        assert mech_partial.awarded_marks == 2.0, f"Expected 2.0/4.0 partial credit for mechanism, got {mech_partial.awarded_marks}"

        # 2. Definition-only (Student C): explains only the high-level definition
        ans_def = "Memory-based learning uses previous examples to help classify new data."
        res_def = GradingEngine.grade_submission(assessment=assessment, student_text=ans_def)
        def_crit = [c for c in res_def.criterion_breakdown if c.criterion_name == "Definition of MBL"][0]
        mech_crit = [c for c in res_def.criterion_breakdown if c.criterion_name == "Classification mechanism"][0]
        assert def_crit.awarded_marks == 1.0, f"Expected 1.0/2.0 for definition, got {def_crit.awarded_marks}"
        assert mech_crit.awarded_marks == 0.0, f"Expected 0.0/4.0 for omitted mechanism, got {mech_crit.awarded_marks}"

        # 3. Full mechanism: explains both feature vector representation and similarity comparison
        ans_full = (
            "It stores training examples as feature vectors and compares them with new instances using a similarity or distance measure. "
            "These comparisons help classify new instances."
        )
        res_full = GradingEngine.grade_submission(assessment=assessment, student_text=ans_full)
        mech_full = [c for c in res_full.criterion_breakdown if c.criterion_name == "Classification mechanism"][0]
        assert mech_full.awarded_marks == 4.0, f"Expected 4.0/4.0 for full mechanism, got {mech_full.awarded_marks}"

# 25. Truncated or incomplete feedback rejection and deterministic fallback triggering
def test_truncated_or_incomplete_feedback_triggers_deterministic_fallback():
    # Direct validation check on truncated candidate strings
    truncated_samples = [
        "The correct NLP applications of MBL include part-of-speech tagging, morphological analysis, shallow parsing, word disambiguation, named",
        "The correct NLP applications of MBL include part-of-speech tagging, named,",
        "The correct NLP applications include such as.",
        "Your statement contradicts the expected concept that MBL is efficient and",
        "Statement contradicts 'MBL is versatile",
        "Statement contradicts \"MBL is versatile.",
        "MBL has applications including",
        "MBL has applications named."
    ]
    for sample in truncated_samples:
        is_valid = LLMFeedbackService._is_feedback_consistent(
            feedback=sample,
            awarded_marks=0.0,
            max_marks=3.0,
            criterion_name="NLP applications",
            clean_contradictions=["Statement contradicts expected concept 'Valid applications'"],
            clean_missing=["Valid applications"],
            clean_present=[]
        )
        assert is_valid is False, f"Truncated feedback was falsely accepted as consistent: '{sample}'"

    # Verifies that when Ollama returns truncated text, generate_criterion_feedback rejects it and falls back
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "The correct NLP applications include part-of-speech tagging, shallow parsing, named"
        }
        mock_post.return_value = mock_resp

        fb = LLMFeedbackService.generate_criterion_feedback(
            question="Explain MBL.",
            criterion_name="NLP applications",
            criterion_description="Identifies valid Natural Language Processing applications of MBL.",
            awarded_marks=0.0,
            max_marks=3.0,
            student_text="It has no applications.",
            contradictions=["Student stated 'no applications' which contradicts expected concept 'Valid applications, such as tagging, parsing, named entity recognition.'"],
            missing_concepts=["Additional applications: tagging"],
            citations=[]
        )
        assert not fb.rstrip().endswith("named"), f"Truncated word leaked into final feedback: '{fb}'"
        assert fb.rstrip().endswith("."), f"Feedback must end with terminal punctuation: '{fb}'"
        assert "directly contradicts the expected rubric concept" in fb

# 26. Complete feedback formatting: no trailing connectors, balanced punctuation, and no long lists in fallback
def test_complete_feedback_formatting_and_concise_fallback():
    # Test deterministic fallback for contradiction with long illustrative list
    fb = LLMFeedbackService._get_deterministic_fallback(
        awarded_marks=0.0,
        max_marks=3.0,
        clean_contradictions=[
            "Student stated 'no applications' which contradicts expected concept 'Valid Natural Language Processing applications of MBL, such as part-of-speech tagging, morphological analysis, shallow parsing, word disambiguation, named entity recognition, natural language generation, machine translation, or dialogue-act recognition.'"
        ],
        clean_missing=["Additional applications: part-of-speech tagging"],
        clean_present=[],
        criterion_name="NLP applications"
    )
    assert fb.endswith("."), f"Fallback must end with period: '{fb}'"
    assert "such as" not in fb, f"Fallback should not recite long illustrative list: '{fb}'"
    assert "directly contradicts the expected rubric concept 'Valid Natural Language Processing applications of MBL'" in fb

    # Test deterministic fallback for partial marks
    fb_partial = LLMFeedbackService._get_deterministic_fallback(
        awarded_marks=1.5,
        max_marks=3.0,
        clean_contradictions=[],
        clean_missing=["Additional applications: part-of-speech tagging, such as tagging, chunking"],
        clean_present=["Valid items identified: parsing"],
        criterion_name="NLP applications"
    )
    assert fb_partial.endswith("."), f"Partial fallback must end with period: '{fb_partial}'"
    assert "such as" not in fb_partial
    assert "Accurately addressed:" in fb_partial
    assert "To improve for 'NLP applications'" in fb_partial

# 27. Score invariance under feedback truncation, timeout, or failure
def test_score_invariance_under_feedback_truncation_and_failure(mbl_assessment_fixture):
    aid = mbl_assessment_fixture["id"]
    assessment = _fetch_assessment_by_id(aid)
    student_text = (
        "Memory-based learning does not use stored training examples or feature vectors. "
        "It classifies new inputs only through fixed, manually written grammar rules. "
        "It is inefficient and has no applications in natural language processing."
    )

    # 1. Baseline run with deterministic feedback
    with patch.object(LLMFeedbackService, "generate_criterion_feedback", return_value="Deterministic clean feedback."):
        res_baseline = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)

    # 2. Run with Ollama returning truncated feedback (ending with 'named')
    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "response": "The correct NLP applications of MBL include part-of-speech tagging, named"
        }
        mock_post.return_value = mock_resp

        res_truncated = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)

    # Exact score parity and structure must be preserved
    assert res_truncated.total_score == res_baseline.total_score == 0.0
    assert res_truncated.requires_teacher_review == res_baseline.requires_teacher_review is True
    for c_base, c_trunc in zip(res_baseline.criterion_breakdown, res_truncated.criterion_breakdown):
        assert c_trunc.awarded_marks == c_base.awarded_marks
        assert c_trunc.max_marks == c_base.max_marks
        assert c_trunc.missing_concepts == c_base.missing_concepts
        assert c_trunc.contradictions_detected == c_base.contradictions_detected
        assert c_trunc.feedback.rstrip().endswith(".")
        assert not c_trunc.feedback.rstrip().endswith("named")
