import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database import init_db
from backend.app.models import AssessmentResponse
from backend.app.services.answer_segmenter import AnswerSegmenter
from backend.app.services.grading_engine import GradingEngine

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_segmenter_explicit_question_markers():
    """
    Tests that explicit question markers (Question 1, Q2, Answer 3) are segmented cleanly,
    and metadata/headers prior to the first anchor are discarded.
    """
    questions = [
        {
            "id": "q1",
            "question_order": 1,
            "question_text": "Distinguish cybersecurity and information security.",
            "model_answer": "Cybersecurity protects digital systems. Information security covers all data formats.",
            "criteria": [
                {"name": "Cybersecurity", "description": "Protects digital systems.", "max_marks": 5.0}
            ]
        },
        {
            "id": "q2",
            "question_order": 2,
            "question_text": "Distinguish passive and active attacks.",
            "model_answer": "Passive attacks eavesdrop without modification. Active attacks modify resources.",
            "criteria": [
                {"name": "Attacks", "description": "Passive vs active distinction.", "max_marks": 5.0}
            ]
        },
        {
            "id": "q3",
            "question_order": 3,
            "question_text": "Explain memory-based learning.",
            "model_answer": "Memory-based learning delays computation until query time.",
            "criteria": [
                {"name": "MBL", "description": "Instance-based lazy evaluation.", "max_marks": 5.0}
            ]
        }
    ]

    submission_text = """
    Student Name: Alice Smith
    Student ID: CS-2026-99
    Course: Information Security

    Question 1:
    Cybersecurity protects internet-connected digital systems and devices. Information security is broader and protects all data formats whether physical or electronic.

    Q2:
    Passive attacks monitor or intercept network communications without modifying data, such as traffic analysis. Active attacks alter system resources or disrupt operations, such as denial of service.

    Answer 3:
    Memory-based learning defers model induction until a query is encountered, storing training instances in memory rather than constructing an eager hypothesis.
    """

    slices = AnswerSegmenter.segment_submission(submission_text, questions)

    assert "q1" in slices
    assert "q2" in slices
    assert "q3" in slices

    # Metadata prior to first anchor must be discarded
    assert "Alice Smith" not in slices["q1"]
    assert "CS-2026-99" not in slices["q1"]

    # Slices contain their respective content
    assert "Cybersecurity protects" in slices["q1"]
    assert "Information security is broader" in slices["q1"]

    assert "Passive attacks monitor" in slices["q2"]
    assert "Active attacks alter" in slices["q2"]

    assert "Memory-based learning defers" in slices["q3"]
    assert "storing training instances" in slices["q3"]

    # Cross-question boundaries must be strictly isolated
    assert "Passive attacks" not in slices["q1"]
    assert "Memory-based learning" not in slices["q2"]


def test_segmenter_unlabeled_semantic_routing():
    """
    Tests semantic routing of unlabeled continuous paragraphs based on embedding similarity,
    and verifies that off-topic paragraphs are discarded.
    """
    questions = [
        {
            "id": "q_sql",
            "question_order": 1,
            "question_text": "Explain SQL joins and relational database normalization.",
            "model_answer": "Relational database normalization organizes tables to minimize data redundancy. Inner joins combine matching rows.",
            "criteria": [
                {"name": "SQL Joins", "description": "Inner and outer joins in relational databases.", "max_marks": 5.0}
            ]
        },
        {
            "id": "q_crypto",
            "question_order": 2,
            "question_text": "Explain symmetric and asymmetric encryption cryptography.",
            "model_answer": "Symmetric encryption uses a single secret key. Asymmetric cryptography uses public and private key pairs.",
            "criteria": [
                {"name": "Cryptography", "description": "Symmetric and asymmetric cipher keys.", "max_marks": 5.0}
            ]
        }
    ]

    submission_text = """Relational database normalization organizes tables to minimize data redundancy. An inner join combines rows from multiple tables based on a shared foreign key condition.

Symmetric encryption uses a single shared secret key for both data encryption and decryption, such as AES. Asymmetric cryptography uses mathematically linked public and private key pairs, such as RSA.

Photosynthesis occurs in plant chloroplasts where light energy converts water and carbon dioxide into oxygen and glucose."""

    slices = AnswerSegmenter.segment_submission(submission_text, questions)

    assert "q_sql" in slices
    assert "q_crypto" in slices

    # SQL paragraph routed to Q1
    assert "normalization organizes tables" in slices["q_sql"]
    assert "inner join combines rows" in slices["q_sql"]

    # Crypto paragraph routed to Q2
    assert "Symmetric encryption" in slices["q_crypto"]
    assert "Asymmetric cryptography" in slices["q_crypto"]

    # Off-topic biology paragraph must be discarded by both questions
    assert "Photosynthesis" not in slices["q_sql"]
    assert "Photosynthesis" not in slices["q_crypto"]


def test_multi_question_end_to_end_grading_and_skipped_question():
    """
    End-to-end multi-question assessment test:
    - Creates a 3-question assessment via the API.
    - Submits an answer covering Q1 and Q2, but skipping Q3.
    - Verifies Q1 receives full/high marks, Q2 partial marks, and Q3 receives exactly 0.00 marks.
    - Verifies aggregated score matches sum of question scores.
    """
    payload = {
        "title": "Computer Science Comprehensive Exam",
        "total_marks": 15.0,
        "doc_scope": "ALL",
        "questions": [
            {
                "question_order": 1,
                "question_text": "Explain the difference between passive and active security attacks.",
                "model_answer": "Passive attacks eavesdrop on communications without altering data, such as traffic analysis. Active attacks modify system data or disrupt services, such as denial of service.",
                "max_marks": 5.0,
                "criteria": [
                    {
                        "name": "Passive Attacks",
                        "description": "Monitors or eavesdrops on transmissions without altering data, including traffic analysis.",
                        "max_marks": 2.5
                    },
                    {
                        "name": "Active Attacks",
                        "description": "Alters system resources, modifies data, or disrupts network operations including denial of service.",
                        "max_marks": 2.5
                    }
                ]
            },
            {
                "question_order": 2,
                "question_text": "Define supervised learning and explain the role of training labels.",
                "model_answer": "Supervised learning learns a mapping function from input features to output targets using labeled training examples.",
                "max_marks": 5.0,
                "criteria": [
                    {
                        "name": "Supervised Learning Definition",
                        "description": "Learning a mapping function from input features to target labels using labeled training pairs.",
                        "max_marks": 2.5
                    },
                    {
                        "name": "Training Labels Role",
                        "description": "Ground-truth target labels guide loss optimization and model parameter updates.",
                        "max_marks": 2.5
                    }
                ]
            },
            {
                "question_order": 3,
                "question_text": "Describe relational database normalization and Boyce-Codd normal form.",
                "model_answer": "Normalization organizes relational tables to eliminate redundancy. BCNF requires every determinant to be a candidate key.",
                "max_marks": 5.0,
                "criteria": [
                    {
                        "name": "Normalization Purpose",
                        "description": "Minimizes redundancy and prevents update anomalies in relational tables.",
                        "max_marks": 2.5
                    },
                    {
                        "name": "Boyce-Codd Normal Form",
                        "description": "Requires every functional determinant to be a superkey or candidate key.",
                        "max_marks": 2.5
                    }
                ]
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()
    assert len(assessment["questions"]) == 3

    # Student submission: Q1 complete, Q2 partial, Q3 completely omitted
    student_text = """
    Question 1:
    Passive attacks eavesdrop on network communications without altering the data transmissions, such as traffic analysis. Active attacks modify system data and disrupt network operations like denial of service attacks.

    Question 2:
    Supervised learning algorithms learn a mapping function from input features to target labels using labeled data examples.
    """

    grade_res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert grade_res.status_code == 200
    res_data = grade_res.json()

    assert "question_results" in res_data
    q_results = res_data["question_results"]
    assert len(q_results) == 3

    # Q1: Both passive and active attacks addressed -> high/full marks
    q1 = q_results[0]
    assert q1["question_order"] == 1
    assert q1["awarded_marks"] >= 4.0

    # Q2: Supervised learning definition addressed, training labels role missing -> partial credit
    q2 = q_results[1]
    assert q2["question_order"] == 2
    assert 1.5 <= q2["awarded_marks"] <= 3.5

    # Q3: Skipped entirely -> exactly 0.00 marks, no crash
    q3 = q_results[2]
    assert q3["question_order"] == 3
    assert q3["awarded_marks"] == 0.00
    assert q3["student_answer_slice"] == ""

    # Mathematical score invariance: Total score must equal sum of question scores
    expected_total = round(q1["awarded_marks"] + q2["awarded_marks"] + q3["awarded_marks"], 2)
    assert abs(res_data["total_score"] - expected_total) < 1e-4

    # Flattened criterion breakdown must contain all 6 criteria
    assert len(res_data["criterion_breakdown"]) == 6


def test_single_question_backward_compatibility():
    """
    Verifies that legacy single-question assessments without a 'questions' payload
    function identically: auto-creates Question 1 record, grades seamlessly,
    and returns backward-compatible fields.
    """
    payload = {
        "title": "Legacy Single Question Exam",
        "question": "What is the primary objective of computer security?",
        "model_answer": "Computer security aims to preserve confidentiality, integrity, and availability of system resources and data assets.",
        "total_marks": 6.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "CIA Triad",
                "description": "Preserves confidentiality, integrity, and availability of system assets.",
                "max_marks": 6.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    assert assessment["question"] == payload["question"]
    assert assessment["model_answer"] == payload["model_answer"]
    assert len(assessment["questions"]) == 1
    assert assessment["questions"][0]["question_order"] == 1

    student_text = "Computer security preserves the confidentiality, integrity, and availability of system information assets."

    grade_res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert grade_res.status_code == 200
    res_data = grade_res.json()

    assert res_data["total_score"] >= 5.0
    assert res_data["max_score"] == 6.0
    assert len(res_data["criterion_breakdown"]) == 1
    assert res_data["criterion_breakdown"][0]["criterion_name"] == "CIA Triad"


def test_intra_criterion_sibling_propositions_no_false_contradiction():
    """
    Verifies that a criterion containing multiple propositions covering distinct subtopics
    (e.g., Windows live response tools vs. Unix live response tools) awards full marks (5.00/5.00)
    without flagging the Windows paragraph as contradicting the Unix proposition.
    """
    payload = {
        "title": "Digital Forensics Live Response Exam",
        "question": "Explain live response tools across Windows and Unix platforms.",
        "model_answer": "Live response on Windows systems uses tools such as winpmem, DumpIt, or FTK Imager Lite for volatile memory acquisition. Live response on Unix systems inspects /proc and /dev/fmem and uses LiME for kernel memory capture.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Live Response Across Windows and Unix",
                "description": "Live response on Windows systems uses tools such as winpmem, DumpIt, or FTK Imager Lite for volatile memory acquisition. Live response on Unix systems inspects /proc, /dev/fmem, and uses LiME for kernel memory capture.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    student_text = """
    Live response on Windows systems uses tools such as winpmem, DumpIt, and FTK Imager Lite to capture physical RAM to external storage.
    Live response on Unix systems inspects /proc virtual files and /dev/fmem, utilizing LiME for kernel memory capture.
    """

    grade_res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert grade_res.status_code == 200
    res_data = grade_res.json()

    crit = res_data["criterion_breakdown"][0]
    assert crit["awarded_marks"] == 5.00, f"Expected 5.00, got {crit['awarded_marks']}"
    assert len(crit["contradictions_detected"]) == 0, f"Expected 0 contradictions, got {crit['contradictions_detected']}"
    assert res_data["requires_teacher_review"] is False


def test_detailed_elaboration_not_penalized_by_reverse_nli():
    """
    Verifies that a detailed student sentence elaborating on a concise rubric proposition
    retains full entailment credit without false Reverse NLI contradiction,
    while an inverted misconception with polarity-inverting qualifiers is penalized.
    """
    payload = {
        "title": "Order of Volatility Assessment",
        "question": "State the core rule regarding the order of volatile versus non-volatile evidence collection.",
        "model_answer": "Collect volatile evidence before non-volatile storage.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Order of Volatility",
                "description": "Collect volatile evidence before non-volatile storage.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # 1. Valid rich elaboration: extra technical details, strongly entails proposition, no polarity inverters
    student_elaboration = (
        "Volatile evidence must be collected in order of volatility from CPU registers, routing tables, "
        "and physical RAM before acquiring non-volatile secondary disk storage."
    )

    res_good = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_elaboration,
        "feedback_mode": "none"
    })
    assert res_good.status_code == 200
    good_data = res_good.json()
    assert good_data["total_score"] >= 4.5, f"Expected elaboration >= 4.50, got {good_data['total_score']}"
    assert len(good_data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert good_data["requires_teacher_review"] is False

    # 2. Inverted misconception: polarity qualifiers invert the required collection sequence
    student_inversion = (
        "Non-volatile disk storage must always be acquired before volatile evidence, "
        "ignoring RAM until after the hard drive is cloned."
    )

    res_bad = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_inversion,
        "feedback_mode": "none"
    })
    assert res_bad.status_code == 200
    bad_data = res_bad.json()
    assert bad_data["total_score"] <= 1.5, f"Expected inversion <= 1.50, got {bad_data['total_score']}"
    assert len(bad_data["criterion_breakdown"][0]["contradictions_detected"]) > 0
    assert bad_data["requires_teacher_review"] is True


def test_paraphrase_guards_and_off_topic_review_suppression():
    """
    Tests:
    1. Guard A: Conditional/contrapositive paraphrase using 'without' does not trip Reverse NLI contradiction.
    2. Guard B: Multi-clause contrastive proposition with 'whereas' allows one-sided answers without contradiction.
    3. Guard C: Completely off-topic submission scores 0.00 and suppresses requires_teacher_review.
    """
    payload = {
        "title": "Digital Forensics Principles",
        "question": "Explain volatile acquisition and live response platforms.",
        "model_answer": "Live response on Windows systems uses winpmem, whereas Unix systems use LiME.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Live Collection Contrast",
                "description": "Live response on Windows systems uses tools such as winpmem, whereas Unix systems inspect /proc and use LiME.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # 1. Guard A & B: Student answers only one contrastive clause using conditional framing
    student_contrapositive = "Without live kernel memory acquisition tools like LiME, Unix system /proc structures cannot be captured."
    res_cp = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_contrapositive,
        "feedback_mode": "none"
    })
    assert res_cp.status_code == 200
    cp_data = res_cp.json()
    assert len(cp_data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert cp_data["requires_teacher_review"] is False

    # 2. Guard C: Completely off-topic submission must score 0.00 and have requires_teacher_review == False
    off_topic_text = "Photosynthesis takes place in the chloroplasts of plant cells using sunlight and carbon dioxide to generate glucose."
    res_off = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": off_topic_text,
        "feedback_mode": "none"
    })
    assert res_off.status_code == 200
    off_data = res_off.json()
    assert off_data["total_score"] == 0.00
    assert off_data["requires_teacher_review"] is False
    assert len(off_data["review_reasons"]) == 0


def test_segmenter_rescues_unlabeled_interior_question():
    """
    Tests that when an interior question is completely unlabeled between explicit anchors
    (e.g., Question 1: and Question 3:), Stage 3 fill empty rescues the semantic block sequence
    belonging to the unlabeled question based on embedding cosine similarity.
    """
    questions = [
        {
            "id": "q1",
            "question_order": 1,
            "question_text": "Discuss regular expressions and finite automata theory.",
            "model_answer": "Regular expressions specify languages recognized by deterministic and non-deterministic finite automata.",
            "criteria": [{"name": "Automata", "description": "Finite automata properties.", "max_marks": 5.0}]
        },
        {
            "id": "q2",
            "question_order": 2,
            "question_text": "Analyze context-free grammars and CYK parsing.",
            "model_answer": "Context-free grammars use production rules with single nonterminals on the left. The CYK parsing algorithm evaluates Chomsky normal form grammars in cubic time.",
            "criteria": [{"name": "CFG", "description": "Context free grammar and CYK.", "max_marks": 5.0}]
        },
        {
            "id": "q3",
            "question_order": 3,
            "question_text": "Explain Turing machines and decidability.",
            "model_answer": "Turing machines model universal computation with an infinite tape and read-write head. Decidability concerns halting problems.",
            "criteria": [{"name": "TM", "description": "Turing computability and halting.", "max_marks": 5.0}]
        }
    ]

    submission_text = """Question 1:
Regular expressions define regular languages recognized by finite state automata.

Deterministic and nondeterministic finite automata share equivalent computational power.

Context-Free Grammars generate context-free languages beyond finite automata.
They use production rules where the left-hand side is a single nonterminal.

The CYK algorithm parses grammars transformed into Chomsky Normal Form.

For Algorithm CYK, Grammar Form is CNF, and Complexity is O(n^3).

Question 3:
Turing machines model universal computation with an infinite tape and read-write head.
The halting problem is undecidable for universal Turing machines."""

    slices = AnswerSegmenter.segment_submission(submission_text, questions)
    assert "q1" in slices and "q2" in slices and "q3" in slices
    assert "finite state automata" in slices["q1"]
    assert "Context-Free Grammars" in slices["q2"]
    assert "CYK" in slices["q2"]
    assert "Turing machines" in slices["q3"]
    assert len(slices["q2"].strip()) > 0


def test_compound_contrastive_propositions_awarded_full_credit():
    """
    Tests that a long compound contrastive proposition (e.g. len > 110 with 'whereas')
    is split into atomic clauses so that a student entailing both clauses in separate sentences
    receives full marks with 0 false-positive contradiction flags.
    """
    payload = {
        "title": "Formal Language Closure Properties",
        "question": "Discuss closure properties of context-free languages under standard operations.",
        "model_answer": "Context-free languages are closed under union and concatenation operations, whereas they are not closed under set intersection or complementation.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Closure Contrast",
                "description": "Context-free languages are closed under union and concatenation operations, whereas they are not closed under set intersection or complementation.",
                "max_marks": 10.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    student_answer = (
        "Context-free languages are closed under union and concatenation operations. "
        "In contrast, they are not closed under set intersection or complementation."
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_answer,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 9.0, f"Expected high score, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_linearized_table_propositions_isolated_from_unrelated_prose():
    """
    Tests that when a criterion includes linearized table propositions,
    unrelated prose sentences in the student submission (even containing polarity inversion or negative words)
    do not falsely contradict linearized table propositions for other entities.
    """
    payload = {
        "title": "Grammar Family Equivalence",
        "question": "Compare regular languages and context-free languages.",
        "model_answer": (
            "For Language Class Regular Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is Yes. "
            "For Language Class Context-Free Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is No."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Equivalence Matrix",
                "description": (
                    "For Language Class Regular Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is Yes. "
                    "For Language Class Context-Free Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is No."
                ),
                "max_marks": 10.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    student_submission = (
        "For Language Class Regular Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is Yes.\n"
        "For Language Class Context-Free Languages, Closed Under Union is Yes, and Deterministic Variant Equivalent is No.\n"
        "Deterministic pushdown automata reject languages that general context-free grammars can generate."
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_submission,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 9.0, f"Expected full/high credit, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_linearized_table_candidate_rows_isolated_from_prose_rubric_propositions():
    """
    Tests that when a student includes a comparison table in their submission
    alongside prose, the linearized table rows for entity B (e.g., Context-Free Languages)
    do not trigger false contradictions against prose rubric propositions for entity A
    (e.g., "Regular languages are closed under union and concatenation operations.").
    """
    payload = {
        "title": "Formal Language Foundations",
        "question": "Explain the closure properties of regular languages.",
        "model_answer": "Regular languages are closed under union and concatenation operations.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Regular Language Closure",
                "description": "Regular languages are closed under union and concatenation operations.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # Student provides the correct prose answer for regular languages,
    # but ALSO includes a comparison table for context-free languages in the same answer slice
    student_text = (
        "Regular languages are closed under union and concatenation operations.\n\n"
        "| Language Class | Closed Under Intersection | Deterministic PDA Equivalent |\n"
        "| --- | --- | --- |\n"
        "| Context-Free Languages | No | No |"
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 4.5, f"Expected high score >= 4.5, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_entity_scoped_semicolon_attribute_lists_preserve_subject():
    """
    Tests that rubric criteria structured as 'Entity: Key1: Val1; Key2: Val2.'
    preserve the leading entity name across all attribute clauses without severing them
    into orphan propositions.
    """
    # Test proposition extraction directly
    criterion_desc = "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2)."
    props = GradingEngine._extract_propositions(criterion_desc)
    assert len(props) == 1
    assert "CKY Algorithm" in props[0]
    assert "Time: O(|G| n^3)" in props[0]
    assert "Space: O(|G| n^2)" in props[0]

    # Test full grading with comparison table
    payload = {
        "title": "Parsing Algorithms Evaluation",
        "question": "Specify the grammar form, time complexity, and space complexity of the CKY algorithm.",
        "model_answer": "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2).",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "CKY Specifications",
                "description": "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2).",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    student_text = (
        "| Algorithm | Grammar Form | Time Complexity | Space Complexity |\n"
        "| --- | --- | --- | --- |\n"
        "| CKY Algorithm | Chomsky Normal Form CFG | O(|G| n^3) | O(|G| n^2) |"
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 4.5, f"Expected high score >= 4.5, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_affirmative_prose_elaborations_and_genuine_inversions():
    """
    Tests that:
    1. A rich affirmative multi-concept sentence discussing DFA transitions and state properties
       does not falsely contradict a sibling affirmative proposition about NFA transitions.
    2. A genuine contradictory inversion (claiming DFAs have non-deterministic multiple transitions)
       is caught as a contradiction with review required.
    """
    payload = {
        "title": "Automata Equivalence and Determinism",
        "question": "Explain determinism in finite automata and transition properties.",
        "model_answer": "Deterministic finite automata have exactly one transition per symbol from each state. Non-deterministic finite automata can have multiple transitions or epsilon transitions.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "DFA Determinism",
                "description": "Deterministic finite automata have exactly one transition per symbol from each state.",
                "max_marks": 5.0
            },
            {
                "name": "NFA Non-Determinism",
                "description": "Non-deterministic finite automata can have multiple transitions or epsilon transitions.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # 1. Affirmative multi-concept student answer (both sentences affirmative, explaining both concepts)
    student_affirmative = (
        "Deterministic finite automata have exactly one transition per symbol from each state, executing linearly. "
        "Non-deterministic finite automata can have multiple transitions or epsilon transitions for a given input symbol."
    )
    res_aff = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_affirmative,
        "feedback_mode": "none"
    })
    assert res_aff.status_code == 200
    aff_data = res_aff.json()
    assert aff_data["total_score"] >= 9.0, f"Expected full marks for affirmative elaboration, got {aff_data['total_score']}"
    assert aff_data["requires_teacher_review"] is False
    for cb in aff_data["criterion_breakdown"]:
        assert len(cb["contradictions_detected"]) == 0

    # 2. Genuine contradiction: student directly inverts DFA determinism
    student_inverted = (
        "Deterministic finite automata have multiple possible transitions for the same symbol from each state. "
        "Non-deterministic finite automata can have multiple transitions or epsilon transitions."
    )
    res_inv = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_inverted,
        "feedback_mode": "none"
    })
    assert res_inv.status_code == 200
    inv_data = res_inv.json()
    dfa_crit = [cb for cb in inv_data["criterion_breakdown"] if cb["criterion_name"] == "DFA Determinism"][0]
    assert len(dfa_crit["contradictions_detected"]) > 0 or dfa_crit["awarded_marks"] <= 1.0
    assert inv_data["requires_teacher_review"] is True


def test_colon_semicolon_entity_extraction_and_prose_isolation():
    """
    Tests that:
    1. _extract_linearized_subject extracts ("Entity", "<EntityName>") from "<EntityName>: <Attr1>: <Val1>; <Attr2>: <Val2>."
    2. In grading, an entity-scoped rubric proposition in this format is isolated from unrelated
       prose sentences containing negative/incompatibility words (e.g., "cannot", "without").
    """
    spec_text = "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2)."
    extracted = GradingEngine._extract_linearized_subject(spec_text)
    assert extracted == ("Entity", "CKY Algorithm")

    payload = {
        "title": "Parsing Algorithms Benchmark",
        "question": "Describe the requirements and complexities of CKY parsing.",
        "model_answer": "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2).",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "CKY Specifications",
                "description": "CKY Algorithm: Chomsky Normal Form CFG; Time: O(|G| n^3); Space: O(|G| n^2).",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # Student describes CKY Algorithm in prose, but ALSO adds unrelated prose discussing limitations of PDAs
    student_submission = (
        "CKY Algorithm operates on Chomsky Normal Form CFG, running in time O(|G| n^3) and space O(|G| n^2).\n\n"
        "Deterministic pushdown automata cannot parse general context-free languages without non-deterministic branching."
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_submission,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 4.5, f"Expected high score >= 4.5, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_same_polarity_paraphrase_not_contradicted_by_lexical_bias():
    """
    Tests that near-verbatim same-polarity paraphrases containing negative-sentiment adjectives
    (e.g., "useless rules" vs. "unreachable rules") are recognized as supporting paraphrases
    and not falsely penalized as contradictions by DeBERTa lexical bias.
    """
    payload = {
        "title": "Grammar Simplification",
        "question": "Explain CFG simplification procedures.",
        "model_answer": "CFG simplification eliminates unreachable rules to produce an equivalent reduced grammar.",
        "total_marks": 5.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Grammar Reduction",
                "description": "CFG simplification eliminates unreachable rules to produce an equivalent reduced grammar.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # Student uses "useless rules" instead of "unreachable rules" (both sentences affirmative, 0 negation markers)
    student_text = "CFG simplification removes useless rules to produce an equivalent reduced grammar."

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 4.5, f"Expected paraphrase >= 4.5, got {data['total_score']}"
    assert len(data["criterion_breakdown"][0]["contradictions_detected"]) == 0
    assert data["requires_teacher_review"] is False


def test_skipped_interior_question_preserves_labeled_host_question():
    """
    Tests that when Question 1: and Question 3: are explicitly labeled and Question 2 is intentionally
    omitted by the student (no Q2 content anywhere in the submission), Stage 3 does not drain or split
    Question 1's slice: Question 1 keeps 100% of its content and Question 2 remains empty ("").
    """
    questions = [
        {
            "id": "q1",
            "question_order": 1,
            "question_text": "Discuss regular expressions and finite automata theory.",
            "model_answer": "Regular expressions specify languages recognized by deterministic and non-deterministic finite automata.",
            "criteria": [{"name": "Automata", "description": "Finite automata properties.", "max_marks": 5.0}]
        },
        {
            "id": "q2",
            "question_order": 2,
            "question_text": "Analyze context-free grammars and CYK parsing.",
            "model_answer": "Context-free grammars use production rules with single nonterminals on the left. The CYK parsing algorithm evaluates Chomsky normal form grammars in cubic time.",
            "criteria": [{"name": "CFG", "description": "Context free grammar and CYK.", "max_marks": 5.0}]
        },
        {
            "id": "q3",
            "question_order": 3,
            "question_text": "Explain Turing machines and decidability.",
            "model_answer": "Turing machines model universal computation with an infinite tape and read-write head. Decidability concerns halting problems.",
            "criteria": [{"name": "TM", "description": "Turing computability and halting.", "max_marks": 5.0}]
        }
    ]

    # Student explicitly answers Q1 (with multiple paragraphs) and Q3, intentionally omitting Q2
    submission_text = """Question 1:
Regular expressions define regular languages recognized by finite state automata.

Deterministic and nondeterministic finite automata share equivalent computational power.

Finite state machines maintain a fixed finite set of internal control states.

Pumping lemma for regular languages establishes minimum repetition lengths for loops.

Question 3:
Turing machines model universal computation with an infinite tape and read-write head.
The halting problem is undecidable for universal Turing machines."""

    slices = AnswerSegmenter.segment_submission(submission_text, questions)
    assert "q1" in slices and "q2" in slices and "q3" in slices

    # Q1 retains all 4 paragraphs
    assert "finite state automata" in slices["q1"]
    assert "equivalent computational power" in slices["q1"]
    assert "Pumping lemma" in slices["q1"]

    # Q2 is completely empty (intentionally omitted)
    assert slices["q2"].strip() == ""

    # Q3 retains its content
    assert "Turing machines" in slices["q3"]
    assert "undecidable" in slices["q3"]


def test_txt_inline_q1_followed_by_q3_preserves_q1_and_leaves_q2_empty():
    """
    Tests that a .txt submission where Question 1: is immediately followed by a paragraph
    (no blank line) and followed later by Question 3: (with Q2 skipped):
    1. Assigns Question 1's text to q1.
    2. Leaves q2 completely empty ("").
    3. Assigns Question 3's text to q3.
    """
    questions = [
        {
            "id": "q1",
            "question_order": 1,
            "question_text": "Explain formal language theory and Chomsky hierarchy.",
            "model_answer": "Formal language theory categorizes languages according to the Chomsky hierarchy.",
            "criteria": [{"name": "Language Theory", "description": "Chomsky hierarchy concepts.", "max_marks": 5.0}]
        },
        {
            "id": "q2",
            "question_order": 2,
            "question_text": "Discuss context-free grammars and pushdown automata.",
            "model_answer": "Context-free grammars are recognized by pushdown automata using a stack.",
            "criteria": [{"name": "CFG and PDA", "description": "Stack automata and CFGs.", "max_marks": 5.0}]
        },
        {
            "id": "q3",
            "question_order": 3,
            "question_text": "Describe Turing machine computability.",
            "model_answer": "Turing machines model universal computation with an infinite tape and read-write head.",
            "criteria": [{"name": "Computability", "description": "Turing machines and halting.", "max_marks": 5.0}]
        }
    ]

    # Notice: Question 1: has no blank line before the paragraph (inline on next line)
    raw_txt_submission = (
        "Question 1:\n"
        "Formal language theory is a branch of theoretical computer science that focuses on the syntactic "
        "and mathematical aspects of formal languages. It categorizes languages according to the Chomsky "
        "hierarchy into regular, context-free, context-sensitive, and recursively enumerable languages.\n\n"
        "Question 3:\n"
        "Turing machines model universal computation with an infinite tape and read-write head. "
        "The halting problem is undecidable for universal Turing machines."
    )

    slices = AnswerSegmenter.segment_submission(raw_txt_submission, questions)
    assert "q1" in slices and "q2" in slices and "q3" in slices
    assert "Formal language theory" in slices["q1"]
    assert "Chomsky hierarchy" in slices["q1"]
    assert slices["q2"].strip() == ""
    assert "Turing machines" in slices["q3"]
    assert "halting problem" in slices["q3"]


def test_motivational_and_scope_boundary_clauses_no_false_contradiction():
    """
    Tests that explanatory sentences starting with motivational/purpose prepositions
    ("To prevent zero probabilities...") or scope-boundary phrases ("Beyond regular languages lies...")
    do not trigger false Reverse NLI contradictions against base definition propositions.
    """
    payload = {
        "title": "Language Models and Automata Boundaries",
        "question": "Explain statistical language models and finite state boundaries.",
        "model_answer": "Statistical language models decompose sentence probability using the chain rule and n-gram Markov assumptions. Regular expressions denote regular languages recognized by deterministic finite automata.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "N-gram Modeling",
                "description": "Statistical language models decompose sentence probability using the chain rule and n-gram Markov assumptions.",
                "max_marks": 5.0
            },
            {
                "name": "Regular Languages",
                "description": "Regular expressions denote regular languages recognized by deterministic finite automata.",
                "max_marks": 5.0
            }
        ]
    }

    create_res = client.post("/assessments", json=payload)
    assert create_res.status_code == 201
    assessment = create_res.json()

    # Student includes motivational sentence with "zero" and scope-boundary sentence with "beyond"
    student_text = (
        "Statistical language models decompose sentence probabilities using chain rule and n-gram Markov assumptions. "
        "To prevent zero probabilities on unseen n-grams, smoothing techniques such as Good-Turing or Kneser-Ney are applied. "
        "Regular expressions denote regular languages recognized by deterministic finite automata. "
        "Beyond regular languages lies the class of context-free languages recognized by pushdown automata."
    )

    res = client.post("/grading/grade-text", json={
        "assessment_id": assessment["id"],
        "student_text": student_text,
        "feedback_mode": "none"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["total_score"] >= 9.0, f"Expected high score >= 9.0, got {data['total_score']}"
    assert data["requires_teacher_review"] is False
    for cb in data["criterion_breakdown"]:
        assert len(cb["contradictions_detected"]) == 0, f"Unexpected contradiction in {cb['criterion_name']}: {cb['contradictions_detected']}"


