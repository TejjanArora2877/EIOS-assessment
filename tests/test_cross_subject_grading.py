import io
import zipfile
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models import AssessmentResponse
from backend.app.routers.assessments import _fetch_assessment_by_id

client = TestClient(app)

# ---------------------------------------------------------------------------
# Assessment Fixtures for 3 Academic Subjects
# (Self-contained in tests only; never inserted as production seed data)
# ---------------------------------------------------------------------------

@pytest.fixture
def mbl_assessment():
    payload = {
        "title": "Natural Language Processing - Memory-Based Learning",
        "question": "Explain Memory-Based Learning (MBL) as a learning approach for NLP and its applications.",
        "model_answer": (
            "Memory-Based Learning (MBL) is a machine-learning classification approach used in Natural Language Processing. "
            "It stores training examples as feature vectors in memory. "
            "To classify a new instance, it compares the feature vector with stored examples using a similarity or distance measure. "
            "MBL is a flexible, versatile, and efficient learning framework. "
            "Its applications in NLP include part-of-speech tagging, shallow parsing, named entity recognition, and machine translation."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Definition of MBL",
                "description": "Defines Memory-Based Learning as a machine-learning classification approach used in Natural Language Processing that classifies new cases using stored training examples.",
                "max_marks": 2.0
            },
            {
                "name": "Classification mechanism",
                "description": "Explains how MBL stores training examples as feature vectors and compares them with new instances using a similarity or distance measure.",
                "max_marks": 4.0
            },
            {
                "name": "Characteristics",
                "description": "Identifies MBL as an efficient, versatile, and flexible learning framework.",
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
    aid = res.json()["id"]
    assessment = _fetch_assessment_by_id(aid)
    yield assessment
    client.delete(f"/assessments/{aid}")


@pytest.fixture
def dbms_assessment():
    payload = {
        "title": "Database Management Systems - Relational Joins",
        "question": "Explain the purpose and operation of relational joins in SQL, comparing INNER JOIN, LEFT OUTER JOIN, and FULL OUTER JOIN, and explain how join conditions avoid Cartesian products.",
        "model_answer": (
            "Relational joins combine columns from multiple tables using related attributes. "
            "An INNER JOIN returns only records that have matching values in both tables. "
            "A LEFT OUTER JOIN preserves all rows from the left table and populates missing right-table columns with NULLs. "
            "A FULL OUTER JOIN preserves all rows from both tables, inserting NULLs wherever matches are absent. "
            "Join conditions match corresponding keys between tables, preventing an unconstrained Cartesian product combining every row from one table with every row from the other."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Join Concepts and Inner Join",
                "description": "Explains that a join combines columns from multiple tables using related attributes. Explains that an INNER JOIN returns only records that have matching values in both tables.",
                "max_marks": 3.0
            },
            {
                "name": "Outer Joins Comparison",
                "description": "Explains that a LEFT OUTER JOIN preserves all rows from the left table and populates missing right-table columns with NULLs. Explains that a FULL OUTER JOIN preserves all rows from both tables, inserting NULLs wherever matches are absent.",
                "max_marks": 4.0
            },
            {
                "name": "Cartesian Product Avoidance",
                "description": "Explains that explicit join conditions match corresponding keys between tables. Explains that omitting a join predicate results in a Cartesian product combining every row from one table with every row from the other.",
                "max_marks": 3.0
            }
        ]
    }
    res = client.post("/assessments/", json=payload)
    assert res.status_code == 201
    aid = res.json()["id"]
    assessment = _fetch_assessment_by_id(aid)
    yield assessment
    client.delete(f"/assessments/{aid}")


@pytest.fixture
def physics_assessment():
    payload = {
        "title": "Classical Mechanics - Newton's Laws of Motion",
        "question": "State and explain Newton's Three Laws of Motion, describing inertia, the relationship between net force, mass, and acceleration, and action-reaction force pairs.",
        "model_answer": (
            "Newton's First Law of Motion, the Law of Inertia, states that an object remains at rest or continues to move with constant velocity in a straight line unless acted upon by a net external force. "
            "Newton's Second Law states that acceleration is directly proportional to net force and inversely proportional to mass, expressed mathematically as F = ma. "
            "Newton's Third Law states that for every action force, there is an equal and opposite reaction force, acting simultaneously on two different interacting bodies."
        ),
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "First Law - Inertia",
                "description": "States Newton's First Law: an object remains at rest or moves with constant velocity unless acted upon by a net external force. Explains inertia as the tendency of matter to resist changes in its state of motion.",
                "max_marks": 3.0
            },
            {
                "name": "Second Law - Force, Mass, Acceleration",
                "description": "States Newton's Second Law: acceleration is directly proportional to net force and inversely proportional to mass, expressed as F = ma. Explains that unbalanced forces cause mass to accelerate.",
                "max_marks": 4.0
            },
            {
                "name": "Third Law - Action and Reaction Pairs",
                "description": "States Newton's Third Law: for every action force, there is an equal and opposite reaction force. Explains that action-reaction force pairs act on two different objects simultaneously.",
                "max_marks": 3.0
            }
        ]
    }
    res = client.post("/assessments/", json=payload)
    assert res.status_code == 201
    aid = res.json()["id"]
    assessment = _fetch_assessment_by_id(aid)
    yield assessment
    client.delete(f"/assessments/{aid}")


# ---------------------------------------------------------------------------
# Test Data Fixtures for 6 Submission Types per Subject
# ---------------------------------------------------------------------------

CROSS_SUBJECT_DATA = {
    "mbl": {
        "correct": (
            "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
            "It represents examples using feature vectors and compares them using a similarity or distance measure. "
            "MBL is a flexible, versatile, and efficient learning framework. "
            "Its uses in NLP include part-of-speech tagging, shallow parsing, named entity recognition, and machine translation."
        ),
        "paraphrase": (
            "In MBL, new instances are categorized by examining their feature representation against cases preserved in the training corpus. "
            "A metric of distance or similarity determines class assignments. "
            "The paradigm operates as an efficient, versatile, and adaptable approach. "
            "It has established effectiveness for language tasks including grammatical tagging, shallow syntactic analysis, named entity extraction, and automatic translation."
        ),
        "partial": (
            "Memory-based learning uses previous examples to help classify new data. "
            "It can be used for part-of-speech tagging and machine translation."
        ),
        "irrelevant": (
            "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into sugars. "
            "Chlorophyll absorbs light energy, and oxygen is released as a by-product."
        ),
        "contradictory": (
            "Memory-based learning does not use stored training examples or feature vectors. "
            "It classifies new inputs only through fixed, manually written grammar rules. "
            "It is inefficient and has no applications in natural language processing."
        ),
        "empty": "   \t\n "
    },
    "dbms": {
        "correct": (
            "Relational joins combine attributes from different tables using common key columns. "
            "An INNER JOIN returns only records that have matching values in both tables. "
            "A LEFT OUTER JOIN preserves every tuple from the left relation, supplementing unmatched attributes with NULL values when right relation tuples do not exist. "
            "A FULL OUTER JOIN retains all tuples from both left and right relations, inserting NULLs for missing data on either side. "
            "Join conditions match corresponding keys between tables using the ON clause, preventing an unconstrained Cartesian product that would combine every row of the first table with every row of the second."
        ),
        "paraphrase": (
            "In relational databases, joins connect columns from distinct relations by linking shared identifier fields. "
            "Executing an INNER JOIN filters for records satisfying equality across both relations, eliminating non-matching entries. "
            "In contrast, a LEFT OUTER JOIN retains the complete set of left-side records, filling right-side columns with NULL indicators whenever matching entries are absent. "
            "A FULL OUTER JOIN encompasses all rows from both tables, providing NULL padding whenever an association is missing. "
            "Specifying an explicit join predicate correlates foreign keys with primary keys, ensuring unconstrained Cartesian combinations—where every tuple in table one pairs with every tuple in table two—are strictly avoided."
        ),
        "partial": (
            "A join combines columns from multiple tables using shared keys. "
            "An inner join returns only rows where matching values exist in both tables."
        ),
        "irrelevant": (
            "Newton's First Law states that an object remains at rest or in uniform straight motion unless acted upon by an external net force. "
            "Inertia represents the resistance of physical mass to changes in velocity."
        ),
        "contradictory": (
            "An INNER JOIN returns all rows from both tables regardless of whether keys match. "
            "A LEFT OUTER JOIN deletes records from the left table that have null values. "
            "Omitting join predicates is standard practice because SQL never generates Cartesian products without explicit cross join keywords."
        ),
        "empty": ""
    },
    "physics": {
        "correct": (
            "Newton's First Law of Motion, the Law of Inertia, states that an object remains at rest or continues to move with constant velocity in a straight line unless acted upon by a net external force. "
            "Newton's Second Law states that acceleration is directly proportional to net force and inversely proportional to mass, expressed mathematically as F = ma. "
            "Newton's Third Law states that for every action force, there is an equal and opposite reaction force, acting simultaneously on two different interacting bodies."
        ),
        "paraphrase": (
            "Under the principle of inertia, a physical body maintains its velocity—either resting stationary or traveling at steady speed along a straight trajectory—until an unbalanced external influence alters its state. "
            "The second law governs dynamics quantitatively: an applied net force induces proportional acceleration in a massive body, following the fundamental relation F = ma. "
            "The third law dictates that mutual physical interactions generate coupled force pairs of identical magnitude pointing in opposing directions, acting concurrently on distinct objects."
        ),
        "partial": (
            "Newton's First Law states that an object remains at rest unless acted upon by an external force. "
            "Newton's Second Law relates force, mass, and acceleration through the formula F = ma."
        ),
        "irrelevant": (
            "In relational databases, an inner join combines matching records from two tables based on a foreign key relationship, while outer joins preserve non-matching rows."
        ),
        "contradictory": (
            "Newton's First Law states that moving objects naturally slow down and stop on their own without any external forces acting on them. "
            "Newton's Second Law proves that greater mass produces greater acceleration for the same force. "
            "Newton's Third Law states that action and reaction forces act on the same object and cancel each other out."
        ),
        "empty": "   "
    }
}


# ===========================================================================
# 1. NLP / Memory-Based Learning Suite
# ===========================================================================

def test_mbl_correct_and_paraphrase_high_marks(mbl_assessment):
    aid = mbl_assessment.id
    # Correct
    res_c = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["mbl"]["correct"]}).json()
    assert res_c["total_score"] >= 9.0
    assert res_c["requires_teacher_review"] is False

    # Paraphrase
    res_p = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["mbl"]["paraphrase"]}).json()
    assert res_p["total_score"] >= 8.0
    assert res_p["requires_teacher_review"] is False


def test_mbl_partial_irrelevant_contradictory_differentiation(mbl_assessment):
    aid = mbl_assessment.id
    # Partial
    res_part = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["mbl"]["partial"]}).json()
    assert 2.0 <= res_part["total_score"] <= 5.5
    assert res_part["requires_teacher_review"] is False

    # Irrelevant
    res_irr = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["mbl"]["irrelevant"]}).json()
    assert res_irr["total_score"] <= 1.0
    assert res_irr["requires_teacher_review"] is False

    # Contradictory
    res_con = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["mbl"]["contradictory"]}).json()
    assert res_con["total_score"] <= 1.0
    assert res_con["requires_teacher_review"] is True


def test_mbl_single_vs_batch_parity(mbl_assessment):
    aid = mbl_assessment.id
    data_dict = CROSS_SUBJECT_DATA["mbl"]

    # 1. Single evaluations
    single_scores = {}
    for name, text in data_dict.items():
        res = client.post(
            "/grading/grade-file",
            data={"assessment_id": aid, "feedback_mode": "hybrid"},
            files={"file": (f"{name}.txt", io.BytesIO(text.encode("utf-8")), "text/plain")}
        )
        if not text.strip():
            assert res.status_code == 400
            single_scores[name] = {"score": None, "review": None}
        else:
            assert res.status_code == 200
            j = res.json()
            single_scores[name] = {"score": j["total_score"], "review": j["requires_teacher_review"]}

    # 2. Batch evaluation
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in data_dict.items():
            zf.writestr(f"{name}.txt", text)

    batch_res = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid, "feedback_mode": "hybrid"},
        files={"file": ("mbl_batch.zip", io.BytesIO(zip_buf.getvalue()), "application/zip")}
    )
    assert batch_res.status_code == 202
    bid = batch_res.json()["batch_id"]

    import time
    for _ in range(180):
        poll = client.get(f"/grading/batch/{bid}").json()
        if poll["status"] in ("completed", "failed"):
            break
        time.sleep(1.0)

    assert poll["status"] == "completed"
    batch_map = {r["filename"]: r for r in poll["results"]}

    for name in data_dict.keys():
        b_item = batch_map[f"{name}.txt"]
        s_item = single_scores[name]
        assert b_item["total_score"] == s_item["score"]
        assert b_item["requires_teacher_review"] == s_item["review"]


# ===========================================================================
# 2. DBMS / Relational Joins Suite
# ===========================================================================

def test_dbms_correct_and_paraphrase_high_marks(dbms_assessment):
    aid = dbms_assessment.id
    # Correct
    res_c = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["dbms"]["correct"]}).json()
    assert res_c["total_score"] == 10.0
    assert res_c["requires_teacher_review"] is False

    # Paraphrase
    res_p = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["dbms"]["paraphrase"]}).json()
    assert res_p["total_score"] >= 8.0
    assert res_p["requires_teacher_review"] is False


def test_dbms_partial_irrelevant_contradictory_differentiation(dbms_assessment):
    aid = dbms_assessment.id
    # Partial (covers inner join, omits outer joins)
    res_part = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["dbms"]["partial"]}).json()
    assert 3.0 <= res_part["total_score"] <= 6.0
    assert res_part["total_score"] < 8.0
    assert res_part["requires_teacher_review"] is False

    # Irrelevant (Newton's laws on DBMS exam)
    res_irr = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["dbms"]["irrelevant"]}).json()
    assert res_irr["total_score"] <= 1.0
    assert res_irr["requires_teacher_review"] is False

    # Contradictory (Claims inner join returns all rows, Cartesian product never happens)
    res_con = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["dbms"]["contradictory"]}).json()
    assert res_con["total_score"] <= 1.0
    assert res_con["requires_teacher_review"] is True


def test_dbms_single_vs_batch_parity(dbms_assessment):
    aid = dbms_assessment.id
    data_dict = CROSS_SUBJECT_DATA["dbms"]

    single_scores = {}
    for name, text in data_dict.items():
        res = client.post(
            "/grading/grade-file",
            data={"assessment_id": aid, "feedback_mode": "hybrid"},
            files={"file": (f"{name}.txt", io.BytesIO(text.encode("utf-8")), "text/plain")}
        )
        if not text.strip():
            assert res.status_code == 400
            single_scores[name] = {"score": None, "review": None}
        else:
            assert res.status_code == 200
            j = res.json()
            single_scores[name] = {"score": j["total_score"], "review": j["requires_teacher_review"]}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in data_dict.items():
            zf.writestr(f"{name}.txt", text)

    batch_res = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid, "feedback_mode": "hybrid"},
        files={"file": ("dbms_batch.zip", io.BytesIO(zip_buf.getvalue()), "application/zip")}
    )
    assert batch_res.status_code == 202
    bid = batch_res.json()["batch_id"]

    import time
    for _ in range(180):
        poll = client.get(f"/grading/batch/{bid}").json()
        if poll["status"] in ("completed", "failed"):
            break
        time.sleep(1.0)

    assert poll["status"] == "completed"
    batch_map = {r["filename"]: r for r in poll["results"]}

    for name in data_dict.keys():
        b_item = batch_map[f"{name}.txt"]
        s_item = single_scores[name]
        assert b_item["total_score"] == s_item["score"]
        assert b_item["requires_teacher_review"] == s_item["review"]


# ===========================================================================
# 3. Physics / Newton's Laws Suite
# ===========================================================================

def test_physics_correct_and_paraphrase_high_marks(physics_assessment):
    aid = physics_assessment.id
    # Correct
    res_c = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["physics"]["correct"]}).json()
    assert res_c["total_score"] >= 8.0
    assert res_c["requires_teacher_review"] is False

    # Paraphrase
    res_p = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["physics"]["paraphrase"]}).json()
    assert res_p["total_score"] >= 8.0
    assert res_p["requires_teacher_review"] is False


def test_physics_partial_irrelevant_contradictory_differentiation(physics_assessment):
    aid = physics_assessment.id
    # Partial (Law 1 and 2 mentioned briefly, Law 3 omitted)
    res_part = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["physics"]["partial"]}).json()
    assert 1.0 <= res_part["total_score"] <= 5.0
    assert res_part["requires_teacher_review"] is False

    # Irrelevant (SQL joins on Physics exam)
    res_irr = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["physics"]["irrelevant"]}).json()
    assert res_irr["total_score"] <= 1.0
    assert res_irr["requires_teacher_review"] is False

    # Contradictory (Claims moving objects stop on their own, greater mass increases acceleration, forces cancel)
    res_con = client.post("/grading/grade-text", json={"assessment_id": aid, "student_text": CROSS_SUBJECT_DATA["physics"]["contradictory"]}).json()
    assert res_con["requires_teacher_review"] is True


def test_physics_single_vs_batch_parity(physics_assessment):
    aid = physics_assessment.id
    data_dict = CROSS_SUBJECT_DATA["physics"]

    single_scores = {}
    for name, text in data_dict.items():
        res = client.post(
            "/grading/grade-file",
            data={"assessment_id": aid, "feedback_mode": "hybrid"},
            files={"file": (f"{name}.txt", io.BytesIO(text.encode("utf-8")), "text/plain")}
        )
        if not text.strip():
            assert res.status_code == 400
            single_scores[name] = {"score": None, "review": None}
        else:
            assert res.status_code == 200
            j = res.json()
            single_scores[name] = {"score": j["total_score"], "review": j["requires_teacher_review"]}

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in data_dict.items():
            zf.writestr(f"{name}.txt", text)

    batch_res = client.post(
        "/grading/batch-zip",
        data={"assessment_id": aid, "feedback_mode": "hybrid"},
        files={"file": ("physics_batch.zip", io.BytesIO(zip_buf.getvalue()), "application/zip")}
    )
    assert batch_res.status_code == 202
    bid = batch_res.json()["batch_id"]

    import time
    for _ in range(180):
        poll = client.get(f"/grading/batch/{bid}").json()
        if poll["status"] in ("completed", "failed"):
            break
        time.sleep(1.0)

    assert poll["status"] == "completed"
    batch_map = {r["filename"]: r for r in poll["results"]}

    for name in data_dict.keys():
        b_item = batch_map[f"{name}.txt"]
        s_item = single_scores[name]
        assert b_item["total_score"] == s_item["score"]
        assert b_item["requires_teacher_review"] == s_item["review"]


# ===========================================================================
# 4. Cross-Mode Scoring Invariance Test
# ===========================================================================

def test_cross_mode_scoring_invariance(dbms_assessment):
    """
    Verifies that numerical marks and teacher review flags remain 100% invariant
    between feedback_mode='hybrid' and feedback_mode='llm'.
    """
    aid = dbms_assessment.id
    text = CROSS_SUBJECT_DATA["dbms"]["paraphrase"]

    res_hybrid = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    ).json()

    res_llm = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "llm"}
    ).json()

    assert res_hybrid["total_score"] == res_llm["total_score"]
    assert res_hybrid["max_score"] == res_llm["max_score"]
    assert res_hybrid["requires_teacher_review"] == res_llm["requires_teacher_review"]

    for c_h, c_l in zip(res_hybrid["criterion_breakdown"], res_llm["criterion_breakdown"]):
        assert c_h["criterion_name"] == c_l["criterion_name"]
        assert c_h["awarded_marks"] == c_l["awarded_marks"]
        assert c_h["max_marks"] == c_l["max_marks"]
