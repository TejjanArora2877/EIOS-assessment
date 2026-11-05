import os
import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routers.assessments import _fetch_assessment_by_id

client = TestClient(app)


@pytest.fixture
def security_fixtures():
    fixtures_path = os.path.join(os.path.dirname(__file__), "fixtures", "security_fixtures.json")
    with open(fixtures_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def security_assessment(security_fixtures):
    payload = security_fixtures["assessment"]
    res = client.post("/assessments/", json=payload)
    assert res.status_code == 201
    aid = res.json()["id"]
    assessment = _fetch_assessment_by_id(aid)
    yield assessment
    client.delete(f"/assessments/{aid}")


# ===========================================================================
# Security Assessment Cross-Subject Grading & Parity Tests (Students A-E)
# ===========================================================================

def test_security_student_a_complete(security_assessment, security_fixtures):
    aid = security_assessment.id
    text = security_fixtures["students"]["Student_A_Complete.txt"]
    res = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    )
    assert res.status_code == 200
    j = res.json()
    assert j["total_score"] >= 18.0
    assert j["requires_teacher_review"] is False


def test_security_student_b_paraphrase(security_assessment, security_fixtures):
    aid = security_assessment.id
    text = security_fixtures["students"]["Student_B_Paraphrase.txt"]
    res = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    )
    assert res.status_code == 200
    j = res.json()
    assert j["total_score"] >= 15.0
    assert j["requires_teacher_review"] is False


def test_security_student_c_relevant_partial(security_assessment, security_fixtures):
    aid = security_assessment.id
    text = security_fixtures["students"]["Student_C_Relevant_Partial.txt"]
    res = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    )
    assert res.status_code == 200
    j = res.json()
    assert 9.0 <= j["total_score"] <= 13.0
    assert j["requires_teacher_review"] is False


def test_security_student_d_irrelevant_sql_joins(security_assessment, security_fixtures):
    aid = security_assessment.id
    text = security_fixtures["students"]["Student_D_Irrelevant_SQL_Joins.txt"]
    res = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    )
    assert res.status_code == 200
    j = res.json()
    assert j["total_score"] == 0.0
    assert j["requires_teacher_review"] is False


def test_security_student_e_contradictory(security_assessment, security_fixtures):
    aid = security_assessment.id
    text = security_fixtures["students"]["Student_E_Contradictory.txt"]
    res = client.post(
        "/grading/grade-text",
        json={"assessment_id": aid, "student_text": text, "feedback_mode": "hybrid"}
    )
    assert res.status_code == 200
    j = res.json()
    assert j["total_score"] <= 6.0
    assert j["requires_teacher_review"] is True
