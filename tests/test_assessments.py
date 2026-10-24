import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_database():
    init_db()

def test_assessment_creation_and_retrieval():
    payload = {
        "title": "Natural Language Processing Midterm",
        "question": "What is semantic parsing and how does it relate to syntax?",
        "model_answer": "Semantic parsing maps natural language sentences into formal meaning representations like lambda calculus or SQL.",
        "total_marks": 10.0,
        "doc_scope": "ALL",
        "criteria": [
            {
                "name": "Definition of Semantic Parsing",
                "description": "Explains mapping text to formal representations",
                "max_marks": 5.0
            },
            {
                "name": "Relationship to Syntax",
                "description": "Explains how syntactic trees feed into compositional semantics",
                "max_marks": 5.0
            }
        ]
    }

    # 1. Create
    res = client.post("/assessments", json=payload)
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["title"] == payload["title"]
    assert len(data["criteria"]) == 2
    assert data["total_marks"] == 10.0
    aid = data["id"]

    # 2. Retrieve by ID
    get_res = client.get(f"/assessments/{aid}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["model_answer"] == payload["model_answer"]
    assert get_data["criteria"][0]["name"] == "Definition of Semantic Parsing"
    assert get_data["criteria"][0]["max_marks"] == 5.0

    # 3. Update
    update_payload = payload.copy()
    update_payload["title"] = "Updated NLP Midterm"
    put_res = client.put(f"/assessments/{aid}", json=update_payload)
    assert put_res.status_code == 200
    assert put_res.json()["title"] == "Updated NLP Midterm"

    # 4. Delete
    del_res = client.delete(f"/assessments/{aid}")
    assert del_res.status_code == 200

    # 5. Verify deleted
    verify_res = client.get(f"/assessments/{aid}")
    assert verify_res.status_code == 404

def test_assessment_validation_mismatched_marks():
    payload = {
        "title": "Invalid Marks Assessment",
        "question": "Explain gradient descent.",
        "model_answer": "An optimization algorithm.",
        "total_marks": 10.0,
        "criteria": [
            {
                "name": "Concept",
                "description": "Desc",
                "max_marks": 4.0  # Sum is 4.0, but total_marks is 10.0
            }
        ]
    }
    res = client.post("/assessments", json=payload)
    assert res.status_code == 400
    assert "Criterion marks sum" in res.json()["detail"]
