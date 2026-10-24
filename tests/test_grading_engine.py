import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db

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
