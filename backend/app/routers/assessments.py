import uuid
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, status
from backend.app.database import get_db
from backend.app.models import (
    AssessmentCreate,
    AssessmentUpdate,
    AssessmentResponse,
    AssessmentListResponse,
    RubricCriterionResponse
)

router = APIRouter(prefix="/assessments", tags=["Assessments & Rubrics"])

def _fetch_assessment_by_id(assessment_id: str) -> AssessmentResponse:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
        a_row = cursor.fetchone()
        if not a_row:
            raise HTTPException(status_code=404, detail=f"Assessment with ID '{assessment_id}' not found.")

        cursor.execute(
            "SELECT * FROM rubric_criteria WHERE assessment_id = ? ORDER BY order_index ASC",
            (assessment_id,)
        )
        c_rows = cursor.fetchall()
        criteria = [
            RubricCriterionResponse(
                id=c["id"],
                order_index=c["order_index"],
                name=c["name"],
                description=c["description"],
                max_marks=float(c["max_marks"])
            )
            for c in c_rows
        ]

        return AssessmentResponse(
            id=a_row["id"],
            title=a_row["title"],
            question=a_row["question"],
            model_answer=a_row["model_answer"],
            total_marks=float(a_row["total_marks"]),
            doc_scope=a_row["doc_scope"],
            criteria=criteria,
            created_at=a_row["created_at"],
            updated_at=a_row["updated_at"]
        )

@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(data: AssessmentCreate):
    # Validation 1: Mandatory fields
    if not data.title.strip():
        raise HTTPException(status_code=400, detail="Assessment title is required.")
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not data.model_answer.strip():
        raise HTTPException(status_code=400, detail="Model answer is required.")
    if not data.criteria or len(data.criteria) == 0:
        raise HTTPException(status_code=400, detail="At least one rubric criterion is required.")

    # Validation 2: Sum of criteria marks must equal total marks
    criteria_sum = sum(c.max_marks for c in data.criteria)
    if abs(criteria_sum - data.total_marks) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Criterion marks sum ({criteria_sum:.2f}) does not equal assessment total marks ({data.total_marks:.2f})."
        )

    assessment_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute("""
            INSERT INTO assessments (
                id, title, question, model_answer, total_marks, doc_scope, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_id,
            data.title.strip(),
            data.question.strip(),
            data.model_answer.strip(),
            data.total_marks,
            data.doc_scope.strip() if data.doc_scope else "ALL",
            now_iso,
            now_iso
        ))

        for idx, crit in enumerate(data.criteria):
            crit_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO rubric_criteria (
                    id, assessment_id, order_index, name, description, max_marks
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                crit_id,
                assessment_id,
                idx + 1,
                crit.name.strip(),
                crit.description.strip(),
                crit.max_marks
            ))

    return _fetch_assessment_by_id(assessment_id)

@router.get("", response_model=AssessmentListResponse)
def list_assessments():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM assessments ORDER BY created_at DESC")
        ids = [row["id"] for row in cursor.fetchall()]

    assessments = [_fetch_assessment_by_id(aid) for aid in ids]
    return AssessmentListResponse(assessments=assessments, total=len(assessments))

@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(assessment_id: str):
    return _fetch_assessment_by_id(assessment_id)

@router.put("/{assessment_id}", response_model=AssessmentResponse)
def update_assessment(assessment_id: str, data: AssessmentUpdate):
    # Check exists
    _fetch_assessment_by_id(assessment_id)

    # Validation
    if not data.title.strip():
        raise HTTPException(status_code=400, detail="Assessment title is required.")
    if not data.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    if not data.model_answer.strip():
        raise HTTPException(status_code=400, detail="Model answer is required.")
    if not data.criteria or len(data.criteria) == 0:
        raise HTTPException(status_code=400, detail="At least one rubric criterion is required.")

    criteria_sum = sum(c.max_marks for c in data.criteria)
    if abs(criteria_sum - data.total_marks) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Criterion marks sum ({criteria_sum:.2f}) does not equal total marks ({data.total_marks:.2f})."
        )

    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute("""
            UPDATE assessments
            SET title = ?,
                question = ?,
                model_answer = ?,
                total_marks = ?,
                doc_scope = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            data.title.strip(),
            data.question.strip(),
            data.model_answer.strip(),
            data.total_marks,
            data.doc_scope.strip() if data.doc_scope else "ALL",
            now_iso,
            assessment_id
        ))

        # Re-populate criteria
        conn.execute("DELETE FROM rubric_criteria WHERE assessment_id = ?", (assessment_id,))
        for idx, crit in enumerate(data.criteria):
            crit_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO rubric_criteria (
                    id, assessment_id, order_index, name, description, max_marks
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                crit_id,
                assessment_id,
                idx + 1,
                crit.name.strip(),
                crit.description.strip(),
                crit.max_marks
            ))

    return _fetch_assessment_by_id(assessment_id)

@router.delete("/{assessment_id}", status_code=status.HTTP_200_OK)
def delete_assessment(assessment_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM assessments WHERE id = ?", (assessment_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Assessment not found.")

        # Deletes assessment and cascades to rubric_criteria
        cursor.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))

    return {"status": "success", "message": f"Assessment '{row['title']}' deleted successfully."}
