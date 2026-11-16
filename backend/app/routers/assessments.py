import os
import uuid
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from backend.app.database import get_db
from backend.app.models import (
    AssessmentCreate,
    AssessmentUpdate,
    AssessmentResponse,
    AssessmentListResponse,
    RubricCriterionResponse,
    AssessmentQuestionCreate,
    AssessmentQuestionResponse
)
from backend.app.services.assessment_parser import AssessmentDocumentParser

logger = logging.getLogger(__name__)

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
                max_marks=float(c["max_marks"]),
                question_id=c["question_id"] if "question_id" in c.keys() else None
            )
            for c in c_rows
        ]

        cursor.execute(
            "SELECT * FROM assessment_questions WHERE assessment_id = ? ORDER BY question_order ASC",
            (assessment_id,)
        )
        q_rows = cursor.fetchall()
        questions: List[AssessmentQuestionResponse] = []
        if q_rows:
            for qr in q_rows:
                q_id = qr["id"]
                q_criteria = [c for c in criteria if c.question_id == q_id]
                questions.append(
                    AssessmentQuestionResponse(
                        id=q_id,
                        assessment_id=assessment_id,
                        question_order=qr["question_order"],
                        question_text=qr["question_text"],
                        model_answer=qr["model_answer"],
                        max_marks=float(qr["max_marks"]),
                        criteria=q_criteria
                    )
                )
        else:
            questions = [
                AssessmentQuestionResponse(
                    id=str(uuid.uuid4()),
                    assessment_id=assessment_id,
                    question_order=1,
                    question_text=a_row["question"],
                    model_answer=a_row["model_answer"],
                    max_marks=float(a_row["total_marks"]),
                    criteria=criteria
                )
            ]

        primary_question = questions[0].question_text if questions else a_row["question"]
        primary_model_answer = (questions[0].model_answer or "") if questions else a_row["model_answer"]

        return AssessmentResponse(
            id=a_row["id"],
            title=a_row["title"],
            question=primary_question,
            model_answer=primary_model_answer,
            total_marks=float(a_row["total_marks"]),
            doc_scope=a_row["doc_scope"],
            criteria=criteria,
            questions=questions,
            created_at=a_row["created_at"],
            updated_at=a_row["updated_at"]
        )

@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(data: AssessmentCreate):
    if not data.title.strip():
        raise HTTPException(status_code=400, detail="Assessment title is required.")

    assessment_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    if data.questions and len(data.questions) > 0:
        total_marks = 0.0
        parsed_questions = []

        for q_idx, q in enumerate(data.questions):
            if not q.question_text.strip():
                raise HTTPException(status_code=400, detail=f"Question #{q_idx + 1} text is required.")
            if not q.criteria or len(q.criteria) == 0:
                raise HTTPException(status_code=400, detail=f"Question #{q_idx + 1} requires at least one rubric criterion.")
            q_crit_sum = sum(c.max_marks for c in q.criteria)
            q_max_marks = q.max_marks if (q.max_marks is not None and q.max_marks > 0) else q_crit_sum
            if abs(q_crit_sum - q_max_marks) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Question #{q_idx + 1} criteria marks sum ({q_crit_sum:.2f}) does not equal question max marks ({q_max_marks:.2f})."
                )
            total_marks += q_max_marks
            parsed_questions.append((q, q_max_marks))

        if data.total_marks is not None and abs(total_marks - data.total_marks) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Sum of question marks ({total_marks:.2f}) does not equal assessment total marks ({data.total_marks:.2f})."
            )

        primary_question = parsed_questions[0][0].question_text.strip()
        primary_model_answer = parsed_questions[0][0].model_answer.strip() if parsed_questions[0][0].model_answer else ""

        with get_db() as conn:
            conn.execute("""
                INSERT INTO assessments (
                    id, title, question, model_answer, total_marks, doc_scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                assessment_id,
                data.title.strip(),
                primary_question,
                primary_model_answer,
                total_marks,
                data.doc_scope.strip() if data.doc_scope else "ALL",
                now_iso,
                now_iso
            ))

            global_order = 1
            for q_idx, (q, q_max) in enumerate(parsed_questions):
                qid = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO assessment_questions (
                        id, assessment_id, question_order, question_text, model_answer, max_marks
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    qid,
                    assessment_id,
                    q_idx + 1,
                    q.question_text.strip(),
                    q.model_answer.strip() if q.model_answer else None,
                    q_max
                ))
                for crit in q.criteria:
                    crit_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO rubric_criteria (
                            id, assessment_id, question_id, order_index, name, description, max_marks
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        crit_id,
                        assessment_id,
                        qid,
                        global_order,
                        crit.name.strip(),
                        crit.description.strip(),
                        crit.max_marks
                    ))
                    global_order += 1
    else:
        # Legacy single-question assessment creation
        if not data.question or not data.question.strip():
            raise HTTPException(status_code=400, detail="Question is required.")
        if not data.model_answer or not data.model_answer.strip():
            raise HTTPException(status_code=400, detail="Model answer is required.")
        if not data.criteria or len(data.criteria) == 0:
            raise HTTPException(status_code=400, detail="At least one rubric criterion is required.")

        criteria_sum = sum(c.max_marks for c in data.criteria)
        total_marks = data.total_marks if data.total_marks is not None else criteria_sum
        if abs(criteria_sum - total_marks) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Criterion marks sum ({criteria_sum:.2f}) does not equal assessment total marks ({total_marks:.2f})."
            )

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
                total_marks,
                data.doc_scope.strip() if data.doc_scope else "ALL",
                now_iso,
                now_iso
            ))

            qid = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO assessment_questions (
                    id, assessment_id, question_order, question_text, model_answer, max_marks
                ) VALUES (?, ?, 1, ?, ?, ?)
            """, (
                qid,
                assessment_id,
                data.question.strip(),
                data.model_answer.strip(),
                total_marks
            ))

            for idx, crit in enumerate(data.criteria):
                crit_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO rubric_criteria (
                        id, assessment_id, question_id, order_index, name, description, max_marks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    crit_id,
                    assessment_id,
                    qid,
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

    if not data.title.strip():
        raise HTTPException(status_code=400, detail="Assessment title is required.")

    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        if data.questions and len(data.questions) > 0:
            total_marks = 0.0
            parsed_questions = []

            for q_idx, q in enumerate(data.questions):
                if not q.question_text.strip():
                    raise HTTPException(status_code=400, detail=f"Question #{q_idx + 1} text is required.")
                if not q.criteria or len(q.criteria) == 0:
                    raise HTTPException(status_code=400, detail=f"Question #{q_idx + 1} requires at least one rubric criterion.")
                q_crit_sum = sum(c.max_marks for c in q.criteria)
                q_max_marks = q.max_marks if (q.max_marks is not None and q.max_marks > 0) else q_crit_sum
                if abs(q_crit_sum - q_max_marks) > 0.01:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Question #{q_idx + 1} criteria marks sum ({q_crit_sum:.2f}) does not equal question max marks ({q_max_marks:.2f})."
                    )
                total_marks += q_max_marks
                parsed_questions.append((q, q_max_marks))

            if data.total_marks is not None and abs(total_marks - data.total_marks) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sum of question marks ({total_marks:.2f}) does not equal assessment total marks ({data.total_marks:.2f})."
                )

            primary_question = parsed_questions[0][0].question_text.strip()
            primary_model_answer = parsed_questions[0][0].model_answer.strip() if parsed_questions[0][0].model_answer else ""

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
                primary_question,
                primary_model_answer,
                total_marks,
                data.doc_scope.strip() if data.doc_scope else "ALL",
                now_iso,
                assessment_id
            ))

            # Cascade delete old criteria and questions
            conn.execute("DELETE FROM rubric_criteria WHERE assessment_id = ?", (assessment_id,))
            conn.execute("DELETE FROM assessment_questions WHERE assessment_id = ?", (assessment_id,))

            global_order = 1
            for q_idx, (q, q_max) in enumerate(parsed_questions):
                qid = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO assessment_questions (
                        id, assessment_id, question_order, question_text, model_answer, max_marks
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    qid,
                    assessment_id,
                    q_idx + 1,
                    q.question_text.strip(),
                    q.model_answer.strip() if q.model_answer else None,
                    q_max
                ))
                for crit in q.criteria:
                    crit_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO rubric_criteria (
                            id, assessment_id, question_id, order_index, name, description, max_marks
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        crit_id,
                        assessment_id,
                        qid,
                        global_order,
                        crit.name.strip(),
                        crit.description.strip(),
                        crit.max_marks
                    ))
                    global_order += 1
        else:
            if not data.question or not data.question.strip():
                raise HTTPException(status_code=400, detail="Question is required.")
            if not data.model_answer or not data.model_answer.strip():
                raise HTTPException(status_code=400, detail="Model answer is required.")
            if not data.criteria or len(data.criteria) == 0:
                raise HTTPException(status_code=400, detail="At least one rubric criterion is required.")

            criteria_sum = sum(c.max_marks for c in data.criteria)
            total_marks = data.total_marks if data.total_marks is not None else criteria_sum
            if abs(criteria_sum - total_marks) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail=f"Criterion marks sum ({criteria_sum:.2f}) does not equal total marks ({total_marks:.2f})."
                )

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
                total_marks,
                data.doc_scope.strip() if data.doc_scope else "ALL",
                now_iso,
                assessment_id
            ))

            conn.execute("DELETE FROM rubric_criteria WHERE assessment_id = ?", (assessment_id,))
            conn.execute("DELETE FROM assessment_questions WHERE assessment_id = ?", (assessment_id,))

            qid = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO assessment_questions (
                    id, assessment_id, question_order, question_text, model_answer, max_marks
                ) VALUES (?, ?, 1, ?, ?, ?)
            """, (
                qid,
                assessment_id,
                data.question.strip(),
                data.model_answer.strip(),
                total_marks
            ))

            for idx, crit in enumerate(data.criteria):
                crit_id = str(uuid.uuid4())
                conn.execute("""
                    INSERT INTO rubric_criteria (
                        id, assessment_id, question_id, order_index, name, description, max_marks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    crit_id,
                    assessment_id,
                    qid,
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


@router.post("/parse-document", status_code=status.HTTP_200_OK)
async def parse_assessment_document(file: UploadFile = File(...)):
    """
    Parses an uploaded question paper or rubric document (.pdf, .docx, .txt, .md),
    linearizing tables and extracting assessment title, questions, model answers,
    and rubric criteria for teacher review before saving.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only PDF, DOCX, TXT, and MD files are supported."
        )

    try:
        file_bytes = await file.read()
        parsed_data = AssessmentDocumentParser.parse_document(file.filename, file_bytes)
        return parsed_data
    except Exception as e:
        logger.exception(f"Error parsing assessment document '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse assessment document: {str(e)}")
