from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Document schemas
# -------------------------------------------------------------
class DocumentResponse(BaseModel):
    id: str
    title: str
    doc_type: str
    filename: str
    file_size: int
    total_pages: Optional[int] = None
    total_chunks: int = 0
    status: str  # queued, processing, ready, failed
    error_message: Optional[str] = None
    created_at: str
    indexed_at: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

# -------------------------------------------------------------
# Rubric and Assessment schemas
# -------------------------------------------------------------
class RubricCriterionCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Criterion name")
    description: str = Field(..., min_length=1, description="Detailed concept explanation")
    max_marks: float = Field(..., gt=0, description="Max marks for this criterion")

class RubricCriterionResponse(BaseModel):
    id: str
    order_index: int
    name: str
    description: str
    max_marks: float

class AssessmentCreate(BaseModel):
    title: str = Field(..., min_length=1, description="Assessment title")
    question: str = Field(..., min_length=1, description="Question or assignment prompt")
    model_answer: str = Field(..., min_length=1, description="Authoritative model answer")
    total_marks: float = Field(..., gt=0, description="Total maximum marks")
    doc_scope: str = Field("ALL", description="'ALL' or comma-separated document IDs")
    criteria: List[RubricCriterionCreate] = Field(..., min_length=1, description="Rubric criteria")

class AssessmentUpdate(BaseModel):
    title: str
    question: str
    model_answer: str
    total_marks: float
    doc_scope: str = "ALL"
    criteria: List[RubricCriterionCreate]

class AssessmentResponse(BaseModel):
    id: str
    title: str
    question: str
    model_answer: str
    total_marks: float
    doc_scope: str
    criteria: List[RubricCriterionResponse]
    created_at: str
    updated_at: str

class AssessmentListResponse(BaseModel):
    assessments: List[AssessmentResponse]
    total: int

# -------------------------------------------------------------
# Grading schemas
# -------------------------------------------------------------
class GradeTextRequest(BaseModel):
    assessment_id: str
    student_text: str = Field(..., description="Student's answer text")
    feedback_mode: Optional[str] = Field(None, description="'hybrid' or 'llm'. If omitted, uses server setting.")

class EvidenceCitation(BaseModel):
    document_id: str
    document_title: str
    page_number: Optional[int] = None
    snippet: str
    similarity_score: float

class UCEScores(BaseModel):
    understanding: float = Field(..., description="Grasp of essential concepts [0.0 - 1.0]")
    clarity: float = Field(..., description="Coherence, structural lucidity [0.0 - 1.0]")
    expression: float = Field(..., description="Technical vocabulary precision, absence of noise [0.0 - 1.0]")

class CriterionGradingResult(BaseModel):
    criterion_name: str
    max_marks: float
    awarded_marks: float
    awarded_percentage: float
    reason: str
    feedback: str
    contradictions_detected: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)
    present_concepts: List[str] = Field(default_factory=list)
    uce_signals: Optional[UCEScores] = None
    supporting_citations: List[EvidenceCitation] = Field(default_factory=list)

class GradingResponse(BaseModel):
    assessment_id: str
    assessment_title: str
    total_score: float
    max_score: float
    percentage: float
    criterion_breakdown: List[CriterionGradingResult]
    overall_feedback: str
    requires_teacher_review: bool
    review_reasons: List[str]
    citations: List[EvidenceCitation]
    provenance: Dict[str, Any]

# -------------------------------------------------------------
# Health and System Status
# -------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    ollama_ready: bool
    ollama_model: str
    models_status: Dict[str, bool]
    storage_status: Dict[str, Any]


# -------------------------------------------------------------
# Batch Grading Schemas
# -------------------------------------------------------------
class BatchItemResult(BaseModel):
    filename: str
    status: str  # "completed", "failed", "skipped"
    total_score: Optional[float] = None
    max_score: Optional[float] = None
    percentage: Optional[float] = None
    requires_teacher_review: Optional[bool] = None
    criterion_breakdown: Optional[List[CriterionGradingResult]] = None
    error_message: Optional[str] = None
    student_text_preview: Optional[str] = None
    duration_s: Optional[float] = None
    ollama_calls: Optional[int] = None

class BatchJobResponse(BaseModel):
    batch_id: str
    assessment_id: str
    assessment_title: str
    total_files: int
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    status: str = "processing"  # "processing", "completed", "failed"
    created_at: str
    completed_at: Optional[str] = None
    total_duration_s: Optional[float] = None
    results: List[BatchItemResult] = []
