import time
import json
import sqlite3
from backend.app.config import settings
from backend.app.routers.assessments import _fetch_assessment_by_id
from backend.app.services.models_loader import ModelsLoader
from backend.app.services.retriever import HybridRetriever
from backend.app.services.grading_engine import GradingEngine
from backend.app.services.llm_feedback import LLMFeedbackService
from backend.app.models import AssessmentResponse

# Connect to database and fetch assessment
conn = sqlite3.connect(str(settings.SQLITE_DB_PATH))
c = conn.cursor()
c.execute("SELECT id FROM assessments LIMIT 1")
row = c.fetchone()
if not row:
    print("No assessment found in database.")
    exit(1)

assessment_id = row[0]
assessment = _fetch_assessment_by_id(assessment_id)
print(f"Profiling assessment: {assessment.title} ({assessment_id})")
print(f"Criteria count: {len(assessment.criteria)}")

student_text = (
    "Memory-based learning is a classification technique that uses previously stored training examples to classify new cases. "
    "It represents examples using feature vectors and compares them using a similarity or distance measure. "
    "MBL is a flexible and efficient alternative to statistical language modeling. "
    "Its uses in NLP include part-of-speech tagging, shallow parsing, morphological analysis, named entity recognition, and machine translation."
)

# 1. Cold start timing
print("\n--- MEASURING COLD START MODEL LOADING ---")
t0 = time.perf_counter()
_ = ModelsLoader.get_embedding_model()
t_emb = time.perf_counter() - t0
print(f"SentenceTransformer load time: {t_emb:.3f}s")

t0 = time.perf_counter()
_ = ModelsLoader.get_reranker_model()
t_rerank = time.perf_counter() - t0
print(f"CrossEncoder Reranker load time: {t_rerank:.3f}s")

t0 = time.perf_counter()
_ = ModelsLoader.get_nli_model()
t_nli = time.perf_counter() - t0
print(f"CrossEncoder NLI load time: {t_nli:.3f}s")

# 2. Detailed pipeline stage profiling (Warmed up)
print("\n--- MEASURING DETAILED STAGES (WARMED UP) ---")

# Stage A: Text preparation and excerpts
t0 = time.perf_counter()
student_sentences = GradingEngine._split_sentences(student_text)
excerpts = GradingEngine._extract_criterion_excerpts(student_text, assessment.criteria)
t_prep = time.perf_counter() - t0
print(f"1. Text extraction & excerpt segmentation: {t_prep*1000:.2f}ms")

# Stage B: Evidence retrieval across criteria
t0 = time.perf_counter()
all_citations = []
doc_scope = [d.strip() for d in assessment.doc_scope.split(",")] if assessment.doc_scope and assessment.doc_scope != "ALL" else None
for crit in assessment.criteria:
    q = f"{assessment.question} {crit.name} {crit.description}"
    cits = HybridRetriever.retrieve_evidence(query=q, top_k=2, doc_scope=doc_scope)
    all_citations.extend(cits)
t_retrieval = time.perf_counter() - t0
print(f"2. Hybrid retrieval (Chroma + BM25 + Reranker) for {len(assessment.criteria)} criteria: {t_retrieval:.3f}s ({t_retrieval/len(assessment.criteria)*1000:.1f}ms/criterion)")

# Stage C: Proposition extraction and numerical grading (Embedding + NLI)
t0 = time.perf_counter()
# We run evaluate_criterion without Ollama call to isolate scoring latency
crit_results = []
for crit in assessment.criteria:
    # Measure just the scoring part
    # We can invoke evaluate_criterion with a mocked LLM feedback to isolate scoring time
    pass
t_eval = time.perf_counter() - t0

# Let's run full evaluate_criterion with profiling
from unittest.mock import patch
t0 = time.perf_counter()
with patch.object(LLMFeedbackService, "generate_criterion_feedback", return_value="Verified criterion."):
    for crit in assessment.criteria:
        res, _ = GradingEngine.evaluate_criterion(
            question=assessment.question,
            criterion_name=crit.name,
            criterion_description=crit.description,
            max_marks=crit.max_marks,
            model_answer=assessment.model_answer,
            student_text=student_text,
            doc_scope=doc_scope,
            relevant_excerpt=excerpts.get(crit.name, student_text)
        )
        crit_results.append(res)
t_scoring_only = time.perf_counter() - t0
print(f"3. Embeddings, candidate window comparisons, & NLI scoring (total for {len(assessment.criteria)} criteria): {t_scoring_only:.3f}s ({t_scoring_only/len(assessment.criteria)*1000:.1f}ms/criterion)")

# Stage D: LLM feedback generation (Ollama)
t0 = time.perf_counter()
crit_feedbacks = []
for crit in assessment.criteria:
    fb = LLMFeedbackService.generate_criterion_feedback(
        question=assessment.question,
        criterion_name=crit.name,
        criterion_description=crit.description,
        awarded_marks=crit.max_marks,
        max_marks=crit.max_marks,
        student_text=student_text,
        contradictions=[],
        missing_concepts=[],
        citations=[],
        target_propositions=[crit.description]
    )
    crit_feedbacks.append(fb)
t_ollama = time.perf_counter() - t0
print(f"4. Ollama LLM feedback generation for {len(assessment.criteria)} criteria: {t_ollama:.3f}s ({t_ollama/len(assessment.criteria):.2f}s/criterion)")

# Stage E: Complete end-to-end grade_submission
t0 = time.perf_counter()
full_res = GradingEngine.grade_submission(assessment=assessment, student_text=student_text)
t_e2e = time.perf_counter() - t0
print(f"\n5. Total End-to-End grade_submission (warmed up): {t_e2e:.3f}s")
print(f"   Score awarded: {full_res.total_score} / {full_res.max_score}")
