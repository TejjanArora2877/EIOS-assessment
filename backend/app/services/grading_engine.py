import math
import re
import time
import logging
import hashlib
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

from backend.app.config import settings
from backend.app.models import (
    AssessmentResponse,
    CriterionGradingResult,
    EvidenceCitation,
    GradingResponse,
    UCEScores
)
from backend.app.services.models_loader import ModelsLoader
from backend.app.services.retriever import HybridRetriever
from backend.app.services.llm_feedback import LLMFeedbackService

class GradingEngine:
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Splits text into meaningful sentences."""
        if not text:
            return []
        raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
        cleaned = [s.strip() for s in raw_sents if len(s.strip()) > 5]
        return cleaned if cleaned else [text.strip()]

    @staticmethod
    def _calculate_shannon_entropy(text: str) -> float:
        """
        Calculates information density via Shannon entropy:
        H(X) = - sum(p(w) * log2(p(w)))
        """
        words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
        if not words:
            return 0.0
        counts = Counter(words)
        total = len(words)
        entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
        return entropy

    @classmethod
    def _evaluate_uce(
        cls,
        student_text: str,
        criterion_description: str,
        semantic_coverage: float,
        contradiction_count: int
    ) -> UCEScores:
        """
        UCE Cognitive Evaluation (Understanding, Clarity, Expression):
        - Understanding: concept alignment, grasp of core mechanisms, lack of contradictions [0.0 - 1.0]
        - Clarity: syntactic coherence, structural readability, appropriate sentence length [0.0 - 1.0]
        - Expression: lexical diversity, domain vocabulary usage, absence of noise [0.0 - 1.0]
        """
        if not student_text or not student_text.strip():
            return UCEScores(understanding=0.0, clarity=0.0, expression=0.0)

        # 1. Understanding
        understanding = max(0.0, min(1.0, semantic_coverage - (0.4 * contradiction_count)))

        # 2. Clarity
        sentences = cls._split_sentences(student_text)
        avg_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        # Ideal sentence length between 10 and 30 words
        if 8 <= avg_len <= 35:
            clarity = 0.9
        elif 5 <= avg_len < 8 or 35 < avg_len <= 50:
            clarity = 0.7
        else:
            clarity = 0.4

        # 3. Expression
        entropy = cls._calculate_shannon_entropy(student_text)
        # Shannon entropy > 3.0 indicates rich technical vocabulary
        if entropy >= 3.5:
            expression = 0.95
        elif entropy >= 2.5:
            expression = 0.80
        elif entropy >= 1.5:
            expression = 0.60
        else:
            expression = 0.30

        return UCEScores(
            understanding=round(understanding, 2),
            clarity=round(clarity, 2),
            expression=round(expression, 2)
        )

    @staticmethod
    def _is_administrative_sentence(text: str) -> bool:
        """
        Detects teacher grading meta-instructions rather than factual subject propositions.
        E.g.: 'Award partial marks...', 'Marks are awarded proportionately...', 'Deduct 1 mark if...'
        """
        patterns = [
            r'(?i)\b(award|deduct|allocate|give|assign)\s+(partial\s+|full\s+)?(marks?|points?|credit)\b',
            r'(?i)\bmarks?\s+(are|will be|should be|to be)\s+(awarded|deducted|allocated|given)\b',
            r'(?i)\b(award|deduct)\s+proportionately\b',
            r'(?i)\b(penalty|penalize)\b',
            r'(?i)\bpartial\s+marks?\b',
            r'(?i)\b(full\s+marks?|zero\s+marks?)\b',
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _clean_proposition(text: str, prev_text: str = "") -> str:
        """
        Strips teacher meta-prefixes from rubric criterion sentences to extract clean,
        authoritative factual propositions suitable for semantic embedding and NLI validation.
        """
        s = text.strip()
        # Remove instructional prefixes like 'The answer should explain that', 'The student must describe'
        s = re.sub(
            r'(?i)^(the\s+)?(answer|student|response)\s+(should|must|needs to|ought to)\s+'
            r'(explain|describe|state|identify|demonstrate|show|mention|discuss|include)\s+(that|how)?\s*',
            '',
            s
        )
        s = re.sub(r'(?i)^(explains?|describes?|states?)\s+(how|that)\s+', '', s)
        s = re.sub(r'(?i)^defines\s+', '', s)
        s = re.sub(r'(?i)^identifies\s+valid\s+', 'Valid ', s)
        s = re.sub(r'(?i)^identifies\s+', '', s)
        s = s.strip()

        # Transform noun-phrase definition prefix "X as a Y" -> "X is a Y"
        s = re.sub(r'(?i)^([A-Za-z0-9_() -]+?)\s+as\s+(an?|the)\s+', r'\1 is \2 ', s)

        # Resolve anaphoric demonstrative "these <noun>" if preceding sentence provides context
        if re.search(r'(?i)^these\s+(\w+)\b', s) and prev_text:
            m = re.search(r'(?i)\b\w+\b.*?\busing\s+(?:an?\s+)?([\w\s]+?)\s+(?:measures?|metrics?|functions?|methods?|techniques?|tools?|algorithms?)', prev_text)
            if m:
                mod = m.group(1).strip()
                s = re.sub(r'(?i)^these\s+(\w+)\b', f'{mod.capitalize()} \\1', s)
            else:
                s = re.sub(r'(?i)^these\s+', 'The ', s)

        if s:
            s = s[0].upper() + s[1:]
        return s

    @classmethod
    def _extract_propositions(
        cls,
        criterion_description: str,
        model_answer: str = ""
    ) -> List[str]:
        """
        Extracts substantive, factual target propositions from the rubric criterion description.
        Filters administrative grading instructions and normalizes instructional language.
        Splits compound clauses on coordinating predicate verbs into atomic propositions.
        """
        raw_sents = cls._split_sentences(criterion_description)
        substantive = [s for s in raw_sents if not cls._is_administrative_sentence(s)]

        # Split compound clauses on coordinating predicate verbs (e.g. "...and recognizes it as...")
        expanded_sents = []
        for s in substantive:
            parts = re.split(
                r'(?i)\s+and\s+(?:recognizes?|identifies?|explains?|describes?|states?|considers?)\s+(?:it\s+as\s+|that\s+)?',
                s
            )
            if len(parts) > 1 and all(len(p.strip()) > 5 for p in parts):
                expanded_sents.append(parts[0].strip())
                for p in parts[1:]:
                    p_clean = p.strip()
                    if re.match(r'(?i)^(an?|the)\s+', p_clean):
                        expanded_sents.append(f"It is {p_clean}")
                    else:
                        expanded_sents.append(p_clean)
            else:
                expanded_sents.append(s)

        cleaned = []
        for i, s in enumerate(expanded_sents):
            prev = expanded_sents[i-1] if i > 0 else ""
            c = cls._clean_proposition(s, prev)
            if c:
                cleaned.append(c)

        # If all sentences were filtered or empty, fall back to cleaned whole description
        if not cleaned:
            clean_whole = cls._clean_proposition(criterion_description)
            cleaned = [clean_whole] if clean_whole else [criterion_description]

        return cleaned

    @staticmethod
    def _extract_content_tokens(text: str) -> Set[str]:
        """
        Extracts substantive content tokens and root stems excluding grammatical stop words.
        Provides vocabulary-level grounding to prevent superficial topical embeddings from
        falsely awarding positive marks for absent requirements.
        """
        stop = {
            "this", "that", "with", "from", "have", "more", "also", "some", "such",
            "than", "they", "them", "their", "should", "could", "would", "must",
            "about", "which", "where", "when", "what", "into", "these", "other",
            "uses", "used", "using", "help", "helps", "well", "step", "steps",
            "answer", "student", "explain", "explains", "describe", "describes",
            "states", "identifies", "defined", "defines", "including", "includes"
        }
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        roots = set()
        for w in words:
            if w not in stop:
                roots.add(w)
                if w.endswith("ing") and len(w) > 5:
                    roots.add(w[:-3])
                elif w.endswith("ed") and len(w) > 4:
                    roots.add(w[:-2])
                elif w.endswith("es") and len(w) > 4:
                    roots.add(w[:-2])
                elif w.endswith("s") and len(w) > 4:
                    roots.add(w[:-1])
        return roots

    @classmethod
    def _extract_enumeration_items(cls, criterion_description: str) -> Tuple[List[str], bool]:
        """
        Extracts illustrative list items (e.g. from 'such as X, Y, Z') and checks if proportional grading applies.
        """
        m = re.search(r'(?i)\b(such as|including|e\.g\.?,?|for example)\s+([^.]*)', criterion_description)
        if not m:
            return [], False

        raw_list = m.group(2)
        raw_items = re.split(r',|\bor\b|\band\b', raw_list)
        items = [re.sub(r'^\s*(such as|including|e\.g\.?)\s*', '', it).strip() for it in raw_items]
        items = [it for it in items if len(it) > 3]

        is_proportional = bool(re.search(r'(?i)\b(proportionately|partial marks|number and relevance|list some)\b', criterion_description))
        return items, is_proportional

    @classmethod
    def _evaluate_enumeration_coverage(cls, items: List[str], student_text: str) -> Tuple[float, List[str], List[str]]:
        """
        Evaluates how many enumerated items are mentioned in the student answer.
        Returns (coverage_ratio, matched_items, missing_items).
        """
        if not items:
            return 1.0, [], []

        chunks = [s.strip() for s in re.split(r'[,.]|\band\b', student_text) if len(s.strip()) > 3]
        if not chunks:
            return 0.0, [], items

        item_embs = ModelsLoader.encode_texts(items)
        chunk_embs = ModelsLoader.encode_texts(chunks)
        sim_matrix = cosine_similarity(item_embs, chunk_embs)

        matched = []
        missing = []

        for i, it in enumerate(items):
            best_sim = float(sim_matrix[i].max())
            it_lower = it.lower()
            text_lower = student_text.lower()

            # Exact or token match or high semantic embedding match
            if it_lower in text_lower or best_sim >= 0.65:
                matched.append(it)
            else:
                missing.append(it)

        # Benchmark: 4 items (or all items if total < 4) yields 100% full marks
        target_benchmark = min(4, len(items))
        ratio = min(1.0, len(matched) / target_benchmark) if target_benchmark > 0 else 1.0
        return ratio, matched, missing

    @classmethod
    def _fuzzy_similarity_multiplier(cls, sim: float, lower_bound: float = 0.35, upper_bound: float = 0.65) -> float:
        """Continuous fuzzy mapping of cosine similarity."""
        if sim >= upper_bound:
            return 1.0
        elif sim <= lower_bound:
            return 0.0
        else:
            return (sim - lower_bound) / (upper_bound - lower_bound)

    @classmethod
    def _extract_criterion_excerpts(
        cls,
        student_text: str,
        criteria: List[Any]
    ) -> Dict[str, str]:
        """
        Dynamically segments student submission into criterion-specific excerpts based on
        proposition semantic affinity. Avoids cross-criterion bleed in feedback generation.
        """
        sentences = cls._split_sentences(student_text)
        if not sentences:
            return {c.name: "" for c in criteria}

        cand_embs = ModelsLoader.encode_texts(sentences)
        crit_sims = {}
        for crit in criteria:
            props = cls._extract_propositions(crit.description, None)
            if not props:
                props = [crit.description]
            prop_embs = ModelsLoader.encode_texts(props)
            sim_matrix = cosine_similarity(prop_embs, cand_embs)
            crit_sims[crit.name] = sim_matrix.max(axis=0)

        crit_names = [c.name for c in criteria]
        all_sims = np.array([crit_sims[name] for name in crit_names])
        best_crit_idx = all_sims.argmax(axis=0)

        criterion_sentences: Dict[str, List[str]] = {name: [] for name in crit_names}
        for s_idx, s in enumerate(sentences):
            b_idx = int(best_crit_idx[s_idx])
            assigned_name = crit_names[b_idx]
            best_sim = float(all_sims[b_idx, s_idx])
            if best_sim >= 0.30:
                criterion_sentences[assigned_name].append(s)
            for c_idx, c_name in enumerate(crit_names):
                if c_idx != b_idx:
                    c_sim = float(all_sims[c_idx, s_idx])
                    if c_sim >= 0.50 and c_sim >= best_sim - 0.08:
                        criterion_sentences[c_name].append(s)

        excerpts = {}
        for name in crit_names:
            sents = criterion_sentences[name]
            excerpts[name] = " ".join(sents) if sents else ""
        return excerpts

    @classmethod
    def evaluate_criterion(
        cls,
        question: str,
        criterion_name: str,
        criterion_description: str,
        max_marks: float,
        model_answer: str,
        student_text: str,
        doc_scope: Optional[List[str]] = None,
        all_assessment_props: Optional[List[str]] = None,
        all_criteria_descriptions: Optional[Dict[str, str]] = None,
        relevant_excerpt: Optional[str] = None,
        precomputed_windows: Optional[List[str]] = None,
        precomputed_cand_embs: Optional[np.ndarray] = None,
        stage_timings: Optional[Dict[str, float]] = None,
        feedback_mode: Optional[str] = None,
        feedback_stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[CriterionGradingResult, List[EvidenceCitation]]:
        """
        Evaluates a single rubric criterion deterministically.
        """
        # Retrieve relevant institutional evidence
        t_ret0 = time.perf_counter()
        evidence_query = f"{question} {criterion_name} {criterion_description}"
        citations = HybridRetriever.retrieve_evidence(
            query=evidence_query,
            top_k=2,
            doc_scope=doc_scope
        )
        t_ret_ms = (time.perf_counter() - t_ret0) * 1000

        # Handle empty student answer contract
        if not student_text or not student_text.strip():
            if stage_timings is not None:
                stage_timings["retrieval_ms"] = t_ret_ms
                stage_timings["scoring_ms"] = 0.0
                stage_timings["feedback_ms"] = 0.0
            res = CriterionGradingResult(
                criterion_name=criterion_name,
                max_marks=max_marks,
                awarded_marks=0.0,
                awarded_percentage=0.0,
                reason="No response provided by student.",
                feedback="No answer submitted. Expected concepts were not addressed.",
                contradictions_detected=[],
                missing_concepts=[criterion_description],
                present_concepts=[],
                uce_signals=None,
                supporting_citations=citations
            )
            return res, citations

        t_score0 = time.perf_counter()
        student_sentences = cls._split_sentences(student_text)
        if precomputed_windows is not None and precomputed_cand_embs is not None:
            candidate_windows = precomputed_windows
            cand_embs = precomputed_cand_embs
        else:
            candidate_windows = list(student_sentences)
            for i in range(len(student_sentences) - 1):
                candidate_windows.append(f"{student_sentences[i]} {student_sentences[i+1]}")
            if len(student_sentences) > 2:
                for i in range(2, len(student_sentences)):
                    candidate_windows.append(f"{student_sentences[0]} {student_sentences[i]}")
            candidate_windows.append(student_text)
            cand_embs = ModelsLoader.encode_texts(candidate_windows)

        # Extract authoritative substantive propositions (administrative instructions filtered)
        target_propositions = cls._extract_propositions(criterion_description, model_answer)

        # Check for enumeration list (e.g. applications, examples)
        enum_items, is_proportional = cls._extract_enumeration_items(criterion_description)

        proposition_multipliers = []
        contradictions_found = []
        missing_concepts = []
        present_concepts = []

        q_tokens = cls._extract_content_tokens(question)
        student_tokens = cls._extract_content_tokens(student_text)

        for prop in target_propositions:
            prop_emb = ModelsLoader.encode_texts([prop])[0]
            sims = cosine_similarity([prop_emb], cand_embs)[0]
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])
            best_cand = candidate_windows[best_idx]

            prop_tokens = cls._extract_content_tokens(prop)
            distinctive_prop_tokens = prop_tokens - q_tokens
            if not distinctive_prop_tokens or len(distinctive_prop_tokens) < 2:
                distinctive_prop_tokens = prop_tokens

            distinctive_overlap = distinctive_prop_tokens & student_tokens
            distinctive_ratio = len(distinctive_overlap) / len(distinctive_prop_tokens) if distinctive_prop_tokens else 1.0

            # 1. Candidate entailment check for technical paraphrases (batched across candidate windows)
            # Candidates that directly contradict the proposition must not count as valid entailment
            fwd_pairs = [(cand, prop) for cand in candidate_windows]
            fwd_results = ModelsLoader.predict_nli(fwd_pairs)
            valid_fwd_ents = [
                r["entailment"] for r in fwd_results
                if not (r["contradiction"] >= 0.50 and r["contradiction"] > r["entailment"])
            ]
            ent_score = max(valid_fwd_ents) if valid_fwd_ents else 0.0

            # Reverse entailment check for complex paraphrases (skipped if forward entailment already >= 0.65)
            if not is_proportional and ent_score < 0.65:
                rev_indices = [i for i, cand in enumerate(candidate_windows) if sims[i] >= 0.50]
                if rev_indices:
                    rev_pairs = [(prop, candidate_windows[i]) for i in rev_indices]
                    rev_results = ModelsLoader.predict_nli(rev_pairs)
                    for r in rev_results:
                        if r["entailment"] >= 0.70 and r["contradiction"] < 0.30:
                            ent_score = max(ent_score, r["entailment"] * 0.90)

            # 2. Contradiction check: explicit negation and high contradiction (batched across candidates with negations)
            is_contradiction = False
            if ent_score < 0.20:
                neg_indices = [
                    i for i, cand in enumerate(candidate_windows)
                    if re.search(r'(?i)\b(never|not|no|neither|nor|cannot|can\'t|won\'t|doesn\'t|don\'t|discards?|avoids?|rejects?|without|fails?|false|incorrect|incompatible|inefficient)\b', cand)
                ]
                if neg_indices:
                    contra_pairs = [(candidate_windows[i], prop) for i in neg_indices]
                    contra_results = ModelsLoader.predict_nli(contra_pairs)
                    for i_idx, r in zip(neg_indices, contra_results):
                        if r["contradiction"] >= 0.70:
                            is_contradiction = True
                            cand = candidate_windows[i_idx]
                            contradictions_found.append(
                                f"Student stated '{cand[:80]}...' which contradicts expected concept '{prop}'"
                            )
                            break

            # 3. Deterministic proposition scoring: concepts require genuine semantic entailment
            if is_contradiction:
                p_mult = 0.0
            elif ent_score >= 0.55:
                p_mult = 1.0 if ent_score >= 0.65 else ent_score
            elif ent_score >= 0.20:
                if best_sim >= 0.80 or (best_sim >= 0.50 and distinctive_ratio >= 0.45):
                    fuzzy = cls._fuzzy_similarity_multiplier(best_sim, 0.50, 0.85)
                    p_mult = min(1.0, 0.50 + 0.50 * max(ent_score, fuzzy, distinctive_ratio))
                else:
                    p_mult = min(0.35, ent_score * 0.8)
            else:
                if best_sim >= 0.85 and distinctive_ratio >= 0.75:
                    p_mult = 0.25
                else:
                    p_mult = 0.0

            if is_contradiction:
                missing_concepts.append(prop)
            elif p_mult >= 0.50:
                present_concepts.append(prop)
            else:
                missing_concepts.append(prop)

            proposition_multipliers.append(p_mult)

        # Base semantic coverage
        prop_coverage = sum(proposition_multipliers) / len(proposition_multipliers) if proposition_multipliers else 0.0

        # If criterion is an enumeration (e.g. applications), evaluate item coverage proportionately
        if enum_items and is_proportional:
            enum_ratio, matched_items, missing_items = cls._evaluate_enumeration_coverage(enum_items, student_text)
            semantic_coverage = enum_ratio
            # Strip full enumeration proposition from present and missing concepts to avoid leaking unrequired examples into LLM prompts
            present_concepts = [p for p in present_concepts if not any(it.lower() in p.lower() for it in enum_items[:2])]
            missing_concepts = [m for m in missing_concepts if not any(it.lower() in m.lower() for it in enum_items[:2])]
            if matched_items:
                present_concepts.append(f"Valid items identified: {', '.join(matched_items)}")
            if missing_items and len(matched_items) < 4:
                missing_concepts.append(f"Additional applications: {', '.join(missing_items[:3])}")
            feedback_target_props = [f"Identifies valid {criterion_name} tasks or applications (such as {', '.join(matched_items[:3] if matched_items else enum_items[:3])})"]
        else:
            semantic_coverage = prop_coverage
            feedback_target_props = target_propositions

        # Contradiction deduction if genuine conflicts occurred
        contra_penalty = 0.50 * len(contradictions_found)
        adjusted_coverage = max(0.0, semantic_coverage - contra_penalty)

        # Deterministic scoring bounded by criterion max
        if semantic_coverage >= 0.85 and not missing_concepts and not contradictions_found:
            awarded_marks = max_marks
        else:
            awarded_marks = round(max_marks * adjusted_coverage, 2)
        awarded_marks = max(0.0, min(max_marks, awarded_marks))
        awarded_percentage = round((awarded_marks / max_marks) * 100, 1) if max_marks > 0 else 0.0

        # Reason synthesis
        if contradictions_found:
            reason = f"Marks deducted due to contradictory statements against expected concepts."
        elif semantic_coverage >= 0.85:
            reason = f"Full or strong conceptual alignment with the rubric criteria."
        elif semantic_coverage >= 0.50:
            reason = f"Partial conceptual coverage; some expected details were omitted."
        else:
            reason = f"Insufficient conceptual coverage; missing primary expected details."

        # UCE cognitive metrics
        uce = cls._evaluate_uce(
            student_text=student_text,
            criterion_description=criterion_description,
            semantic_coverage=semantic_coverage,
            contradiction_count=len(contradictions_found)
        )

        # Ensure relevant_excerpt is established if not provided
        if relevant_excerpt is None:
            if len(student_sentences) > 2 and target_propositions:
                matched_s_indices = set()
                sent_embs = cand_embs[:len(student_sentences)]
                for prop in target_propositions:
                    prop_emb = ModelsLoader.encode_texts([prop])[0]
                    sims = cosine_similarity([prop_emb], sent_embs)[0]
                    best_s_idx = int(sims.argmax())
                    if sims[best_s_idx] >= 0.30:
                        matched_s_indices.add(best_s_idx)
                if matched_s_indices:
                    relevant_excerpt = " ".join([student_sentences[i] for i in sorted(matched_s_indices)])
                else:
                    relevant_excerpt = student_text
            else:
                relevant_excerpt = student_text

        t_score_ms = (time.perf_counter() - t_score0) * 1000

        # LLM Feedback generation
        t_fb0 = time.perf_counter()
        crit_fb_stats = {}
        feedback = LLMFeedbackService.generate_criterion_feedback(
            question=question,
            criterion_name=criterion_name,
            criterion_description=criterion_description,
            awarded_marks=awarded_marks,
            max_marks=max_marks,
            student_text=student_text,
            contradictions=contradictions_found,
            missing_concepts=missing_concepts,
            present_concepts=present_concepts,
            citations=citations,
            target_propositions=feedback_target_props,
            relevant_excerpt=relevant_excerpt,
            all_criteria_descriptions=all_criteria_descriptions,
            feedback_mode=feedback_mode,
            feedback_stats=crit_fb_stats
        )
        t_fb_ms = (time.perf_counter() - t_fb0) * 1000

        if feedback_stats is not None:
            feedback_stats.update(crit_fb_stats)

        if stage_timings is not None:
            stage_timings["retrieval_ms"] = t_ret_ms
            stage_timings["scoring_ms"] = t_score_ms
            stage_timings["feedback_ms"] = t_fb_ms
            stage_timings["ollama_called"] = crit_fb_stats.get("ollama_called", False)

        result = CriterionGradingResult(
            criterion_name=criterion_name,
            max_marks=max_marks,
            awarded_marks=awarded_marks,
            awarded_percentage=awarded_percentage,
            reason=reason,
            feedback=feedback,
            contradictions_detected=contradictions_found,
            missing_concepts=missing_concepts[:3],
            present_concepts=present_concepts[:3],
            uce_signals=None,
            supporting_citations=citations
        )
        return result, citations

    @classmethod
    def grade_submission(
        cls,
        assessment: AssessmentResponse,
        student_text: str,
        submission_type: str = "text",
        submission_filename: Optional[str] = None,
        feedback_mode: Optional[str] = None
    ) -> GradingResponse:
        """
        Main grading pipeline for both pasted text and file uploads.
        """
        t_overall_start = time.perf_counter()

        effective_feedback_mode = (feedback_mode or settings.FEEDBACK_MODE).lower()
        if effective_feedback_mode not in ("hybrid", "llm"):
            effective_feedback_mode = "hybrid"

        # Determine doc scope
        doc_scope = None
        if assessment.doc_scope and assessment.doc_scope != "ALL":
            doc_scope = [d.strip() for d in assessment.doc_scope.split(",") if d.strip()]

        criterion_breakdowns: List[CriterionGradingResult] = []
        all_citations: List[EvidenceCitation] = []
        review_reasons: List[str] = []

        total_awarded = 0.0

        t_prep0 = time.perf_counter()
        all_assessment_props = [
            p for c in assessment.criteria
            for p in cls._extract_propositions(c.description, assessment.model_answer)
        ]
        all_criteria_descriptions = {c.name: c.description for c in assessment.criteria}
        criterion_excerpts = cls._extract_criterion_excerpts(student_text, assessment.criteria)

        candidate_windows = []
        cand_embs = None
        if student_text and student_text.strip():
            student_sentences = cls._split_sentences(student_text)
            candidate_windows = list(student_sentences)
            for i in range(len(student_sentences) - 1):
                candidate_windows.append(f"{student_sentences[i]} {student_sentences[i+1]}")
            if len(student_sentences) > 2:
                for i in range(2, len(student_sentences)):
                    candidate_windows.append(f"{student_sentences[0]} {student_sentences[i]}")
            candidate_windows.append(student_text)
            cand_embs = ModelsLoader.encode_texts(candidate_windows)
        t_prep_ms = (time.perf_counter() - t_prep0) * 1000

        total_retrieval_ms = 0.0
        total_scoring_ms = 0.0
        total_feedback_ms = 0.0
        total_ollama_calls = 0

        for crit in assessment.criteria:
            crit_timings = {}
            crit_res, citations = cls.evaluate_criterion(
                question=assessment.question,
                criterion_name=crit.name,
                criterion_description=crit.description,
                max_marks=crit.max_marks,
                model_answer=assessment.model_answer,
                student_text=student_text,
                doc_scope=doc_scope,
                all_assessment_props=all_assessment_props,
                all_criteria_descriptions=all_criteria_descriptions,
                relevant_excerpt=criterion_excerpts.get(crit.name, student_text),
                precomputed_windows=candidate_windows if candidate_windows else None,
                precomputed_cand_embs=cand_embs,
                stage_timings=crit_timings,
                feedback_mode=effective_feedback_mode
            )
            total_retrieval_ms += crit_timings.get("retrieval_ms", 0.0)
            total_scoring_ms += crit_timings.get("scoring_ms", 0.0)
            total_feedback_ms += crit_timings.get("feedback_ms", 0.0)
            if crit_timings.get("ollama_called", False):
                total_ollama_calls += 1

            criterion_breakdowns.append(crit_res)
            total_awarded += crit_res.awarded_marks
            all_citations.extend(citations)

            if crit_res.contradictions_detected:
                review_reasons.append(f"Contradiction detected in {crit.name}")

        # Deduplicate citations by document_id + snippet
        unique_citations = []
        seen_snippets = set()
        for c in all_citations:
            key = (c.document_id, c.snippet[:50])
            if key not in seen_snippets:
                seen_snippets.add(key)
                unique_citations.append(c)

        # Flag for teacher review if citations are insufficient or contradictions occurred
        requires_teacher_review = False
        if not unique_citations and len(student_text.strip()) > 20:
            requires_teacher_review = True
            review_reasons.append("No supporting institutional evidence was retrieved from the knowledge base.")
        if any(len(c.contradictions_detected) > 0 for c in criterion_breakdowns):
            requires_teacher_review = True

        total_score = round(min(assessment.total_marks, max(0.0, total_awarded)), 2)
        percentage = round((total_score / assessment.total_marks) * 100, 1) if assessment.total_marks > 0 else 0.0

        overall_feedback = LLMFeedbackService.generate_overall_feedback(
            question=assessment.question,
            total_score=total_score,
            max_score=assessment.total_marks,
            review_reasons=review_reasons
        )

        # Provenance audit block
        text_hash = hashlib.sha256(student_text.encode("utf-8")).hexdigest()
        provenance = {
            "text_sha256": text_hash,
            "submission_type": submission_type,
            "submission_filename": submission_filename,
            "feedback_mode": effective_feedback_mode,
            "ollama_calls": total_ollama_calls,
            "embedding_model": "all-MiniLM-L6-v2",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "total_citations_analyzed": len(unique_citations)
        }

        raw_response = GradingResponse(
            assessment_id=assessment.id,
            assessment_title=assessment.title,
            total_score=total_score,
            max_score=assessment.total_marks,
            percentage=percentage,
            criterion_breakdown=criterion_breakdowns,
            overall_feedback=overall_feedback,
            requires_teacher_review=requires_teacher_review,
            review_reasons=review_reasons,
            citations=unique_citations[:5],
            provenance=provenance
        )

        t_sanit0 = time.perf_counter()
        sanitized_response = cls.validate_and_sanitize_response(raw_response, assessment, student_text)
        t_sanit_ms = (time.perf_counter() - t_sanit0) * 1000
        t_total_ms = (time.perf_counter() - t_overall_start) * 1000

        stage_durations = {
            "text_preparation_ms": round(t_prep_ms, 2),
            "retrieval_ms": round(total_retrieval_ms, 2),
            "numerical_scoring_ms": round(total_scoring_ms, 2),
            "feedback_generation_ms": round(total_feedback_ms, 2),
            "sanitization_ms": round(t_sanit_ms, 2),
            "total_grading_ms": round(t_total_ms, 2),
            "ollama_calls": total_ollama_calls
        }
        sanitized_response.provenance["stage_durations"] = stage_durations

        logger.info(
            f"[Grading Pipeline] Graded submission '{submission_filename or submission_type}' "
            f"score={total_score:.2f}/{assessment.total_marks:.2f} ({percentage:.1f}%) "
            f"total_ms={t_total_ms:.1f} stages={stage_durations}"
        )

        return sanitized_response

    @classmethod
    def validate_and_sanitize_response(
        cls,
        response: GradingResponse,
        assessment: AssessmentResponse,
        student_text: str
    ) -> GradingResponse:
        """
        Validates the entire final response object before returning to API caller or saving to DB.
        Ensures NO criterion contains inconsistent or rejected feedback.
        Numerical scores and marks are NEVER altered.
        """
        all_criteria_descriptions = {c.name: c.description for c in assessment.criteria}
        q_tokens = LLMFeedbackService._get_criterion_tokens(assessment.question)
        sub_irrelevant = LLMFeedbackService._is_submission_irrelevant(
            student_text=student_text,
            question=assessment.question,
            all_criteria_descriptions=all_criteria_descriptions
        )

        for crit in response.criterion_breakdown:
            is_valid = LLMFeedbackService._is_feedback_consistent(
                feedback=crit.feedback,
                awarded_marks=crit.awarded_marks,
                max_marks=crit.max_marks,
                criterion_name=crit.criterion_name,
                all_criteria_descriptions=all_criteria_descriptions,
                clean_missing=crit.missing_concepts,
                clean_present=crit.present_concepts,
                student_text=student_text,
                context_stopwords=q_tokens
            )
            if not is_valid:
                fallback = LLMFeedbackService._get_deterministic_fallback(
                    awarded_marks=crit.awarded_marks,
                    max_marks=crit.max_marks,
                    clean_contradictions=crit.contradictions_detected,
                    clean_missing=crit.missing_concepts,
                    clean_present=crit.present_concepts,
                    criterion_name=crit.criterion_name,
                    is_submission_irrelevant=sub_irrelevant
                )
                if LLMFeedbackService._is_feedback_consistent(
                    feedback=fallback,
                    awarded_marks=crit.awarded_marks,
                    max_marks=crit.max_marks,
                    criterion_name=crit.criterion_name,
                    all_criteria_descriptions=all_criteria_descriptions,
                    clean_missing=crit.missing_concepts,
                    clean_present=crit.present_concepts,
                    student_text=student_text,
                    context_stopwords=q_tokens
                ):
                    crit.feedback = fallback
                else:
                    crit.feedback = f"Feedback is currently unavailable for '{crit.criterion_name}'. Please refer to the assigned score ({crit.awarded_marks:.2f}/{crit.max_marks:.2f}) and rubric requirements."

        return response
