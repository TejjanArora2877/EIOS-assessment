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
            r'(?i)\b(maximum|max)\s+marks?\b',
            r'(?i)\bmarks?\s+distribution\b',
            r'(?i)\bgrading\s+(criteria|rubric|guidelines?)\b',
            r'(?i)\bthe\s+answer\s+must\s+(include|contain|mention)\b',
        ]
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _clean_proposition(text: str, prev_text: str = "") -> str:
        """
        Strips teacher meta-prefixes and instructional commands from rubric criterion sentences
        to extract clean, authoritative declarative propositions suitable for semantic embedding
        and NLI validation.
        """
        s = text.strip()
        # 1. Strip inline administrative / marking instructions
        s = re.sub(r'(?i)[;,]\s*(?:award|deduct)\s+(?:partial\s+|full\s+)?(?:marks?|points?|credit).*$', '', s)
        s = re.sub(r'(?i)\s*\((?:award|deduct|partial\s+marks?|marks?\s+are).*?\)', '', s)

        # 2. Strip teacher meta-framing prefixes
        s = re.sub(
            r'(?i)^(?:the\s+)?(?:answer|student|response|submission)\s+(?:should|must|needs to|ought to|has to)\s+'
            r'(?:explain|describe|state|identify|demonstrate|show|mention|discuss|include|distinguish|differentiate|provide|cover)\s+(?:that|how|between)?\s*',
            '',
            s
        )
        s = re.sub(r'(?i)^(?:the\s+)?criterion\s+(?:evaluates|assesses|requires|checks)\s+(?:that|whether|how)?\s*', '', s)
        s = re.sub(r'(?i)^(?:must\s+include|should\s+cover|requires\s+mention\s+of|must\s+contain)\s+', '', s)

        # 3. Strip trailing meta-instructions (e.g. 'using relevant examples', 'with examples')
        s = re.sub(r'(?i)[,;]?\s*(?:providing|giving|using)\s+(?:relevant\s+|appropriate\s+)?examples?(?:\s+of\s+each)?\.?$', '', s)
        s = re.sub(r'(?i)[,;]?\s*with\s+examples?\.?$', '', s)
        s = re.sub(r'(?i)[,;]?\s*and\s+(?:discuss|explain)\s+(?:important\s+|their\s+)?limitations\.?$', '', s)

        # 4. Transform imperative constructions into declarative propositions
        # 'Distinguish A from B' -> 'A are distinguished from B'
        m_from = re.match(r'(?i)^distinguish\s+(.*?)\s+from\s+(.*)', s)
        if m_from:
            s = f"{m_from.group(1).strip()} are distinguished from {m_from.group(2).strip()}"
        # 'Distinguish between A and B' -> 'A and B are distinguished'
        m_between = re.match(r'(?i)^distinguish\s+between\s+(.*?)\s+and\s+(.*)', s)
        if m_between:
            s = f"{m_between.group(1).strip()} and {m_between.group(2).strip()} are distinguished"
        # 'Distinguish A, B, and C' -> 'A, B, and C are distinguished'
        m_dist = re.match(r'(?i)^distinguish\s+(.*)', s)
        if m_dist:
            s = f"{m_dist.group(1).strip()} are distinguished"

        # 'Compare and contrast A and B' -> 'A and B are compared and contrasted'
        m_comp = re.match(r'(?i)^compare\s+and\s+contrast\s+(.*)', s)
        if m_comp:
            s = f"{m_comp.group(1).strip()} are compared and contrasted"

        # 'Explain why <clause>' -> '<clause>'
        s = re.sub(r'(?i)^(?:explain|describe|discuss|detail)\s+why\s+', '', s)
        # 'Explain how <clause>' -> '<clause>'
        s = re.sub(r'(?i)^(?:explain|describe|discuss|detail)\s+how\s+', '', s)
        # 'Explain that <clause>' -> '<clause>'
        s = re.sub(r'(?i)^(?:explain|describe|discuss|detail)\s+that\s+', '', s)
        # 'Explain the relationship between A and B' -> 'A and B are related concepts'
        m_rel = re.match(r'(?i)^(?:explain|describe|discuss)?\s*(?:the\s+)?relationship\s+between\s+(.*?)(?:\.|$)', s)
        if m_rel:
            s = f"{m_rel.group(1).strip()} are related concepts"

        # 'Explain the difference between A and B' -> 'A and B differ in scope and function'
        m_diff = re.match(r'(?i)^(?:explain|describe|discuss)?\s*(?:the\s+)?difference\s+between\s+(.*?)(?:\.|$)', s)
        if m_diff:
            s = f"{m_diff.group(1).strip()} differ in scope and function"

        # General imperative prefix strip
        s = re.sub(r'(?i)^(?:explain|describe|discuss|detail|state|outline|clarify|elaborate\s+on)\s+', '', s)
        s = re.sub(r'(?i)^(?:explains?|describes?|states?|discusses?)\s+(?:how|that)\s+', '', s)
        s = re.sub(r'(?i)^(?:defines?|definition\s+of)\s+', '', s)
        s = re.sub(r'(?i)^(?:identifies|identify|list|lists|name|names|provide|provides)\s+valid\s+', 'Valid ', s)
        s = re.sub(r'(?i)^(?:identifies|identify|list|lists|name|names|provide|provides)\s+(?:the\s+)?', 'The ', s)
        s = re.sub(r'(?i)^(?:apply|applies)\s+these\s+concepts\s+to\s+', 'Applied to ', s)
        s = re.sub(r'(?i)^(?:propose|proposes)\s+a\s+', 'A ', s)

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
            s = s.rstrip(' ;,')
            if not s.endswith('.'):
                s += '.'
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
            if c and not cls._is_administrative_sentence(c) and len(c.strip()) > 6 and len(cls._extract_content_tokens(c)) > 0:
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
            "states", "identifies", "defined", "defines", "including", "includes",
            # Broad technical and generic domain stop words
            "system", "systems", "data", "information", "service", "services",
            "process", "processes", "approach", "approaches", "table", "tables",
            "record", "records", "application", "applications", "consequence",
            "consequences", "example", "examples", "portal", "key", "keys"
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
        Extracts illustrative list items (e.g. from 'such as X, Y, Z' or 'applications in X, Y, Z') and checks if proportional grading applies.
        """
        m = re.search(r'(?i)\b(such as|including|e\.g\.?,?|for example|applications in|tasks such as|tasks in|examples in)\s+([^.]*)', criterion_description)
        if not m:
            return [], False

        raw_list = m.group(2)
        raw_items = re.split(r',|\bor\b|\band\b', raw_list)
        items = [re.sub(r'^\s*(such as|including|e\.g\.?|applications in)\s*', '', it).strip() for it in raw_items]
        items = [it for it in items if len(it) > 3]

        is_proportional = bool(items) and (len(items) >= 3 or bool(re.search(r'(?i)\b(proportionately|partial marks|number and relevance|list some|applications|examples|tasks)\b', criterion_description)))
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

        # Benchmark: 3 items (or all items if total < 3) yields 100% full marks
        target_benchmark = min(3, len(items))
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
    def _is_conceptually_related(
        cls,
        text: str,
        prop: str,
        sim: Optional[float] = None,
        distinctive_prop_tokens: Optional[Set[str]] = None,
        prop_tokens: Optional[Set[str]] = None
    ) -> bool:
        """
        Requires evidence that the candidate statement is conceptually related to
        the expected rubric proposition before running contradiction inference.
        Combines lexical content token grounding and semantic embedding similarity.
        """
        if not text or not text.strip() or not prop or not prop.strip():
            return False

        tokens = cls._extract_content_tokens(text)
        target_tokens = distinctive_prop_tokens if distinctive_prop_tokens is not None else (
            prop_tokens if prop_tokens is not None else cls._extract_content_tokens(prop)
        )
        overlap = tokens & target_tokens
        has_substantive_lexical = len(overlap) >= 2 or (len(overlap) >= 1 and len(target_tokens) <= 2)
        has_partial_lexical = len(overlap) >= 1

        if sim is None:
            prop_emb = ModelsLoader.encode_texts([prop])[0]
            cand_emb = ModelsLoader.encode_texts([text])[0]
            sim = float(cosine_similarity([prop_emb], [cand_emb])[0][0])

        # Conceptual grounding requires either:
        # a) Substantive domain lexical grounding (>= 2 distinctive tokens) with semantic floor (sim >= 0.35)
        # b) Single distinctive token lexical match requiring moderate semantic affinity (sim >= 0.50)
        # c) High semantic similarity (sim >= 0.68) specifically to the proposition when lexical grounding is absent
        if has_substantive_lexical and sim >= 0.35:
            return True
        if has_partial_lexical and sim >= 0.50:
            return True
        if sim >= 0.68:
            return True
        return False

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
        feedback_stats: Optional[Dict[str, Any]] = None,
        citations: Optional[List[EvidenceCitation]] = None
    ) -> Tuple[CriterionGradingResult, List[EvidenceCitation]]:
        """
        Evaluates a single rubric criterion deterministically.
        """
        # Retrieve relevant institutional evidence if not provided
        t_ret0 = time.perf_counter()
        if citations is None:
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
                max_skip = min(len(student_sentences), 8)
                for i in range(2, max_skip):
                    candidate_windows.append(f"{student_sentences[0]} {student_sentences[i]}")
            if len(student_text.split()) <= 400:
                candidate_windows.append(student_text)
            cand_embs = ModelsLoader.encode_texts(candidate_windows)

        # Extract authoritative substantive propositions (administrative instructions filtered)
        target_propositions = cls._extract_propositions(criterion_description, model_answer)

        # Extract model answer auxiliary propositions and key terminology
        model_propositions = []
        if model_answer and model_answer.strip():
            raw_model_sents = cls._split_sentences(model_answer)
            expanded_model_sents = []
            for ms in raw_model_sents:
                sub_parts = re.split(r'(?i)[;,]\s*(?:but|however|whereas|while|yet)\s+', ms)
                for sp in sub_parts:
                    if len(sp.strip()) > 8:
                        expanded_model_sents.append(sp.strip())
            for ms in expanded_model_sents:
                if not cls._is_administrative_sentence(ms) and len(ms.strip()) > 8:
                    clean_m = cls._clean_proposition(ms)
                    if clean_m:
                        model_propositions.append(clean_m)

        # Check for enumeration list (e.g. applications, examples)
        enum_items, is_proportional = cls._extract_enumeration_items(criterion_description)

        proposition_multipliers = []
        contradictions_found = []
        missing_concepts = []
        present_concepts = []

        q_tokens = cls._extract_content_tokens(question)
        student_tokens = cls._extract_content_tokens(student_text)

        # Precompute citation embeddings once per criterion to avoid redundant encodings
        cite_snippets = [c.snippet for c in citations if c.snippet and c.snippet.strip()] if citations else []
        cite_tokens = set().union(*(cls._extract_content_tokens(cs) for cs in cite_snippets)) - q_tokens if cite_snippets else set()
        precomputed_cite_embs = ModelsLoader.encode_texts(cite_snippets) if cite_snippets else None

        # Precompute embeddings of other criteria and all assessment propositions for isolation
        isolation_texts = []
        if all_criteria_descriptions:
            for cname, cdesc in all_criteria_descriptions.items():
                if cname != criterion_name and cdesc and cdesc.strip():
                    isolation_texts.append(cdesc)
        if all_assessment_props:
            for other_p in all_assessment_props:
                if other_p and len(other_p.strip()) > 5 and other_p not in isolation_texts:
                    isolation_texts.append(other_p)

        isolation_embs = ModelsLoader.encode_texts(isolation_texts) if isolation_texts else None

        for prop in target_propositions:
            prop_emb = ModelsLoader.encode_texts([prop])[0]
            sims = cosine_similarity([prop_emb], cand_embs)[0]
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])
            best_cand = candidate_windows[best_idx]

            if isolation_embs is not None and len(isolation_texts) > 0:
                active_indices = [idx for idx, t in enumerate(isolation_texts) if t != prop]
                prop_other_embs = isolation_embs[active_indices] if active_indices else None
            else:
                prop_other_embs = None

            prop_tokens = cls._extract_content_tokens(prop)
            distinctive_prop_tokens = prop_tokens - q_tokens

            # Grounding against teacher's model answer:
            # Identify model answer statements aligned with this rubric proposition
            aligned_model_props = []
            dist_prop_toks = prop_tokens - q_tokens
            for mp in model_propositions:
                mp_emb = ModelsLoader.encode_texts([mp])[0]
                mp_sim = float(cosine_similarity([prop_emb], [mp_emb])[0][0])
                mp_tokens = cls._extract_content_tokens(mp)
                if mp_sim >= 0.50 or (mp_sim >= 0.35 and bool(dist_prop_toks & mp_tokens)):
                    aligned_model_props.append((mp, mp_sim, mp_tokens))

            # Pre-filter aligned model propositions and cap to top 10
            if aligned_model_props:
                aligned_model_props = sorted(aligned_model_props, key=lambda x: x[1], reverse=True)[:10]

            # Augment distinctive tokens with auxiliary vocabulary from the model answer
            if aligned_model_props:
                aux_model_tokens = set().union(*(tokens for _, _, tokens in aligned_model_props)) - q_tokens
                augmented_distinctive_tokens = distinctive_prop_tokens | aux_model_tokens
            else:
                augmented_distinctive_tokens = distinctive_prop_tokens

            distinctive_overlap = augmented_distinctive_tokens & student_tokens
            distinctive_ratio = len(distinctive_overlap) / len(augmented_distinctive_tokens) if augmented_distinctive_tokens else 1.0

            # 1. Candidate entailment check for technical paraphrases (batched across candidate windows)
            # Candidate pre-filtering: filter out windows with sims[i] < 0.25 unless lexically grounded,
            # always include the highest-similarity candidate, and cap to top 20 candidates.
            candidate_indices = []
            for i, cand in enumerate(candidate_windows):
                if i == best_idx or sims[i] >= 0.25 or cls._is_conceptually_related(cand, prop, float(sims[i]), augmented_distinctive_tokens, prop_tokens):
                    candidate_indices.append(i)
            candidate_indices = sorted(candidate_indices, key=lambda idx: sims[idx], reverse=True)[:20]
            if best_idx not in candidate_indices and best_idx < len(candidate_windows):
                candidate_indices.insert(0, best_idx)

            fwd_pairs = [(candidate_windows[i], prop) for i in candidate_indices]
            fwd_results = ModelsLoader.predict_nli(fwd_pairs) if fwd_pairs else []
            cand_prop_nli = {idx: res for idx, res in zip(candidate_indices, fwd_results)}
            best_ent_cand = best_cand
            best_ent_sim = best_sim
            valid_fwd_ents = []
            best_fwd_ent = 0.0
            for cand_i, r in zip(candidate_indices, fwd_results):
                c_contra = r["contradiction"]
                c_ent = r["entailment"]
                is_invalid_ent = (c_ent < 0.40) and ((c_contra >= 0.70 and c_contra > c_ent) or (c_contra >= 0.60 and c_contra > c_ent and c_ent < 0.30))
                if not is_invalid_ent:
                    valid_fwd_ents.append(c_ent)
                    if c_ent > best_fwd_ent:
                        best_fwd_ent = c_ent
                        best_ent_cand = candidate_windows[cand_i]
                        best_ent_sim = float(sims[cand_i])

            # Auxiliary model answer proposition entailment check:
            # Rewards student answers using teacher's model-answer specific explanations
            if aligned_model_props:
                for mp, _, _ in aligned_model_props:
                    mp_emb = ModelsLoader.encode_texts([mp])[0]
                    mp_sims = cosine_similarity([mp_emb], cand_embs)[0]
                    mp_best_idx = int(mp_sims.argmax())
                    mp_indices = [
                        i for i in range(len(candidate_windows))
                        if i == mp_best_idx or mp_sims[i] >= 0.25
                    ]
                    mp_indices = sorted(mp_indices, key=lambda idx: mp_sims[idx], reverse=True)[:10]
                    if mp_best_idx not in mp_indices and mp_best_idx < len(candidate_windows):
                        mp_indices.insert(0, mp_best_idx)

                    mp_pairs = [(candidate_windows[i], mp) for i in mp_indices]
                    mp_results = ModelsLoader.predict_nli(mp_pairs) if mp_pairs else []
                    for cand_i, r in zip(mp_indices, mp_results):
                        c_contra = r["contradiction"]
                        c_ent = r["entailment"]
                        is_invalid_ent = (c_ent < 0.40) and ((c_contra >= 0.70 and c_contra > c_ent) or (c_contra >= 0.60 and c_contra > c_ent and c_ent < 0.30))

                        # Verify against main proposition: candidate must not explicitly contradict the main proposition
                        direct_r = cand_prop_nli.get(cand_i)
                        if direct_r is None:
                            direct_r = ModelsLoader.predict_nli([(candidate_windows[cand_i], prop)])[0]
                            cand_prop_nli[cand_i] = direct_r
                        neg_cand_regex = r'(?i)\b(never|not|no|neither|nor|cannot|can\'t|won\'t|doesn\'t|don\'t|discards?|avoids?|rejects?|without|fails?|false|incorrect|incompatible|inefficient)\b'
                        is_cand_neg = bool(re.search(neg_cand_regex, candidate_windows[cand_i]))
                        is_direct_prop_contra = is_cand_neg and bool(direct_r and (direct_r["contradiction"] >= 0.50 and direct_r["contradiction"] > direct_r["entailment"]))

                        if not is_invalid_ent and not is_direct_prop_contra:
                            cand_to_prop_sim = float(sims[cand_i])
                            if cand_to_prop_sim >= 0.35 or cls._is_conceptually_related(candidate_windows[cand_i], prop, cand_to_prop_sim, augmented_distinctive_tokens, prop_tokens):
                                if c_ent > best_fwd_ent:
                                    best_fwd_ent = c_ent
                                    best_ent_cand = candidate_windows[cand_i]
                                    best_ent_sim = cand_to_prop_sim

            ent_score = best_fwd_ent

            # Reverse entailment check for complex paraphrases (skipped if forward entailment already >= 0.65)
            if not is_proportional and ent_score < 0.65:
                rev_indices = [i for i, cand in enumerate(candidate_windows) if sims[i] >= 0.50]
                rev_indices = sorted(rev_indices, key=lambda idx: sims[idx], reverse=True)[:5]
                if rev_indices:
                    rev_pairs = [(prop, candidate_windows[i]) for i in rev_indices]
                    rev_results = ModelsLoader.predict_nli(rev_pairs)
                    for r_idx, r in zip(rev_indices, rev_results):
                        if r["entailment"] >= 0.70 and r["contradiction"] < 0.30:
                            rev_ent = r["entailment"] * 0.90
                            if rev_ent > ent_score:
                                ent_score = rev_ent
                                best_ent_cand = candidate_windows[r_idx]
                                best_ent_sim = float(sims[r_idx])

            # 3. Deterministic proposition scoring: concepts require genuine semantic entailment
            # High-entailment credit is ONLY awarded if conceptually grounded
            is_concept_grounded = bool(
                (best_ent_cand and cls._is_conceptually_related(best_ent_cand, prop, best_ent_sim, augmented_distinctive_tokens, prop_tokens))
                or (best_cand and cls._is_conceptually_related(best_cand, prop, best_sim, augmented_distinctive_tokens, prop_tokens))
                or (best_sim >= 0.68)
                or (best_sim >= 0.45 and distinctive_ratio >= 0.40 and bool(distinctive_overlap))
            )

            # 2. Contradiction check: unmask contradictions across candidate windows (both negated and affirmative misconceptions)
            is_contradiction = False
            is_prop_demonstrated = False

            # Verify if apparent high forward entailment is an affirmative misconception / inversion
            if is_concept_grounded and (ent_score >= 0.55 or best_fwd_ent >= 0.65):
                verify_cand = best_ent_cand or best_cand
                if verify_cand:
                    rev_check = ModelsLoader.predict_nli([(prop, verify_cand)])[0]
                    if rev_check["contradiction"] >= 0.60 and rev_check["contradiction"] > rev_check["entailment"]:
                        is_contradiction = True
                        cand = verify_cand
                        contradictions_found.append(
                            f"Student stated '{cand[:80]}...' which contradicts expected concept '{prop}'"
                        )
                    else:
                        is_prop_demonstrated = True
                else:
                    is_prop_demonstrated = True
            neg_regex = r'(?i)\b(never|not|no|neither|nor|cannot|can\'t|won\'t|doesn\'t|don\'t|discards?|avoids?|rejects?|fails?|false|incorrect|incompatible|inefficient|unnecessary|unneeded|pointless|interchangeable|same\s+thing)\b'
            contra_candidates: List[Tuple[int, str]] = []
            num_sentences = len(student_sentences)

            # 1. Inspect individual student sentences:
            for s_idx in range(num_sentences):
                s_text = student_sentences[s_idx]
                s_sim = float(sims[s_idx]) if s_idx < len(sims) else None
                if s_sim is None:
                    continue
                s_tokens = cls._extract_content_tokens(s_text)
                has_prop_overlap = bool(s_tokens & (prop_tokens | augmented_distinctive_tokens))
                if not ((s_sim >= 0.45) or (s_sim >= 0.35 and has_prop_overlap)):
                    continue
                if not has_prop_overlap and s_sim < 0.55:
                    continue

                # d) Isolation from Other Criteria and Propositions: A sentence that aligns with a different criterion or proposition must NEVER be treated as a contradiction of the current proposition
                if prop_other_embs is not None and len(prop_other_embs) > 0:
                    s_emb = cand_embs[s_idx] if s_idx < len(cand_embs) else ModelsLoader.encode_texts([s_text])[0]
                    other_sims = cosine_similarity([s_emb], prop_other_embs)[0]
                    if len(other_sims) > 0 and float(other_sims.max()) >= s_sim:
                        continue

                contra_candidates.append((s_idx, s_text))

            # 2. Inspect adjacent sentence pairs:
            for j in range(num_sentences - 1):
                pair_idx = num_sentences + j
                if pair_idx < len(candidate_windows):
                    pair_text = candidate_windows[pair_idx]
                    pair_sim = float(sims[pair_idx]) if pair_idx < len(sims) else None
                    if pair_sim is None:
                        continue
                    pair_tokens = cls._extract_content_tokens(pair_text)
                    has_pair_overlap = bool(pair_tokens & (prop_tokens | augmented_distinctive_tokens))
                    if not ((pair_sim >= 0.45) or (pair_sim >= 0.35 and has_pair_overlap)):
                        continue
                    if not has_pair_overlap and pair_sim < 0.55:
                        continue

                    if prop_other_embs is not None and len(prop_other_embs) > 0:
                        pair_emb = cand_embs[pair_idx] if pair_idx < len(cand_embs) else ModelsLoader.encode_texts([pair_text])[0]
                        other_sims = cosine_similarity([pair_emb], prop_other_embs)[0]
                        if len(other_sims) > 0 and float(other_sims.max()) >= pair_sim:
                            continue

                    contra_candidates.append((pair_idx, pair_text))

            if contra_candidates and not is_contradiction:
                seen_cands = set()
                unique_contra_candidates = []
                for c_idx, c_text in contra_candidates:
                    if c_text not in seen_cands:
                        seen_cands.add(c_text)
                        unique_contra_candidates.append((c_idx, c_text))

                # Cap contradiction candidates to top 10
                unique_contra_candidates = unique_contra_candidates[:10]

                contra_pairs = [(c_text, prop) for _, c_text in unique_contra_candidates]
                contra_results = ModelsLoader.predict_nli(contra_pairs) if contra_pairs else []
                for (c_idx, c_text), r in zip(unique_contra_candidates, contra_results):
                    c_contra = r["contradiction"]
                    c_ent = r["entailment"]

                    # If candidate has moderate forward entailment to prop, it expresses the concept, not contradicts it!
                    if c_ent >= 0.35:
                        continue

                    is_explicit_neg = bool(re.search(neg_regex, c_text))

                    # For enumerations/proportional criteria or list of examples:
                    # An affirmative sentence providing examples or applications must NEVER be treated as a contradiction!
                    # Only explicit negations can contradict an enumeration list.
                    if (is_proportional or bool(enum_items)) and not is_explicit_neg:
                        continue

                    # Entailment Protection: If student successfully and unambiguously entails prop
                    # in their submission (ent_score >= 0.55 with conceptual grounding),
                    # that proposition is DEMONSTRATED. An affirmative sentence elsewhere in a multi-sentence
                    # essay must NEVER wipe out a demonstrated concept! Only an explicit negation
                    # with high contradiction can flag a demonstrated concept.
                    if is_prop_demonstrated:
                        c_tokens = cls._extract_content_tokens(c_text)
                        target_tokens = prop_tokens | augmented_distinctive_tokens
                        if len(c_tokens & target_tokens) >= 2:
                            continue
                        if is_explicit_neg and c_contra >= 0.80 and c_contra > c_ent and c_ent < 0.15:
                            is_contradiction = True
                            cand = c_text
                            contradictions_found.append(
                                f"Student stated '{cand[:80]}...' which contradicts expected concept '{prop}'"
                            )
                            break
                        continue

                    # Otherwise, proposition was not strongly demonstrated:
                    if is_explicit_neg:
                        if c_contra >= 0.55 and c_contra > c_ent:
                            is_contradiction = True
                            cand = c_text
                            contradictions_found.append(
                                f"Student stated '{cand[:80]}...' which contradicts expected concept '{prop}'"
                            )
                            break
                    else:
                        # Affirmative misconception: requires genuine reverse contradiction (Prop -> Cand)
                        # proving the student statement is incompatible with the rubric proposition,
                        # avoiding spurious forward NLI penalties on cross-criterion or partial answers.
                        rev_nli = ModelsLoader.predict_nli([(prop, c_text)])[0]
                        if rev_nli["contradiction"] >= 0.70 and rev_nli["contradiction"] > rev_nli["entailment"] and rev_nli["entailment"] < 0.25:
                            is_contradiction = True
                            cand = c_text
                            contradictions_found.append(
                                f"Student stated '{cand[:80]}...' which contradicts expected concept '{prop}'"
                            )
                            break

            # 4. Citation evidence verification: check lexical and semantic support from retrieved textbook citations
            has_citation_verification = False
            if (
                precomputed_cite_embs is not None
                and len(cite_snippets) > 0
                and is_concept_grounded
                and not is_contradiction
                and ent_score >= 0.35
            ):
                eval_cand_text = best_ent_cand or best_cand or student_text
                cand_tokens = cls._extract_content_tokens(eval_cand_text)
                cite_distinctive_tokens = cite_tokens & (augmented_distinctive_tokens if augmented_distinctive_tokens else prop_tokens)
                cite_token_overlap = cand_tokens & cite_distinctive_tokens

                cand_emb_single = ModelsLoader.encode_texts([eval_cand_text])[0]
                cite_sims = cosine_similarity([cand_emb_single], precomputed_cite_embs)[0]
                max_cite_sim = float(cite_sims.max()) if len(cite_sims) > 0 else 0.0

                if max_cite_sim >= 0.55 and len(cite_token_overlap) >= 2:
                    has_citation_verification = True

            if is_contradiction:
                p_mult = 0.0
            elif ent_score >= 0.55 and is_concept_grounded:
                if ent_score >= 0.65 or has_citation_verification:
                    p_mult = 1.0
                else:
                    p_mult = ent_score
            elif ent_score >= 0.20 and is_concept_grounded:
                if best_sim >= 0.80 or (best_sim >= 0.50 and distinctive_ratio >= 0.45):
                    fuzzy = cls._fuzzy_similarity_multiplier(best_sim, 0.50, 0.85)
                    base_mult = min(1.0, 0.50 + 0.50 * max(ent_score, fuzzy, distinctive_ratio))
                    p_mult = 1.0 if has_citation_verification else base_mult
                else:
                    p_mult = 0.0
            else:
                if is_concept_grounded and best_sim >= 0.85 and distinctive_ratio >= 0.75:
                    p_mult = 0.50 if has_citation_verification else 0.25
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
        if feedback_mode == "none":
            feedback = ""
            crit_fb_stats = {
                "ollama_called": False,
                "duration_ms": 0.0,
                "mode": "none"
            }
        else:
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
        t_fb_ms = (time.perf_counter() - t_fb0) * 1000 if feedback_mode != "none" else 0.0

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
            uce_signals=uce,
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
        if effective_feedback_mode not in ("hybrid", "llm", "none"):
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
                max_skip = min(len(student_sentences), 8)
                for i in range(2, max_skip):
                    candidate_windows.append(f"{student_sentences[0]} {student_sentences[i]}")
            if len(student_text.split()) <= 400:
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

        if effective_feedback_mode == "none":
            overall_feedback = ""
        else:
            overall_feedback = LLMFeedbackService.generate_overall_feedback(
                question=assessment.question,
                total_score=total_score,
                max_score=assessment.total_marks,
                review_reasons=review_reasons,
                feedback_mode=effective_feedback_mode
            )

        # Five-Lens Cognitive Evaluation & Shannon Entropy
        overall_entropy = round(cls._calculate_shannon_entropy(student_text), 2)
        if criterion_breakdowns:
            avg_understanding = sum(c.uce_signals.understanding for c in criterion_breakdowns if c.uce_signals) / len(criterion_breakdowns)
            avg_coherence = sum(c.uce_signals.clarity for c in criterion_breakdowns if c.uce_signals) / len(criterion_breakdowns)
            avg_expression = sum(c.uce_signals.expression for c in criterion_breakdowns if c.uce_signals) / len(criterion_breakdowns)
        else:
            avg_understanding = 0.0
            avg_coherence = 0.0
            avg_expression = 0.0

        concept_coverage = round(percentage / 100.0, 2)
        evidence_use = round(min(1.0, len(unique_citations) / max(1, len(assessment.criteria))), 2)

        five_lenses = {
            "understanding": round(avg_understanding, 2),
            "concept_coverage": concept_coverage,
            "evidence_use": evidence_use,
            "coherence": round(avg_coherence, 2),
            "expression": round(avg_expression, 2)
        }

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
            "total_citations_analyzed": len(unique_citations),
            "shannon_entropy": overall_entropy,
            "five_lens_cognitive_evaluation": five_lenses
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
        if response.provenance and response.provenance.get("feedback_mode") == "none":
            return response
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
