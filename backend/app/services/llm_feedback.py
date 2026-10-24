import time
import logging
import re
import httpx
from typing import Optional, List, Dict, Any
from backend.app.config import settings
from backend.app.models import EvidenceCitation

logger = logging.getLogger(__name__)

class LLMFeedbackService:
    _client: Optional[httpx.Client] = None

    @classmethod
    def get_client(cls) -> httpx.Client:
        if cls._client is None or cls._client.is_closed:
            cls._client = httpx.Client(
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=60.0)
            )
        return cls._client

    @classmethod
    def close_client(cls):
        if cls._client is not None and not cls._client.is_closed:
            cls._client.close()
            cls._client = None
    @staticmethod
    def _get_criterion_tokens(text: str, context_stopwords: Optional[set] = None) -> set:
        """Extracts substantive tokens (len >= 4, lowercase) excluding grammatical stop words."""
        if not text:
            return set()
        words = set(re.findall(r'[a-zA-Z]{4,}', text.lower()))
        stop = {
            "this", "that", "with", "from", "have", "more", "also", "some", "such",
            "than", "they", "them", "their", "should", "could", "would", "must",
            "about", "which", "where", "when", "what", "learning", "student",
            "answer", "criterion", "award", "marks", "explain", "describes",
            "identifies", "approach", "using", "uses", "used", "help", "helps",
            "describe", "valid", "system", "into", "these", "other", "well", "response",
            "including", "includes", "addressed", "requires", "requirement", "points"
        }
        if context_stopwords:
            stop = stop | {w.lower() for w in context_stopwords if len(w) >= 4}
        return words - stop

    @classmethod
    def _is_feedback_consistent(
        cls,
        feedback: str,
        awarded_marks: float,
        max_marks: float,
        criterion_name: Optional[str] = None,
        all_criteria_descriptions: Optional[Dict[str, str]] = None,
        clean_missing: Optional[List[str]] = None,
        clean_present: Optional[List[str]] = None,
        clean_contradictions: Optional[List[str]] = None,
        student_text: Optional[str] = None,
        context_stopwords: Optional[set] = None
    ) -> bool:
        """
        Validates whether LLM feedback is logically consistent with structured grading scores.
        """
        if not feedback or not feedback.strip():
            return False

        # 0. Completeness and truncation check: must not end abruptly or mid-sentence
        s_fb = feedback.strip()
        if not re.search(r'[.!?]["\'\)\]]*$', s_fb):
            logger.warning(f"Feedback rejected for '{criterion_name}': text lacks terminal punctuation (truncated).")
            return False
        if re.search(r'[,:;—–-]\s*$', s_fb):
            logger.warning(f"Feedback rejected for '{criterion_name}': text ends with trailing delimiter.")
            return False
        bare_fb = re.sub(r'[\s.!?\"\'\)\]]+$', '', s_fb).strip()
        if re.search(r'(?i)\b(?:and|or|but|such as|including|for example|e\.g\.|i\.e\.|with|without|that|which|to|of|in|at|by|from|as|like|the|a|an|named)\b$', bare_fb):
            logger.warning(f"Feedback rejected for '{criterion_name}': text ends with dangling connector or incomplete phrase.")
            return False
        if s_fb.count('"') % 2 != 0:
            logger.warning(f"Feedback rejected for '{criterion_name}': text contains unbalanced quotes (truncated quote).")
            return False

        fb_lower = feedback.lower()

        # 1. Full marks: must not suggest omissions, partial explanation, or needed additions
        if awarded_marks >= max_marks:
            forbidden_full = [
                "to receive full marks", "to earn full marks", "to receive", "to earn", "to get",
                "partially explained", "only partially", "partially", "only", "however",
                "missing", "omitted", "omits", "omit", "omission", "omissions", "omitting",
                "should have", "needed", "needs", "needing",
                "fails to", "failed to", "fails", "failed", "lacks", "lacked", "lacking", "incomplete",
                "consider adding", "consider incorporating", "could benefit", "could have",
                "could include", "could also", "would improve", "further improve", "additional",
                "did not", "does not include", "did not include", "did not mention", "does not mention",
                "not included", "not mentioned", "unmentioned", "unaddressed", "left out", "leaves out", "leaving out",
                "suggest expanding", "could expand", "expanding on", "you could add", "you could also",
                "you could mention", "consider expanding", "could further", "can expand", "can further",
                "suggest adding", "expand on", "expand upon", "is noted", "was noted", "are noted", "were noted",
                "noted as", "noting that", "gap", "gaps", "deficiency", "deficiencies", "deficit",
                "absent", "absence", "excluded", "excluding", "exclusion", "neglected", "neglecting"
            ]
            if any(f in fb_lower for f in forbidden_full):
                logger.warning(f"Feedback rejected for '{criterion_name}': full marks awarded but contains restrictive/omission phrasing.")
                return False

            if re.search(r'(?i)\b(?:omiss(?:ion|ions)|omit(?:ted|s|ting)?|miss(?:ing|ed)?|lack(?:s|ed|ing)?|absent|absence|unmentioned|unaddressed|incomplete|deficien(?:cy|t|cies)|gap(?:s)?|neglect(?:ed|ing)?)\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': full marks awarded but regex matched omission terminology.")
                return False

            if re.search(r'(?i)\b(?:not|did not|does not|without|neither|nor)\s+(?:mention|include|provide|state|contain|address|identify|explain|describe)\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': full marks awarded but regex matched negative statement.")
                return False

            if re.search(r'(?i)\b(?:is|was|are|were)\s+noted\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': full marks awarded but regex matched 'is/was noted'.")
                return False

            # Invalidation of unmentioned illustrative examples in full marks feedback:
            if student_text and all_criteria_descriptions and criterion_name in all_criteria_descriptions:
                target_desc = all_criteria_descriptions[criterion_name]
                enum_match = re.search(r'(?i)\b(?:such as|for example|e\.g\.|including)\s+([^.]+)', target_desc)
                if enum_match:
                    example_items = [it.strip().lower() for it in re.split(r'[,;]|\band\b|\bor\b', enum_match.group(1)) if len(it.strip()) >= 3]
                    st_lower = student_text.lower()
                    for ex in example_items:
                        ex_tokens = [w for w in re.findall(r'[a-zA-Z]{4,}', ex) if w not in {"such", "with", "from", "that", "this", "some", "other", "valid", "tasks", "applications"}]
                        if ex_tokens and not any(t in st_lower for t in ex_tokens):
                            # The illustrative example was NOT in the student's text
                            if any(t in fb_lower for t in ex_tokens):
                                logger.warning(f"Feedback rejected for '{criterion_name}': full marks awarded but mentions unmentioned illustrative example token from '{ex}'.")
                                return False

        # 2. Zero marks / irrelevant response: must not claim student explained concepts
        elif awarded_marks == 0.0:
            forbidden_zero = [
                "you explained", "you partially explained", "you correctly",
                "you described how", "you have described", "partially addressed",
                "good foundational", "you noted", "you identified", "you mentioned"
            ]
            if any(f in fb_lower for f in forbidden_zero):
                return False

        # 3. If there are no missing concepts and no contradictions identified by the grading engine,
        # candidate feedback must NEVER assert or imply omissions or deficiencies.
        if not clean_missing and not clean_contradictions:
            forbidden_no_missing = [
                "missing", "omitted", "omits", "omit", "omission", "omissions", "omitting",
                "failed to", "did not", "does not", "not included", "not mentioned", "unmentioned", "unaddressed",
                "needed", "needs to", "needs", "needing", "lacks", "lacked", "lacking", "incomplete", "should have",
                "consider adding", "however", "only", "partially", "to receive full marks",
                "suggest expanding", "could expand", "expanding on", "you could add", "you could also",
                "you could mention", "consider expanding", "could further", "can expand", "can further",
                "suggest adding", "expand on", "expand upon", "is noted", "was noted", "are noted", "were noted",
                "noted as", "noting that", "gap", "gaps", "deficiency", "deficiencies", "deficit",
                "absent", "absence", "excluded", "excluding", "exclusion", "neglected", "neglecting"
            ]
            if any(f in fb_lower for f in forbidden_no_missing):
                logger.warning(
                    f"Feedback rejected for '{criterion_name}': "
                    f"missing_concepts is empty but feedback claims omissions or deficiencies."
                )
                return False

            if re.search(r'(?i)\b(?:omiss(?:ion|ions)|omit(?:ted|s|ting)?|miss(?:ing|ed)?|lack(?:s|ed|ing)?|absent|absence|unmentioned|unaddressed|incomplete|deficien(?:cy|t|cies)|gap(?:s)?|neglect(?:ed|ing)?)\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': missing_concepts is empty but regex matched omission terminology.")
                return False

            if re.search(r'(?i)\b(?:not|did not|does not|without|neither|nor)\s+(?:mention|include|provide|state|contain|address|identify|explain|describe)\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': missing_concepts is empty but regex matched negative statement.")
                return False

            if re.search(r'(?i)\b(?:is|was|are|were)\s+noted\b', feedback):
                logger.warning(f"Feedback rejected for '{criterion_name}': missing_concepts is empty but regex matched 'is/was noted'.")
                return False

        # 4. Invalidation of false omission claims for terms present in student text:
        if student_text:
            st_lower = student_text.lower()
            omission_matches = re.finditer(
                r'(?i)\b(?:did not|does not|failed to|fails to|omitted|omits|omission of|omissions of|missing|missed|lacks?|needs?(?:\s+to)?|needed(?:\s+to)?)\s+'
                r'(?:mention|include|state|describe|provide)?\s*([a-zA-Z\s-]+?)(?:[.,;]|\band\b|\bor\b|$)',
                feedback
            )
            for match in omission_matches:
                phrase = match.group(1).strip().lower()
                phrase_tokens = [w for w in re.findall(r'[a-zA-Z]{4,}', phrase) if w not in {"that", "this", "with", "from", "have", "more", "also", "some", "such"}]
                if phrase and len(phrase) >= 4 and phrase in st_lower:
                    logger.warning(
                        f"Feedback rejected for '{criterion_name}': "
                        f"claimed '{phrase}' was missing, but it is present in student text."
                    )
                    return False
                if len(phrase_tokens) >= 2 and all(tok in st_lower for tok in phrase_tokens):
                    logger.warning(
                        f"Feedback rejected for '{criterion_name}': "
                        f"claimed tokens '{phrase_tokens}' were missing, but all are present in student text."
                    )
                    return False

        # 5. Prevent false attribution of missing concepts:
        # If candidate feedback claims the student explained/described/mentioned concepts
        # that are actually missing and absent from the student's submission, reject it.
        if clean_missing and (student_text is not None or clean_present is not None):
            student_tokens = cls._get_criterion_tokens(student_text or "")
            present_tokens = cls._get_criterion_tokens(" ".join(clean_present or []))
            allowed_tokens = student_tokens | present_tokens

            for miss in clean_missing:
                miss_tokens = cls._get_criterion_tokens(miss, context_stopwords)
                unaddressed_tokens = miss_tokens - allowed_tokens
                if not unaddressed_tokens:
                    continue

                for tok in unaddressed_tokens:
                    clauses = re.split(r'[,;.!?]|\b(?:but|however|although|though|while|whereas|yet)\b', feedback, flags=re.IGNORECASE)
                    for clause in clauses:
                        has_negation = bool(re.search(
                            r'(?i)\b(?:failed|fails|omitted|omits|missing|missed|lacks|lacked|without|neither|nor|did not|does not|not|needs?|should have)\b',
                            clause
                        ))
                        if has_negation:
                            continue

                        tok_pattern = (
                            rf'(?i)\b(?:you|student|answer|submission|response)\b[^.?!]*\b'
                            rf'(?:correctly|accurately|well|clearly)?\s*'
                            rf'(?:explained|explains|described|describes|stated|states|identified|identifies|'
                            rf'noted|notes|mentioned|mentions|provided|provides|included|includes|discussed|discusses)\b'
                            rf'[^.?!]*\b{re.escape(tok)}\b'
                        )
                        if re.search(tok_pattern, clause):
                            logger.warning(
                                f"False attribution rejected for '{criterion_name}': "
                                f"feedback claimed student explained unaddressed concept token '{tok}'."
                            )
                            return False

        # 6. Dynamic cross-criterion drift validation
        if criterion_name and all_criteria_descriptions and len(all_criteria_descriptions) > 1:
            target_desc = all_criteria_descriptions.get(criterion_name, "")
            target_tokens = cls._get_criterion_tokens(f"{criterion_name} {target_desc}", context_stopwords)
            # Strip out quoted strings to avoid false drift flags from student quotes or rubric references
            unquoted_feedback = re.sub(r"['\"].*?['\"]", "", feedback)
            fb_tokens = cls._get_criterion_tokens(unquoted_feedback, context_stopwords)

            # Check if feedback contains distinctive tokens of another criterion while having 0 overlap with target
            for other_name, other_desc in all_criteria_descriptions.items():
                if other_name == criterion_name:
                    continue
                other_tokens = cls._get_criterion_tokens(f"{other_name} {other_desc}", context_stopwords)
                distinctive_other = other_tokens - target_tokens
                overlap_other = fb_tokens & distinctive_other

                # If feedback has >= 2 distinctive tokens of another criterion, reject cross-criterion drift
                if len(overlap_other) >= 2:
                    logger.warning(
                        f"Cross-criterion drift detected for '{criterion_name}': "
                        f"feedback mentions distinctive tokens {overlap_other} from '{other_name}'."
                    )
                    return False

        return True

    @classmethod
    def _is_submission_irrelevant(
        cls,
        student_text: str,
        question: str,
        all_criteria_descriptions: Optional[Dict[str, str]] = None
    ) -> bool:
        """Determines whether the student's submission is completely off-topic or irrelevant to the assessment."""
        if not student_text or not student_text.strip():
            return True
        st_tokens = cls._get_criterion_tokens(student_text)
        if not st_tokens:
            return True

        assessment_tokens = cls._get_criterion_tokens(question) if question else set()
        if all_criteria_descriptions:
            for name, desc in all_criteria_descriptions.items():
                assessment_tokens |= cls._get_criterion_tokens(f"{name} {desc}")

        overlap = st_tokens & assessment_tokens
        # If there are fewer than 2 substantive domain tokens overlapping with the assessment
        return len(overlap) < 2

    @classmethod
    def _get_deterministic_fallback(
        cls,
        awarded_marks: float,
        max_marks: float,
        clean_contradictions: List[str],
        clean_missing: List[str],
        clean_present: Optional[List[str]] = None,
        criterion_name: str = "this criterion",
        is_submission_irrelevant: bool = False
    ) -> str:
        """Truthful deterministic fallback matching the exact computed evaluation."""
        c_label = f"'{criterion_name}'" if criterion_name and criterion_name != "this criterion" else "this criterion"
        if clean_contradictions:
            contra_str = clean_contradictions[0]
            match = re.search(r"contradicts expected concept ['\"](.+?)['\"]", contra_str)
            if match:
                concept_name = match.group(1).rstrip('.')
                concept_name = re.sub(r'(?i)\s*,?\s*(?:such as|for example|e\.g\.).*', '', concept_name).strip(' ,;-')
                return f"Review recommended: Statement directly contradicts the expected rubric concept '{concept_name}' for {c_label}."
            return f"Review recommended: Statement conflicts with the expected rubric concepts for {c_label}."
        elif awarded_marks >= max_marks:
            return f"All expected concepts for {c_label} were accurately and comprehensively addressed."
        elif is_submission_irrelevant:
            return f"The submitted response does not address the expected concepts for {c_label}."
        elif awarded_marks == 0.0:
            filtered_missing = []
            for m in clean_missing:
                m_clean = re.sub(r'(?i)\b(?:such as|for example|e\.g\.).*', '', m).strip(' ,;-')
                if m_clean and m_clean not in filtered_missing:
                    filtered_missing.append(m_clean)
            if filtered_missing:
                return f"The response does not address {c_label}. Expected concepts to address: {'; '.join(filtered_missing[:2])}."
            return f"The response does not address the expected concepts for {c_label}."
        elif not clean_missing:
            return f"All expected concepts for {c_label} were accurately addressed in the student's submission."
        elif clean_missing:
            filtered_present = []
            for p in (clean_present or []):
                p_clean = re.sub(r'(?i)\b(?:such as|for example|e\.g\.).*', '', p).strip(' ,;-')
                if p_clean and p_clean not in filtered_present:
                    filtered_present.append(p_clean)

            filtered_missing = []
            for m in clean_missing:
                m_clean = re.sub(r'(?i)\b(?:such as|for example|e\.g\.).*', '', m).strip(' ,;-')
                if m_clean and m_clean not in filtered_missing:
                    filtered_missing.append(m_clean)

            addressed_part = ""
            if filtered_present:
                addressed_part = f"Accurately addressed: {'; '.join(filtered_present[:2])}. "
            return f"{addressed_part}To improve for {c_label}, address the following expected concepts: {'; '.join(filtered_missing[:2])}."
        else:
            return f"Partially addressed for {c_label} based on the rubric criteria."

    @classmethod
    def generate_criterion_feedback(
        cls,
        question: str,
        criterion_name: str,
        criterion_description: str,
        awarded_marks: float,
        max_marks: float,
        student_text: str,
        contradictions: List[str],
        missing_concepts: List[str],
        citations: List[EvidenceCitation],
        present_concepts: Optional[List[str]] = None,
        target_propositions: Optional[List[str]] = None,
        relevant_excerpt: Optional[str] = None,
        all_criteria_descriptions: Optional[Dict[str, str]] = None,
        feedback_mode: Optional[str] = None,
        feedback_stats: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Calls local Ollama or deterministic fallback to generate 1-2 constructive, grounded feedback sentences.
        Supports configurable 'hybrid' (deterministic for full marks and irrelevant answers without contradictions;
        LLM for partial marks and contradictions) or 'llm' (full LLM evaluation for all criteria).
        Guarantees score consistency, validates output against numerical marks,
        prevents cross-criterion drift, allows at most one bounded regeneration attempt,
        and uses a truthful deterministic fallback WITHOUT affecting numerical scores.
        """
        if feedback_stats is not None:
            feedback_stats["ollama_called"] = False
            feedback_stats["duration_ms"] = 0.0

        if not student_text or not student_text.strip():
            if feedback_stats is not None:
                feedback_stats["deterministic_reason"] = "empty_student_text"
            return f"No response provided for '{criterion_name}'. Expected concepts were not addressed."

        clean_missing = [m for m in missing_concepts if m.strip()]
        clean_present = [p for p in (present_concepts or []) if p.strip()]
        clean_contradictions = [c for c in contradictions if c.strip()]

        is_full = awarded_marks >= max_marks
        is_zero = awarded_marks == 0.0

        effective_mode = (feedback_mode or settings.FEEDBACK_MODE).lower()
        if effective_mode not in ("hybrid", "llm"):
            effective_mode = "hybrid"

        # Context stopwords from the assessment question to prevent domain-level drift false positives
        q_tokens = cls._get_criterion_tokens(question) if question else set()

        # Scope student excerpt to criterion-relevant text to prevent cross-criterion bleed
        if relevant_excerpt is not None:
            eval_student_text = relevant_excerpt.strip() if relevant_excerpt.strip() else "[No statements addressing this criterion were identified in the student's submission]"
        else:
            eval_student_text = student_text

        has_no_statements = (
            not eval_student_text
            or eval_student_text == "[No statements addressing this criterion were identified in the student's submission]"
        )

        sub_irrelevant = cls._is_submission_irrelevant(
            student_text=student_text,
            question=question,
            all_criteria_descriptions=all_criteria_descriptions
        )

        # Safe hybrid feedback strategy:
        # Avoid expensive LLM generation when the evaluation result is deterministic and unambiguous
        if effective_mode == "hybrid":
            # 1. Contradictions: always retain the LLM path to evaluate misconceptions & flag teacher review
            if clean_contradictions:
                pass

            # 2. Full marks, no missing concepts, no contradictions:
            # Generate concise, criterion-specific positive feedback deterministically.
            elif is_full and not clean_missing and not clean_contradictions:
                fallback = cls._get_deterministic_fallback(
                    awarded_marks=awarded_marks,
                    max_marks=max_marks,
                    clean_contradictions=clean_contradictions,
                    clean_missing=clean_missing,
                    clean_present=clean_present,
                    criterion_name=criterion_name,
                    is_submission_irrelevant=False
                )
                if not cls._is_feedback_consistent(
                    fallback, awarded_marks, max_marks, clean_contradictions,
                    clean_missing, clean_present, student_text, criterion_name
                ):
                    return f"Feedback is currently unavailable for '{criterion_name}'. Please refer to the assigned score ({awarded_marks:.2f}/{max_marks:.2f}) and rubric requirements."
                if feedback_stats is not None:
                    feedback_stats["ollama_called"] = False
                    feedback_stats["deterministic_reason"] = "full_marks"
                    feedback_stats["duration_ms"] = 0.0
                logger.info(
                    f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
                    f"duration_ms=0.0 mode=hybrid (deterministic full marks applied)"
                )
                return fallback

            # 3. Clearly irrelevant submission (entire answer is off-topic with no contradictions, e.g. Student E):
            elif sub_irrelevant and not clean_contradictions:
                fallback = cls._get_deterministic_fallback(
                    awarded_marks=0.0,
                    max_marks=max_marks,
                    clean_contradictions=clean_contradictions,
                    clean_missing=clean_missing,
                    clean_present=clean_present,
                    criterion_name=criterion_name,
                    is_submission_irrelevant=True
                )
                if not cls._is_feedback_consistent(
                    fallback, 0.0, max_marks, clean_contradictions,
                    clean_missing, clean_present, student_text, criterion_name
                ):
                    return f"Feedback is currently unavailable for '{criterion_name}'. Please refer to the assigned score (0.00/{max_marks:.2f}) and rubric requirements."
                if feedback_stats is not None:
                    feedback_stats["ollama_called"] = False
                    feedback_stats["deterministic_reason"] = "irrelevant_submission"
                    feedback_stats["duration_ms"] = 0.0
                logger.info(
                    f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
                    f"duration_ms=0.0 mode=hybrid (deterministic irrelevant submission applied)"
                )
                return fallback

            # 4. Relevant submission with an unaddressed criterion (zero marks, no statements identified):
            elif is_zero and has_no_statements and not clean_contradictions:
                fallback = cls._get_deterministic_fallback(
                    awarded_marks=0.0,
                    max_marks=max_marks,
                    clean_contradictions=clean_contradictions,
                    clean_missing=clean_missing,
                    clean_present=clean_present,
                    criterion_name=criterion_name,
                    is_submission_irrelevant=False
                )
                if not cls._is_feedback_consistent(
                    fallback, 0.0, max_marks, clean_contradictions,
                    clean_missing, clean_present, student_text, criterion_name
                ):
                    return f"Feedback is currently unavailable for '{criterion_name}'. Please refer to the assigned score (0.00/{max_marks:.2f}) and rubric requirements."
                if feedback_stats is not None:
                    feedback_stats["ollama_called"] = False
                    feedback_stats["deterministic_reason"] = "unaddressed_criterion"
                    feedback_stats["duration_ms"] = 0.0
                logger.info(
                    f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
                    f"duration_ms=0.0 mode=hybrid (deterministic unaddressed criterion applied)"
                )
                return fallback

            # 5. Partial marks, mixed coverage, or ambiguous evidence:
            # (e.g. candidate statements were found for this criterion, or 0.0 < awarded_marks < max_marks)
            # Retain the LLM feedback path below.

        # Format evaluated concepts without leaking unrequired illustrative examples
        if target_propositions:
            clean_props = []
            for p in target_propositions:
                if is_full or not clean_missing:
                    p_clean = re.sub(r'(?i)\b(?:such as|for example|e\.g\.).*', '(satisfied by valid examples provided)', p).strip(' ,;-')
                    clean_props.append(p_clean)
                else:
                    clean_props.append(p)
            expected_concepts_str = "\n".join([f"- {p}" for p in clean_props])
        else:
            if is_full or not clean_missing:
                desc_clean = re.sub(r'(?i)\b(?:such as|for example|e\.g\.).*', '(satisfied by valid examples provided)', criterion_description).strip(' ,;-')
                expected_concepts_str = f"- {desc_clean}"
            else:
                expected_concepts_str = f"- {criterion_description}"

        # Grounded institutional evidence context (only provide when appropriate for remediation)
        evidence_context = ""
        if citations and not is_zero and not is_full:
            evidence_context = "Institutional Reference Context (for teacher grounding; not student claims):\n" + "\n".join(
                [f"- {c.document_title} (p. {c.page_number or 'N/A'}): {c.snippet}" for c in citations[:2]]
            )

        # Score-anchored, criterion-specific directives
        if is_full:
            directive = (
                f"STATUS: FULL MARKS AWARDED ({awarded_marks:.2f} / {max_marks:.2f}). "
                f"The student's answer completely satisfied the requirements for '{criterion_name}'. "
                f"Acknowledge specifically how the student addressed '{criterion_name}'. "
                f"CRITICAL: Focus EXCLUSIVELY on '{criterion_name}'. Do NOT mention, summarize, or evaluate concepts belonging to other criteria. "
                f"CRITICAL: Illustrative examples (e.g. introduced by 'such as') are NOT mandatory requirements. Do NOT note or mention omissions of any unmentioned examples. "
                f"CRITICAL: You MUST NOT state, imply, or note that anything was omitted, missing, unmentioned, or needed to receive full marks. "
                f"CRITICAL: You MUST NOT suggest adding unmentioned examples, optional topics, or expanding further. Confirm that the student fully satisfied the criterion."
            )
        elif clean_contradictions:
            directive = (
                f"STATUS: CONTRADICTION DETECTED ({awarded_marks:.2f} / {max_marks:.2f}). "
                f"Explain the specific factual misconception identified in 'Contradictions detected' for '{criterion_name}' without reciting long lists of examples. "
                f"Focus EXCLUSIVELY on '{criterion_name}'."
            )
        elif is_zero:
            directive = (
                f"STATUS: ZERO MARKS AWARDED (0.00 / {max_marks:.2f}). "
                f"The submitted answer does not address '{criterion_name}'. "
                f"State clearly that the response does not address the required topic for '{criterion_name}'. "
                f"CRITICAL: Focus EXCLUSIVELY on '{criterion_name}'. "
                f"CRITICAL: You MUST NOT claim the student explained or partially explained any concepts, "
                f"and you MUST NOT confuse the student's text with the reference evidence."
            )
        elif not clean_missing:
            directive = (
                f"STATUS: CONCEPTS VERIFIED ({awarded_marks:.2f} / {max_marks:.2f}). "
                f"The student accurately addressed the expected concepts for '{criterion_name}'. "
                f"Acknowledge the student's explanation of '{criterion_name}'. "
                f"CRITICAL: Focus EXCLUSIVELY on '{criterion_name}'. "
                f"CRITICAL: There are NO missing concepts. You MUST NOT state or imply that anything was missing, omitted, incomplete, or needed."
            )
        else:
            directive = (
                f"STATUS: PARTIAL MARKS AWARDED ({awarded_marks:.2f} / {max_marks:.2f}). "
                f"State what was accurately addressed for '{criterion_name}' based ONLY on 'Verified concepts present'. "
                f"Identify the missing concepts listed in 'Missing concepts'. "
                f"CRITICAL: Focus EXCLUSIVELY on '{criterion_name}'. "
                f"CRITICAL: Do NOT claim the student described, explained, or mentioned any concept listed in 'Missing concepts' or absent from their text. "
                f"CRITICAL: Do NOT invent missing requirements not listed under 'Missing concepts'."
            )

        prompt_contra = [
            re.sub(r'(?i)\s*,?\s*(?:such as|for example|e\.g\.).*?([\'"])', r'\1', c)
            for c in clean_contradictions
        ]

        base_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an objective academic evaluator providing constructive feedback on an exam criterion.
Rules:
1. Respond strictly in 1 or 2 concise, factual sentences addressed directly to the student.
2. Focus EXCLUSIVELY on evaluating '{criterion_name}' and its expected concepts. Do NOT discuss or evaluate concepts belonging to other criteria.
3. Adhere strictly to the instruction in the DIRECTIVE block.
4. Do NOT invent missing concepts or contradictions not present in the Evaluation Findings.
5. Do NOT enumerate or repeat long lists of examples, applications, or tasks. Missing concepts and contradictions are displayed in separate UI sections; keep feedback concise and complete.
6. Do NOT output greetings, headings, or conversational filler.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Assessed Criterion: {criterion_name}
Target Rubric Requirement: {criterion_description}
Marks: {awarded_marks:.2f} / {max_marks:.2f}

DIRECTIVE:
{directive}

Expected Concepts for '{criterion_name}':
{expected_concepts_str}

Student's Submitted Answer Relevant to '{criterion_name}':
\"\"\"{eval_student_text}\"\"\"

Evaluation Findings for '{criterion_name}':
- Verified concepts present in student answer: {'; '.join(clean_present) if clean_present else 'None'}
- Contradictions detected: {'; '.join(prompt_contra) if prompt_contra else 'None'}
- Missing concepts: {'; '.join(clean_missing) if clean_missing else 'None'}
{evidence_context}

Provide 1-2 concise, constructive feedback sentences for the student adhering to the DIRECTIVE:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

        # Attempt bounded generation with local LLM (num_predict=112 for complete, concise feedback)
        t_start = time.perf_counter()
        gen_duration_ms = 0.0
        ollama_status = None
        prompt_tokens = 0
        eval_tokens = 0
        fallback_used = False

        if feedback_stats is not None:
            feedback_stats["ollama_called"] = True

        try:
            client = cls.get_client()
            response = client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": base_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 112
                    }
                }
            )
            ollama_status = response.status_code
            gen_duration_ms = (time.perf_counter() - t_start) * 1000
            if response.status_code == 200:
                resp_data = response.json()
                candidate_feedback = resp_data.get("response", "").strip()
                prompt_tokens = resp_data.get("prompt_eval_count", 0)
                eval_tokens = resp_data.get("eval_count", 0)
                if cls._is_feedback_consistent(
                    candidate_feedback,
                    awarded_marks,
                    max_marks,
                    criterion_name=criterion_name,
                    all_criteria_descriptions=all_criteria_descriptions,
                    clean_missing=clean_missing,
                    clean_present=clean_present,
                    clean_contradictions=clean_contradictions,
                    student_text=eval_student_text,
                    context_stopwords=q_tokens
                ):
                    if feedback_stats is not None:
                        feedback_stats["duration_ms"] = gen_duration_ms
                        feedback_stats["fallback_used"] = False
                    logger.info(
                        f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
                        f"status={ollama_status} prompt_tokens={prompt_tokens} eval_tokens={eval_tokens} "
                        f"duration_ms={gen_duration_ms:.1f} fallback_used=False"
                    )
                    return candidate_feedback
                logger.warning(f"Ollama feedback failed consistency check: {candidate_feedback}. Falling back to deterministic feedback.")
                fallback_used = True
            else:
                logger.warning(f"Ollama returned HTTP status {response.status_code}")
                fallback_used = True
        except Exception as e:
            gen_duration_ms = (time.perf_counter() - t_start) * 1000
            logger.warning(f"Ollama feedback generation failed: {str(e)}")
            fallback_used = True

        if feedback_stats is not None:
            feedback_stats["duration_ms"] = gen_duration_ms
            feedback_stats["fallback_used"] = True

        # Fallback if Ollama fails, times out, or violates consistency
        fallback = cls._get_deterministic_fallback(
            awarded_marks,
            max_marks,
            clean_contradictions,
            clean_missing,
            clean_present=clean_present,
            criterion_name=criterion_name,
            is_submission_irrelevant=sub_irrelevant
        )
        if cls._is_feedback_consistent(
            feedback=fallback,
            awarded_marks=awarded_marks,
            max_marks=max_marks,
            criterion_name=criterion_name,
            all_criteria_descriptions=all_criteria_descriptions,
            clean_missing=clean_missing,
            clean_present=clean_present,
            clean_contradictions=clean_contradictions,
            student_text=eval_student_text,
            context_stopwords=q_tokens
        ):
            logger.info(
                f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
                f"duration_ms={gen_duration_ms:.1f} fallback_used=True (deterministic fallback applied)"
            )
            return fallback

        logger.warning(f"Deterministic fallback violated consistency for '{criterion_name}'. Returning explicit unavailable state.")
        logger.info(
            f"[LLM Feedback] criterion='{criterion_name}' marks={awarded_marks:.2f}/{max_marks:.2f} "
            f"duration_ms={gen_duration_ms:.1f} fallback_used=True (unavailable state applied)"
        )
        return f"Feedback is currently unavailable for '{criterion_name}'. Please refer to the assigned score ({awarded_marks:.2f}/{max_marks:.2f}) and rubric requirements."

    @classmethod
    def generate_overall_feedback(
        cls,
        question: str,
        total_score: float,
        max_score: float,
        review_reasons: List[str]
    ) -> str:
        """Generates a concise overall summary of student performance."""
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        if total_score == 0.0 or percentage == 0.0:
            summary = "The submission does not address the assessed topic or meet any rubric criteria."
        elif percentage >= 85:
            summary = "The submission demonstrates a strong, accurate comprehension of the topic with precise terminology and sound reasoning."
        elif percentage >= 60:
            summary = "The submission demonstrates good foundational understanding, but omits some key mechanisms or application areas."
        elif percentage >= 30:
            summary = "The submission provides only partial concepts and requires substantial elaboration on mechanisms and core definitions."
        else:
            summary = "The submission does not meet the rubric criteria. Fundamental concepts are either missing or contradicted."

        if review_reasons:
            summary += f" Note: Teacher review recommended ({'; '.join(review_reasons)})."
        return summary
