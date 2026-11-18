import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.services.models_loader import ModelsLoader

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "explain", "describe", "discuss", "distinguish", "define", "identify", "compare", "provide",
    "question", "answer", "student", "rubric", "criterion", "marks"
}

ANCHOR_REGEX = re.compile(
    r'(?im)^\s*(?:'
    r'(?:question|ques|q|answer|ans|part)\s*[\.\:\-\)]*\s*(\d+|[a-zA-Z])\b'
    r'|(\d+)[\.\)]\s+'
    r'|\((\d+)\)\s+'
    r')'
)

INLINE_ANCHOR_REGEX = re.compile(
    r'(?im)^\s*(?:question|ques|q)\s*(\d+)\s*[:\.\-\)]\s+'
)

EXPLICIT_PREFIX_REGEX = re.compile(
    r'(?im)^\s*(?:question|ques|q|answer|ans|part)\s*[\.\:\-\)]*\s*(\d+|[a-zA-Z])\b'
)


def _get_q_field(q: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(q, dict):
        return q.get(field_name, default)
    return getattr(q, field_name, default)


def _get_qid(q: Any, idx: int = 0) -> str:
    qid = _get_q_field(q, "id")
    if qid is not None:
        return str(qid)
    order = _get_q_field(q, "question_order", idx + 1)
    return str(order)


class AnswerSegmenter:
    """
    3-Stage Automatic Student Submission Segmenter for Multi-Question Assessments.
    - Fast Path: 0 overhead for single-question assessments.
    - Stage 1: Structural Anchor Detection (Q1, Question 2, Part A, 1., etc.)
    - Stage 2: Semantic Paragraph Routing (Embedding cosine similarity + token overlap)
    - Stage 3: Empty Slice Handling for skipped questions.
    """

    @classmethod
    def segment_submission(cls, submission_text: str, questions: List[Any]) -> Dict[str, str]:
        if not questions:
            return {}

        # Fast Path: Single-question assessment
        if len(questions) <= 1:
            qid = _get_qid(questions[0], 0)
            return {qid: (submission_text or "").strip()}

        cleaned_text = (submission_text or "").strip()
        if not cleaned_text:
            return {_get_qid(q, idx): "" for idx, q in enumerate(questions)}

        # Build order -> q mapping
        order_to_qid: Dict[int, str] = {}
        for idx, q in enumerate(questions):
            order = _get_q_field(q, "question_order", idx + 1)
            order_to_qid[int(order)] = _get_qid(q, idx)

        # Stage 1: Structural Anchor Detection
        stage1_slices = cls._stage1_structural_anchors(cleaned_text, questions, order_to_qid)
        if stage1_slices is not None:
            # Stage 3: Ensure all questions exist in dict
            return cls._stage3_fill_empty(stage1_slices, questions)

        # Stage 2: Semantic Paragraph Routing
        stage2_slices = cls._stage2_semantic_routing(cleaned_text, questions)
        return cls._stage3_fill_empty(stage2_slices, questions)

    @classmethod
    def _stage1_structural_anchors(
        cls,
        text: str,
        questions: List[Any],
        order_to_qid: Dict[int, str]
    ) -> Optional[Dict[str, str]]:
        matches = []
        seen_starts = set()

        # Check inline explicit Question N: <body text> anchors
        for m in INLINE_ANCHOR_REGEX.finditer(text):
            raw_lbl = m.group(1)
            if raw_lbl and raw_lbl.isdigit():
                target_order = int(raw_lbl)
                if target_order in order_to_qid:
                    matches.append({
                        "start": m.start(),
                        "end": m.end(),
                        "order": target_order,
                        "qid": order_to_qid[target_order],
                        "is_explicit": True,
                    })
                    seen_starts.add(m.start())

        # Check general anchor regex
        for m in ANCHOR_REGEX.finditer(text):
            if m.start() in seen_starts:
                continue
            raw_lbl = m.group(1) or m.group(2) or m.group(3)
            if not raw_lbl:
                continue
            is_explicit = bool(m.group(1))

            target_order: Optional[int] = None
            if raw_lbl.isdigit():
                target_order = int(raw_lbl)
            elif len(raw_lbl) == 1 and raw_lbl.isalpha():
                target_order = ord(raw_lbl.upper()) - ord('A') + 1

            if target_order is not None and target_order in order_to_qid:
                matches.append({
                    "start": m.start(),
                    "end": m.end(),
                    "order": target_order,
                    "qid": order_to_qid[target_order],
                    "is_explicit": is_explicit,
                })
                seen_starts.add(m.start())

        if not matches:
            return None

        # Check if matches are valid structural boundaries
        distinct_qids = {m["qid"] for m in matches}
        has_explicit = any(m["is_explicit"] for m in matches)

        # If we have only 1 match and it's a weak number (e.g. "1. "), it's likely an internal list item
        if len(distinct_qids) < 2 and not has_explicit:
            return None

        # If explicit markers are present, filter out weak markers to avoid numbered bullet collision
        if has_explicit:
            matches = [m for m in matches if m["is_explicit"]]
            distinct_qids = {m["qid"] for m in matches}

        # Sort matches by start position
        matches.sort(key=lambda x: x["start"])

        # Slice text between consecutive boundaries
        slices: Dict[str, str] = {}
        for i, m in enumerate(matches):
            qid = m["qid"]
            content_start = m["end"]
            content_end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)
            chunk = text[content_start:content_end].strip()
            chunk = re.sub(r'^[\s:\-\)\.]+', '', chunk).strip()

            if qid in slices and slices[qid]:
                slices[qid] = f"{slices[qid]}\n\n{chunk}"
            else:
                slices[qid] = chunk

        # If the first matched anchor has anchor_num > 1 (e.g. Question 3: is the first explicit anchor found)
        # and non-trivial pre-anchor text (>= 80 chars) precedes it while q1 is empty,
        # assign that pre-anchor text to q1 rather than discarding it as cover-page noise.
        if matches and matches[0]["order"] > 1 and 1 in order_to_qid:
            qid_1 = order_to_qid[1]
            if not slices.get(qid_1, "").strip():
                pre_text = text[:matches[0]["start"]].strip()
                if len(pre_text) >= 80:
                    m_inline_q1 = re.search(
                        r'(?im)^\s*(?:question|ques|q)\s*1\s*[:\.\-\)]\s*(.*)',
                        pre_text,
                        re.DOTALL
                    )
                    if m_inline_q1:
                        q1_body = m_inline_q1.group(1).strip()
                    else:
                        q1_body = pre_text
                    q1_body = re.sub(r'^[\s:\-\)\.]+', '', q1_body).strip()
                    if q1_body:
                        slices[qid_1] = q1_body

        return slices

    _stage1_explicit_anchors = _stage1_structural_anchors

    @classmethod
    def _stage2_semantic_routing(cls, text: str, questions: List[Any]) -> Dict[str, str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            return {}

        # Build target profile for each question
        q_profiles: List[str] = []
        q_token_sets: List[set] = []
        q_ids: List[str] = []

        for idx, q in enumerate(questions):
            qid = _get_qid(q, idx)
            q_ids.append(qid)

            q_text = _get_q_field(q, "question_text", "") or ""
            model_ans = _get_q_field(q, "model_answer", "") or ""
            criteria = _get_q_field(q, "criteria", []) or []

            crit_strs = []
            for c in criteria:
                c_name = _get_q_field(c, "name", "") or ""
                c_desc = _get_q_field(c, "description", "") or ""
                crit_strs.append(f"{c_name} {c_desc}")

            full_profile = f"{q_text} {model_ans} " + " ".join(crit_strs)
            q_profiles.append(full_profile.strip())

            tokens = {
                w for w in re.findall(r'[a-zA-Z]{3,}', full_profile.lower())
                if w not in STOPWORDS
            }
            q_token_sets.append(tokens)

        # Encode question profiles
        profile_embs = ModelsLoader.encode_texts(q_profiles)  # Shape (N, D)

        # Encode paragraphs
        para_embs = ModelsLoader.encode_texts(paragraphs)  # Shape (P, D)

        assigned_paras: Dict[str, List[str]] = {qid: [] for qid in q_ids}

        for p_idx, p_text in enumerate(paragraphs):
            p_emb = para_embs[p_idx]
            sims = np.dot(profile_embs, p_emb)  # Cosine similarities since normalized

            p_tokens = {
                w for w in re.findall(r'[a-zA-Z]{3,}', p_text.lower())
                if w not in STOPWORDS
            }

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_overlap = len(p_tokens & q_token_sets[best_idx])

            # Similarity threshold >= 0.30 or token overlap >= 2
            if best_sim >= 0.30 or best_overlap >= 2:
                best_qid = q_ids[best_idx]
                assigned_paras[best_qid].append(p_text)
            else:
                logger.info(
                    f"[AnswerSegmenter] Discarded off-topic paragraph: "
                    f"sim={best_sim:.3f}, overlap={best_overlap}, preview='{p_text[:60]}...'"
                )

        slices: Dict[str, str] = {}
        for qid, p_list in assigned_paras.items():
            slices[qid] = "\n\n".join(p_list).strip()

        return slices

    @classmethod
    def _stage3_fill_empty(cls, slices: Dict[str, str], questions: List[Any]) -> Dict[str, str]:
        final_slices: Dict[str, str] = {}
        for idx, q in enumerate(questions):
            qid = _get_qid(q, idx)
            raw = (slices.get(qid) or "").strip()
            final_slices[qid] = re.sub(r'^[\s:\-\)\.]+', '', raw).strip()

        # Check if one or more interior or trailing questions are empty
        empty_entries = [
            (idx, q, _get_qid(q, idx))
            for idx, q in enumerate(questions)
            if not final_slices.get(_get_qid(q, idx), "").strip()
        ]
        if not empty_entries:
            return final_slices

        # Check if any host question holds a long multi-block slice (>= 4 blocks)
        from backend.app.services.models_loader import ModelsLoader
        from sklearn.metrics.pairwise import cosine_similarity

        for host_idx, host_q in enumerate(questions):
            host_qid = _get_qid(host_q, host_idx)
            host_text = final_slices.get(host_qid, "").strip()
            if not host_text:
                continue

            raw_blocks = [b.strip() for b in re.split(r'\n\s*\n+', host_text) if b.strip()]
            if len(raw_blocks) < 4:
                continue

            host_prompt = (
                (_get_q_field(host_q, "question_text", "") or "")
                + " "
                + (_get_q_field(host_q, "model_answer", "") or "")
            ).strip()
            if not host_prompt:
                continue

            host_emb = ModelsLoader.encode_texts([host_prompt])[0]
            block_embs = ModelsLoader.encode_texts(raw_blocks)
            sims_host = cosine_similarity(block_embs, [host_emb]).flatten()

            for empty_idx, empty_q, empty_qid in empty_entries:
                if final_slices.get(empty_qid, "").strip():
                    continue

                empty_prompt = (
                    (_get_q_field(empty_q, "question_text", "") or "")
                    + " "
                    + (_get_q_field(empty_q, "model_answer", "") or "")
                ).strip()
                if not empty_prompt:
                    continue

                empty_emb = ModelsLoader.encode_texts([empty_prompt])[0]
                sims_empty = cosine_similarity(block_embs, [empty_emb]).flatten()

                # Find longest contiguous sequence of blocks where sim_empty > sim_host + 0.02
                best_seq_start = -1
                best_seq_len = 0
                curr_start = -1
                curr_len = 0

                for b_i in range(len(raw_blocks)):
                    if sims_empty[b_i] > sims_host[b_i] + 0.02 and sims_empty[b_i] >= 0.28:
                        if curr_start == -1:
                            curr_start = b_i
                            curr_len = 1
                        else:
                            curr_len += 1
                    else:
                        if curr_len > best_seq_len:
                            best_seq_start = curr_start
                            best_seq_len = curr_len
                        curr_start = -1
                        curr_len = 0

                if curr_len > best_seq_len:
                    best_seq_start = curr_start
                    best_seq_len = curr_len

                if best_seq_start >= 0 and best_seq_len >= 1 and (len(raw_blocks) - best_seq_len) >= 1:
                    rescued_blocks = raw_blocks[best_seq_start : best_seq_start + best_seq_len]
                    remaining_blocks = raw_blocks[:best_seq_start] + raw_blocks[best_seq_start + best_seq_len:]

                    final_slices[empty_qid] = "\n\n".join(rescued_blocks).strip()
                    final_slices[host_qid] = "\n\n".join(remaining_blocks).strip()

                    raw_blocks = remaining_blocks
                    if len(raw_blocks) >= 4:
                        block_embs = ModelsLoader.encode_texts(raw_blocks)
                        sims_host = cosine_similarity(block_embs, [host_emb]).flatten()
                    else:
                        break

        return final_slices
