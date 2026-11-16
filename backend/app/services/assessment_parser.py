import io
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import httpx

from backend.app.config import settings
from backend.app.services.table_linearizer import TableLinearizer

logger = logging.getLogger(__name__)


class AssessmentDocumentParser:
    """
    Parses instructor question papers, marking schemes, and rubric documents
    (.pdf, .docx, .txt, .md) to extract structured assessment questions,
    model answers, and rubric criteria for UI auto-fill.
    """

    CRITERION_NAME_PATTERN = re.compile(
        r'(?i)\b(criterion|criteria|component|area|aspect|topic|title|name)\b'
    )
    CRITERION_DESC_PATTERN = re.compile(
        r'(?i)\b(description|expected|requirement|details?|explanation|definition|concept|descriptors?|guidelines?)\b'
    )
    CRITERION_MARKS_PATTERN = re.compile(
        r'(?i)\b(marks?|points?|score|weight|weightage|max\s*marks?)\b'
    )

    QUESTION_HEADER_PATTERN = re.compile(
        r'(?i)^\s*(?:question|ques|q|problem|task|part)\s*[\.\:\-\)]*\s*(\d+|[a-zA-Z])(?:[:.\-\)]\s*(.*))?$'
    )
    MODEL_ANSWER_PATTERN = re.compile(
        r'(?i)^\s*(?:model\s+answer|authoritative\s+answer|expected\s+answer|sample\s+answer|answer\s+key|marking\s+scheme|solution|answer)\s*[:\-]?\s*(.*)$'
    )
    RUBRIC_HEADER_PATTERN = re.compile(
        r'(?i)^\s*(?:rubric|grading\s+criteria|marking\s+criteria|evaluation\s+criteria|criteria)\s*[:\-]?\s*(.*)$'
    )
    BULLET_CRITERION_PATTERN = re.compile(
        r'(?i)^\s*(?:[-*•]|\d+[\.)])\s*(.+?)(?:\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)\s*[\)\]])?\s*[:\-]\s*(.*)$'
    )
    TITLE_PREFIX_PATTERN = re.compile(
        r'^(?:assessment\s+title|exam\s+title|course\s+title|title|subject|paper)\s*[:\-]\s*(.*)$',
        re.IGNORECASE
    )

    @classmethod
    def _parse_rubric_table(cls, headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Extracts structured criteria from a rubric table:
        [{"name": ..., "description": ..., "max_marks": ...}]
        """
        extracted_criteria: List[Dict[str, Any]] = []
        if not headers or not rows:
            return extracted_criteria

        name_idx: Optional[int] = None
        desc_idx: Optional[int] = None
        marks_idx: Optional[int] = None

        for idx, h in enumerate(headers):
            h_clean = (h or "").strip()
            if cls.CRITERION_MARKS_PATTERN.search(h_clean):
                marks_idx = idx
            elif cls.CRITERION_DESC_PATTERN.search(h_clean):
                desc_idx = idx
            elif cls.CRITERION_NAME_PATTERN.search(h_clean):
                name_idx = idx

        if name_idx is None:
            name_idx = 0
        if desc_idx is None:
            desc_idx = 1 if len(headers) > 1 else 0

        for r_idx, row in enumerate(rows):
            if not row:
                continue
            name_val = row[name_idx].strip() if name_idx < len(row) else ""
            desc_val = row[desc_idx].strip() if desc_idx < len(row) else ""

            marks_val = 5.0
            if marks_idx is not None and marks_idx < len(row):
                m = re.search(r'(\d+(?:\.\d+)?)', row[marks_idx])
                if m:
                    try:
                        marks_val = float(m.group(1))
                    except ValueError:
                        marks_val = 5.0

            if not name_val and not desc_val:
                continue
            if not name_val:
                name_val = f"Criterion {r_idx + 1}"
            if not desc_val:
                desc_val = name_val

            extracted_criteria.append({
                "name": name_val,
                "description": desc_val,
                "max_marks": marks_val if marks_val > 0 else 5.0
            })

        return extracted_criteria

    @classmethod
    def detect_rubric_tables(cls, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detects tables whose headers qualify as a rubric table and returns their criteria.
        """
        all_criteria: List[Dict[str, Any]] = []
        for tbl in tables:
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            if TableLinearizer.is_rubric_table(headers):
                all_criteria.extend(cls._parse_rubric_table(headers, rows))
        return all_criteria

    @classmethod
    def _start_new_question(cls, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_obj = {
            "question_order": len(questions) + 1,
            "question_text_parts": [],
            "model_answer_parts": [],
            "criteria": [],
            "has_real_criteria": False,
            "active_section": "question"
        }
        questions.append(q_obj)
        return q_obj

    @classmethod
    def _parse_blocks_deterministically(
        cls,
        blocks: List[Dict[str, Any]],
        filename: str
    ) -> Dict[str, Any]:
        """
        Deterministically groups ordered blocks into questions Q1..Qn, routes
        non-rubric tables (linearized) to model_answer or question_text, and parses
        rubric tables directly into criteria.
        """
        base_name = os.path.splitext(os.path.basename(filename or ""))[0]
        default_title = re.sub(r'[_ -]+', ' ', base_name).strip() or "Assessment"
        title = default_title

        # Check if the document contains any explicit question headers (Q1, Question 1, etc.)
        has_explicit_questions = False
        for b in blocks:
            if b.get("type") == "text":
                for l in b.get("text", "").splitlines():
                    if cls.QUESTION_HEADER_PATTERN.match(l.strip()):
                        has_explicit_questions = True
                        break
            if has_explicit_questions:
                break

        questions: List[Dict[str, Any]] = []
        curr_q: Optional[Dict[str, Any]] = None
        unassigned_rubric_tables: List[List[Dict[str, Any]]] = []
        title_found = False

        for block in blocks:
            b_type = block.get("type")

            if b_type == "table":
                headers = block.get("headers", [])
                rows = block.get("rows", [])

                if TableLinearizer.is_rubric_table(headers):
                    tbl_criteria = cls._parse_rubric_table(headers, rows)
                    if not questions:
                        unassigned_rubric_tables.append(tbl_criteria)
                    else:
                        empty_questions = [q for q in questions if not q["has_real_criteria"]]
                        if len(empty_questions) > 1 and len(tbl_criteria) >= len(empty_questions):
                            if len(tbl_criteria) == len(empty_questions):
                                for target_q, crit in zip(empty_questions, tbl_criteria):
                                    target_q["criteria"].append(crit)
                                    target_q["has_real_criteria"] = True
                            else:
                                for idx, crit in enumerate(tbl_criteria):
                                    target_q = empty_questions[idx % len(empty_questions)]
                                    target_q["criteria"].append(crit)
                                    target_q["has_real_criteria"] = True
                        elif empty_questions:
                            empty_questions[-1]["criteria"].extend(tbl_criteria)
                            empty_questions[-1]["has_real_criteria"] = True
                        elif curr_q:
                            curr_q["criteria"].extend(tbl_criteria)
                            curr_q["has_real_criteria"] = True
                        else:
                            questions[-1]["criteria"].extend(tbl_criteria)
                            questions[-1]["has_real_criteria"] = True
                else:
                    # Non-rubric comparison or data table: linearize and append to active subsection
                    linearized = TableLinearizer.linearize_table(headers, rows)
                    if not curr_q and not has_explicit_questions:
                        curr_q = cls._start_new_question(questions)
                    if curr_q:
                        if curr_q["active_section"] == "model_answer":
                            curr_q["model_answer_parts"].append(linearized)
                        else:
                            curr_q["question_text_parts"].append(linearized)

            elif b_type == "text":
                text_content = block.get("text", "")
                if not text_content:
                    continue

                lines = text_content.splitlines()
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue

                    # Check for explicit Title prefix
                    m_title = cls.TITLE_PREFIX_PATTERN.match(stripped)
                    if m_title:
                        title = re.sub(
                            r'^(?:assessment\s+title|exam\s+title|course\s+title|title|subject|paper)\s*[:\-]\s*',
                            '',
                            stripped,
                            flags=re.IGNORECASE
                        ).strip()
                        title_found = True
                        continue

                    # Check for Question Header
                    m_q = cls.QUESTION_HEADER_PATTERN.match(stripped)
                    if m_q:
                        curr_q = cls._start_new_question(questions)
                        curr_q["active_section"] = "question"
                        trailing = m_q.group(2).strip() if m_q.group(2) else ""

                        # Check if Model Answer appears on the same line as Question Header
                        ma_match = re.search(
                            r'(?i)\b(?:Model\s+Answer|Authoritative\s+Answer|Expected\s+Answer|Sample\s+Answer|Answer\s+Key)\s*[:\-]\s*(.*)$',
                            trailing
                        )
                        if ma_match:
                            q_part = trailing[:ma_match.start()].strip()
                            ans_part = ma_match.group(1).strip()
                            if q_part:
                                curr_q["question_text_parts"].append(q_part)
                            curr_q["active_section"] = "model_answer"
                            if ans_part:
                                curr_q["model_answer_parts"].append(ans_part)
                        elif trailing:
                            curr_q["question_text_parts"].append(trailing)
                        continue

                    # If no question has started yet and the document has explicit questions,
                    # this text is preamble or document title
                    if not curr_q and has_explicit_questions:
                        if not title_found and len(stripped) < 120 and "|" not in stripped:
                            title = stripped
                            title_found = True
                        continue

                    # Check for Model Answer Header
                    m_ma = cls.MODEL_ANSWER_PATTERN.match(stripped)
                    if m_ma:
                        if not curr_q:
                            curr_q = cls._start_new_question(questions)
                        curr_q["active_section"] = "model_answer"
                        trailing = m_ma.group(1).strip() if m_ma.group(1) else ""
                        if trailing:
                            curr_q["model_answer_parts"].append(trailing)
                        continue

                    # Check for Rubric Header
                    m_rub = cls.RUBRIC_HEADER_PATTERN.match(stripped)
                    if m_rub:
                        if not curr_q:
                            curr_q = cls._start_new_question(questions)
                        curr_q["active_section"] = "rubric"
                        trailing = m_rub.group(1).strip() if m_rub.group(1) else ""
                        if trailing:
                            b_m = cls.BULLET_CRITERION_PATTERN.match(trailing)
                            if b_m:
                                c_name = b_m.group(1).strip()
                                c_marks = float(b_m.group(2)) if b_m.group(2) else 5.0
                                c_desc = b_m.group(3).strip() if b_m.group(3) else c_name
                                curr_q["criteria"].append({
                                    "name": c_name,
                                    "description": c_desc,
                                    "max_marks": c_marks
                                })
                                curr_q["has_real_criteria"] = True
                        continue

                    # Normal text line inside active section
                    if not curr_q:
                        if not title_found and len(stripped) < 120 and "|" not in stripped:
                            title = stripped
                            title_found = True
                            continue
                        curr_q = cls._start_new_question(questions)

                    if curr_q["active_section"] == "rubric":
                        b_m = cls.BULLET_CRITERION_PATTERN.match(stripped)
                        if b_m:
                            c_name = b_m.group(1).strip()
                            c_marks = float(b_m.group(2)) if b_m.group(2) else 5.0
                            c_desc = b_m.group(3).strip() if b_m.group(3) else c_name
                            curr_q["criteria"].append({
                                "name": c_name,
                                "description": c_desc,
                                "max_marks": c_marks
                            })
                            curr_q["has_real_criteria"] = True
                        elif curr_q["criteria"]:
                            curr_q["criteria"][-1]["description"] += " " + stripped
                    elif curr_q["active_section"] == "model_answer":
                        curr_q["model_answer_parts"].append(stripped)
                    else:
                        curr_q["question_text_parts"].append(stripped)

        # 3. Post-process unassigned rubric tables
        if unassigned_rubric_tables and questions:
            for tbl_criteria in unassigned_rubric_tables:
                empty_questions = [q for q in questions if not q["has_real_criteria"]]
                if empty_questions:
                    for idx, crit in enumerate(tbl_criteria):
                        target_q = empty_questions[idx % len(empty_questions)]
                        target_q["criteria"].append(crit)
                        target_q["has_real_criteria"] = True
                else:
                    for idx, crit in enumerate(tbl_criteria):
                        target_q = questions[idx % len(questions)]
                        target_q["criteria"].append(crit)

        # 4. Finalize question fields and criteria fallbacks
        final_questions: List[Dict[str, Any]] = []
        for q in questions:
            q_text = "\n\n".join(q["question_text_parts"]).strip()
            m_ans = "\n\n".join(q["model_answer_parts"]).strip()
            if not m_ans:
                m_ans = q_text

            crits = q["criteria"]
            if not crits:
                crits = [{
                    "name": "Core Concept Alignment",
                    "description": f"Accurately explains and addresses the core requirements of: {q_text[:80]}...",
                    "max_marks": 5.0
                }]

            q_max = sum(c["max_marks"] for c in crits)
            final_questions.append({
                "question_order": q["question_order"],
                "question_text": q_text,
                "model_answer": m_ans,
                "max_marks": q_max,
                "criteria": crits,
                "has_real_criteria": q["has_real_criteria"]
            })

        total_marks = sum(q["max_marks"] for q in final_questions)

        return {
            "title": title,
            "total_marks": total_marks,
            "doc_scope": "ALL",
            "questions": final_questions
        }

    @classmethod
    def _call_ollama_parser(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Invokes Ollama (llama3.2) with a strict JSON format prompt to extract
        assessment structure. Returns parsed dict or None if unavailable.
        """
        bounded_text = text[:8000] if len(text) > 8000 else text

        prompt = f"""You are an educational assessment parser. Extract the structured assessment questions and grading rubric criteria from the text below into JSON.

Return ONLY a single valid JSON object matching this exact schema:
{{
  "title": "Assessment Title",
  "questions": [
    {{
      "question_text": "The question prompt or problem statement",
      "model_answer": "Authoritative model answer or expected key solution points",
      "criteria": [
        {{
          "name": "Short criterion name",
          "description": "Factual grading criteria description of expected concepts",
          "max_marks": 5.0
        }}
      ]
    }}
  ]
}}

Guidelines:
- Each question must have at least one rubric criterion with max_marks > 0.
- If multiple questions exist, extract each as a separate item in the questions list.
- Return ONLY valid JSON, with no preamble or code fences.

Document text:
{bounded_text}
"""

        try:
            with httpx.Client(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
                res = client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "keep_alive": "15m",
                        "options": {
                            "num_ctx": 4096,
                            "num_predict": 1024,
                            "temperature": 0.1,
                            "top_p": 0.9
                        }
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    response_text = data.get("response", "").strip()
                    parsed = json.loads(response_text)
                    if isinstance(parsed, dict) and "questions" in parsed:
                        return parsed
        except Exception as e:
            logger.info(f"[AssessmentParser] Ollama LLM extraction skipped or failed ({e}); falling back to deterministic parser.")

        return None

    @classmethod
    def parse_document(cls, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """
        Main entry point for parsing an assessment document (.pdf, .docx, .txt, .md).
        Extracts ordered document blocks (paragraphs and raw tables), applies deterministic
        question-scoped block grouping, and returns immediately if valid criteria are found.
        Otherwise falls back to Ollama LLM extraction.
        """
        # 1. Extract ordered document blocks in exact reading order
        blocks = TableLinearizer.extract_document_blocks(filename, file_bytes)
        if not blocks:
            base_name = os.path.splitext(os.path.basename(filename or ""))[0]
            title = re.sub(r'[_ -]+', ' ', base_name).strip() or "Assessment"
            return {
                "title": title,
                "total_marks": 5.0,
                "doc_scope": "ALL",
                "questions": [
                    {
                        "question_order": 1,
                        "question_text": "Assessment Question",
                        "model_answer": "Model Answer",
                        "max_marks": 5.0,
                        "criteria": [
                            {
                                "name": "Criterion 1 - Core Accuracy",
                                "description": "Accurately addresses the question requirements.",
                                "max_marks": 5.0
                            }
                        ]
                    }
                ]
            }

        # 2. Deterministic block-level parsing
        det_result = cls._parse_blocks_deterministically(blocks, filename)
        questions = det_result.get("questions", [])

        # Check if deterministic parsing succeeded with real criteria
        has_real_crit = any(q.get("has_real_criteria", False) for q in questions)
        has_valid_q_text = all(bool(q.get("question_text", "").strip()) for q in questions)

        if questions and has_real_crit and has_valid_q_text:
            # Clean up internal tracking flag
            for q in questions:
                q.pop("has_real_criteria", None)
            return det_result

        # 3. If no real rubric criteria were found, try Ollama LLM extraction
        full_text_blocks = []
        for b in blocks:
            if b.get("type") == "text":
                txt = b.get("text", "").strip()
                if txt:
                    full_text_blocks.append(txt)
            elif b.get("type") == "table":
                full_text_blocks.append(
                    TableLinearizer.linearize_table(b.get("headers", []), b.get("rows", []))
                )
        full_text = "\n\n".join(full_text_blocks)

        llm_result = None
        if full_text:
            llm_result = cls._call_ollama_parser(full_text)

        if llm_result and isinstance(llm_result.get("questions"), list) and len(llm_result["questions"]) > 0:
            parsed_title = llm_result.get("title") or det_result.get("title")
            normalized_questions: List[Dict[str, Any]] = []

            for idx, q in enumerate(llm_result["questions"]):
                q_text = (q.get("question_text") or "").strip()
                m_ans = (q.get("model_answer") or "").strip() or q_text
                raw_crits = q.get("criteria", [])

                norm_crits: List[Dict[str, Any]] = []
                if isinstance(raw_crits, list) and raw_crits:
                    for c_idx, c in enumerate(raw_crits):
                        c_name = (c.get("name") or f"Criterion {c_idx + 1}").strip()
                        c_desc = (c.get("description") or c_name).strip()
                        try:
                            c_marks = float(c.get("max_marks", 5.0))
                            if c_marks <= 0:
                                c_marks = 5.0
                        except (ValueError, TypeError):
                            c_marks = 5.0
                        norm_crits.append({
                            "name": c_name,
                            "description": c_desc,
                            "max_marks": c_marks
                        })

                if not norm_crits:
                    norm_crits = [{
                        "name": "Core Concept Alignment",
                        "description": f"Accurately addresses the core concepts: {q_text[:80]}...",
                        "max_marks": 5.0
                    }]

                q_max = sum(c["max_marks"] for c in norm_crits)
                normalized_questions.append({
                    "question_order": idx + 1,
                    "question_text": q_text or f"Question {idx + 1}",
                    "model_answer": m_ans,
                    "max_marks": q_max,
                    "criteria": norm_crits
                })

            total_marks = sum(q["max_marks"] for q in normalized_questions)
            return {
                "title": parsed_title,
                "total_marks": total_marks,
                "doc_scope": "ALL",
                "questions": normalized_questions
            }

        # 4. Fallback: return deterministic result
        for q in questions:
            q.pop("has_real_criteria", None)
        return det_result
