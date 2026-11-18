import io
import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    pdfplumber = None
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    fitz = None
    HAS_FITZ = False

try:
    from docx import Document as DocxDocument
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    HAS_DOCX = True
except ImportError:
    DocxDocument = None
    HAS_DOCX = False


class TableLinearizer:
    """
    Unified structured table extraction and propositional linearization service.
    Transforms 2D table grids and Markdown pipe tables into self-contained declarative
    propositions binding column headers to cell values.
    """

    KEY_HEADERS_PATTERN = re.compile(
        r'(?i)^(?:key|property|field|item|parameter|attribute|name|setting|criterion|component|question|metric|term)$'
    )
    VALUE_HEADERS_PATTERN = re.compile(
        r'(?i)^(?:value|val|description|details|definition|explanation|content|marks|score|points|meaning|expected)$'
    )
    CRITERION_HEADER_PATTERN = re.compile(
        r'(?i)\b(criterion|criteria|component|rubric|aspect|dimension|topic)\b'
    )
    MARKS_HEADER_PATTERN = re.compile(
        r'(?i)\b(mark|marks|point|points|score|weight|weightage|max)\b'
    )

    PDF_TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 4,
        "join_tolerance": 4,
    }

    @classmethod
    def clean_cell_text(cls, cell: Optional[str]) -> str:
        """
        Normalizes table cell text by joining line-wrap hyphens and replacing
        internal newlines with spaces.
        """
        if not cell:
            return ""
        c = re.sub(r'([A-Za-z]{3,})-\s*\n\s*([a-z]{2,})', r'\1-\2', str(cell))
        c = re.sub(r'-\s*\n\s*', '-', c)
        c = re.sub(r'\s*\n\s*', ' ', c)
        return re.sub(r'\s+', ' ', c).strip()

    @classmethod
    def normalize_pdf_text(cls, raw_text: str) -> str:
        """
        Normalizes extracted PDF text by:
        1. Cleaning mid-word hyphenated line breaks (e.g. 'subword-\nsegmentation' -> 'subword-segmentation').
        2. Rejoining soft line wraps within visual paragraphs into single spaces while
           preserving genuine paragraph breaks (\n\n) and structural line boundaries
           (headers, list bullets, table pipes).
        """
        if not raw_text:
            return ""

        # 1. Clean mid-word hyphenated line breaks
        text = re.sub(r'([A-Za-z]{3,})-\s*\n\s*([a-z]{2,})', r'\1-\2', raw_text)
        text = re.sub(r'-\s*\n\s*', '-', text)

        # 2. Split by genuine paragraph breaks (2 or more newlines)
        paragraphs = re.split(r'\n\s*\n+', text)
        normalized_paragraphs: List[str] = []

        question_label_pattern = re.compile(
            r'(?i)^\s*(?:question|ques|q|problem|part)\s*[\.\:\-\)]*\s*(?:\d+|[a-zA-Z])\s*[:\.\-\)]?\s*$'
        )
        header_re = re.compile(
            r'(?i)^\s*(?:question|ques|q|problem|part|model\s+answer|authoritative\s+answer|expected\s+answer|sample\s+answer|rubric|grading\s+criteria|marking\s+criteria|marking\s+scheme|criteria|title|assessment\s+title)\b'
        )
        bullet_re = re.compile(r'^\s*(?:[-*•]|\d+[\.)]|\([0-9a-zA-Z]+\))\s+')

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            lines = [l.strip() for l in p_clean.splitlines() if l.strip()]
            if not lines:
                continue

            joined_lines: List[str] = []
            for line in lines:
                is_boundary = (
                    bool(question_label_pattern.match(line))
                    or bool(header_re.match(line))
                    or bool(bullet_re.match(line))
                    or line.startswith("|")
                    or bool(re.match(r'^(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|\d+|-+\s*\d+\s*-+|page\s*[-–—]\s*\d+)$', line, re.IGNORECASE))
                )

                if is_boundary or not joined_lines:
                    joined_lines.append(line)
                else:
                    prev = joined_lines[-1]
                    if (
                        prev.startswith("|")
                        or bool(question_label_pattern.match(prev))
                        or bool(header_re.match(prev))
                        or bool(re.search(r'\b(?:continued|cont\.?)\b', prev, re.IGNORECASE))
                    ):
                        joined_lines.append(line)
                    else:
                        joined_lines[-1] = prev + " " + line

            normalized_paragraphs.append("\n".join(joined_lines))

        return "\n\n".join(normalized_paragraphs)

    @classmethod
    def _merge_wrapped_rows(
        cls,
        headers: List[str],
        rows: List[List[str]]
    ) -> List[List[str]]:
        """
        Merges wrapped partial rows inside extracted tables. If a row has an empty
        first column (and empty last column for 3+ column tables), merge its non-empty
        cell strings into the preceding row.
        """
        if not rows:
            return []
        merged: List[List[str]] = []
        for row in rows:
            if not row:
                continue
            row_clean = [cls.clean_cell_text(c) for c in row]
            is_continuation = False
            if merged:
                first_empty = not row_clean[0]
                if len(row_clean) >= 3:
                    last_empty = not row_clean[-1]
                    if first_empty and last_empty:
                        is_continuation = True
                elif len(row_clean) == 2:
                    if first_empty:
                        is_continuation = True

            if is_continuation:
                prev = merged[-1]
                for c in range(min(len(prev), len(row_clean))):
                    val = row_clean[c]
                    if val:
                        prev[c] = (prev[c] + " " + val).strip()
            else:
                merged.append(row_clean)
        return merged

    @classmethod
    def _stitch_continuation_tables(cls, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stitches tables that span across page breaks or are separated at most by
        page number/noise text.
        """
        if not blocks or len(blocks) < 2:
            return blocks

        def _is_ignorable_page_noise(b: Dict[str, Any]) -> bool:
            if b.get("type") != "text":
                return False
            t = (b.get("text") or "").strip()
            if not t:
                return True
            return bool(re.match(
                r'^(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|\d+|-+\s*\d+\s*-+|page\s*[-–—]\s*\d+)$',
                t,
                re.IGNORECASE
            ))

        i = 0
        while i < len(blocks):
            b1 = blocks[i]
            if b1.get("type") == "table":
                H1 = b1.get("headers", [])
                R1 = b1.get("rows", [])

                # Look ahead for next non-noise block
                j = i + 1
                noise_indices = []
                while j < len(blocks) and _is_ignorable_page_noise(blocks[j]):
                    noise_indices.append(j)
                    j += 1

                if j < len(blocks) and blocks[j].get("type") == "table":
                    b2 = blocks[j]
                    H2 = b2.get("headers", [])
                    R2 = b2.get("rows", [])

                    if len(H1) == len(H2) and len(H1) > 0:
                        H1_clean = [cls.clean_cell_text(c) for c in H1]
                        H2_clean = [cls.clean_cell_text(c) for c in H2]

                        H1_norm = [c.strip().lower() for c in H1_clean]
                        H2_norm = [c.strip().lower() for c in H2_clean]

                        if H1_norm == H2_norm:
                            # Repeated header row on the new page
                            R1.extend(R2)
                        else:
                            # Continuation data row of the split table
                            # (applicable to both rubric tables and comparison/data tables)
                            R1.append(H2_clean)
                            R1.extend(R2)

                        b1["rows"] = cls._merge_wrapped_rows(H1_clean, R1)
                        indices_to_remove = set([j] + noise_indices)
                        blocks = [b for idx, b in enumerate(blocks) if idx not in indices_to_remove]
                        continue
            i += 1
        return blocks

    @classmethod
    def is_rubric_table(cls, headers: List[str]) -> bool:
        """
        Returns True when normalized headers contain at least one criterion-label header
        (criterion, criteria, component, rubric, aspect, dimension, topic)
        AND at least one marks/weight header (mark, marks, point, points, score, weight, max).
        """
        if not headers:
            return False
        has_crit = any(bool(cls.CRITERION_HEADER_PATTERN.search(h or "")) for h in headers)
        has_marks = any(bool(cls.MARKS_HEADER_PATTERN.search(h or "")) for h in headers)
        return has_crit and has_marks

    @classmethod
    def linearize_table(
        cls,
        headers: List[str],
        rows: List[List[str]],
        key_value: Optional[bool] = None
    ) -> str:
        """
        Converts a 2D table grid (headers + rows) into self-contained declarative propositions.

        - 2-column key-value tables: "<Key>: <Value>."
        - Standard data/comparison tables:
          "For <H0> <V0>, <H1> is <V1>, ..., and <Hn> is <Vn>."
        """
        if not rows:
            return ""

        clean_headers = [re.sub(r'\s+', ' ', (h or "")).strip() for h in headers]

        # Determine key-value vs comparison table
        is_kv = key_value
        if is_kv is None:
            if len(clean_headers) == 2:
                h0_match = bool(cls.KEY_HEADERS_PATTERN.match(clean_headers[0]))
                h1_match = bool(cls.VALUE_HEADERS_PATTERN.match(clean_headers[1]))
                is_kv = h0_match or h1_match or (not clean_headers[0] and not clean_headers[1])
            elif not clean_headers and all(len(r) == 2 for r in rows):
                is_kv = True
            else:
                is_kv = False

        row_sentences = []

        if is_kv:
            for row in rows:
                if not row:
                    continue
                k = re.sub(r'\s+', ' ', (row[0] or "")).strip()
                v = re.sub(r'\s+', ' ', (row[1] or "")).strip() if len(row) > 1 else ""
                if not k and not v:
                    continue
                if k and v:
                    clean_v = v.rstrip('.')
                    row_sentences.append(f"{k}: {clean_v}.")
                elif k:
                    row_sentences.append(f"{k.rstrip('.')}.")
                else:
                    row_sentences.append(f"{v.rstrip('.')}.")
        else:
            # Comparison / Data table
            for row in rows:
                if not row:
                    continue

                pairs: List[Tuple[str, str]] = []
                for idx, cell in enumerate(row):
                    cell_val = re.sub(r'\s+', ' ', (cell or "")).strip().rstrip('.')
                    if not cell_val:
                        continue
                    header_name = clean_headers[idx] if idx < len(clean_headers) else f"Column {idx + 1}"
                    pairs.append((header_name, cell_val))

                if not pairs:
                    continue

                if len(pairs) == 1:
                    h0, v0 = pairs[0]
                    sentence = f"For {h0} {v0}." if h0 else f"{v0}."
                elif len(pairs) == 2:
                    h0, v0 = pairs[0]
                    h1, v1 = pairs[1]
                    first_part = f"For {h0} {v0}" if h0 else f"For {v0}"
                    second_part = f"{h1} is {v1}" if h1 else f"is {v1}"
                    sentence = f"{first_part}, {second_part}."
                else:
                    h0, v0 = pairs[0]
                    first_part = f"For {h0} {v0}" if h0 else f"For {v0}"
                    rest_clauses = [f"{h} is {v}" if h else f"is {v}" for h, v in pairs[1:]]
                    sentence = f"{first_part}, " + ", ".join(rest_clauses[:-1]) + f", and {rest_clauses[-1]}."

                row_sentences.append(sentence)

        return "\n".join(row_sentences)

    @classmethod
    def linearize_markdown_tables(cls, text: str) -> str:
        """
        Scans text for contiguous Markdown pipe tables (| Col1 | Col2 | ... with separator |---|---|),
        linearizes their rows into declarative sentences, and replaces table blocks inline.
        """
        if not text or "|" not in text:
            return text

        lines = text.split("\n")
        output_lines: List[str] = []
        i = 0
        separator_pattern = re.compile(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$')

        def parse_pipe_row(line_str: str) -> List[str]:
            stripped = line_str.strip()
            if stripped.startswith("|"):
                stripped = stripped[1:]
            if stripped.endswith("|"):
                stripped = stripped[:-1]

            # Mask mathematical cardinality/norm expressions e.g. |G|, |N|, |V_D|, |w|
            math_pipes: List[str] = []
            def mask_math(m):
                idx = len(math_pipes)
                math_pipes.append(m.group(0))
                return f"__MATH_CARD_{idx}__"

            masked = re.sub(r'\|([A-Za-z0-9_\\^]+)\|', mask_math, stripped)

            cells = []
            for c in masked.split("|"):
                cleaned_c = re.sub(r'\s+', ' ', c).strip()
                for idx, orig in enumerate(math_pipes):
                    cleaned_c = cleaned_c.replace(f"__MATH_CARD_{idx}__", orig)
                cells.append(cleaned_c)
            return cells

        while i < len(lines):
            line = lines[i].strip()
            # Check if current line contains pipes and next line is a markdown separator row
            if "|" in line and i + 1 < len(lines) and separator_pattern.match(lines[i + 1].strip()):
                raw_headers = parse_pipe_row(line)
                i += 2  # Skip header and separator rows

                table_rows: List[List[str]] = []
                while i < len(lines) and "|" in lines[i] and not separator_pattern.match(lines[i].strip()):
                    r = parse_pipe_row(lines[i])
                    # Pad or trim row to match header length
                    if len(r) < len(raw_headers):
                        r.extend([""] * (len(raw_headers) - len(r)))
                    table_rows.append(r)
                    i += 1

                linearized = cls.linearize_table(raw_headers, table_rows)
                if linearized:
                    output_lines.append(linearized)
            else:
                output_lines.append(lines[i])
                i += 1

        return "\n".join(output_lines)

    @classmethod
    def render_blocks_to_text(cls, blocks: List[Dict[str, Any]]) -> str:
        """
        Renders ordered document blocks into unified linearized text:
        - For 'text' blocks: emits block['text'].
        - For 'table' blocks: emits cls.linearize_table(block['headers'], block['rows']).
        """
        elements: List[str] = []
        for b in blocks:
            b_type = b.get("type")
            if b_type == "text":
                txt = (b.get("text") or "").strip()
                if txt:
                    elements.append(txt)
            elif b_type == "table":
                headers = b.get("headers", [])
                rows = b.get("rows", [])
                linearized = cls.linearize_table(headers, rows)
                if linearized:
                    elements.append(linearized)
        return "\n\n".join(elements)

    @classmethod
    def extract_linearized_text(cls, filename: str, file_bytes: bytes) -> str:
        """
        Extracts ordered document blocks with multi-page table stitching and wrapped row
        merging, and renders them into declarative linearized text.
        """
        blocks = cls.extract_document_blocks(filename, file_bytes)
        return cls.render_blocks_to_text(blocks)

    @classmethod
    def extract_docx_with_tables(cls, file_input: Any) -> Dict[str, Any]:
        """
        Extracts paragraphs and tables from a DOCX file in exact document reading order.
        Linearizes tables into declarative sentences inline.
        """
        if not HAS_DOCX:
            raise RuntimeError("python-docx is not installed in the environment.")
        return cls.extract_from_file(file_input, filename="document.docx")

    @classmethod
    def extract_pdf_with_tables(cls, file_input: Any) -> Dict[str, Any]:
        """
        Extracts text and tables from a PDF document in exact vertical reading order.
        Linearizes tables into declarative sentences inline with multi-page table stitching.
        """
        return cls.extract_from_file(file_input, filename="document.pdf")


    @classmethod
    def _parse_text_blocks(cls, raw_text: str) -> List[Dict[str, Any]]:
        """
        Splits text containing Markdown pipe tables into ordered blocks of:
          - {"type": "text", "text": "..."}
          - {"type": "table", "headers": [...], "rows": [[...], ...]}
        """
        if not raw_text or not raw_text.strip():
            return []

        lines = raw_text.splitlines()
        blocks: List[Dict[str, Any]] = []
        current_text_lines: List[str] = []
        separator_pattern = re.compile(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$')

        def parse_pipe_row(line_str: str) -> List[str]:
            stripped = line_str.strip()
            if stripped.startswith("|"):
                stripped = stripped[1:]
            if stripped.endswith("|"):
                stripped = stripped[:-1]

            # Mask mathematical cardinality/norm expressions e.g. |G|, |N|, |V_D|, |w|
            math_pipes: List[str] = []
            def mask_math(m):
                idx = len(math_pipes)
                math_pipes.append(m.group(0))
                return f"__MATH_CARD_{idx}__"

            masked = re.sub(r'\|([A-Za-z0-9_\\^]+)\|', mask_math, stripped)

            cells = []
            for c in masked.split("|"):
                cleaned_c = re.sub(r'\s+', ' ', c).strip()
                for idx, orig in enumerate(math_pipes):
                    cleaned_c = cleaned_c.replace(f"__MATH_CARD_{idx}__", orig)
                cells.append(cleaned_c)
            return cells

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            has_sep = (i + 1 < len(lines) and bool(separator_pattern.match(lines[i + 1].strip())))
            is_pipe_row = (stripped_line.startswith("|") and stripped_line.endswith("|") and stripped_line.count("|") >= 2)
            has_next_pipe_row = (i + 1 < len(lines) and lines[i+1].strip().startswith("|") and lines[i+1].strip().endswith("|") and lines[i+1].strip().count("|") >= 2)

            if "|" in stripped_line and (has_sep or (is_pipe_row and has_next_pipe_row)):
                # Flush accumulated text
                if current_text_lines:
                    text_content = "\n".join(current_text_lines).strip()
                    if text_content:
                        blocks.append({"type": "text", "text": text_content})
                    current_text_lines = []

                raw_headers = parse_pipe_row(stripped_line)
                i += 2 if has_sep else 1

                table_rows: List[List[str]] = []
                while i < len(lines):
                    s_curr = lines[i].strip()
                    if separator_pattern.match(s_curr):
                        i += 1
                        continue
                    if not (s_curr.startswith("|") and s_curr.endswith("|") and s_curr.count("|") >= 2):
                        if not ("|" in s_curr and len(parse_pipe_row(s_curr)) >= 2):
                            break
                    r = parse_pipe_row(s_curr)
                    if len(r) < len(raw_headers):
                        r.extend([""] * (len(raw_headers) - len(r)))
                    table_rows.append(r)
                    i += 1

                blocks.append({
                    "type": "table",
                    "headers": raw_headers,
                    "rows": table_rows
                })
            else:
                current_text_lines.append(line)
                i += 1

        if current_text_lines:
            text_content = "\n".join(current_text_lines).strip()
            if text_content:
                blocks.append({"type": "text", "text": text_content})

        return blocks

    @classmethod
    def extract_document_blocks(
        cls,
        filename: str,
        file_bytes: bytes
    ) -> List[Dict[str, Any]]:
        """
        Extracts ordered document blocks (paragraphs and raw tables) in exact reading order.
        Returns a list of dicts:
          - {"type": "text", "text": "..."}
          - {"type": "table", "headers": [...], "rows": [[...], ...]}
        """
        _, ext = os.path.splitext(filename or "")
        ext = ext.lower()

        if ext == ".docx":
            if not HAS_DOCX:
                raise RuntimeError("python-docx is not installed in the environment.")
            doc = DocxDocument(io.BytesIO(file_bytes))
            blocks: List[Dict[str, Any]] = []
            separator_pattern = re.compile(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$')

            for child in doc.element.body:
                if isinstance(child, CT_P):
                    p = Paragraph(child, doc)
                    p_text = p.text.strip()
                    if p_text:
                        if "|" in p_text and separator_pattern.search(p_text):
                            blocks.extend(cls._parse_text_blocks(p_text))
                        else:
                            blocks.append({"type": "text", "text": p_text})
                elif isinstance(child, CT_Tbl):
                    tbl = Table(child, doc)
                    raw_rows: List[List[str]] = []
                    for row in tbl.rows:
                        row_cells = [cls.clean_cell_text(cell.text) for cell in row.cells]
                        raw_rows.append(row_cells)
                    if raw_rows:
                        headers = raw_rows[0]
                        data_rows = cls._merge_wrapped_rows(headers, raw_rows[1:])
                        blocks.append({
                            "type": "table",
                            "headers": headers,
                            "rows": data_rows
                        })
            return cls._stitch_continuation_tables(blocks)

        elif ext == ".pdf":
            blocks: List[Dict[str, Any]] = []
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page_idx, page in enumerate(pdf.pages):
                            try:
                                settings_tables = page.find_tables(table_settings=cls.PDF_TABLE_SETTINGS)
                            except Exception:
                                settings_tables = []
                            try:
                                default_tables = page.find_tables()
                            except Exception:
                                default_tables = []

                            if len(settings_tables) >= len(default_tables) and len(settings_tables) > 0:
                                tables = settings_tables
                                active_settings = cls.PDF_TABLE_SETTINGS
                            else:
                                tables = default_tables
                                active_settings = None

                            if not tables:
                                page_text = page.extract_text() or ""
                                page_text = cls.normalize_pdf_text(page_text)
                                if page_text.strip():
                                    blocks.extend([{**b, "page": page_idx + 1} for b in cls._parse_text_blocks(page_text.strip())])
                                continue

                            sorted_tables = sorted(tables, key=lambda t: t.bbox[1])
                            last_bottom = 0.0

                            for t in sorted_tables:
                                t_top, t_bottom = t.bbox[1], t.bbox[3]
                                if t_top > last_bottom:
                                    try:
                                        crop = page.crop((0, last_bottom, page.width, t_top))
                                        txt = crop.extract_text() or ""
                                        txt = cls.normalize_pdf_text(txt)
                                        if txt.strip():
                                            blocks.extend([{**b, "page": page_idx + 1} for b in cls._parse_text_blocks(txt.strip())])
                                    except Exception:
                                        pass

                                t_data = None
                                if active_settings:
                                    try:
                                        t_data = t.extract(table_settings=active_settings)
                                    except Exception:
                                        t_data = None
                                if not t_data:
                                    t_data = t.extract()

                                if t_data and len(t_data) >= 1:
                                    headers = [cls.clean_cell_text(c) for c in t_data[0]]
                                    data_rows = [
                                        [cls.clean_cell_text(c) for c in row]
                                        for row in t_data[1:]
                                    ]
                                    data_rows = cls._merge_wrapped_rows(headers, data_rows)
                                    blocks.append({
                                        "type": "table",
                                        "headers": headers,
                                        "rows": data_rows,
                                        "page": page_idx + 1
                                    })
                                last_bottom = max(last_bottom, t_bottom)

                            if last_bottom < page.height:
                                try:
                                    crop = page.crop((0, last_bottom, page.width, page.height))
                                    txt = crop.extract_text() or ""
                                    txt = cls.normalize_pdf_text(txt)
                                    if txt.strip():
                                        blocks.extend([{**b, "page": page_idx + 1} for b in cls._parse_text_blocks(txt.strip())])
                                except Exception:
                                    pass
                    return cls._stitch_continuation_tables(blocks)
                except Exception as e:
                    logger.warning(f"[TableLinearizer] pdfplumber block extraction failed, falling back to PyMuPDF: {e}")

            if HAS_FITZ:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page_idx, page in enumerate(doc):
                    txt = page.get_text("text").strip()
                    txt = cls.normalize_pdf_text(txt)
                    if txt:
                        blocks.extend([{**b, "page": page_idx + 1} for b in cls._parse_text_blocks(txt)])
                return cls._stitch_continuation_tables(blocks)

            raise RuntimeError("No PDF extraction library available.")

        else:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            return cls._stitch_continuation_tables(cls._parse_text_blocks(raw_text))

    @classmethod
    def extract_from_file(
        cls,
        file_input: Any,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified document extractor for .docx, .pdf, .txt, .md, and .csv files.
        Routes all formats through extract_document_blocks with multi-page table stitching,
        wrapped row merging, and propositional table linearization.
        """
        name = "document"
        if filename:
            name = filename
        elif isinstance(file_input, str):
            name = file_input

        _, ext = os.path.splitext(name)
        ext = ext.lower()

        if isinstance(file_input, (bytes, bytearray)):
            file_bytes = bytes(file_input)
        elif isinstance(file_input, str):
            with open(file_input, "rb") as f:
                file_bytes = f.read()
        elif hasattr(file_input, "read"):
            file_bytes = file_input.read()
        else:
            raise ValueError(f"Unsupported file_input type: {type(file_input)}")

        if ext in [".pdf", ".docx", ".txt", ".md", ".csv", ""]:
            blocks = cls.extract_document_blocks(name, file_bytes)
            full_text = cls.render_blocks_to_text(blocks)

            raw_tables = [
                {"headers": b["headers"], "rows": b["rows"]}
                for b in blocks if b.get("type") == "table"
            ]

            if ext == ".pdf":
                # Build per-page text list
                page_map: Dict[int, List[Dict[str, Any]]] = {}
                for b in blocks:
                    p_num = b.get("page", 1)
                    page_map.setdefault(p_num, []).append(b)

                total_pages = max(page_map.keys()) if page_map else 1
                pages = [
                    {"page_number": p, "text": cls.render_blocks_to_text(p_blocks)}
                    for p, p_blocks in sorted(page_map.items())
                ]
            else:
                total_pages = None
                pages = [{"page_number": None, "text": full_text}]

            return {
                "total_pages": total_pages,
                "pages": pages,
                "full_text": full_text,
                "tables": raw_tables
            }
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Only PDF, DOCX, TXT, and MD are supported.")
