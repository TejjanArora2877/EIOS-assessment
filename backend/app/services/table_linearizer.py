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
            return [re.sub(r'\s+', ' ', c).strip() for c in stripped.split("|")]

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
    def extract_docx_with_tables(cls, file_input: Any) -> Dict[str, Any]:
        """
        Extracts paragraphs and tables from a DOCX file in exact document reading order.
        Linearizes tables into declarative sentences inline.
        """
        if not HAS_DOCX:
            raise RuntimeError("python-docx is not installed in the environment.")

        if isinstance(file_input, (bytes, bytearray)):
            doc = DocxDocument(io.BytesIO(file_input))
        else:
            doc = DocxDocument(file_input)

        elements: List[str] = []
        raw_tables: List[Dict[str, Any]] = []

        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                p_text = p.text.strip()
                if p_text:
                    elements.append(cls.linearize_markdown_tables(p_text))
            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                raw_rows: List[List[str]] = []
                for row in tbl.rows:
                    row_cells = [re.sub(r'\s+', ' ', cell.text).strip() for cell in row.cells]
                    raw_rows.append(row_cells)

                if raw_rows:
                    headers = raw_rows[0]
                    data_rows = raw_rows[1:]
                    raw_tables.append({
                        "headers": headers,
                        "rows": data_rows
                    })
                    linearized = cls.linearize_table(headers, data_rows)
                    if linearized:
                        elements.append(linearized)

        full_text = "\n\n".join(e for e in elements if e)
        return {
            "total_pages": None,
            "pages": [{"page_number": None, "text": full_text}],
            "full_text": full_text,
            "tables": raw_tables
        }

    @classmethod
    def extract_pdf_with_tables(cls, file_input: Any) -> Dict[str, Any]:
        """
        Extracts text and tables from a PDF document.
        Uses pdfplumber to detect table bounding boxes and interleave linearized
        tables with surrounding page text in vertical reading order.
        Falls back to PyMuPDF if pdfplumber is unavailable or encounters parsing errors.
        """
        pages_data: List[Dict[str, Any]] = []
        raw_tables: List[Dict[str, Any]] = []
        total_pages = 0

        # Primary extraction: pdfplumber
        if HAS_PDFPLUMBER:
            try:
                pdf_source = io.BytesIO(file_input) if isinstance(file_input, (bytes, bytearray)) else file_input
                with pdfplumber.open(pdf_source) as pdf:
                    total_pages = len(pdf.pages)
                    for page_idx, page in enumerate(pdf.pages):
                        tables = page.find_tables()
                        if not tables:
                            page_text = page.extract_text() or ""
                            page_text = cls.linearize_markdown_tables(page_text.strip())
                            if page_text:
                                pages_data.append({
                                    "page_number": page_idx + 1,
                                    "text": page_text
                                })
                            continue

                        # Interleave text and tables by vertical y0 (bbox[1])
                        sorted_tables = sorted(tables, key=lambda t: t.bbox[1])
                        page_parts: List[str] = []
                        last_bottom = 0.0

                        for t in sorted_tables:
                            t_top = t.bbox[1]
                            t_bottom = t.bbox[3]

                            # Extract text slice above this table
                            if t_top > last_bottom:
                                try:
                                    slice_crop = page.crop((0, last_bottom, page.width, t_top))
                                    slice_text = slice_crop.extract_text() or ""
                                    if slice_text.strip():
                                        page_parts.append(cls.linearize_markdown_tables(slice_text.strip()))
                                except Exception:
                                    pass

                            # Extract and linearize table
                            t_data = t.extract()
                            if t_data and len(t_data) >= 1:
                                headers = [re.sub(r'\s+', ' ', (c or "")).strip() for c in t_data[0]]
                                data_rows = [
                                    [re.sub(r'\s+', ' ', (c or "")).strip() for c in row]
                                    for row in t_data[1:]
                                ]
                                raw_tables.append({
                                    "page": page_idx + 1,
                                    "headers": headers,
                                    "rows": data_rows
                                })
                                linearized = cls.linearize_table(headers, data_rows)
                                if linearized:
                                    page_parts.append(linearized)

                            last_bottom = max(last_bottom, t_bottom)

                        # Extract text slice below the last table
                        if last_bottom < page.height:
                            try:
                                bottom_crop = page.crop((0, last_bottom, page.width, page.height))
                                bottom_text = bottom_crop.extract_text() or ""
                                if bottom_text.strip():
                                    page_parts.append(cls.linearize_markdown_tables(bottom_text.strip()))
                            except Exception:
                                pass

                        combined_page_text = "\n\n".join(p for p in page_parts if p)
                        if combined_page_text:
                            pages_data.append({
                                "page_number": page_idx + 1,
                                "text": combined_page_text
                            })

                full_text = "\n\n".join(p["text"] for p in pages_data)
                return {
                    "total_pages": total_pages,
                    "pages": pages_data,
                    "full_text": full_text,
                    "tables": raw_tables
                }
            except Exception as e:
                logger.warning(f"[TableLinearizer] pdfplumber extraction failed, falling back to PyMuPDF: {e}")

        # Fallback extraction: PyMuPDF (fitz)
        if HAS_FITZ:
            doc = fitz.open(stream=file_input, filetype="pdf") if isinstance(file_input, (bytes, bytearray)) else fitz.open(file_input)
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_text_parts: List[str] = []

                # Check if PyMuPDF table finder is supported
                tab_finder = None
                try:
                    tab_finder = page.find_tables()
                except Exception:
                    tab_finder = None

                if tab_finder and hasattr(tab_finder, "tables") and tab_finder.tables:
                    for t in tab_finder.tables:
                        t_data = t.extract()
                        if t_data and len(t_data) >= 1:
                            headers = [re.sub(r'\s+', ' ', (c or "")).strip() for c in t_data[0]]
                            data_rows = [
                                [re.sub(r'\s+', ' ', (c or "")).strip() for c in row]
                                for row in t_data[1:]
                            ]
                            raw_tables.append({
                                "page": page_idx + 1,
                                "headers": headers,
                                "rows": data_rows
                            })
                            linearized = cls.linearize_table(headers, data_rows)
                            if linearized:
                                page_text_parts.append(linearized)

                raw_text = page.get_text("text").strip()
                if raw_text:
                    page_text_parts.insert(0, cls.linearize_markdown_tables(raw_text))

                comb = "\n\n".join(page_text_parts)
                if comb:
                    pages_data.append({
                        "page_number": page_idx + 1,
                        "text": comb
                    })

            full_text = "\n\n".join(p["text"] for p in pages_data)
            return {
                "total_pages": total_pages,
                "pages": pages_data,
                "full_text": full_text,
                "tables": raw_tables
            }

        raise RuntimeError("No PDF extraction library (pdfplumber or PyMuPDF) is available.")

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
            return [re.sub(r'\s+', ' ', c).strip() for c in stripped.split("|")]

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
                        row_cells = [re.sub(r'\s+', ' ', cell.text).strip() for cell in row.cells]
                        raw_rows.append(row_cells)
                    if raw_rows:
                        headers = raw_rows[0]
                        data_rows = raw_rows[1:]
                        blocks.append({
                            "type": "table",
                            "headers": headers,
                            "rows": data_rows
                        })
            return blocks

        elif ext == ".pdf":
            blocks: List[Dict[str, Any]] = []
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        for page in pdf.pages:
                            tables = page.find_tables()
                            if not tables:
                                page_text = page.extract_text() or ""
                                if page_text.strip():
                                    blocks.extend(cls._parse_text_blocks(page_text.strip()))
                                continue

                            sorted_tables = sorted(tables, key=lambda t: t.bbox[1])
                            last_bottom = 0.0

                            for t in sorted_tables:
                                t_top, t_bottom = t.bbox[1], t.bbox[3]
                                if t_top > last_bottom:
                                    try:
                                        crop = page.crop((0, last_bottom, page.width, t_top))
                                        txt = crop.extract_text() or ""
                                        if txt.strip():
                                            blocks.extend(cls._parse_text_blocks(txt.strip()))
                                    except Exception:
                                        pass

                                t_data = t.extract()
                                if t_data and len(t_data) >= 1:
                                    headers = [re.sub(r'\s+', ' ', (c or "")).strip() for c in t_data[0]]
                                    data_rows = [
                                        [re.sub(r'\s+', ' ', (c or "")).strip() for c in row]
                                        for row in t_data[1:]
                                    ]
                                    blocks.append({
                                        "type": "table",
                                        "headers": headers,
                                        "rows": data_rows
                                    })
                                last_bottom = max(last_bottom, t_bottom)

                            if last_bottom < page.height:
                                try:
                                    crop = page.crop((0, last_bottom, page.width, page.height))
                                    txt = crop.extract_text() or ""
                                    if txt.strip():
                                        blocks.extend(cls._parse_text_blocks(txt.strip()))
                                except Exception:
                                    pass
                    return blocks
                except Exception as e:
                    logger.warning(f"[TableLinearizer] pdfplumber block extraction failed, falling back to PyMuPDF: {e}")

            if HAS_FITZ:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    txt = page.get_text("text").strip()
                    if txt:
                        blocks.extend(cls._parse_text_blocks(txt))
                return blocks

            raise RuntimeError("No PDF extraction library available.")

        else:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            return cls._parse_text_blocks(raw_text)

    @classmethod
    def extract_from_file(
        cls,
        file_input: Any,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified document extractor for .docx, .pdf, .txt, .md, and .csv files.
        Accepts either a filesystem path (str) or raw bytes.
        """
        ext = ""
        if filename:
            _, ext = os.path.splitext(filename)
        elif isinstance(file_input, str):
            _, ext = os.path.splitext(file_input)

        ext = ext.lower()

        if ext == ".docx":
            return cls.extract_docx_with_tables(file_input)
        elif ext == ".pdf":
            return cls.extract_pdf_with_tables(file_input)
        elif ext in [".txt", ".md", ".csv", ""]:
            if isinstance(file_input, (bytes, bytearray)):
                raw_text = file_input.decode("utf-8", errors="replace")
            else:
                with open(file_input, "r", encoding="utf-8", errors="replace") as f:
                    raw_text = f.read()

            linearized_text = cls.linearize_markdown_tables(raw_text.strip())
            return {
                "total_pages": None,
                "pages": [{"page_number": None, "text": linearized_text}],
                "full_text": linearized_text,
                "tables": []
            }
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Only PDF, DOCX, TXT, and MD are supported.")
