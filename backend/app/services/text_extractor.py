import os
import re
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from docx import Document as DocxDocument

class TextExtractor:
    @staticmethod
    def extract_from_file(file_path: str) -> Dict[str, Any]:
        """
        Extracts text from a PDF, DOCX, or plain text file.
        Returns a dict with:
          - 'total_pages': int or None
          - 'pages': list of dicts with 'page_number' and 'text'
          - 'full_text': full concatenated text
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pdf":
            return TextExtractor._extract_pdf(file_path)
        elif ext == ".docx":
            return TextExtractor._extract_docx(file_path)
        elif ext in [".txt", ".md", ".csv"]:
            return TextExtractor._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Only PDF, DOCX, and TXT are supported.")

    @staticmethod
    def _extract_pdf(file_path: str) -> Dict[str, Any]:
        pages = []
        with fitz.open(file_path) as pdf:
            total_pages = len(pdf)
            for page_num in range(total_pages):
                page = pdf[page_num]
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text
                    })

        full_text = "\n\n".join(p["text"] for p in pages)
        return {
            "total_pages": total_pages,
            "pages": pages,
            "full_text": full_text
        }

    @staticmethod
    def _extract_docx(file_path: str) -> Dict[str, Any]:
        doc = DocxDocument(file_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        # Word docs do not expose rigid pagination in raw XML
        return {
            "total_pages": None,
            "pages": [{"page_number": None, "text": full_text}],
            "full_text": full_text
        }

    @staticmethod
    def _extract_txt(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            full_text = f.read().strip()
        return {
            "total_pages": None,
            "pages": [{"page_number": None, "text": full_text}],
            "full_text": full_text
        }
