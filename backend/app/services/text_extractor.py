import os
from typing import Dict, Any
from backend.app.services.table_linearizer import TableLinearizer

class TextExtractor:
    @staticmethod
    def extract_from_file(file_path: str) -> Dict[str, Any]:
        """
        Extracts text from a PDF, DOCX, or plain text file with unified structured
        table extraction and propositional linearization.
        Returns a dict with:
          - 'total_pages': int or None
          - 'pages': list of dicts with 'page_number' and 'text'
          - 'full_text': full concatenated text
          - 'tables': list of raw extracted table dictionaries (optional)
        """
        return TableLinearizer.extract_from_file(file_path)
