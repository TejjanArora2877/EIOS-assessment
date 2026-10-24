import re
from typing import List, Dict, Any, Optional

class Preprocessor:
    # Patterns for PII detection
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    STUDENT_ID_PATTERN = re.compile(r'\b(?:ID|Roll\s*(?:No|Number)|Student\s*ID)[:\s]+[A-Za-z0-9_-]{5,15}\b', re.IGNORECASE)

    @classmethod
    def redact_pii(cls, text: str) -> str:
        """Removes or pseudonymizes sensitive personal information."""
        text = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        text = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        text = cls.SSN_PATTERN.sub("[REDACTED_SSN]", text)
        text = cls.STUDENT_ID_PATTERN.sub("[REDACTED_STUDENT_ID]", text)
        return text

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Cleans, normalizes, and de-noises text extracted from documents."""
        if not text:
            return ""

        # Replace non-breaking spaces and other special spaces
        text = text.replace('\xa0', ' ').replace('\u200b', '')

        # Fix hyphenation across line breaks: e.g. "clas- \nsification" -> "classification"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

        # Replace solitary linebreaks inside paragraphs with a single space
        # but preserve genuine paragraph breaks (2 or more newlines)
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for p in paragraphs:
            cleaned_p = re.sub(r'\s+', ' ', p).strip()
            if cleaned_p:
                cleaned_paragraphs.append(cleaned_p)

        return "\n\n".join(cleaned_paragraphs)

    @classmethod
    def elc_chunk_document(cls, pages: List[Dict[str, Any]], target_chunk_words: int = 150) -> List[Dict[str, Any]]:
        """
        ELC (Epistemic Logical Chunking):
        Extracts rule-based epistemic units that represent complete propositions.
        Preserves page boundaries and sentence integrity.
        """
        chunks = []
        chunk_idx = 0

        for page_data in pages:
            page_num = page_data.get("page_number")
            raw_text = page_data.get("text", "")

            clean_text = cls.normalize_text(cls.redact_pii(raw_text))
            if not clean_text:
                continue

            # Split into sentences using punctuation boundaries
            # Handles abbreviations and decimal numbers reasonably
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', clean_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

            if not sentences:
                continue

            current_chunk_sentences = []
            current_word_count = 0

            for sent in sentences:
                sent_words = len(sent.split())

                # If adding this sentence exceeds target size and we already have content, yield chunk
                if current_word_count + sent_words > target_chunk_words and current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "page_number": page_num,
                        "text": chunk_text,
                        "char_count": len(chunk_text),
                        "word_count": current_word_count
                    })
                    chunk_idx += 1
                    current_chunk_sentences = [sent]
                    current_word_count = sent_words
                else:
                    current_chunk_sentences.append(sent)
                    current_word_count += sent_words

            # Add any trailing sentences as final chunk for this page
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": page_num,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                    "word_count": current_word_count
                })
                chunk_idx += 1

        return chunks
