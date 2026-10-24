import os
import json
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from backend.app.config import settings

class BM25Service:
    _index: Optional[BM25Okapi] = None
    _corpus_records: List[Dict[str, Any]] = []  # [{id, document_id, document_title, page_number, chunk_index, text, tokens}]
    _initialized = False

    @classmethod
    def _get_corpus_path(cls) -> str:
        return str(settings.BM25_PERSIST_DIR / "bm25_corpus.json")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Simple, deterministic tokenization for lexical retrieval."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text)
        return tokens

    @classmethod
    def initialize(cls):
        """Explicitly initializes or rebuilds BM25 index on startup."""
        cls._initialize_if_needed()

    @classmethod
    def reset(cls):
        """Resets in-memory state (used for clean test isolation)."""
        cls._index = None
        cls._corpus_records = []
        cls._initialized = False

    @classmethod
    def rebuild_from_db(cls):
        """
        Reconstructs the BM25 index and corpus file directly from persisted
        SQLite document chunks without re-running embedding or re-extracting PDFs.
        """
        from backend.app.database import get_db
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.id, c.document_id, d.title as document_title,
                           c.page_number, c.chunk_index, c.text
                    FROM document_chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY c.document_id, c.chunk_index
                """)
                rows = cursor.fetchall()
        except Exception:
            rows = []

        records = []
        for r in rows:
            tokens = cls.tokenize(r["text"])
            records.append({
                "id": r["id"],
                "document_id": r["document_id"],
                "document_title": r["document_title"],
                "page_number": r["page_number"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "tokens": tokens
            })

        cls._corpus_records = records
        if records:
            tokenized_corpus = [r["tokens"] for r in records]
            cls._index = BM25Okapi(tokenized_corpus)
            cls._save_to_disk()
        else:
            cls._index = None
        cls._initialized = True

    @classmethod
    def _initialize_if_needed(cls):
        if cls._initialized:
            return
        corpus_path = cls._get_corpus_path()
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    cls._corpus_records = json.load(f)
                if cls._corpus_records:
                    tokenized_corpus = [r["tokens"] for r in cls._corpus_records]
                    cls._index = BM25Okapi(tokenized_corpus)
            except Exception as e:
                cls._corpus_records = []
                cls._index = None

        # If corpus file was absent or empty, attempt reconstruction from persisted SQLite chunks
        if not cls._corpus_records:
            cls.rebuild_from_db()

        cls._initialized = True

    @classmethod
    def _save_to_disk(cls):
        corpus_path = cls._get_corpus_path()
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(cls._corpus_records, f, ensure_ascii=False)

    @classmethod
    def add_document(
        cls,
        document_id: str,
        document_title: str,
        chunks: List[Dict[str, Any]]
    ):
        cls._initialize_if_needed()
        # Remove any existing records for this document
        cls._corpus_records = [r for r in cls._corpus_records if r["document_id"] != document_id]

        for c in chunks:
            tokens = cls.tokenize(c["text"])
            cls._corpus_records.append({
                "id": f"{document_id}_{c['chunk_index']}",
                "document_id": document_id,
                "document_title": document_title,
                "page_number": c.get("page_number"),
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "tokens": tokens
            })

        if cls._corpus_records:
            tokenized_corpus = [r["tokens"] for r in cls._corpus_records]
            cls._index = BM25Okapi(tokenized_corpus)
        else:
            cls._index = None

        cls._save_to_disk()

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 15,
        doc_scope: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        cls._initialize_if_needed()
        if cls._index is None or not cls._corpus_records:
            return []

        query_tokens = cls.tokenize(query)
        if not query_tokens:
            return []

        raw_scores = cls._index.get_scores(query_tokens)
        scored_records = []

        for i, score in enumerate(raw_scores):
            if score <= 0:
                continue
            rec = cls._corpus_records[i]
            # Check doc_scope filter
            if doc_scope and len(doc_scope) > 0 and "ALL" not in doc_scope:
                if rec["document_id"] not in doc_scope:
                    continue

            scored_records.append({
                "id": rec["id"],
                "document_id": rec["document_id"],
                "document_title": rec["document_title"],
                "page_number": rec["page_number"],
                "chunk_index": rec["chunk_index"],
                "text": rec["text"],
                "bm25_score": float(score)
            })

        scored_records.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_records[:top_k]

    @classmethod
    def delete_document(cls, document_id: str):
        cls._initialize_if_needed()
        cls._corpus_records = [r for r in cls._corpus_records if r["document_id"] != document_id]
        if cls._corpus_records:
            tokenized_corpus = [r["tokens"] for r in cls._corpus_records]
            cls._index = BM25Okapi(tokenized_corpus)
        else:
            cls._index = None
        cls._save_to_disk()
