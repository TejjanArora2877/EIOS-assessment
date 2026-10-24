import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service

logger = logging.getLogger(__name__)

class StorageReconciler:
    @classmethod
    def reconcile(cls):
        """
        Durable persistence contract reconciler executed on application startup.
        Ensures consistency across:
          1. SQLite metadata and chunk tables (eios.db)
          2. Persistent ChromaDB collections (chroma/)
          3. Lexical BM25 index (bm25/)
          4. Physical upload files (uploads/)
        """
        logger.info("Running storage reconciler...")

        # 1. Inspect ChromaDB collection
        try:
            collection = VectorStore.get_collection()
            chroma_count = collection.count()
        except Exception as e:
            logger.warning(f"Could not connect to ChromaDB collection on startup: {e}")
            chroma_count = 0

        # 2. Reconcile ChromaDB into SQLite if documents table was cleared or desynchronized
        if chroma_count > 0:
            try:
                # Fetch all metadata from Chroma collection
                chroma_data = collection.get(limit=chroma_count + 100, include=["metadatas", "documents"])
                metadatas = chroma_data.get("metadatas", [])
                documents_text = chroma_data.get("documents", [])
                ids = chroma_data.get("ids", [])

                # Group by document_id
                doc_groups: Dict[str, Dict[str, Any]] = {}
                for idx, meta in enumerate(metadatas):
                    if not meta:
                        continue
                    doc_id = meta.get("document_id")
                    if not doc_id:
                        continue
                    if doc_id not in doc_groups:
                        doc_groups[doc_id] = {
                            "document_id": doc_id,
                            "title": meta.get("document_title", "Untitled Document"),
                            "chunks": [],
                            "max_page": -1
                        }

                    page = meta.get("page_number", -1)
                    if page is not None and page > doc_groups[doc_id]["max_page"]:
                        doc_groups[doc_id]["max_page"] = page

                    doc_groups[doc_id]["chunks"].append({
                        "id": ids[idx],
                        "chunk_index": meta.get("chunk_index", 0),
                        "page_number": page if page != -1 else None,
                        "text": documents_text[idx] if idx < len(documents_text) else ""
                    })

                with get_db() as conn:
                    cursor = conn.cursor()
                    for doc_id, doc_info in doc_groups.items():
                        cursor.execute("SELECT id FROM documents WHERE id = ?", (doc_id,))
                        row = cursor.fetchone()

                        # Find physical upload file if present
                        matched_file = None
                        matched_filename = f"{doc_info['title']}.pdf"
                        file_size = 0
                        if settings.UPLOADS_DIR.exists():
                            for f in settings.UPLOADS_DIR.iterdir():
                                if f.name.startswith(doc_id):
                                    matched_file = str(f.resolve())
                                    # Strip doc_id prefix for human-readable original filename
                                    matched_filename = f.name[len(doc_id):].lstrip("_")
                                    file_size = f.stat().st_size
                                    break

                        now_iso = datetime.now(timezone.utc).isoformat()
                        total_pages = doc_info["max_page"] if doc_info["max_page"] > 0 else None
                        total_chunks = len(doc_info["chunks"])

                        if not row:
                            logger.info(f"Reconciling document '{doc_info['title']}' ({doc_id}) into SQLite documents table...")
                            cursor.execute("""
                                INSERT INTO documents (
                                    id, title, doc_type, filename, file_path, file_size,
                                    total_pages, total_chunks, status, created_at, indexed_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                            """, (
                                doc_id,
                                doc_info["title"],
                                "textbook",
                                matched_filename,
                                matched_file or "",
                                file_size,
                                total_pages,
                                total_chunks,
                                now_iso,
                                now_iso
                            ))

                        # Check if document_chunks has records
                        cursor.execute("SELECT count(*) FROM document_chunks WHERE document_id = ?", (doc_id,))
                        chunk_count_in_db = cursor.fetchone()[0]
                        if chunk_count_in_db < total_chunks:
                            logger.info(f"Reconciling {total_chunks} chunks into SQLite document_chunks table for doc {doc_id}...")
                            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))
                            for c in doc_info["chunks"]:
                                cursor.execute("""
                                    INSERT INTO document_chunks (
                                        id, document_id, chunk_index, page_number, text, char_count
                                    ) VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    c["id"],
                                    doc_id,
                                    c["chunk_index"],
                                    c["page_number"],
                                    c["text"],
                                    len(c["text"])
                                ))

            except Exception as e:
                logger.error(f"Error during ChromaDB to SQLite reconciliation: {e}", exc_info=True)

        # 3. Reconcile BM25 Index
        try:
            BM25Service.initialize()
        except Exception as e:
            logger.error(f"Error initializing BM25 during startup reconciliation: {e}", exc_info=True)

        logger.info("Storage reconciliation completed successfully.")
