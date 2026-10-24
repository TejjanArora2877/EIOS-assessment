import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import DocumentResponse, DocumentListResponse
from backend.app.services.text_extractor import TextExtractor
from backend.app.services.preprocessor import Preprocessor
from backend.app.services.models_loader import ModelsLoader
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service
from backend.app.services.retriever import HybridRetriever

router = APIRouter(prefix="/documents", tags=["Reference Materials"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    doc_type: str = Form(...)
):
    if not title or not title.strip():
        raise HTTPException(status_code=400, detail="Document title cannot be empty.")
    if not doc_type or not doc_type.strip():
        raise HTTPException(status_code=400, detail="Document type cannot be empty.")

    clean_filename = os.path.basename(file.filename or "uploaded_doc")
    ext = os.path.splitext(clean_filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Only PDF, DOCX, and TXT/MD files are supported."
        )

    doc_id = str(uuid.uuid4())
    save_filename = f"{doc_id}_{clean_filename}"
    file_path = str(settings.UPLOADS_DIR / save_filename)
    now_iso = datetime.now(timezone.utc).isoformat()

    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    # Save to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Initial SQLite insert with 'processing' status
    with get_db() as conn:
        conn.execute("""
            INSERT INTO documents (
                id, title, doc_type, filename, file_path, file_size,
                total_pages, total_chunks, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, title.strip(), doc_type.strip(), clean_filename, file_path, file_size, None, 0, "processing", now_iso))

    # Process and index document
    try:
        # 1. Text extraction
        extraction = TextExtractor.extract_from_file(file_path)
        total_pages = extraction.get("total_pages")
        pages = extraction.get("pages", [])

        # 2. Preprocessing & ELC chunking
        chunks = Preprocessor.elc_chunk_document(pages)
        if not chunks:
            raise ValueError("No readable text could be extracted or chunked from the uploaded document.")

        # 3. Store chunks in SQLite
        with get_db() as conn:
            for c in chunks:
                chunk_id = f"{doc_id}_{c['chunk_index']}"
                conn.execute("""
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, page_number, text, char_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, doc_id, c["chunk_index"], c.get("page_number"), c["text"], c["char_count"]))

        # 4. Dense Embeddings & ChromaDB storage
        chunk_texts = [c["text"] for c in chunks]
        embeddings = ModelsLoader.encode_texts(chunk_texts).tolist()
        VectorStore.add_chunks(
            document_id=doc_id,
            document_title=title.strip(),
            chunks=chunks,
            embeddings=embeddings
        )

        # 5. BM25 Lexical Indexing
        BM25Service.add_document(
            document_id=doc_id,
            document_title=title.strip(),
            chunks=chunks
        )

        indexed_iso = datetime.now(timezone.utc).isoformat()

        # 6. Update document status to 'ready'
        HybridRetriever.clear_cache()
        with get_db() as conn:
            conn.execute("""
                UPDATE documents
                SET status = 'ready',
                    total_pages = ?,
                    total_chunks = ?,
                    indexed_at = ?
                WHERE id = ?
            """, (total_pages, len(chunks), indexed_iso, doc_id))

        return DocumentResponse(
            id=doc_id,
            title=title.strip(),
            doc_type=doc_type.strip(),
            filename=clean_filename,
            file_size=file_size,
            total_pages=total_pages,
            total_chunks=len(chunks),
            status="ready",
            error_message=None,
            created_at=now_iso,
            indexed_at=indexed_iso
        )

    except Exception as e:
        error_msg = str(e)
        with get_db() as conn:
            conn.execute("""
                UPDATE documents
                SET status = 'failed',
                    error_message = ?
                WHERE id = ?
            """, (error_msg, doc_id))

        # Rollback chunks from vector store and BM25 if partially added
        VectorStore.delete_document(doc_id)
        BM25Service.delete_document(doc_id)

        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {error_msg}"
        )

@router.get("", response_model=DocumentListResponse)
def list_documents():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        docs = [
            DocumentResponse(
                id=r["id"],
                title=r["title"],
                doc_type=r["doc_type"],
                filename=r["filename"],
                file_size=r["file_size"],
                total_pages=r["total_pages"],
                total_chunks=r["total_chunks"],
                status=r["status"],
                error_message=r["error_message"],
                created_at=r["created_at"],
                indexed_at=r["indexed_at"]
            )
            for r in rows
        ]
        return DocumentListResponse(documents=docs, total=len(docs))

@router.get("/{doc_id}/status", response_model=DocumentResponse)
def get_document_status(doc_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        r = cursor.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Document not found.")
        return DocumentResponse(
            id=r["id"],
            title=r["title"],
            doc_type=r["doc_type"],
            filename=r["filename"],
            file_size=r["file_size"],
            total_pages=r["total_pages"],
            total_chunks=r["total_chunks"],
            status=r["status"],
            error_message=r["error_message"],
            created_at=r["created_at"],
            indexed_at=r["indexed_at"]
        )

@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(doc_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        doc = cursor.fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")

        file_path = doc["file_path"]

        # 1. Delete from SQLite (cascades to document_chunks)
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # 2. Delete from ChromaDB
    VectorStore.delete_document(doc_id)

    # 3. Delete from BM25 index
    BM25Service.delete_document(doc_id)

    # 4. Invalidate retrieval cache
    HybridRetriever.clear_cache()

    # 5. Remove physical file
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    return {"status": "success", "message": f"Document '{doc['title']}' and all its indexed chunks have been deleted."}
