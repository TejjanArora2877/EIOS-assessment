import chromadb
from typing import List, Dict, Any, Optional
from backend.app.config import settings

class VectorStore:
    _client = None
    _collection = None

    @classmethod
    def reset(cls):
        """Resets cached client and collection (used for test isolation and clean shutdown)."""
        cls._client = None
        cls._collection = None

    @classmethod
    def get_collection(cls):
        if cls._collection is None:
            cls._client = chromadb.PersistentClient(path=str(settings.CHROMA_PERSIST_DIR))
            cls._collection = cls._client.get_or_create_collection(
                name="eios_institutional_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
        return cls._collection

    @classmethod
    def add_chunks(
        cls,
        document_id: str,
        document_title: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ):
        """Adds embedded chunks with metadata to the persistent ChromaDB collection."""
        if not chunks:
            return

        collection = cls.get_collection()
        ids = [f"{document_id}_{c['chunk_index']}" for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "document_title": document_title,
                "page_number": c.get("page_number") if c.get("page_number") is not None else -1,
                "chunk_index": c["chunk_index"]
            }
            for c in chunks
        ]

        # Chroma batch upsert in safe chunks of 500
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

    @classmethod
    def search(
        cls,
        query_embedding: List[float],
        top_k: int = 15,
        doc_scope: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Dense semantic search across institutional documents."""
        collection = cls.get_collection()
        count = collection.count()
        if count == 0:
            return []

        where_filter = None
        if doc_scope and len(doc_scope) > 0 and "ALL" not in doc_scope:
            if len(doc_scope) == 1:
                where_filter = {"document_id": doc_scope[0]}
            else:
                where_filter = {"document_id": {"$in": doc_scope}}

        actual_k = min(top_k, count)
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
        }
        if where_filter:
            query_args["where"] = where_filter

        results = collection.query(**query_args)

        matches = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                page = meta.get("page_number")
                matches.append({
                    "id": results["ids"][0][i],
                    "document_id": meta.get("document_id"),
                    "document_title": meta.get("document_title"),
                    "page_number": page if page != -1 else None,
                    "chunk_index": meta.get("chunk_index"),
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                })
        return matches

    @classmethod
    def delete_document(cls, document_id: str):
        """Removes all indexed chunks associated with a deleted document."""
        collection = cls.get_collection()
        try:
            collection.delete(where={"document_id": document_id})
        except Exception:
            pass
