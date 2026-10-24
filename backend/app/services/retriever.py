import time
import logging
from typing import List, Dict, Any, Optional
from backend.app.services.models_loader import ModelsLoader
from backend.app.services.vector_store import VectorStore
from backend.app.services.bm25_index import BM25Service
from backend.app.models import EvidenceCitation

logger = logging.getLogger(__name__)

class HybridRetriever:
    RRF_K = 60  # Standard RRF constant
    _cache: Dict[Any, List[EvidenceCitation]] = {}

    @classmethod
    def clear_cache(cls):
        """Clears the evidence retrieval cache."""
        cls._cache.clear()

    @classmethod
    def retrieve_evidence(
        cls,
        query: str,
        top_k: int = 4,
        doc_scope: Optional[List[str]] = None,
        candidate_pool_size: int = 15
    ) -> List[EvidenceCitation]:
        """
        Executes the full RAG Retrieval & Evidence Selection Pipeline:
          1. Dense semantic retrieval via ChromaDB
          2. Lexical retrieval via BM25
          3. Reciprocal Rank Fusion (RRF)
          4. Cross-Encoder reranking
          5. Return top-k evidence citations
        """
        if not query or not query.strip():
            return []

        t0 = time.perf_counter()
        cache_key = (
            query.strip(),
            top_k,
            tuple(sorted(doc_scope)) if doc_scope else None,
            candidate_pool_size
        )
        if cache_key in cls._cache:
            hit_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Retrieval] CACHE_HIT duration_ms={hit_ms:.2f} top_k={top_k} query='{query[:40]}...'")
            return cls._cache[cache_key]

        # 1. Dense Semantic Retrieval
        query_embedding = ModelsLoader.encode_texts([query])[0].tolist()
        dense_results = VectorStore.search(
            query_embedding=query_embedding,
            top_k=candidate_pool_size,
            doc_scope=doc_scope
        )

        # 2. Lexical Retrieval (BM25)
        lexical_results = BM25Service.search(
            query=query,
            top_k=candidate_pool_size,
            doc_scope=doc_scope
        )

        # If both empty, return empty list
        if not dense_results and not lexical_results:
            return []

        # 3. Reciprocal Rank Fusion (RRF)
        # Store candidate by unique chunk id
        candidate_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, item in enumerate(dense_results):
            cid = item["id"]
            candidate_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (cls.RRF_K + (rank + 1)))

        for rank, item in enumerate(lexical_results):
            cid = item["id"]
            if cid not in candidate_map:
                candidate_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (cls.RRF_K + (rank + 1)))

        # Sort fused candidates by RRF score
        fused_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        top_fused_candidates = [candidate_map[cid] for cid in fused_cids[:candidate_pool_size]]

        if not top_fused_candidates:
            return []

        # 4. Cross-Encoder Reranking
        passages = [c["text"] for c in top_fused_candidates]
        cross_scores = ModelsLoader.rerank_passages(query, passages)

        # Combine candidate metadata with cross-encoder scores
        scored_candidates = []
        for cand, score in zip(top_fused_candidates, cross_scores):
            scored_candidates.append({
                "candidate": cand,
                "score": score
            })

        # Sort by cross-encoder score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        # 5. Build Top-k Evidence Citations
        citations: List[EvidenceCitation] = []
        for item in scored_candidates[:top_k]:
            c = item["candidate"]
            # Generate a clean, informative snippet (first 280 chars or full sentence)
            text_snippet = c["text"]
            if len(text_snippet) > 300:
                text_snippet = text_snippet[:300].rsplit(' ', 1)[0] + "..."

            citations.append(EvidenceCitation(
                document_id=c["document_id"],
                document_title=c["document_title"],
                page_number=c.get("page_number"),
                snippet=text_snippet,
                similarity_score=round(float(item["score"]), 4)
            ))

        cls._cache[cache_key] = citations
        miss_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[Retrieval] CACHE_MISS duration_ms={miss_ms:.2f} top_k={top_k} citations={len(citations)} query='{query[:40]}...'")
        return citations
