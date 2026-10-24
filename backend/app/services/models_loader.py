import time
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.app.config import settings

logger = logging.getLogger(__name__)

class ModelsLoader:
    _embedding_model: Optional[SentenceTransformer] = None
    _reranker_model: Optional[CrossEncoder] = None
    _nli_model: Optional[CrossEncoder] = None

    _embedding_cache: Dict[str, np.ndarray] = {}
    _nli_cache: Dict[Tuple[str, str], Dict[str, float]] = {}

    @classmethod
    def clear_caches(cls):
        """Clears in-memory embedding and NLI inference caches."""
        cls._embedding_cache.clear()
        cls._nli_cache.clear()

    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        if cls._embedding_model is None:
            t0 = time.perf_counter()
            logger.info(f"[Model Init] Loading SentenceTransformer: {settings.EMBEDDING_MODEL_NAME}")
            cls._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            load_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Model Init] SentenceTransformer '{settings.EMBEDDING_MODEL_NAME}' loaded in {load_ms:.1f}ms")
        return cls._embedding_model

    @classmethod
    def get_reranker_model(cls) -> CrossEncoder:
        if cls._reranker_model is None:
            t0 = time.perf_counter()
            logger.info(f"[Model Init] Loading CrossEncoder Reranker: {settings.RERANKER_MODEL_NAME}")
            cls._reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
            load_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Model Init] CrossEncoder Reranker '{settings.RERANKER_MODEL_NAME}' loaded in {load_ms:.1f}ms")
        return cls._reranker_model

    @classmethod
    def get_nli_model(cls) -> CrossEncoder:
        if cls._nli_model is None:
            t0 = time.perf_counter()
            logger.info(f"[Model Init] Loading CrossEncoder NLI: {settings.NLI_MODEL_NAME}")
            cls._nli_model = CrossEncoder(settings.NLI_MODEL_NAME)
            load_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[Model Init] CrossEncoder NLI '{settings.NLI_MODEL_NAME}' loaded in {load_ms:.1f}ms")
        return cls._nli_model

    @classmethod
    def encode_texts(cls, texts: List[str], batch_size: int = 128) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        uncached = [t for t in texts if t not in cls._embedding_cache]
        if uncached:
            model = cls.get_embedding_model()
            unique_uncached = list(dict.fromkeys(uncached))
            new_embs = model.encode(unique_uncached, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
            for t, emb in zip(unique_uncached, new_embs):
                cls._embedding_cache[t] = emb

        return np.array([cls._embedding_cache[t] for t in texts])

    @classmethod
    def rerank_passages(cls, query: str, passages: List[str]) -> List[float]:
        """
        Computes relevance scores for (query, passage) pairs.
        Higher score = higher relevance.
        """
        if not passages:
            return []
        reranker = cls.get_reranker_model()
        pairs = [(query, p) for p in passages]
        scores = reranker.predict(pairs)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(s) for s in scores]

    @classmethod
    def predict_nli(cls, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Evaluates NLI for (premise, hypothesis) pairs with verified softmax.
        Label mapping from model config:
          0: contradiction
          1: entailment
          2: neutral
        """
        if not pairs:
            return []

        uncached = [p for p in pairs if p not in cls._nli_cache]
        if uncached:
            nli = cls.get_nli_model()
            unique_uncached = list(dict.fromkeys(uncached))
            probs = nli.predict(unique_uncached, apply_softmax=True, batch_size=32)
            if hasattr(probs, "ndim") and probs.ndim == 1:
                probs = [probs]
            for p, prob in zip(unique_uncached, probs):
                cls._nli_cache[p] = {
                    "contradiction": float(prob[0]),
                    "entailment": float(prob[1]),
                    "neutral": float(prob[2])
                }

        return [cls._nli_cache[p] for p in pairs]
