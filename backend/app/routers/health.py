import httpx
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("", response_model=HealthResponse)
async def check_health():
    # 1. Check Ollama readiness
    ollama_ready = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                tags_data = resp.json()
                models = [m.get("name") for m in tags_data.get("models", [])]
                # Check if target model or prefix matches
                for m in models:
                    if settings.OLLAMA_MODEL in m or m.startswith("llama3.2"):
                        ollama_ready = True
                        break
    except Exception:
        ollama_ready = False

    # 2. Check Database and local storage
    db_ok = False
    doc_count = 0
    assessment_count = 0
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM documents")
            doc_count = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM assessments")
            assessment_count = c.fetchone()[0]
            db_ok = True
    except Exception:
        db_ok = False

    storage_status = {
        "database_connected": db_ok,
        "sqlite_path": str(settings.SQLITE_DB_PATH),
        "total_documents": doc_count,
        "total_assessments": assessment_count,
        "chroma_dir_exists": settings.CHROMA_PERSIST_DIR.exists(),
        "bm25_dir_exists": settings.BM25_PERSIST_DIR.exists(),
        "uploads_dir_exists": settings.UPLOADS_DIR.exists(),
    }

    # Model status (names configured)
    models_status = {
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "nli_model": settings.NLI_MODEL_NAME,
    }

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        ollama_ready=ollama_ready,
        ollama_model=settings.OLLAMA_MODEL,
        models_status={k: True for k in models_status},
        storage_status=storage_status
    )
