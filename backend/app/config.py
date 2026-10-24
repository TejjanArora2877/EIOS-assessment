import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False

    # Storage paths
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    SQLITE_DB_PATH: Path = BASE_DIR / "backend" / "data" / "eios.db"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "backend" / "data" / "chroma"
    BM25_PERSIST_DIR: Path = BASE_DIR / "backend" / "data" / "bm25"
    UPLOADS_DIR: Path = BASE_DIR / "backend" / "data" / "uploads"

    # AI Models
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-small"

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0

    # Feedback Generation Strategy: "hybrid" (deterministic for full/zero marks, LLM for partial/contradictions) or "llm" (all LLM)
    FEEDBACK_MODE: str = "hybrid"

    @field_validator("DATA_DIR", "SQLITE_DB_PATH", "CHROMA_PERSIST_DIR", "BM25_PERSIST_DIR", "UPLOADS_DIR", mode="after")
    @classmethod
    def enforce_project_rooted_paths(cls, v: Path) -> Path:
        """Ensures storage paths are always absolute and rooted at the project directory."""
        if not v.is_absolute():
            return (BASE_DIR / v).resolve()
        return v.resolve()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()

# Ensure directories exist
for path in [settings.DATA_DIR, settings.CHROMA_PERSIST_DIR, settings.BM25_PERSIST_DIR, settings.UPLOADS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
