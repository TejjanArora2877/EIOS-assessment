from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.database import init_db
from backend.app.services.storage_reconciler import StorageReconciler
from backend.app.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database on startup
    init_db()
    # Reconcile persistent storage across SQLite, ChromaDB, BM25, and Uploads
    StorageReconciler.reconcile()
    yield

app = FastAPI(
    title="EIOS RAG Grading MVP",
    description="Local, single-user educational grading system following the EIOS RAG Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.routers import health, documents, assessments, grading
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(assessments.router)
app.include_router(grading.router)

@app.get("/")
def root():
    return {
        "name": "EIOS RAG Grading System API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
