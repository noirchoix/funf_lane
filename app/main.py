from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.ingest import router as ingest_router
from routes.ingest_pdf import router as ingest_pdf_router
from routes.ingest_phytochemistry import router as ingest_phytochemistry_router
from routes.query import router as query_router
from routes.memory import router as memory_router
from routes.chat import router as chat_router
from routes.health import router as health_router
from routes.metrics import router as metrics_router
from routes.pro import router as pro_router
from routes.funf import router as funf_router

app = FastAPI(title="Fünf RAG / ChemRAG Pro Microservice", version="0.3.0")

# Development CORS for the Svelte/Vite frontend.
# Vite can move between 5173, 5174, 5178, etc.; the regex keeps local dev stable
# while still allowing API-key auth headers and multipart uploads.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(pro_router)
app.include_router(funf_router)
app.include_router(ingest_router)
app.include_router(ingest_pdf_router)
app.include_router(ingest_phytochemistry_router)
app.include_router(query_router)
app.include_router(memory_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"ok": True}
