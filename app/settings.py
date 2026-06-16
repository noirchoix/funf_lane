from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -------------------------
    # Service auth
    # -------------------------
    SERVICE_API_KEY: str = "8b7d3b2e6b1a4d2f9a3c1c7f0a2e9c44"

    # -------------------------
    # Provider keys
    # -------------------------
    VOYAGEAI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    DEEPSEEK_API_KEY: Optional[str] = None

    # -------------------------
    # Generation provider routing
    # -------------------------
    DEFAULT_GENERATION_PROVIDER: str = "gemini"
    FALLBACK_GENERATION_PROVIDER: str = "deepseek"
    LLM_PROVIDER_PREFER: str = "gemini"  # backwards-compatible alias used by older code

    GEMINI_MODEL: str = "gemini-2.0-flash"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # -------------------------
    # Embedding provider routing
    # -------------------------
    DEFAULT_EMBED_PROVIDER: str = "gemini"
    FALLBACK_EMBED_PROVIDER: str = "voyage"
    EMBED_PROVIDER_PREFER: str = "gemini"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"
    VOYAGE_EMBED_MODEL: str = "voyage-3"

    # -------------------------
    # Qdrant
    # -------------------------
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_TIMEOUT_S: float = 30.0

    # Legacy/general collection alias retained for backward compatibility
    QDRANT_COLLECTION: str = "chem_docs_v1"

    # ChemRAG v2 lane collections
    RAG_COLLECTION_PHYTO_CONTEXT: str = "rag_phytochemistry_context"
    RAG_COLLECTION_REACTION: str = "rag_reaction_orgchem"
    RAG_COLLECTION_QC: str = "rag_quality_control"
    RAG_COLLECTION_PHYSICAL_CHEM: str = "rag_physical_chem"
    RAG_COLLECTION_INTERNAL_NOTES: str = "rag_internal_notes"
    RAG_COLLECTION_GENERAL: str = "chem_docs_v1"
    RAG_COLLECTION_FUNF: str = "funf_rag_chunks"

    # -------------------------
    # Artifact registry / wheel-system context
    # -------------------------
    ARTIFACT_REGISTRY_PATH: Optional[str] = None

    # -------------------------
    # RAG defaults
    # -------------------------
    RAG_TOP_K: int = 8
    RAG_MAX_CONTEXT_CHARS: int = 12000
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_MB: int = 50

    RAG_QUERY_EXPANSION: bool = True
    RAG_EXPANSION_COUNT: int = 4
    RAG_RERANK: bool = True
    RAG_RERANK_TOP_N: int = 8
    ENABLE_QUERY_EXPANSION: bool = False  # legacy route-level gate


    # -------------------------
    # ChemRAG Pro retrieval/evaluation
    # -------------------------
    RAG_PRO_DEFAULT_MODE: str = "hybrid_rerank"  # vector | sparse | hybrid | hybrid_rerank
    RAG_PRO_FETCH_K: int = 30
    RAG_PRO_RRF_K: int = 60
    RAG_PRO_SPARSE_MAX_DOCS_PER_LANE: int = 5000
    RAG_PRO_SPARSE_MIN_TOKEN_LEN: int = 2
    RAG_PRO_ENABLE_CROSS_ENCODER: bool = False
    RAG_PRO_CROSS_ENCODER_MODEL: str = "BAAI/bge-reranker-base"
    RAG_PRO_CROSS_ENCODER_TOP_N: int = 12
    RAG_PRO_TRACE_MAX_CHARS: int = 600

    # CI evaluation gate thresholds
    RAG_EVAL_MIN_RECALL_AT_K: float = 0.70
    RAG_EVAL_MIN_MRR: float = 0.55
    RAG_EVAL_MIN_NDCG_AT_K: float = 0.55
    RAG_EVAL_MAX_NO_CITATION_RATE: float = 0.15

    # -------------------------
    # Memory
    # -------------------------
    MEMORY_DB_PATH: str = "data/memory.db"
    MEMORY_COLLECTION: str = "chem_memory_v1"
    MEMORY_TOP_K: int = 8
    MEMORY_MIN_IMPORTANCE: int = 3
    MEMORY_MAX_EVENT_TEXT_CHARS: int = 4000
    MEMORY_ENABLED: bool = True

    # -------------------------
    # External compute layer
    # -------------------------
    COMPUTE_BASE_URL: Optional[str] = None
    COMPUTE_API_KEY: Optional[str] = None
    COMPUTE_TIMEOUT_S: float = 30.0
    COMPUTE_ENDPOINT: str = "api/compute"

    # -------------------------
    # OpenTelemetry
    # -------------------------
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_SERVICE_NAME: str = "chemrag"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
