from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


RetrievalMode = Literal["vector", "sparse", "hybrid", "hybrid_rerank"]


class ProQueryRequest(BaseModel):
    query: str
    top_k: int = 8
    fetch_k: Optional[int] = None
    doc_id: Optional[str] = None
    retrieval_lanes: Optional[List[str]] = None
    rag_lane: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    generate_answer: bool = True
    return_trace: bool = True


class RetrievalHit(BaseModel):
    rank: int
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    page: Optional[int] = None
    chunk: str = ""
    source: Dict[str, Any] = Field(default_factory=dict)
    rag_lane: Optional[str] = None
    collection: Optional[str] = None
    score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrieval_stage: Optional[str] = None


class RetrievalTrace(BaseModel):
    mode: RetrievalMode
    selected_lanes: List[str] = Field(default_factory=list)
    dense_hits: List[RetrievalHit] = Field(default_factory=list)
    sparse_hits: List[RetrievalHit] = Field(default_factory=list)
    fused_hits: List[RetrievalHit] = Field(default_factory=list)
    reranked_hits: List[RetrievalHit] = Field(default_factory=list)
    final_context: List[RetrievalHit] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class ProQueryResponse(BaseModel):
    answer: str
    provider: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    selected_lanes: List[str] = Field(default_factory=list)
    trace: Optional[RetrievalTrace] = None


class GoldenCase(BaseModel):
    id: str
    question: str
    expected_doc_ids: List[str] = Field(default_factory=list)
    expected_chunk_ids: List[str] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    retrieval_lanes: Optional[List[str]] = None
    tags: List[str] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    cases: List[GoldenCase]
    top_k: int = 8
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    generate_answers: bool = False
    fail_on_threshold: bool = False


class EvalCaseResult(BaseModel):
    id: str
    question: str
    passed_retrieval: bool
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    citation_count: int
    first_relevant_rank: Optional[int] = None
    retrieved_doc_ids: List[str] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    missing_expected_doc_ids: List[str] = Field(default_factory=list)
    missing_expected_chunk_ids: List[str] = Field(default_factory=list)
    answer: Optional[str] = None


class EvalRunResponse(BaseModel):
    ok: bool
    summary: Dict[str, Any] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    cases: List[EvalCaseResult] = Field(default_factory=list)
