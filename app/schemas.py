from __future__ import annotations

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RagLaneMetadata(BaseModel):
    rag_lane: str = "general"
    module_relevance: List[str] = Field(default_factory=list)
    evidence_type: Optional[str] = None
    source_quality: Optional[str] = None
    artifact_versions: Dict[str, Any] = Field(default_factory=dict)


class ArtifactContext(BaseModel):
    artifact_versions: Dict[str, str] = Field(default_factory=dict)
    artifact_registry_snapshot: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    text: str
    source_uri: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # ChemRAG v2 lane metadata
    rag_lane: str = "general"
    module_relevance: List[str] = Field(default_factory=list)
    evidence_type: Optional[str] = "internal_note"
    source_quality: Optional[str] = "internal"
    artifact_versions: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    doc_id: Optional[str] = None
    rag_lane: Optional[str] = None
    retrieval_lanes: Optional[List[str]] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    provider: Optional[str] = None
    selected_lanes: List[str] = Field(default_factory=list)


class PdfIngestMeta(BaseModel):
    doc_id: str
    title: str
    source_uri: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    rag_lane: str = "general"
    module_relevance: List[str] = Field(default_factory=list)
    evidence_type: Optional[str] = "textbook"
    source_quality: Optional[str] = "secondary"
    artifact_versions: Dict[str, Any] = Field(default_factory=dict)


class PhytochemistryExportIngestRequest(BaseModel):
    path: str
    project_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    artifact_versions: Dict[str, Any] = Field(default_factory=dict)


class MemoryEventRequest(BaseModel):
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None

    event_type: str
    importance: Optional[int] = None

    title: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None

    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    ml: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)

    rag_lane: Optional[str] = None
    artifact_versions: Dict[str, Any] = Field(default_factory=dict)


class MemoryEventResponse(BaseModel):
    stored: bool
    event_id: Optional[int] = None
    vector_id: Optional[str] = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = 8
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class MemorySearchResponse(BaseModel):
    memories: List[Dict[str, Any]]


class MemoryCompactRequest(BaseModel):
    session_id: str
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    max_events: int = 50


class PlannerRequest(BaseModel):
    query: str

    # identity context for memory scoping
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None

    # optional runtime state from server (formulation, constraints, etc.)
    runtime_context: Optional[Dict[str, Any]] = None

    # artifact-aware context from reaction_framework/formulation_engine wheel system
    artifact_versions: Optional[Dict[str, str]] = None
    artifact_registry_snapshot: Optional[Dict[str, Any]] = None

    # explicit lane override; if omitted, planner uses intent-to-lane routing
    retrieval_lanes: Optional[List[str]] = None

    # allow caller to provide compute output or ask ChemRAG to call a compute endpoint
    require_compute: bool = False
    compute_payload: Optional[Dict[str, Any]] = None
    compute_result: Optional[Dict[str, Any]] = None

    # retrieval parameters
    top_k_docs: int = 8
    top_k_memory: int = 6


class PlannerResponse(BaseModel):
    intent: str
    answer: str
    provider: Optional[str] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    memories_used: List[Dict[str, Any]] = Field(default_factory=list)
    compute_result: Optional[Dict[str, Any]] = None
    selected_lanes: List[str] = Field(default_factory=list)
    artifact_versions_used: Dict[str, Any] = Field(default_factory=dict)
