from __future__ import annotations

from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.settings import settings
from rag.collections import get_collection_for_lane, normalize_lane


# Backward-compatible constant for old scripts/tests.
COLLECTION_DOCS = settings.RAG_COLLECTION_GENERAL


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=settings.QDRANT_TIMEOUT_S,
    )


def _resolve_collection(collection_name: str | None = None, lane: str | None = None) -> str:
    if collection_name:
        return collection_name
    return get_collection_for_lane(lane)


def ensure_collection(
    client: QdrantClient,
    vector_size: int,
    *,
    collection_name: str | None = None,
    lane: str | None = None,
) -> None:
    name = _resolve_collection(collection_name, lane)
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    *,
    ids: List[str],
    vectors: List[List[float]],
    payloads: List[dict],
    collection_name: str | None = None,
    lane: str | None = None,
) -> None:
    name = _resolve_collection(collection_name, lane)
    points = [
        PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=name, points=points)


def _build_filter(doc_id: Optional[str], filters: dict | None = None, lane: str | None = None) -> Filter | None:
    must: list[FieldCondition] = []
    if lane:
        must.append(FieldCondition(key="rag_lane", match=MatchValue(value=normalize_lane(lane))))
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    # Lightweight exact-match filter support for flat payload keys.
    for key, value in (filters or {}).items():
        if value is None:
            continue
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=must) if must else None


def search(
    client: QdrantClient,
    *,
    query_vector: List[float],
    top_k: int,
    doc_id: Optional[str] = None,
    collection_name: str | None = None,
    lane: str | None = None,
    filters: dict | None = None,
) -> List[Dict[str, Any]]:
    name = _resolve_collection(collection_name, lane)
    qfilter = _build_filter(doc_id=doc_id, filters=filters, lane=lane)

    res = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )

    hits = res.points or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        source = payload.get("source", {}) or {}
        out.append(
            {
                "score": float(h.score or 0.0),
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "page": payload.get("page"),
                "chunk": payload.get("chunk", ""),
                "source": source,
                "rag_lane": payload.get("rag_lane") or source.get("rag_lane"),
                "collection": name,
                "module_relevance": payload.get("module_relevance") or source.get("module_relevance"),
                "evidence_type": payload.get("evidence_type") or source.get("evidence_type"),
                "source_quality": payload.get("source_quality") or source.get("source_quality"),
                "artifact_versions": payload.get("artifact_versions") or source.get("artifact_versions"),
            }
        )
    return out


def search_multiple(
    client: QdrantClient,
    *,
    query_vector: List[float],
    top_k: int,
    lanes: list[str],
    doc_id: Optional[str] = None,
    filters: dict | None = None,
) -> List[Dict[str, Any]]:
    merged: list[dict] = []
    for lane in lanes:
        try:
            merged.extend(
                search(
                    client,
                    query_vector=query_vector,
                    top_k=top_k,
                    doc_id=doc_id,
                    lane=lane,
                    filters=filters,
                )
            )
        except Exception:
            # Missing/empty collections should not crash multi-lane planner retrieval.
            continue
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged
