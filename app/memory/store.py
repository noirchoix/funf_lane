from __future__ import annotations

from typing import Optional
import uuid

from app.settings import settings
from rag.embed_router import embed_texts
from rag.qdrant_store import get_client
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

_MEMORY_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000002")


def _memory_point_id(*, project_id: Optional[str], event_id: int) -> str:
    name = f"{project_id or 'default'}:{event_id}"
    return str(uuid.uuid5(_MEMORY_NAMESPACE, name))


def ensure_memory_collection(vector_size: int) -> None:
    client = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.MEMORY_COLLECTION in existing:
        return
    client.create_collection(
        collection_name=settings.MEMORY_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_memory_item(
    *,
    event_id: int,
    project_id: str | None,
    user_id: str | None,
    session_id: str | None,
    run_id: str | None,
    event_type: str,
    importance: int,
    narrative: str,
    title: str | None,
    tags: list[str] | None,
    created_at: str,
    rag_lane: str | None = None,
    artifact_versions: dict | None = None,
) -> str:
    vec = embed_texts([narrative])[0]
    ensure_memory_collection(vector_size=len(vec))
    point_id = _memory_point_id(project_id=project_id, event_id=event_id)

    payload = {
        "event_id": event_id,
        "project_id": project_id,
        "user_id": user_id,
        "session_id": session_id,
        "run_id": run_id,
        "event_type": event_type,
        "importance": importance,
        "title": title,
        "tags": tags or [],
        "created_at": created_at,
        "narrative": narrative,
        "rag_lane": rag_lane,
        "artifact_versions": artifact_versions or {},
    }

    client = get_client()
    client.upsert(
        collection_name=settings.MEMORY_COLLECTION,
        points=[PointStruct(id=point_id, vector=vec, payload=payload)],
    )
    return point_id


def search_memory(
    *,
    query: str,
    top_k: int,
    project_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[dict]:
    qvec = embed_texts([query])[0]
    client = get_client()

    must = []
    if project_id:
        must.append(FieldCondition(key="project_id", match=MatchValue(value=project_id)))
    if user_id:
        must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
    if session_id:
        must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

    qfilter = Filter(must=must) if must else None

    res = client.query_points(
        collection_name=settings.MEMORY_COLLECTION,
        query=qvec,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )

    out: list[dict] = []
    for h in res.points or []:
        payload = h.payload or {}
        out.append(
            {
                "score": float(h.score or 0.0),
                "vector_id": str(h.id),
                "event_id": payload.get("event_id"),
                "event_type": payload.get("event_type"),
                "importance": payload.get("importance"),
                "title": payload.get("title"),
                "tags": payload.get("tags"),
                "created_at": payload.get("created_at"),
                "narrative": payload.get("narrative"),
                "rag_lane": payload.get("rag_lane"),
                "artifact_versions": payload.get("artifact_versions") or {},
            }
        )
    return out
