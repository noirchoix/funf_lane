from __future__ import annotations

import json
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from app.deps import require_service_key
from app.settings import settings
from app.schemas import (
    MemoryEventRequest, MemoryEventResponse,
    MemorySearchRequest, MemorySearchResponse,
    MemoryCompactRequest
)

from app.memory.db import init_db, insert_event, fetch_recent_events, mark_compacted, update_vector_id
from app.memory.policy import normalize_event_type, compute_importance, should_store
from app.memory.summarize import build_narrative
from app.memory.store import upsert_memory_item, search_memory

router = APIRouter(prefix="/v1/memory", tags=["memory"])


@router.on_event("startup")
def _startup() -> None:
    init_db()


@router.post("/event", response_model=MemoryEventResponse)
def create_event(req: MemoryEventRequest, _=Depends(require_service_key)):
    try:
        event_type = normalize_event_type(req.event_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    importance = compute_importance(event_type, req.importance)
    if not should_store(event_type, importance):
        return MemoryEventResponse(stored=False)

    created_at = datetime.now(timezone.utc).isoformat()

    narrative = build_narrative(
        event_type=event_type,
        title=req.title,
        decision=req.decision,
        reason=req.reason,
        inputs=req.inputs,
        outputs=req.outputs,
        ml=req.ml,
        tags=req.tags,
    )

    # 1) Store structured log (SQLite) FIRST (vector_id unknown for now)
    row = {
        "created_at": created_at,
        "project_id": req.project_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "run_id": req.run_id,
        "event_type": event_type,
        "importance": importance,
        "title": req.title,
        "narrative": narrative,
        "inputs_json": json.dumps(req.inputs, ensure_ascii=False) if req.inputs is not None else None,
        "outputs_json": json.dumps(req.outputs, ensure_ascii=False) if req.outputs is not None else None,
        "model_json": json.dumps(req.ml, ensure_ascii=False) if req.ml is not None else None,
        "tags_json": json.dumps(req.tags, ensure_ascii=False) if req.tags is not None else None,
        "vector_id": None,   # <-- important
        "is_compacted": 0,
    }

    event_id = insert_event(row)

    # 2) Store semantic memory (Qdrant) using event_id to generate UUID point id
    try:
        vector_id = upsert_memory_item(
            event_id=event_id,                 # <-- NEW
            project_id=req.project_id,
            user_id=req.user_id,
            session_id=req.session_id,
            run_id=req.run_id,
            event_type=event_type,
            importance=importance,
            narrative=narrative,
            title=req.title,
            tags=req.tags,
            created_at=created_at,
            rag_lane=req.rag_lane,
            artifact_versions=req.artifact_versions,
        )
    except Exception as e:
        # Keep the SQLite log even if Qdrant fails; return stored=True but vector_id None
        # (Or you can choose to return stored=False; this approach keeps audit trail.)
        raise HTTPException(status_code=500, detail=f"Vector store upsert failed: {e!r}")

    # 3) Persist the vector_id back into SQLite
    update_vector_id(event_id=event_id, vector_id=vector_id)

    return MemoryEventResponse(stored=True, event_id=event_id, vector_id=vector_id)


@router.post("/search", response_model=MemorySearchResponse)
def search_events(req: MemorySearchRequest, _=Depends(require_service_key)):
    top_k = req.top_k or settings.MEMORY_TOP_K
    hits = search_memory(
        query=req.query,
        top_k=top_k,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )
    return MemorySearchResponse(memories=hits)


@router.post("/compact")
def compact_session(req: MemoryCompactRequest, _=Depends(require_service_key)):
    """
    Optional: compact last N events for a session into one high-importance 'summary' memory node,
    then mark those events as compacted in SQLite.
    """
    events = fetch_recent_events(session_id=req.session_id, project_id=req.project_id, limit=req.max_events)
    if not events:
        return {"ok": True, "compacted": 0, "summary_vector_id": None}

    # Build a compact summary narrative (no LLM required; deterministic compaction)
    lines = []
    for e in reversed(events):  # oldest -> newest
        if int(e.get("is_compacted") or 0) == 1:
            continue
        narrative = e.get("narrative") or ""
        lines.append(f"- [{e.get('event_type')}] {e.get('title') or ''} :: {narrative[:400]}")

    summary_text = "SESSION SUMMARY\n" + "\n".join(lines)
    summary_text = summary_text[: settings.MEMORY_MAX_EVENT_TEXT_CHARS]

    created_at = datetime.now(timezone.utc).isoformat()

    # 1) Insert summary row FIRST (vector_id unknown)
    summary_row = {
        "created_at": created_at,
        "project_id": req.project_id,
        "user_id": req.user_id,
        "session_id": req.session_id,
        "run_id": None,
        "event_type": "summary",
        "importance": 5,
        "title": "Session summary",
        "narrative": summary_text,
        "inputs_json": None,
        "outputs_json": None,
        "model_json": None,
        "tags_json": json.dumps(["summary"], ensure_ascii=False),
        "vector_id": None,
        "is_compacted": 0,
    }
    summary_event_id = insert_event(summary_row)

    # 2) Upsert into Qdrant using summary_event_id
    vector_id = upsert_memory_item(
        event_id=summary_event_id,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
        run_id=None,
        event_type="summary",
        importance=5,
        narrative=summary_text,
        title="Session summary",
        tags=["summary"],
        created_at=created_at,
    )

    # 3) Update sqlite with vector_id
    update_vector_id(event_id=summary_event_id, vector_id=vector_id)

    compacted = mark_compacted(req.session_id)
    return {"ok": True, "compacted": compacted, "summary_vector_id": vector_id}