from __future__ import annotations

from fastapi import APIRouter, Depends
from app.deps import require_service_key
from app.memory.db import get_conn, init_db
from rag.qdrant_store import get_client
from rag.collections import get_lane_configs
from app.settings import settings

router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


@router.on_event("startup")
def _startup() -> None:
    init_db()


@router.get("")
def metrics(_=Depends(require_service_key)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM memory_events;")
    total_events = int(cur.fetchone()["n"])
    cur.execute("SELECT COUNT(*) AS n FROM memory_events WHERE created_at >= datetime('now','-1 day');")
    events_24h = int(cur.fetchone()["n"])
    conn.close()

    client = get_client()
    qdrant: dict[str, dict[str, int | None] | None] = {}
    for lane, cfg in get_lane_configs().items():
        try:
            info = client.get_collection(cfg.collection_name)
            qdrant[lane] = {"collection": cfg.collection_name, "points_count": getattr(info, "points_count", None)}
        except Exception:
            qdrant[lane] = {"collection": cfg.collection_name, "points_count": None}

    try:
        mem_info = client.get_collection(settings.MEMORY_COLLECTION)
        qdrant["memory"] = {"collection": settings.MEMORY_COLLECTION, "points_count": getattr(mem_info, "points_count", None)}
    except Exception:
        qdrant["memory"] = {"collection": settings.MEMORY_COLLECTION, "points_count": None}

    return {
        "memory": {"total_events": total_events, "events_last_24h": events_24h},
        "qdrant": qdrant,
    }
