from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, List

from app.settings import settings


def _ensure_parent_dir(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def get_conn() -> sqlite3.Connection:
    _ensure_parent_dir(settings.MEMORY_DB_PATH)
    conn = sqlite3.connect(settings.MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            project_id TEXT,
            user_id TEXT,
            session_id TEXT,
            run_id TEXT,
            event_type TEXT NOT NULL,
            importance INTEGER NOT NULL,
            title TEXT,
            narrative TEXT NOT NULL,
            inputs_json TEXT,
            outputs_json TEXT,
            model_json TEXT,
            tags_json TEXT,
            vector_id TEXT,
            is_compacted INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON memory_events(session_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_project ON memory_events(project_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_events_run ON memory_events(run_id);")
    conn.commit()
    conn.close()


def insert_event(row: Dict[str, Any]) -> int:
    """
    Inserts row into sqlite and returns new integer primary key.

    Important: sqlite3.Cursor.lastrowid is typed as Optional[int] by type checkers.
    We guard it to satisfy Pylance and avoid runtime surprises.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memory_events (
            created_at, project_id, user_id, session_id, run_id,
            event_type, importance, title, narrative,
            inputs_json, outputs_json, model_json, tags_json,
            vector_id, is_compacted
        )
        VALUES (
            :created_at, :project_id, :user_id, :session_id, :run_id,
            :event_type, :importance, :title, :narrative,
            :inputs_json, :outputs_json, :model_json, :tags_json,
            :vector_id, :is_compacted
        );
        """,
        row,
    )
    conn.commit()

    last = cur.lastrowid  # Optional[int] per typing
    conn.close()

    if last is None:
        raise RuntimeError("Failed to obtain lastrowid after insert into memory_events")

    return int(last)


def mark_compacted(session_id: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE memory_events SET is_compacted=1 WHERE session_id=? AND is_compacted=0;",
        (session_id,),
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return int(n)


def fetch_recent_events(
    *,
    session_id: Optional[str],
    project_id: Optional[str],
    limit: int = 50,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()

    where: list[str] = []
    params: list[Any] = []
    if session_id:
        where.append("session_id=?")
        params.append(session_id)
    if project_id:
        where.append("project_id=?")
        params.append(project_id)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    cur.execute(
        f"""
        SELECT * FROM memory_events
        {where_sql}
        ORDER BY id DESC
        LIMIT ?;
        """,
        (*params, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_vector_id(*, event_id: int, vector_id: str) -> None:
    """
    Optional helper (recommended): persist the Qdrant point id (UUID string)
    back into sqlite after upsert succeeds.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE memory_events SET vector_id=? WHERE id=?;",
        (vector_id, event_id),
    )
    conn.commit()
    conn.close()