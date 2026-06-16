from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

from app.settings import settings


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return ""


def build_narrative(
    *,
    event_type: str,
    title: str | None,
    decision: str | None,
    reason: str | None,
    inputs: dict | None,
    outputs: dict | None,
    ml: dict | None,
    tags: list[str] | None,
) -> str:
    """
    Compact, retrieval-friendly event narrative.
    Keep it short, concrete, and searchable.
    """
    parts: list[str] = []
    parts.append(f"type={event_type}")

    if title:
        parts.append(f"title={title.strip()}")

    if tags:
        parts.append("tags=" + ", ".join([t.strip() for t in tags if t.strip()]))

    if decision:
        parts.append(f"decision={decision.strip()}")
    if reason:
        parts.append(f"reason={reason.strip()}")

    if inputs:
        parts.append("inputs=" + _safe_json(inputs))
    if outputs:
        parts.append("outputs=" + _safe_json(outputs))
    if ml:
        parts.append("ml=" + _safe_json(ml))

    text = "\n".join(parts).strip()
    # cap length
    return text[: settings.MEMORY_MAX_EVENT_TEXT_CHARS]
