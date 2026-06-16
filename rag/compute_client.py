from __future__ import annotations

from typing import Any, Dict, Optional
import httpx

from app.settings import settings

def _headers() -> dict:
    h = {}
    if settings.COMPUTE_API_KEY:
        h["X-API-Key"] = settings.COMPUTE_API_KEY
    return h

async def call_compute(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your external server computation endpoint.
    You control the contract. We treat response as JSON.
    """
    if not settings.COMPUTE_BASE_URL:
        return {"ok": False, "error": "COMPUTE_BASE_URL not configured"}

    url = settings.COMPUTE_BASE_URL.rstrip("/") + "/" + endpoint.lstrip("/")
    timeout = settings.COMPUTE_TIMEOUT_S

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()
