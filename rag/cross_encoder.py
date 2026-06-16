from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List

from app.settings import settings
from rag.sparse_index import tokenize


def _lexical_cross_score(query: str, chunk: str) -> float:
    q = set(tokenize(query))
    c = tokenize(chunk)
    if not q or not c:
        return 0.0
    c_set = set(c)
    overlap = len(q & c_set) / max(1, len(q))
    density = sum(1 for t in c if t in q) / max(1, len(c))
    return float((0.75 * overlap) + (0.25 * min(1.0, density * 8)))


@lru_cache(maxsize=1)
def _load_cross_encoder():
    if not settings.RAG_PRO_ENABLE_CROSS_ENCODER:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        return CrossEncoder(settings.RAG_PRO_CROSS_ENCODER_MODEL)
    except Exception:
        return None


def rerank_with_cross_encoder(query: str, hits: List[Dict[str, Any]], *, top_n: int | None = None) -> List[Dict[str, Any]]:
    if not hits:
        return []
    top_n = top_n or min(settings.RAG_PRO_CROSS_ENCODER_TOP_N, len(hits))
    candidates = hits[: max(top_n, min(len(hits), settings.RAG_PRO_CROSS_ENCODER_TOP_N))]
    model = _load_cross_encoder()

    scored: list[tuple[float, int, Dict[str, Any]]] = []
    if model is not None:
        try:
            pairs = [(query, h.get("chunk") or "") for h in candidates]
            scores = model.predict(pairs)
            for i, (score, hit) in enumerate(zip(scores, candidates)):
                h = dict(hit)
                h["rerank_score"] = float(score)
                h["retrieval_stage"] = "cross_encoder"
                scored.append((float(score), i, h))
        except Exception:
            scored = []

    if not scored:
        for i, hit in enumerate(candidates):
            h = dict(hit)
            score = _lexical_cross_score(query, h.get("chunk") or "")
            h["rerank_score"] = score
            h["retrieval_stage"] = "lexical_cross_encoder_fallback"
            scored.append((score, i, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [h for _, _, h in scored]
    seen = {str(h.get("chunk_id") or (h.get("chunk") or "")[:160]) for h in reranked}
    for hit in hits:
        key = str(hit.get("chunk_id") or (hit.get("chunk") or "")[:160])
        if key not in seen:
            reranked.append(hit)
            seen.add(key)
    return reranked[:top_n]
