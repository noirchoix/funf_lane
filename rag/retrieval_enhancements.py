from __future__ import annotations

import json
from typing import List, Dict, Any

from app.settings import settings
from .llm_router import generate_answer


def expand_queries(user_query: str) -> List[str]:
    """
    Generate a few search variants for better recall.
    Safe: if LLM fails, degrade to [user_query].
    """
    if not settings.RAG_QUERY_EXPANSION:
        return [user_query]

    prompt = f"""Generate {settings.RAG_EXPANSION_COUNT} alternative search queries
for retrieving chemistry textbook and journal content.

Rules:
- Keep each query short (3-12 words).
- Preserve chemistry terms (pKa, polarity, oxidation, miscibility, solubility, esterification, etc.).
- Output ONLY a numbered list, one query per line.
User query: {user_query}
"""
    try:
        text, _provider = generate_answer(prompt)
    except Exception:
        return [user_query]

    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("0123456789. )-").strip()
        if line:
            out.append(line)

    merged = [user_query] + [q for q in out if q.lower() != user_query.lower()]

    seen = set()
    uniq: List[str] = []
    for q in merged:
        k = q.lower().strip()
        if k and k not in seen:
            uniq.append(q)
            seen.add(k)

    return uniq[: (1 + settings.RAG_EXPANSION_COUNT)]


def rerank_chunks(user_query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM-based rerank (optional). If LLM fails, keep original order.
    """
    if not settings.RAG_RERANK or not hits:
        return hits

    top_n = min(settings.RAG_RERANK_TOP_N, len(hits))

    candidates: List[str] = []
    cap = min(len(hits), 30)
    for i, h in enumerate(hits[:cap]):
        chunk = (h.get("chunk") or "")[:800]
        src = h.get("source", {}) or {}
        page = h.get("page", None)
        candidates.append(
            f"[{i}] title={src.get('title')} page={page} score={h.get('score')}\n{chunk}"
        )

    prompt = f"""You are reranking retrieval chunks for a chemistry RAG system.

Return ONLY valid JSON with this schema:
{{
  "ranked_indices": [0, 3, 2]
}}

Rules:
- ranked_indices are 0-based integers referencing the candidate IDs in brackets.
- Return at most {top_n} indices.
- Prefer passages that directly answer the query with chemical mechanism / causal explanation.

User query:
{user_query}

Candidates:
{chr(10).join(candidates)}
"""

    try:
        text, _provider = generate_answer(prompt)
        data = json.loads(text)
        ids = data.get("ranked_indices", [])
        if not isinstance(ids, list):
            return hits[:top_n]

        parsed: List[int] = []
        for x in ids:
            if isinstance(x, int) and 0 <= x < cap:
                parsed.append(x)

        if not parsed:
            return hits[:top_n]

        ranked = [hits[i] for i in parsed]
        used = set(parsed)

        for i in range(len(hits)):
            if len(ranked) >= top_n:
                break
            if i not in used:
                ranked.append(hits[i])

        return ranked
    except Exception:
        return hits[:top_n]