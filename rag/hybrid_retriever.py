from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.settings import settings
from rag.balanced_retriever import retrieve_balanced_context
from rag.collections import normalize_lanes
from rag.cross_encoder import rerank_with_cross_encoder
from rag.sparse_index import sparse_search

Hit = Dict[str, Any]


def _hit_key(hit: Hit) -> str:
    return str(hit.get("chunk_id") or f"{hit.get('doc_id')}|{hit.get('page')}|{(hit.get('chunk') or '')[:160]}")


def _rank_map(hits: List[Hit]) -> dict[str, int]:
    return {_hit_key(hit): i + 1 for i, hit in enumerate(hits)}


def _merge_payloads(primary: Hit, secondary: Hit) -> Hit:
    merged = dict(primary)
    for key, value in secondary.items():
        if merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def reciprocal_rank_fusion(
    *,
    dense_hits: List[Hit],
    sparse_hits: List[Hit],
    k: int | None = None,
) -> List[Hit]:
    k = int(k or settings.RAG_PRO_RRF_K)
    dense_rank = _rank_map(dense_hits)
    sparse_rank = _rank_map(sparse_hits)
    payloads: dict[str, Hit] = {}

    for hit in dense_hits:
        key = _hit_key(hit)
        h = dict(hit)
        h["dense_score"] = float(hit.get("score") or hit.get("dense_score") or 0.0)
        h["retrieval_stage"] = "dense"
        payloads[key] = h

    for hit in sparse_hits:
        key = _hit_key(hit)
        h = dict(hit)
        h["sparse_score"] = float(hit.get("score") or hit.get("sparse_score") or 0.0)
        h["retrieval_stage"] = "sparse"
        payloads[key] = _merge_payloads(payloads[key], h) if key in payloads else h

    fused: list[Hit] = []
    all_keys = set(dense_rank) | set(sparse_rank)
    for key in all_keys:
        score = 0.0
        if key in dense_rank:
            score += 1.0 / (k + dense_rank[key])
        if key in sparse_rank:
            score += 1.0 / (k + sparse_rank[key])
        h = dict(payloads[key])
        h["fused_score"] = score
        h["score"] = score
        h["retrieval_stage"] = "rrf_fused"
        fused.append(h)

    fused.sort(key=lambda h: float(h.get("fused_score") or 0.0), reverse=True)
    return fused


def _ranked(hits: List[Hit], limit: int | None = None) -> List[Hit]:
    out = []
    for i, hit in enumerate(hits[:limit] if limit else hits, start=1):
        h = dict(hit)
        h["rank"] = i
        out.append(h)
    return out


def hybrid_retrieve(
    *,
    query: str,
    top_k: int,
    fetch_k: Optional[int] = None,
    doc_id: Optional[str] = None,
    lanes: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    mode: str = "hybrid_rerank",
) -> Tuple[List[Hit], Dict[str, Any]]:
    selected_lanes = normalize_lanes(lanes or ["general"])
    fetch_k = int(fetch_k or settings.RAG_PRO_FETCH_K or max(top_k * 3, top_k + 8))
    dense_hits: List[Hit] = []
    sparse_hits: List[Hit] = []
    fused_hits: List[Hit] = []
    reranked_hits: List[Hit] = []

    if mode in {"vector", "hybrid", "hybrid_rerank"}:
        dense_hits, _quotas, selected_lanes = retrieve_balanced_context(
            queries=[query],
            top_k=fetch_k,
            doc_id=doc_id,
            lanes=selected_lanes,
        )
        for h in dense_hits:
            h["dense_score"] = float(h.get("score") or 0.0)
            h["retrieval_stage"] = "dense"

    if mode in {"sparse", "hybrid", "hybrid_rerank"}:
        sparse_hits = sparse_search(
            query=query,
            top_k=fetch_k,
            lanes=selected_lanes,
            doc_id=doc_id,
            filters=filters,
        )

    if mode == "vector":
        final = dense_hits[:top_k]
    elif mode == "sparse":
        final = sparse_hits[:top_k]
    else:
        fused_hits = reciprocal_rank_fusion(dense_hits=dense_hits, sparse_hits=sparse_hits)
        if mode == "hybrid_rerank":
            reranked_hits = rerank_with_cross_encoder(query, fused_hits, top_n=max(top_k, min(len(fused_hits), settings.RAG_PRO_CROSS_ENCODER_TOP_N)))
            final = reranked_hits[:top_k]
        else:
            final = fused_hits[:top_k]

    trace = {
        "mode": mode,
        "selected_lanes": selected_lanes,
        "dense_hits": _ranked(dense_hits, 30),
        "sparse_hits": _ranked(sparse_hits, 30),
        "fused_hits": _ranked(fused_hits, 30),
        "reranked_hits": _ranked(reranked_hits, 30),
        "final_context": _ranked(final, top_k),
        "diagnostics": {
            "fetch_k": fetch_k,
            "top_k": top_k,
            "dense_count": len(dense_hits),
            "sparse_count": len(sparse_hits),
            "fused_count": len(fused_hits),
            "cross_encoder_enabled": bool(settings.RAG_PRO_ENABLE_CROSS_ENCODER),
        },
    }
    return _ranked(final, top_k), trace
