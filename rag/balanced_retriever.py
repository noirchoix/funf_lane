from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from rag.pipeline import retrieve_context


Hit = Dict[str, Any]
LaneQuotas = Dict[str, int]


def normalize_retrieval_lanes(retrieval_lanes: Optional[List[str]]) -> List[str]:
    """
    Normalize retrieval lane names while preserving caller order.

    Note:
    - This function is intentionally named differently from rag.collections.normalize_lanes
      to avoid import/name confusion.
    - Empty/None means no explicit lane-balanced retrieval.
    """
    if not retrieval_lanes:
        return []

    out: List[str] = []
    seen: set[str] = set()

    for lane in retrieval_lanes:
        lane = (lane or "").strip()
        if not lane:
            continue
        if lane not in seen:
            out.append(lane)
            seen.add(lane)

    return out


def allocate_lane_quotas(total_k: int, lanes: List[str]) -> LaneQuotas:
    """
    Allocate the final result budget across explicit lanes.

    Examples:
      total_k=10, lanes=[phytochemistry_context, quality_control]
      => 5 + 5

      total_k=12, 2 lanes
      => 6 + 6

      total_k=10, 3 lanes
      => 4 + 3 + 3
    """
    if total_k <= 0 or not lanes:
        return {}

    n = len(lanes)
    base = total_k // n
    remainder = total_k % n

    quotas: LaneQuotas = {}
    for i, lane in enumerate(lanes):
        quotas[lane] = base + (1 if i < remainder else 0)

    if total_k >= n:
        for lane in lanes:
            quotas[lane] = max(1, quotas[lane])

    return quotas


def _hit_lane(hit: Hit) -> Optional[str]:
    """
    Read lane from ChemRAG hit payloads.

    Supports:
    - top-level hit["rag_lane"]
    - nested hit["source"]["rag_lane"]
    - collection names like rag_quality_control
    """
    lane = hit.get("rag_lane")
    if lane:
        return str(lane)

    src = hit.get("source") or {}
    lane = src.get("rag_lane")
    if lane:
        return str(lane)

    collection = hit.get("collection")
    if isinstance(collection, str) and collection.startswith("rag_"):
        return collection.removeprefix("rag_")

    return None


def _hit_key(hit: Hit) -> str:
    """
    Stable de-duplication key.
    Prefer chunk_id; fall back to lane/doc/page/text prefix.
    """
    chunk_id = hit.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    lane = _hit_lane(hit) or ""
    doc_id = str(hit.get("doc_id") or "")
    page = str(hit.get("page") or "")
    chunk = (hit.get("chunk") or "")[:256]
    return f"{lane}|{doc_id}|{page}|{chunk}"


def dedupe_hits(hits: List[Hit]) -> List[Hit]:
    seen: set[str] = set()
    out: List[Hit] = []

    for hit in hits:
        key = _hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)

    return out


def sort_hits_by_score(hits: List[Hit]) -> List[Hit]:
    return sorted(hits, key=lambda h: float(h.get("score") or 0.0), reverse=True)


def retrieve_balanced_context(
    *,
    queries: List[str],
    top_k: int,
    doc_id: Optional[str] = None,
    lanes: Optional[List[str]] = None,
    fetch_multiplier: int = 3,
) -> Tuple[List[Hit], LaneQuotas, List[str]]:
    """
    Retrieve a balanced candidate set across explicit lanes.

    This function matches the current project API where rag.pipeline.retrieve_context
    accepts `lanes=...`, not `retrieval_lanes=...`.

    If explicit lanes are passed:
      1. Allocate top_k across lanes.
      2. Retrieve each lane independently.
      3. Deduplicate within each lane.
      4. Keep each lane's quota-sized candidate group.
      5. Return a balanced merged candidate set.

    This prevents one semantically dominant lane from eliminating the other
    before reranking/prompting.
    """
    clean_queries = [q.strip() for q in queries if q and q.strip()]
    if not clean_queries:
        return [], {}, []

    selected_lanes = normalize_retrieval_lanes(lanes)

    # Preserve old behavior if no explicit lanes were selected.
    if not selected_lanes:
        merged: List[Hit] = []
        for q in clean_queries:
            merged.extend(
                retrieve_context(
                    query=q,
                    top_k=top_k,
                    doc_id=doc_id,
                    lanes=None,
                )
            )
        merged = sort_hits_by_score(dedupe_hits(merged))
        return merged[:top_k], {}, []

    quotas = allocate_lane_quotas(top_k, selected_lanes)
    balanced: List[Hit] = []

    for lane in selected_lanes:
        quota = quotas.get(lane, 0)
        if quota <= 0:
            continue

        # Fetch more than the final lane quota because query expansion may
        # introduce duplicates or weak candidates.
        fetch_k = max(quota * fetch_multiplier, quota + 3, 5)

        lane_hits: List[Hit] = []
        for q in clean_queries:
            lane_hits.extend(
                retrieve_context(
                    query=q,
                    top_k=fetch_k,
                    doc_id=doc_id,
                    lanes=[lane],
                )
            )

        lane_hits = sort_hits_by_score(dedupe_hits(lane_hits))
        balanced.extend(lane_hits[:quota])

    balanced = sort_hits_by_score(dedupe_hits(balanced))
    return balanced[:top_k], quotas, selected_lanes


def enforce_lane_balance_after_rerank(
    *,
    ranked_hits: List[Hit],
    candidate_hits: List[Hit],
    lane_quotas: LaneQuotas,
    top_k: int,
) -> List[Hit]:
    """
    Restore lane balance after optional reranking.

    Rerankers may return fewer hits or over-concentrate one lane.
    This function keeps explicit lanes represented according to their quota,
    then fills remaining capacity with the best available hits.
    """
    if not lane_quotas:
        return ranked_hits[:top_k]

    ranked_pool = dedupe_hits(ranked_hits)
    fallback_pool = dedupe_hits(candidate_hits)
    all_pool = dedupe_hits(ranked_pool + fallback_pool)

    by_lane: Dict[str, List[Hit]] = defaultdict(list)
    for hit in all_pool:
        lane = _hit_lane(hit)
        if lane:
            by_lane[lane].append(hit)

    for lane in by_lane:
        by_lane[lane] = sort_hits_by_score(by_lane[lane])

    selected: List[Hit] = []
    selected_keys: set[str] = set()
    counts: Dict[str, int] = defaultdict(int)

    def add_hit(hit: Hit) -> bool:
        key = _hit_key(hit)
        if key in selected_keys:
            return False
        if len(selected) >= top_k:
            return False

        selected.append(hit)
        selected_keys.add(key)

        lane = _hit_lane(hit)
        if lane:
            counts[lane] += 1

        return True

    # Pass 1: preserve reranker order but cap each explicit lane to quota.
    for hit in ranked_pool:
        lane = _hit_lane(hit)
        if lane and lane in lane_quotas:
            if counts[lane] < lane_quotas[lane]:
                add_hit(hit)
        else:
            add_hit(hit)

    # Pass 2: backfill lanes that did not reach quota.
    for lane, quota in lane_quotas.items():
        if counts[lane] >= quota:
            continue

        for hit in by_lane.get(lane, []):
            if counts[lane] >= quota:
                break
            add_hit(hit)

    # Pass 3: fill remaining space with best available hits.
    for hit in sort_hits_by_score(all_pool):
        if len(selected) >= top_k:
            break
        add_hit(hit)

    return selected[:top_k]