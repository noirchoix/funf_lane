from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends

from app.deps import require_service_key
from app.schemas import PlannerRequest, PlannerResponse
from app.settings import settings
from rag.intent import classify_intent, map_intent_to_lanes
from rag.collections import normalize_lanes
from rag.retrieval_enhancements import expand_queries, rerank_chunks
from rag.compute_client import call_compute
from rag.prompting import build_planner_prompt
from rag.balanced_retriever import retrieve_balanced_context, enforce_lane_balance_after_rerank
from rag.llm_router import generate_answer as generate_answer_with_provider
from app.memory.store import search_memory, upsert_memory_item
from app.memory.policy import compute_importance, should_store
from app.memory.summarize import build_narrative
from app.memory.db import init_db, insert_event, update_vector_id

logger = logging.getLogger("chemrag.chat")

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.on_event("startup")
def _startup() -> None:
    init_db()


def _is_degradation_query(user_query: str) -> bool:
    q = (user_query or "").lower()
    trigger_terms = [
        "degradation",
        "degrade",
        "instability",
        "unstable",
        "stability",
        "oxidation",
        "oxidative",
        "acidic",
        "alkaline",
        "product base",
        "antiperspirant",
        "linalool",
        "linalyl acetate",
        "citrus oil",
        "citrus oils",
        "phenolic",
        "phenols",
        "aldehyde",
        "aldehydes",
        "essential oil",
        "essential oils",
        "fragrance formulation",
    ]
    return any(term in q for term in trigger_terms)


def _add_degradation_expansions(user_query: str, queries: list[str]) -> list[str]:
    """
    Targeted recall booster for degradation / stability / product-base questions.
    This broadens semantic retrieval vocabulary without changing answer logic.
    """
    if not _is_degradation_query(user_query):
        return queries

    additions = [
        "acidic antiperspirant base linalool linalyl acetate phenolic materials reactive aldehydes essential oils citrus oils spices undergo chemical reactions",
        "empirical testing antiperspirant bases unstable perfumery ingredients phenolic materials linalool linalyl acetate reactive aldehydes citrus oils spices",
        "unsaturated terpene alcohols esters linalool linalyl acetate acidic antiperspirant active chemical reactions",
        "aerosol antiperspirant unstable perfumery ingredients citrus oils essential oils phenolic materials reactive aldehydes",
    ]

    out: list[str] = []
    seen: set[str] = set()

    for item in list(queries) + additions:
        item = (item or "").strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)

    return out


def _degradation_anchor_queries(user_query: str) -> list[str]:
    """
    High-precision anchor queries for degradation/stability/product-base questions.

    These guarantee that class-level instability evidence is queried directly when
    broad semantic retrieval might otherwise prefer aldehyde-only or general QC chunks.
    """
    if not _is_degradation_query(user_query):
        return []

    return [
        "acidic antiperspirant base linalool linalyl acetate phenolic materials reactive aldehydes essential oils citrus oils spices undergo chemical reactions",
        "empirical testing antiperspirant bases unstable perfumery ingredients phenolic materials linalool linalyl acetate reactive aldehydes citrus oils spices",
        "unsaturated terpene alcohols esters linalool linalyl acetate acidic antiperspirant active chemical reactions",
        "aerosol antiperspirant unstable perfumery ingredients citrus oils essential oils phenolic materials reactive aldehydes",
    ]


def _merge_hits(primary: list[dict], extra: list[dict]) -> list[dict]:
    """
    Merge retrieval hits without duplicates.
    """
    out: list[dict] = []
    seen: set[str] = set()

    for h in primary + extra:
        key = str(h.get("chunk_id") or h.get("doc_id") or (h.get("chunk") or "")[:160])
        if key in seen:
            continue
        out.append(h)
        seen.add(key)

    return out


def _select_degradation_anchors(user_query: str, hits: list[dict], max_anchors: int = 2) -> list[dict]:
    """
    Preserve high-value class-level instability evidence for degradation/stability queries.
    """
    if not _is_degradation_query(user_query):
        return []

    required_any = [
        "linalool",
        "linalyl acetate",
        "phenolic materials",
        "reactive aldehydes",
        "essential oils",
        "citrus oils",
        "spices",
    ]

    required_context = [
        "acidic",
        "antiperspirant",
        "undergo chemical reactions",
        "unstable perfumery ingredients",
    ]

    anchors: list[dict] = []
    seen: set[str] = set()

    for h in hits:
        chunk = (h.get("chunk") or "").lower()
        if not chunk:
            continue

        has_material = any(term in chunk for term in required_any)
        has_context = any(term in chunk for term in required_context)

        if not (has_material and has_context):
            continue

        key = str(h.get("chunk_id") or h.get("doc_id") or chunk[:160])
        if key in seen:
            continue

        anchors.append(h)
        seen.add(key)

        if len(anchors) >= max_anchors:
            break

    return anchors


def _merge_anchors_with_final_hits(anchors: list[dict], final_hits: list[dict], top_k: int) -> list[dict]:
    """
    Put selected degradation anchors at the front of final context while preserving
    the rest of the balanced/reranked evidence.
    """
    if not anchors:
        return final_hits[:top_k]

    merged: list[dict] = []
    seen: set[str] = set()

    for h in anchors + final_hits:
        key = str(h.get("chunk_id") or h.get("doc_id") or (h.get("chunk") or "")[:160])
        if key in seen:
            continue
        merged.append(h)
        seen.add(key)

        if len(merged) >= top_k:
            break

    return merged


def _dedupe_hits(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for h in hits:
        key = str(h.get("chunk_id") or f"{h.get('collection')}:{(h.get('chunk') or '')[:128]}")
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


@router.post("/planner", response_model=PlannerResponse)
async def planner(req: PlannerRequest, _=Depends(require_service_key)) -> PlannerResponse:
    intent = classify_intent(req.query)
    selected_lanes = normalize_lanes(req.retrieval_lanes or map_intent_to_lanes(intent))

    top_k_memory = int(req.top_k_memory or getattr(settings, "MEMORY_TOP_K", 6))
    memories = search_memory(
        query=req.query,
        top_k=top_k_memory,
        project_id=req.project_id,
        user_id=req.user_id,
        session_id=req.session_id,
    )

    top_k_docs = int(req.top_k_docs or 8)

    try:
        queries = expand_queries(req.query)
    except Exception as e:
        logger.exception("Planner query expansion failed; using original query. err=%r", e)
        queries = [req.query]

    # Same degradation/stability recall strategy validated in /v1/query.
    queries = _add_degradation_expansions(req.query, queries)

    candidate_docs: list[dict] = []
    lane_quotas: dict[str, int] = {}
    ranked_docs: list[dict] = []
    docs: list[dict] = []

    try:
        # 1) Primary balanced multi-lane retrieval.
        candidate_docs, lane_quotas, selected_lanes = retrieve_balanced_context(
            queries=queries,
            top_k=top_k_docs,
            doc_id=None,
            lanes=selected_lanes,
        )

        # 2) High-precision anchor retrieval for degradation/stability questions.
        anchor_queries = _degradation_anchor_queries(req.query)
        if anchor_queries:
            try:
                anchor_docs, _, _ = retrieve_balanced_context(
                    queries=anchor_queries,
                    top_k=min(8, max(4, top_k_docs // 2)),
                    doc_id=None,
                    lanes=selected_lanes,
                )
                candidate_docs = _merge_hits(anchor_docs, candidate_docs)
            except Exception as e:
                logger.exception(
                    "Planner degradation anchor retrieval failed; continuing without anchors. err=%r",
                    e,
                )

        logger.info(
            "Planner candidate docs before rerank: %s",
            [
                {
                    "doc_id": h.get("doc_id"),
                    "page": h.get("page"),
                    "lane": h.get("rag_lane") or (h.get("source") or {}).get("rag_lane"),
                    "score": h.get("score"),
                    "preview": (h.get("chunk") or "")[:180],
                }
                for h in candidate_docs[:30]
            ],
        )

        # 3) Select class-level degradation anchors from merged candidate pool.
        anchors = _select_degradation_anchors(req.query, candidate_docs)

        # 4) Rerank merged candidate pool.
        ranked_docs = rerank_chunks(req.query, candidate_docs) if candidate_docs else []

        # 5) Enforce balanced representation across lanes.
        docs = enforce_lane_balance_after_rerank(
            ranked_hits=ranked_docs,
            candidate_hits=candidate_docs,
            lane_quotas=lane_quotas,
            top_k=top_k_docs,
        )

        # 6) Preserve high-value class-level instability anchors in final docs.
        docs = _merge_anchors_with_final_hits(
            anchors=anchors,
            final_hits=docs,
            top_k=top_k_docs,
        )

    except Exception as e:
        logger.exception("Planner balanced retrieval failed. err=%r", e)
        docs = []

    compute_result: Any | None = req.compute_result
    needs_compute = bool(req.require_compute) or intent in (
        "artifact_explanation",
        "formulation_qc",
        "recommendation",
    )

    if compute_result is None and needs_compute and req.compute_payload is not None:
        endpoint = getattr(settings, "COMPUTE_ENDPOINT", "api/compute")
        compute_result = await call_compute(endpoint, req.compute_payload)

    artifact_versions = req.artifact_versions or {}
    artifact_registry_snapshot = req.artifact_registry_snapshot or {}

    prompt = build_planner_prompt(
        user_query=req.query,
        intent=intent,
        docs=docs,
        memories=memories,
        runtime_context=req.runtime_context,
        compute_result=compute_result if isinstance(compute_result, dict) else None,
        artifact_versions=artifact_versions,
        artifact_registry_snapshot=artifact_registry_snapshot,
        selected_lanes=selected_lanes,
    )

    answer, provider = generate_answer_with_provider(prompt)

    created_at = datetime.now(timezone.utc).isoformat()
    event_type = "system"
    importance = compute_importance(event_type, 3)

    if should_store(event_type, importance):
        narrative = build_narrative(
            event_type=event_type,
            title="Chat turn",
            decision=None,
            reason=None,
            inputs={
                "query": req.query,
                "intent": intent,
                "selected_lanes": selected_lanes,
                "runtime_context": req.runtime_context,
                "artifact_versions": artifact_versions,
                "compute_requested": bool(needs_compute),
                "provider": provider,
            },
            outputs={
                "used_sources": [
                    {
                        "title": (h.get("source", {}) or {}).get("title"),
                        "page": h.get("page"),
                        "score": h.get("score"),
                        "chunk_id": h.get("chunk_id"),
                        "rag_lane": h.get("rag_lane"),
                        "collection": h.get("collection"),
                    }
                    for h in docs[:8]
                ],
                "memory_hits": [
                    {
                        "event_id": m.get("event_id"),
                        "title": m.get("title"),
                        "event_type": m.get("event_type"),
                        "score": m.get("score"),
                    }
                    for m in memories[:8]
                ],
                "answer_preview": answer[:400],
            },
            ml=compute_result if isinstance(compute_result, dict) else None,
            tags=["chat", intent, provider, *selected_lanes],
        )

        row = {
            "created_at": created_at,
            "project_id": req.project_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "run_id": req.run_id,
            "event_type": event_type,
            "importance": importance,
            "title": "Chat turn",
            "narrative": narrative,
            "inputs_json": json.dumps(
                {
                    "runtime_context": req.runtime_context,
                    "query": req.query,
                    "intent": intent,
                    "selected_lanes": selected_lanes,
                    "provider": provider,
                    "artifact_versions": artifact_versions,
                },
                ensure_ascii=False,
            ),
            "outputs_json": json.dumps({"answer_preview": answer[:400]}, ensure_ascii=False),
            "model_json": json.dumps(compute_result, ensure_ascii=False) if compute_result else None,
            "tags_json": json.dumps(["chat", intent, provider, *selected_lanes], ensure_ascii=False),
            "vector_id": None,
            "is_compacted": 0,
        }

        event_id = insert_event(row)

        vector_id = upsert_memory_item(
            event_id=event_id,
            project_id=req.project_id,
            user_id=req.user_id,
            session_id=req.session_id,
            run_id=req.run_id,
            event_type=event_type,
            importance=importance,
            narrative=narrative,
            title="Chat turn",
            tags=["chat", intent, provider, *selected_lanes],
            created_at=created_at,
            rag_lane="planner",
            artifact_versions=artifact_versions,
        )

        update_vector_id(event_id=event_id, vector_id=vector_id)

    return PlannerResponse(
        intent=intent,
        answer=answer,
        provider=provider,
        citations=docs[:top_k_docs],
        memories_used=memories,
        compute_result=compute_result if isinstance(compute_result, dict) else None,
        selected_lanes=selected_lanes,
        artifact_versions_used=artifact_versions,
    )