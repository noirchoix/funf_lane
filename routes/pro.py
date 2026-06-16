from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.deps import require_service_key
from app.pro_schemas import EvalRunRequest, EvalRunResponse, ProQueryRequest, ProQueryResponse, RetrievalTrace
from app.settings import settings
from rag.collections import normalize_lanes
from rag.evaluation import default_thresholds, gate_summary, score_retrieval_case, summarize_eval
from rag.hybrid_retriever import hybrid_retrieve
from rag.llm_router import generate_answer

logger = logging.getLogger("funf_rag.pro")
router = APIRouter(prefix="/v1/pro", tags=["funf-rag-pro"])


def _clip_hit(hit: dict) -> dict:
    out = dict(hit)
    max_chars = int(settings.RAG_PRO_TRACE_MAX_CHARS)
    if isinstance(out.get("chunk"), str) and len(out["chunk"]) > max_chars:
        out["chunk"] = out["chunk"][:max_chars] + "…"
    return out


def _build_prompt(user_query: str, hits: list[dict]) -> str:
    context_blocks: list[str] = []
    used = 0
    for h in hits:
        chunk = h.get("chunk", "") or ""
        if not chunk:
            continue
        if used + len(chunk) > settings.RAG_MAX_CONTEXT_CHARS:
            break
        used += len(chunk)
        src = h.get("source", {}) or {}
        page0 = h.get("page", None)
        page_display = (page0 + 1) if isinstance(page0, int) else page0
        context_blocks.append(
            "\n".join(
                [
                    f"[SOURCE lane={h.get('rag_lane')} collection={h.get('collection')} title={src.get('title')} page={page_display} rank={h.get('rank')} score={h.get('score')} fused={h.get('fused_score')} rerank={h.get('rerank_score')} ]",
                    f"doc_id={h.get('doc_id')} chunk_id={h.get('chunk_id')} evidence_type={h.get('evidence_type')} source_quality={h.get('source_quality')}",
                    f"uri={src.get('source_uri')}",
                    chunk,
                ]
            )
        )
    context = "\n---\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    return f"""You are Fünf RAG, a five-lane production retrieval-augmented generation assistant.

Use ONLY the provided SOURCES for factual claims. Every factual claim must be attributable to one or more retrieved sources. If the retrieved sources are insufficient, say exactly what is missing.

USER QUESTION:
{user_query}

SOURCES:
{context}

OUTPUT RULES:
1. Start with a direct answer.
2. Use evidence-bounded language; do not invent facts, mechanisms, protocols, thresholds, dates, controls, or policies.
3. Cite source titles and pages/sections where available.
4. Include a short "Evidence limits" section if the context is incomplete.
"""


@router.post("/query", response_model=ProQueryResponse)
def pro_query(req: ProQueryRequest, _=Depends(require_service_key)) -> ProQueryResponse:
    mode = req.retrieval_mode or settings.RAG_PRO_DEFAULT_MODE
    selected_lanes = req.retrieval_lanes or ([req.rag_lane] if req.rag_lane else ["general"])
    selected_lanes = normalize_lanes(selected_lanes)

    hits, trace_dict = hybrid_retrieve(
        query=req.query,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        doc_id=req.doc_id,
        lanes=selected_lanes,
        filters=req.filters,
        mode=mode,
    )

    provider = "retrieval_only"
    if req.generate_answer:
        try:
            answer, provider = generate_answer(_build_prompt(req.query, hits))
        except Exception as e:
            logger.exception("ChemRAG Pro answer generation failed: %r", e)
            provider = "extractive_fallback"
            if not hits:
                answer = "No relevant context was retrieved from the selected lanes."
            else:
                snippets = [(h.get("chunk") or "").strip()[:700] for h in hits[:5] if (h.get("chunk") or "").strip()]
                answer = "Relevant excerpts:\n\n" + "\n\n---\n\n".join(snippets)
    else:
        answer = "Retrieval completed. Answer generation was disabled for this run."

    trace = None
    if req.return_trace:
        clean = dict(trace_dict)
        for key in ["dense_hits", "sparse_hits", "fused_hits", "reranked_hits", "final_context"]:
            clean[key] = [_clip_hit(h) for h in clean.get(key, [])]
        trace = RetrievalTrace(**clean)

    return ProQueryResponse(
        answer=answer,
        provider=provider,
        citations=hits[: req.top_k],
        selected_lanes=selected_lanes,
        trace=trace,
    )


@router.post("/eval/run", response_model=EvalRunResponse)
def eval_run(req: EvalRunRequest, _=Depends(require_service_key)) -> EvalRunResponse:
    results: list[dict] = []
    for case in req.cases:
        qreq = ProQueryRequest(
            query=case.question,
            top_k=req.top_k,
            retrieval_mode=req.retrieval_mode,
            retrieval_lanes=case.retrieval_lanes,
            generate_answer=req.generate_answers,
            return_trace=False,
        )
        qres = pro_query(qreq, _)
        scored = score_retrieval_case(case.model_dump(), qres.citations, req.top_k)
        if req.generate_answers:
            scored["answer"] = qres.answer
        results.append(scored)

    summary = summarize_eval(results)
    thresholds = default_thresholds()
    ok, checks = gate_summary(summary, thresholds)
    summary["gate_checks"] = checks

    if req.fail_on_threshold and not ok:
        raise HTTPException(status_code=422, detail={"summary": summary, "thresholds": thresholds, "cases": results})

    return EvalRunResponse(ok=ok, summary=summary, thresholds=thresholds, cases=results)


@router.get("/capabilities")
def capabilities(_=Depends(require_service_key)) -> dict:
    return {
        "retrieval_modes": ["vector", "sparse", "hybrid", "hybrid_rerank"],
        "fusion": "reciprocal_rank_fusion",
        "sparse_search": "in-process BM25 over Qdrant payload chunks with lane metadata filtering",
        "product": "Fünf RAG",
        "lanes": ["research", "technical_docs", "policy_compliance", "product_business", "custom", "chemrag_demo"],
        "reranker": "sentence-transformers CrossEncoder when enabled; lexical cross-encoder fallback otherwise",
        "evaluation_metrics": ["recall@k", "MRR", "nDCG@k", "no_citation_rate"],
        "ci_gate_thresholds": default_thresholds(),
    }
