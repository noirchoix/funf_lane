from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.deps import require_service_key
from app.schemas import QueryRequest, QueryResponse
from app.settings import settings
from rag.balanced_retriever import (
    enforce_lane_balance_after_rerank,
    retrieve_balanced_context,
)
from rag.collections import normalize_lanes
from rag.intent import classify_intent, map_intent_to_lanes
from rag.llm_router import generate_answer
from rag.retrieval_enhancements import expand_queries, rerank_chunks

logger = logging.getLogger("chemrag.query")
router = APIRouter(prefix="/v1/query", tags=["query"])


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


def _degradation_anchor_queries(user_query: str) -> list[str]:
    """
    High-precision anchor queries for degradation/stability/product-base questions.

    These are used only to retrieve known class-level instability evidence that
    broad semantic retrieval may miss.
    """
    if not _is_degradation_query(user_query):
        return []

    return [
        "acidic antiperspirant base linalool linalyl acetate phenolic materials reactive aldehydes essential oils citrus oils spices undergo chemical reactions",
        "empirical testing antiperspirant bases unstable perfumery ingredients phenolic materials linalool linalyl acetate reactive aldehydes citrus oils spices",
        "unsaturated terpene alcohols esters linalool linalyl acetate acidic antiperspirant active chemical reactions",
        "aerosol antiperspirant unstable perfumery ingredients citrus oils essential oils phenolic materials reactive aldehydes",
    ]


def _add_degradation_expansions(user_query: str, queries: list[str]) -> list[str]:
    """
    Targeted recall booster for degradation / stability / product-base questions.

    This broadens semantic retrieval vocabulary without changing scoring or answer logic.
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


def _select_degradation_anchors(
    user_query: str,
    hits: list[dict],
    max_anchors: int = 2,
) -> list[dict]:
    """
    Preserve high-value class-level instability evidence for degradation/stability queries.

    This is intentionally narrow. It protects chunks containing the exact formulation-
    instability evidence around acidic antiperspirant/product-base instability.
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


def _merge_anchors_with_final_hits(
    anchors: list[dict],
    final_hits: list[dict],
    top_k: int,
) -> list[dict]:
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

        lane = h.get("rag_lane") or src.get("rag_lane") or "unknown"
        collection = h.get("collection") or "unknown_collection"
        evidence_type = h.get("evidence_type") or src.get("evidence_type")
        source_quality = h.get("source_quality") or src.get("source_quality")
        module_relevance = h.get("module_relevance") or src.get("module_relevance") or []

        context_blocks.append(
            "\n".join(
                [
                    f"[SOURCE lane={lane} collection={collection} title={src.get('title')} page={page_display} score={h.get('score')}]",
                    f"evidence_type={evidence_type} source_quality={source_quality} module_relevance={module_relevance}",
                    f"uri={src.get('source_uri')}",
                    chunk,
                ]
            )
        )

    context = "\n---\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    return f"""You are ChemRAG, a chemistry and formulation reference assistant.

Use ONLY the provided SOURCES for factual claims.

USER QUESTION:
{user_query}

SOURCES:
{context}

ANSWER RULES:
1. Use only the provided SOURCES for factual chemistry, formulation, safety, stability, QC, note, ingredient, or vocabulary claims.
2. Cite sources by title + page/section where available, and mention lane if helpful.
3. Do not invent mechanisms, controls, analytical methods, stabilizers, additives, solvents, test conditions, regulatory rules, or formulation protocols.

4. Distinguish evidence strength:
   A. Direct mechanism evidence:
      - A source gives a specific pathway, reaction, intermediate, catalyst, condition, radical, ion, product, named-compound transformation, or measured degradation/stability result.
      - Example: aldehyde loss in an ethanol-based acidic antiperspirant, acetal formation, aldehyde autoxidation, or a named stability comparison.

   B. Class-level instability or risk evidence:
      - A source says a material class is unstable, reactive, oxidizable, hydrolysable, phototoxic, sensitizing, restricted, or formulation-sensitive without giving a full mechanism for every named molecule.
      - Example: a source says unsaturated terpene alcohols and esters such as linalool and linalyl acetate, citrus oils, essential oils, phenolic materials, spices, or reactive aldehydes can undergo chemical reactions in acidic antiperspirant bases.

   C. Directly retrieved formulation/QC controls:
      - You may name only controls, substitutions, tests, materials, temperatures, packaging checks, evaluation methods, warning labels, or restrictions explicitly stated in the SOURCES.
      - If the source says "avoid these materials," say avoid.
      - If the source says "storage testing in production specification cans," say that.
      - If the source says "evaluate with the finished formulation and valve system," say that.
      - If the source gives exact temperatures, report only those retrieved temperatures.
      - If the source gives a suggested warning label, you may summarize it.

   D. Evidence-bounded general implications:
      - Do not create a separate "Reasonable implications" section.
      - Prefer directly retrieved controls.
      - General implications may appear only as short generic wording inside "Next actions" when needed.
      - Acceptable generic phrasing:
        * run product-base compatibility testing
        * screen substitutions for unstable classes
        * monitor parent-material loss where parent-loss evidence is retrieved
        * monitor odour, colour, discoloration, precipitation, separation, viscosity, corrosion, or appearance where retrieved
        * request targeted mechanism evidence for molecule-specific pathways
      - Do NOT name solvent replacement, pH management, antioxidant screening, chelator use, water-level control, packaging screening, analytical methods, precise storage protocols, phototoxicity mitigation, regulatory thresholds, or predicted degradants unless those exact concepts appear in the SOURCES.

   E. Missing details:
      - If the source gives class-level evidence but not a molecule-specific mechanism, do NOT say the evidence is absent.
      - Instead say: "The retrieved sources provide class-level evidence, but not a full molecule-specific mechanism."
      - If a specific mitigation, analytical method, antioxidant, chelator, solvent, packaging material, regulatory threshold, or test protocol is not retrieved, say it is not specified in the retrieved evidence.

5. For degradation, stability, QC, safety, or formulation-control questions, structure the answer as:
   - Direct answer
   - Direct mechanism evidence
   - Class-level instability or risk evidence
   - Directly retrieved formulation/QC controls
   - Evidence limits / missing details
   - Next actions

6. Next-action rules:
   - Next actions must be evidence-bounded.
   - Prefer generic operational actions unless the SOURCES name a specific method.
   - Do not add examples in parentheses unless those examples appear in the SOURCES.
   - Do not introduce new test variables, such as pH, light exposure, storage temperature, solvent replacement, antioxidant screening, chelator use, analytical methods, packaging screening, phototoxicity mitigation, IFRA thresholds, or predicted degradants, unless those exact items appear in the SOURCES.
   - Allowed generic next actions include:
     * run product-base compatibility testing
     * screen substitutions for unstable classes
     * monitor parent-material loss where parent-loss evidence is retrieved
     * monitor odour, colour, discoloration, precipitation, separation, viscosity, corrosion, or appearance where retrieved
     * request targeted mechanism evidence for molecule-specific pathways

7. For internal note/vocabulary questions:
   - Treat datasets and internal notes as vocabulary evidence, not chemical mechanism evidence.
   - Do not infer safety, stability, or mechanism from note/family vocabulary alone.
   - Separate note/family/accord vocabulary from chemistry or QC evidence.

8. Do not overclaim:
   - Say "the sources directly show..." only for direct evidence.
   - Say "the sources support a class-level concern..." for class-level evidence.
   - Say "a retrieved control is..." only for controls explicitly present in the SOURCES.
   - If retrieved evidence is insufficient, say exactly what is missing.
"""


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest, _=Depends(require_service_key)) -> QueryResponse:
    intent = classify_intent(req.query)

    selected_lanes = req.retrieval_lanes or (
        [req.rag_lane] if req.rag_lane else map_intent_to_lanes(intent)
    )
    selected_lanes = normalize_lanes(selected_lanes or ["general"])

    queries = [req.query]
    if settings.ENABLE_QUERY_EXPANSION:
        try:
            queries = expand_queries(req.query)
        except Exception as e:
            logger.exception("Query expansion failed; using original query. err=%r", e)
            queries = [req.query]

    queries = _add_degradation_expansions(req.query, queries)

    candidate_hits: list[dict] = []
    lane_quotas: dict[str, int] = {}
    final_hits: list[dict] = []

    try:
        candidate_hits, lane_quotas, selected_lanes = retrieve_balanced_context(
            queries=queries,
            top_k=req.top_k,
            doc_id=req.doc_id,
            lanes=selected_lanes,
        )

        anchor_queries = _degradation_anchor_queries(req.query)
        if anchor_queries:
            try:
                anchor_hits, _, _ = retrieve_balanced_context(
                    queries=anchor_queries,
                    top_k=min(8, max(4, req.top_k // 2)),
                    doc_id=req.doc_id,
                    lanes=selected_lanes,
                )
                candidate_hits = _merge_hits(anchor_hits, candidate_hits)
            except Exception as e:
                logger.exception(
                    "Degradation anchor retrieval failed; continuing without anchors. err=%r",
                    e,
                )

        logger.info(
            "Candidate hits before rerank: %s",
            [
                {
                    "doc_id": h.get("doc_id"),
                    "page": h.get("page"),
                    "lane": h.get("rag_lane") or (h.get("source") or {}).get("rag_lane"),
                    "score": h.get("score"),
                    "preview": (h.get("chunk") or "")[:180],
                }
                for h in candidate_hits[:30]
            ],
        )

        anchors = _select_degradation_anchors(req.query, candidate_hits)
        ranked_hits = rerank_chunks(req.query, candidate_hits) if candidate_hits else []

        final_hits = enforce_lane_balance_after_rerank(
            ranked_hits=ranked_hits,
            candidate_hits=candidate_hits,
            lane_quotas=lane_quotas,
            top_k=req.top_k,
        )

        final_hits = _merge_anchors_with_final_hits(
            anchors=anchors,
            final_hits=final_hits,
            top_k=req.top_k,
        )

    except Exception as e:
        logger.exception("Balanced retrieval failed. err=%r", e)
        final_hits = []

    provider = None
    try:
        answer, provider = generate_answer(_build_prompt(req.query, final_hits))
    except Exception as e:
        logger.exception("LLM generation failed; falling back. err=%r", e)
        provider = "extractive_fallback"

        if not final_hits:
            answer = "No relevant context was retrieved from the selected vector lanes for this question."
        else:
            snippets = [
                (h.get("chunk") or "").strip()[:500]
                for h in final_hits[:5]
                if (h.get("chunk") or "").strip()
            ]
            answer = (
                "Relevant excerpts:\n\n" + "\n\n---\n\n".join(snippets)
                if snippets
                else "Relevant context was retrieved, but no usable text snippets were available."
            )

    return QueryResponse(
        answer=answer,
        citations=final_hits[: req.top_k],
        provider=provider,
        selected_lanes=selected_lanes,
    )