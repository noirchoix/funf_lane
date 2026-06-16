from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from app.settings import settings


@dataclass(frozen=True)
class RagLaneConfig:
    lane: str
    collection_name: str
    description: str
    default_top_k: int = 8
    rerank: bool = True


LANE_ALIASES = {
    # Fünf RAG generic lanes
    "research": "research",
    "lane_1": "research",
    "technical_docs": "technical_docs",
    "technical": "technical_docs",
    "docs": "technical_docs",
    "lane_2": "technical_docs",
    "policy_compliance": "policy_compliance",
    "policy": "policy_compliance",
    "compliance": "policy_compliance",
    "lane_3": "policy_compliance",
    "product_business": "product_business",
    "product": "product_business",
    "business": "product_business",
    "lane_4": "product_business",
    "custom": "custom",
    "lane_5": "custom",

    # Seeded / backward-compatible ChemRAG lanes
    "demo": "chemrag_demo",
    "chemrag": "chemrag_demo",
    "phytochemistry_demo": "chemrag_demo",
    "phyto": "chemrag_demo",
    "phytochemistry": "chemrag_demo",
    "phytochemistry_context": "chemrag_demo",
    "rag_phytochemistry_context": "chemrag_demo",
    "reaction": "chemrag_demo",
    "reaction_orgchem": "chemrag_demo",
    "organic": "chemrag_demo",
    "organic_chemistry": "chemrag_demo",
    "rag_reaction_orgchem": "chemrag_demo",
    "qc": "chemrag_demo",
    "quality": "chemrag_demo",
    "quality_control": "chemrag_demo",
    "rag_quality_control": "chemrag_demo",
    "physical": "chemrag_demo",
    "physical_chem": "chemrag_demo",
    "physical_chemistry": "chemrag_demo",
    "rag_physical_chem": "chemrag_demo",
    "internal": "chemrag_demo",
    "internal_notes": "chemrag_demo",
    "notes": "chemrag_demo",
    "perfumery_notes": "chemrag_demo",
    "rag_internal_notes": "chemrag_demo",
    "general": "research",
    "chem_docs_v1": "research",
}


def get_lane_configs() -> dict[str, RagLaneConfig]:
    funf_collection = settings.RAG_COLLECTION_FUNF
    return {
        "research": RagLaneConfig(
            lane="research",
            collection_name=funf_collection,
            description="Research papers, PDFs, literature notes, evidence packs, and source-heavy documents.",
        ),
        "technical_docs": RagLaneConfig(
            lane="technical_docs",
            collection_name=funf_collection,
            description="Engineering docs, READMEs, API specs, runbooks, system manuals, and implementation notes.",
        ),
        "policy_compliance": RagLaneConfig(
            lane="policy_compliance",
            collection_name=funf_collection,
            description="Policies, compliance documents, legal standards, procedures, and governance evidence.",
        ),
        "product_business": RagLaneConfig(
            lane="product_business",
            collection_name=funf_collection,
            description="Product strategy, business plans, requirements, market notes, roadmaps, and operating documents.",
        ),
        "custom": RagLaneConfig(
            lane="custom",
            collection_name=funf_collection,
            description="User-defined fifth lane for any document corpus that should remain separately selectable.",
        ),
        "chemrag_demo": RagLaneConfig(
            lane="chemrag_demo",
            collection_name=settings.RAG_COLLECTION_GENERAL,
            description="Optional seeded ChemRAG scientific/demo corpus retained for backward compatibility.",
        ),
    }


def normalize_lane(lane: str | None) -> str:
    if not lane:
        return "general"
    key = lane.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in LANE_ALIASES:
        valid = ", ".join(sorted(set(LANE_ALIASES.values())))
        raise ValueError(f"Unknown RAG lane {lane!r}. Valid lanes: {valid}")
    return LANE_ALIASES[key]


def get_collection_for_lane(lane: str | None) -> str:
    norm = normalize_lane(lane)
    return get_lane_configs()[norm].collection_name


def normalize_lanes(lanes: Iterable[str] | None) -> list[str]:
    if not lanes:
        return ["general"]
    out: list[str] = []
    seen: set[str] = set()
    for lane in lanes:
        norm = normalize_lane(lane)
        if norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out or ["general"]


def collection_names_for_lanes(lanes: Iterable[str] | None) -> list[str]:
    return [get_collection_for_lane(lane) for lane in normalize_lanes(lanes)]
