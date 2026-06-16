from __future__ import annotations

from typing import Literal

Intent = Literal[
    "reaction_mechanism",
    "formulation_qc",
    "recommendation",
    "physical_chemistry",
    "phytochemical_context",
    "artifact_explanation",
    "memory_recall",
    "general_chat",
]


def classify_intent(query: str) -> Intent:
    q = (query or "").strip().lower()

    if any(w in q for w in ["last time", "previously", "what did we", "why did we", "remember", "accepted", "rejected"]):
        return "memory_recall"

    if any(w in q for w in ["artifact", "version", "registry", "score", "model output", "compute result", "why did the system"]):
        return "artifact_explanation"

    if any(w in q for w in ["mechanism", "reaction", "oxidation", "reduction", "esterification", "hydrolysis", "radical", "rxn", "template"]):
        return "reaction_mechanism"

    if any(w in q for w in ["thermodynamics", "kinetics", "volatility", "vapor pressure", "polarity", "solubility", "miscibility", "partition", "dess", "molecular physics", "intermolecular"]):
        return "physical_chemistry"

    if any(w in q for w in ["qc", "quality", "stability", "safety", "irritation", "sensitization", "ifra", "regulatory", "storage", "shelf"]):
        return "formulation_qc"

    if any(w in q for w in ["substitute", "recommend", "replacement", "alternative", "better ingredient", "blend", "formulate", "accord", "top note", "middle note", "base note"]):
        return "recommendation"

    if any(w in q for w in ["phytochemical", "plant", "botanical", "terpene", "flavonoid", "phenylpropanoid", "coco", "taxonomy", "fooddb", "essential oil"]):
        return "phytochemical_context"

    return "general_chat"


INTENT_LANE_MAP: dict[str, list[str]] = {
    "reaction_mechanism": ["reaction_orgchem", "physical_chem", "phytochemistry_context"],
    "formulation_qc": ["quality_control", "physical_chem", "phytochemistry_context", "internal_notes"],
    "recommendation": ["internal_notes", "quality_control", "phytochemistry_context", "physical_chem"],
    "physical_chemistry": ["physical_chem", "reaction_orgchem", "phytochemistry_context"],
    "phytochemical_context": ["phytochemistry_context", "physical_chem", "reaction_orgchem"],
    "artifact_explanation": ["phytochemistry_context", "quality_control", "physical_chem", "reaction_orgchem"],
    "memory_recall": [],
    "general_chat": ["phytochemistry_context", "reaction_orgchem", "physical_chem", "internal_notes"],
}


def map_intent_to_lanes(intent: str) -> list[str]:
    return list(INTENT_LANE_MAP.get(intent, INTENT_LANE_MAP["general_chat"]))
