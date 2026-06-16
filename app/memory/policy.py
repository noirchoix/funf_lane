from __future__ import annotations

from app.settings import settings


DEFAULT_IMPORTANCE = {
    "decision": 4,
    "deterministic": 3,
    "ml_output": 3,
    "system": 2,
    "summary": 5,
    # ChemRAG v2 artifact-aware event vocabulary
    "formulation_score": 4,
    "recommendation_generated": 4,
    "recommendation_accepted": 5,
    "recommendation_rejected": 4,
    "reaction_hypothesis": 4,
    "stability_warning": 5,
    "taxonomy_fallback_used": 3,
    "fooddb_similarity_used": 4,
    "dess_physics_support": 4,
    "calibration_update": 5,
    "anchor_formulation_selected": 5,
    "artifact_explanation": 3,
}

ALLOWED_EVENT_TYPES = set(DEFAULT_IMPORTANCE.keys())


def normalize_event_type(event_type: str) -> str:
    et = (event_type or "").strip().lower()
    if et not in ALLOWED_EVENT_TYPES:
        valid = ", ".join(sorted(ALLOWED_EVENT_TYPES))
        raise ValueError(f"Unsupported event_type: {event_type!r}. Valid event types: {valid}")
    return et


def compute_importance(event_type: str, importance: int | None) -> int:
    if importance is None:
        importance = DEFAULT_IMPORTANCE.get(event_type, 3)
    importance = int(importance)
    if importance < 1:
        importance = 1
    if importance > 5:
        importance = 5
    return importance


def should_store(event_type: str, importance: int) -> bool:
    if not settings.MEMORY_ENABLED:
        return False
    return importance >= settings.MEMORY_MIN_IMPORTANCE
