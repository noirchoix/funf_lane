from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.pipeline import ingest_text_document


SECTION_LANE_MAP: dict[str, str] = {
    "reaction_framework_summary": "phytochemistry_context",
    "rxnutils_evidence": "reaction_orgchem",
    "reaction_curation_context": "reaction_orgchem",
    "dess_physics_summary": "physical_chem",
    "taxonomy_predictions": "phytochemistry_context",
    "fooddb_ingredient_chemistry_profile": "phytochemistry_context",
    "formulation_engine_summary": "quality_control",
    "formulation_qc": "quality_control",
    "internal_notes": "internal_notes",
}

SECTION_MODULE_MAP: dict[str, list[str]] = {
    "reaction_framework_summary": ["reaction_framework"],
    "rxnutils_evidence": ["rxnutils", "reaction_framework"],
    "reaction_curation_context": ["reaction_curation"],
    "dess_physics_summary": ["dess_physics"],
    "taxonomy_predictions": ["coco_taxonomy"],
    "fooddb_ingredient_chemistry_profile": ["fooddb"],
    "formulation_engine_summary": ["formulation_engine"],
    "formulation_qc": ["formulation_engine", "quality_control"],
    "internal_notes": ["internal_notes"],
}


def load_phytochemistry_rag_export(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Phytochemistry RAG export must be a JSON object at the top level")
    return data


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _detect_artifact_versions(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ["artifact_versions", "artifacts", "artifact_context", "artifact_registry_snapshot"]:
        val = payload.get(key)
        if isinstance(val, dict):
            return val
    return {}


def split_phytochemistry_context(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Split latest_subsystem_rag_export.json into lane-aware ingest records."""
    artifact_versions = _detect_artifact_versions(payload)
    records: list[dict[str, Any]] = []

    for key, value in payload.items():
        if key in {"artifact_versions", "artifact_context", "artifact_registry_snapshot", "metadata"}:
            continue
        text = _to_text(value).strip()
        if not text:
            continue
        lane = SECTION_LANE_MAP.get(key, "phytochemistry_context")
        records.append(
            {
                "section": key,
                "title": f"Phytochemistry subsystem export: {key}",
                "text": f"SECTION: {key}\n\n{text}",
                "rag_lane": lane,
                "module_relevance": SECTION_MODULE_MAP.get(key, ["phytochemistry_context"]),
                "evidence_type": "artifact_note",
                "source_quality": "runtime",
                "artifact_versions": artifact_versions,
            }
        )
    return records


def ingest_phytochemistry_context(
    *,
    path: str | Path,
    project_id: str | None = None,
    tags: list[str] | None = None,
    artifact_versions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_phytochemistry_rag_export(path)
    records = split_phytochemistry_context(payload)
    tags = tags or []
    extra_versions = artifact_versions or {}

    results: list[dict[str, Any]] = []
    stem = Path(path).stem
    for idx, rec in enumerate(records):
        merged_versions = {**(rec.get("artifact_versions") or {}), **extra_versions}
        doc_id = f"{project_id or 'phyto'}:{stem}:{rec['section']}:{idx}"
        result = ingest_text_document(
            doc_id=doc_id,
            title=rec["title"],
            text=rec["text"],
            source_uri=str(path),
            tags=["phytochemistry_export", rec["section"], *tags],
            rag_lane=rec["rag_lane"],
            module_relevance=rec["module_relevance"],
            evidence_type=rec["evidence_type"],
            source_quality=rec["source_quality"],
            artifact_versions=merged_versions,
        )
        results.append(result)

    return {"ok": True, "path": str(path), "records": len(records), "results": results}
