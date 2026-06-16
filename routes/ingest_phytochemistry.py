from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import require_service_key
from app.schemas import PhytochemistryExportIngestRequest
from rag.phytochemistry_context import ingest_phytochemistry_context

router = APIRouter(prefix="/v1/ingest/phytochemistry-export", tags=["ingest"])


@router.post("")
def ingest_export(req: PhytochemistryExportIngestRequest, _=Depends(require_service_key)):
    """Ingest structured latest_subsystem_rag_export.json into lane-routed collections."""
    return ingest_phytochemistry_context(
        path=req.path,
        project_id=req.project_id,
        tags=req.tags,
        artifact_versions=req.artifact_versions,
    )
