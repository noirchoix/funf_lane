from fastapi import APIRouter, Depends

from app.deps import require_service_key
from app.schemas import IngestRequest
from rag.pipeline import ingest_text_document

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post("")
def ingest(req: IngestRequest, _=Depends(require_service_key)):
    """Text-based ingestion endpoint with ChemRAG v2 lane metadata."""
    return ingest_text_document(
        doc_id=req.doc_id,
        title=req.title,
        text=req.text,
        source_uri=req.source_uri,
        tags=req.tags,
        rag_lane=req.rag_lane,
        module_relevance=req.module_relevance,
        evidence_type=req.evidence_type,
        source_quality=req.source_quality,
        artifact_versions=req.artifact_versions,
    )
