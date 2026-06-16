from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from app.deps import require_service_key
from app.settings import settings
from rag.pipeline import ingest_pdf_document

router = APIRouter(prefix="/v1/ingest/pdf", tags=["ingest"])


def _csv(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _json_obj(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@router.post("")
async def ingest_pdf(
    doc_id: str = Form(...),
    title: str = Form(...),
    source_uri: str | None = Form(default=None),
    tags: str = Form(default=""),
    max_pages: int | None = Form(default=None),
    start_page: int = Form(default=0),
    rag_lane: str = Form(default="general"),
    module_relevance: str = Form(default=""),
    evidence_type: str = Form(default="textbook"),
    source_quality: str = Form(default="secondary"),
    artifact_versions: str | None = Form(default=None),
    file: UploadFile = File(...),
    _=Depends(require_service_key),
):
    if start_page < 0:
        raise HTTPException(status_code=400, detail="start_page must be >= 0")
    if max_pages is not None and max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be > 0")

    suffix = Path(file.filename or "doc.pdf").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf supported")

    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "upload.pdf"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        mb = pdf_path.stat().st_size / (1024 * 1024)
        if mb > settings.MAX_UPLOAD_MB:
            raise HTTPException(status_code=400, detail=f"File too large ({mb:.1f} MB). Max is {settings.MAX_UPLOAD_MB} MB.")

        return ingest_pdf_document(
            doc_id=doc_id,
            title=title,
            pdf_path=str(pdf_path),
            source_uri=source_uri,
            tags=_csv(tags),
            max_pages=max_pages,
            start_page=start_page,
            rag_lane=rag_lane,
            module_relevance=_csv(module_relevance),
            evidence_type=evidence_type,
            source_quality=source_quality,
            artifact_versions=_json_obj(artifact_versions),
        )
