from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.deps import require_service_key
from app.settings import settings
from rag.collections import get_lane_configs, normalize_lane
from rag.pipeline import ingest_pdf_document, ingest_text_document
from routes.pro import pro_query
from app.pro_schemas import ProQueryRequest, RetrievalMode

router = APIRouter(prefix="/v1/funf", tags=["funf-rag"])


class LaneUpdate(BaseModel):
    lane: str
    name: Optional[str] = None
    description: Optional[str] = None


class FunfChatRequest(BaseModel):
    query: str
    selected_lanes: list[str] = Field(default_factory=lambda: ["research"])
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = 8
    fetch_k: Optional[int] = None
    generate_answer: bool = True
    return_trace: bool = True


def _lane_catalog() -> list[dict]:
    configs = get_lane_configs()
    defaults = [
        ("research", "Research", "Papers, PDFs, source-heavy notes and evidence documents."),
        ("technical_docs", "Technical Docs", "READMEs, API specs, architecture docs and runbooks."),
        ("policy_compliance", "Policy / Compliance", "Policies, procedures, standards and governance documents."),
        ("product_business", "Product / Business", "Requirements, strategy, market and operating documents."),
        ("custom", "Custom", "A flexible fifth lane for project-specific knowledge."),
        ("chemrag_demo", "ChemRAG Demo", "Optional seeded scientific/demo corpus from the original ChemRAG app."),
    ]
    out = []
    for lane, name, fallback_description in defaults:
        cfg = configs.get(lane)
        out.append({
            "lane": lane,
            "name": name,
            "description": cfg.description if cfg else fallback_description,
            "collection": cfg.collection_name if cfg else settings.RAG_COLLECTION_FUNF,
            "editable": lane != "chemrag_demo",
        })
    return out


@router.get("/lanes")
def lanes(_=Depends(require_service_key)) -> dict:
    return {"product": "Fünf RAG", "lanes": _lane_catalog()}


@router.post("/upload")
async def upload_to_lane(
    lane: str = Form(default="research"),
    title: str = Form(...),
    doc_id: str | None = Form(default=None),
    source_uri: str | None = Form(default=None),
    tags: str = Form(default=""),
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    _=Depends(require_service_key),
):
    norm_lane = normalize_lane(lane)
    clean_tags = [x.strip() for x in (tags or "").split(",") if x.strip()]
    safe_doc_id = doc_id or f"{norm_lane}_{uuid.uuid4().hex[:12]}"

    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / ("upload" + suffix)
            with fpath.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            mb = fpath.stat().st_size / (1024 * 1024)
            if mb > settings.MAX_UPLOAD_MB:
                raise HTTPException(status_code=400, detail=f"File too large ({mb:.1f} MB). Max is {settings.MAX_UPLOAD_MB} MB.")

            if suffix == ".pdf":
                return ingest_pdf_document(
                    doc_id=safe_doc_id,
                    title=title,
                    pdf_path=str(fpath),
                    source_uri=source_uri,
                    tags=clean_tags,
                    rag_lane=norm_lane,
                    evidence_type="uploaded_document",
                    source_quality="user_uploaded",
                )
            if suffix in {".txt", ".md", ".markdown"}:
                body = fpath.read_text(encoding="utf-8", errors="ignore")
                return ingest_text_document(
                    doc_id=safe_doc_id,
                    title=title,
                    text=body,
                    source_uri=source_uri,
                    tags=clean_tags,
                    rag_lane=norm_lane,
                    evidence_type="uploaded_document",
                    source_quality="user_uploaded",
                )
            raise HTTPException(status_code=400, detail="Supported uploads: .pdf, .txt, .md")

    if text and text.strip():
        return ingest_text_document(
            doc_id=safe_doc_id,
            title=title,
            text=text,
            source_uri=source_uri,
            tags=clean_tags,
            rag_lane=norm_lane,
            evidence_type="pasted_note",
            source_quality="user_uploaded",
        )

    raise HTTPException(status_code=400, detail="Upload a PDF/TXT/MD file or paste text to index a lane.")


@router.post("/chat")
def chat(req: FunfChatRequest, _=Depends(require_service_key)):
    qreq = ProQueryRequest(
        query=req.query,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        retrieval_lanes=req.selected_lanes,
        retrieval_mode=req.retrieval_mode,
        generate_answer=req.generate_answer,
        return_trace=req.return_trace,
    )
    return pro_query(qreq, _)
