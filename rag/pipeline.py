from __future__ import annotations

from typing import Optional, List, Dict, Any
import uuid

from app.settings import settings
from .embed_router import embed_texts
from rag.qdrant_store import get_client, ensure_collection, upsert_chunks, search as qdrant_search, search_multiple
from .langchain_loaders import load_pdf_pages, split_documents
from rag.collections import normalize_lane, get_collection_for_lane, normalize_lanes

_DOC_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _chunk_uuid(doc_id: str, page: int, chunk_index: int, lane: str) -> str:
    name = f"{lane}:{doc_id}:{page}:{chunk_index}"
    return str(uuid.uuid5(_DOC_NAMESPACE, name))


def _source_payload(
    *,
    title: str,
    source_uri: str | None,
    tags: list[str],
    content_type: str,
    rag_lane: str,
    module_relevance: list[str] | None,
    evidence_type: str | None,
    source_quality: str | None,
    artifact_versions: dict | None,
) -> dict:
    return {
        "title": title,
        "source_uri": source_uri,
        "tags": tags,
        "content_type": content_type,
        "rag_lane": rag_lane,
        "module_relevance": module_relevance or [],
        "evidence_type": evidence_type,
        "source_quality": source_quality,
        "artifact_versions": artifact_versions or {},
    }


def ingest_text_document(
    *,
    doc_id: str,
    title: str,
    text: str,
    source_uri: Optional[str],
    tags: Optional[List[str]] = None,
    rag_lane: str = "general",
    module_relevance: Optional[List[str]] = None,
    evidence_type: Optional[str] = "internal_note",
    source_quality: Optional[str] = "internal",
    artifact_versions: Optional[Dict[str, Any]] = None,
) -> dict:
    """Text ingestion with ChemRAG v2 lane-aware metadata."""
    tags = tags or []
    lane = normalize_lane(rag_lane)
    collection_name = get_collection_for_lane(lane)

    from langchain_core.documents import Document
    lc_docs = [Document(page_content=text, metadata={"page": 0})]

    source = _source_payload(
        title=title,
        source_uri=source_uri,
        tags=tags,
        content_type="text",
        rag_lane=lane,
        module_relevance=module_relevance,
        evidence_type=evidence_type,
        source_quality=source_quality,
        artifact_versions=artifact_versions,
    )

    chunks = split_documents(
        lc_docs,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        extra_metadata={
            "doc_id": doc_id,
            **source,
        },
    )
    if not chunks:
        return {"doc_id": doc_id, "chunks": 0, "rag_lane": lane, "collection": collection_name}

    texts = [c.page_content for c in chunks]
    vectors = embed_texts(texts)
    vector_size = len(vectors[0])

    client = get_client()
    ensure_collection(client, vector_size, collection_name=collection_name)

    ids: list[str] = []
    payloads: list[dict] = []
    for i, ch in enumerate(chunks):
        page = int(ch.metadata.get("page", 0))
        chunk_id = _chunk_uuid(doc_id, page, i, lane)
        payloads.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "page": page,
                "chunk": ch.page_content,
                "rag_lane": lane,
                "module_relevance": module_relevance or [],
                "evidence_type": evidence_type,
                "source_quality": source_quality,
                "artifact_versions": artifact_versions or {},
                "source": source,
            }
        )
        ids.append(chunk_id)

    upsert_chunks(client, ids=ids, vectors=vectors, payloads=payloads, collection_name=collection_name)
    return {"doc_id": doc_id, "chunks": len(chunks), "vector_size": vector_size, "rag_lane": lane, "collection": collection_name}


def ingest_pdf_document(
    *,
    doc_id: str,
    title: str,
    pdf_path: str,
    source_uri: Optional[str],
    tags: Optional[List[str]] = None,
    max_pages: int | None = None,
    start_page: int = 0,
    rag_lane: str = "general",
    module_relevance: Optional[List[str]] = None,
    evidence_type: Optional[str] = "textbook",
    source_quality: Optional[str] = "secondary",
    artifact_versions: Optional[Dict[str, Any]] = None,
) -> dict:
    tags = tags or []
    lane = normalize_lane(rag_lane)
    collection_name = get_collection_for_lane(lane)

    pages = load_pdf_pages(pdf_path)
    total_pages = len(pages)

    if start_page < 0:
        start_page = 0
    if start_page > total_pages:
        start_page = total_pages

    if max_pages is None:
        end_page = total_pages
    else:
        if max_pages < 0:
            max_pages = 0
        end_page = min(total_pages, start_page + max_pages)

    pages = pages[start_page:end_page]
    if not pages:
        return {"doc_id": doc_id, "chunks": 0, "vector_size": 0, "pages_ingested": 0, "total_pages": total_pages, "start_page": start_page, "end_page": end_page, "max_pages": max_pages, "rag_lane": lane, "collection": collection_name}

    for i, p in enumerate(pages):
        p.metadata["page"] = start_page + i
        p.metadata["page_rel"] = i

    source = _source_payload(
        title=title,
        source_uri=source_uri,
        tags=tags,
        content_type="pdf",
        rag_lane=lane,
        module_relevance=module_relevance,
        evidence_type=evidence_type,
        source_quality=source_quality,
        artifact_versions=artifact_versions,
    )

    chunks = split_documents(
        pages,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        extra_metadata={"doc_id": doc_id, **source},
    )
    if not chunks:
        return {"doc_id": doc_id, "chunks": 0, "vector_size": 0, "pages_ingested": len(pages), "total_pages": total_pages, "start_page": start_page, "end_page": end_page, "max_pages": max_pages, "rag_lane": lane, "collection": collection_name}

    texts = [c.page_content for c in chunks]
    vectors = embed_texts(texts)
    if not vectors:
        return {"doc_id": doc_id, "chunks": 0, "vector_size": 0, "pages_ingested": len(pages), "total_pages": total_pages, "start_page": start_page, "end_page": end_page, "max_pages": max_pages, "rag_lane": lane, "collection": collection_name}

    vector_size = len(vectors[0])
    client = get_client()
    ensure_collection(client, vector_size, collection_name=collection_name)

    ids: list[str] = []
    payloads: list[dict] = []
    for i, ch in enumerate(chunks):
        page = int(ch.metadata.get("page", 0))
        chunk_id = _chunk_uuid(doc_id, page, i, lane)
        ids.append(chunk_id)
        payloads.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "page": page,
                "chunk": ch.page_content,
                "rag_lane": lane,
                "module_relevance": module_relevance or [],
                "evidence_type": evidence_type,
                "source_quality": source_quality,
                "artifact_versions": artifact_versions or {},
                "source": source,
            }
        )

    upsert_chunks(client, ids=ids, vectors=vectors, payloads=payloads, collection_name=collection_name)
    return {"doc_id": doc_id, "chunks": len(chunks), "vector_size": vector_size, "pages_ingested": len(pages), "total_pages": total_pages, "start_page": start_page, "end_page": end_page, "max_pages": max_pages, "rag_lane": lane, "collection": collection_name}


def retrieve_context(
    *,
    query: str,
    top_k: int,
    doc_id: Optional[str] = None,
    lane: str | None = None,
    lanes: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    qvec = embed_texts([query])[0]
    client = get_client()

    if lanes:
        return search_multiple(client, query_vector=qvec, top_k=top_k, lanes=normalize_lanes(lanes), doc_id=doc_id, filters=filters)

    selected_lane = normalize_lane(lane or "general")
    return qdrant_search(client, query_vector=qvec, top_k=top_k, doc_id=doc_id, lane=selected_lane, filters=filters)
