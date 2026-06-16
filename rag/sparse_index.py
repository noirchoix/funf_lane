from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.settings import settings
from rag.collections import get_collection_for_lane, normalize_lane, normalize_lanes
from rag.qdrant_store import get_client

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-/\.]{1,}")


def tokenize(text: str) -> list[str]:
    min_len = int(getattr(settings, "RAG_PRO_SPARSE_MIN_TOKEN_LEN", 2))
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= min_len]


def _hit_key(hit: dict) -> str:
    return str(hit.get("chunk_id") or f"{hit.get('doc_id')}|{hit.get('page')}|{(hit.get('chunk') or '')[:120]}")


@dataclass
class SparseDocument:
    key: str
    payload: Dict[str, Any]
    tokens: List[str]


class BM25Index:
    def __init__(self, docs: list[SparseDocument], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freqs: list[Counter[str]] = []
        self.doc_lens: list[int] = []

        for doc in docs:
            tf = Counter(doc.tokens)
            self.term_freqs.append(tf)
            self.doc_lens.append(len(doc.tokens))
            for term in tf:
                self.doc_freq[term] += 1

        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.N = len(docs)

    def score(self, query: str) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self.docs:
            return []

        q_counts = Counter(q_terms)
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self.term_freqs):
            dl = self.doc_lens[idx] or 1
            score = 0.0
            for term, qf in q_counts.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                score += idf * ((f * (self.k1 + 1)) / denom) * max(1, qf)
            if score > 0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def _build_scroll_filter(doc_id: Optional[str], filters: Optional[Dict[str, Any]]) -> Filter | None:
    must: list[FieldCondition] = []
    if doc_id:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    for key, value in (filters or {}).items():
        if value is None:
            continue
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must) if must else None


def _payload_to_hit(payload: Dict[str, Any], *, collection: str, score: float, lane: str) -> Dict[str, Any]:
    source = payload.get("source", {}) or {}
    return {
        "score": float(score),
        "sparse_score": float(score),
        "chunk_id": payload.get("chunk_id"),
        "doc_id": payload.get("doc_id"),
        "page": payload.get("page"),
        "chunk": payload.get("chunk", ""),
        "source": source,
        "rag_lane": payload.get("rag_lane") or source.get("rag_lane") or lane,
        "collection": collection,
        "module_relevance": payload.get("module_relevance") or source.get("module_relevance"),
        "evidence_type": payload.get("evidence_type") or source.get("evidence_type"),
        "source_quality": payload.get("source_quality") or source.get("source_quality"),
        "artifact_versions": payload.get("artifact_versions") or source.get("artifact_versions"),
        "retrieval_stage": "sparse",
    }


def load_sparse_documents(
    *,
    lanes: Iterable[str],
    doc_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    max_docs_per_lane: Optional[int] = None,
) -> list[SparseDocument]:
    client = get_client()
    docs: list[SparseDocument] = []
    seen: set[str] = set()
    limit = int(max_docs_per_lane or settings.RAG_PRO_SPARSE_MAX_DOCS_PER_LANE)

    for lane in normalize_lanes(lanes):
        lane = normalize_lane(lane)
        collection = get_collection_for_lane(lane)
        lane_filters = dict(filters or {})
        lane_filters["rag_lane"] = lane
        scroll_filter = _build_scroll_filter(doc_id, lane_filters)
        offset = None
        loaded = 0
        while loaded < limit:
            try:
                points, offset = client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=min(256, limit - loaded),
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
            except Exception:
                break

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                chunk = payload.get("chunk") or ""
                if not chunk.strip():
                    continue
                hit = _payload_to_hit(payload, collection=collection, score=0.0, lane=lane)
                key = _hit_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(SparseDocument(key=key, payload=hit, tokens=tokenize(chunk)))
                loaded += 1
                if loaded >= limit:
                    break
            if offset is None:
                break

    return docs


def sparse_search(
    *,
    query: str,
    top_k: int,
    lanes: Iterable[str],
    doc_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> list[dict]:
    docs = load_sparse_documents(lanes=lanes, doc_id=doc_id, filters=filters)
    index = BM25Index(docs)
    scored = index.score(query)[:top_k]
    hits: list[dict] = []
    for idx, score in scored:
        hit = dict(docs[idx].payload)
        hit["score"] = float(score)
        hit["sparse_score"] = float(score)
        hit["retrieval_stage"] = "sparse"
        hits.append(hit)
    return hits
