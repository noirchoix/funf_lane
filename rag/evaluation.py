from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.settings import settings


def load_golden_jsonl(path: str | Path) -> list[dict]:
    cases: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def _ids(citations: list[dict], field: str) -> list[str]:
    out: list[str] = []
    for c in citations:
        v = c.get(field)
        if v is not None:
            out.append(str(v))
    return out


def _first_relevant_rank(citations: list[dict], expected_doc_ids: set[str], expected_chunk_ids: set[str]) -> int | None:
    for idx, c in enumerate(citations, start=1):
        doc_hit = str(c.get("doc_id")) in expected_doc_ids if c.get("doc_id") is not None else False
        chunk_hit = str(c.get("chunk_id")) in expected_chunk_ids if c.get("chunk_id") is not None else False
        if doc_hit or chunk_hit:
            return idx
    return None


def _dcg(relevances: list[int]) -> float:
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(relevances))


def score_retrieval_case(case: dict, citations: list[dict], top_k: int) -> dict:
    expected_doc_ids = {str(x) for x in case.get("expected_doc_ids", []) if str(x)}
    expected_chunk_ids = {str(x) for x in case.get("expected_chunk_ids", []) if str(x)}
    retrieved_doc_ids = _ids(citations, "doc_id")
    retrieved_chunk_ids = _ids(citations, "chunk_id")

    total_expected = len(expected_doc_ids) + len(expected_chunk_ids)
    if total_expected == 0:
        recall = 1.0 if citations else 0.0
    else:
        matched_docs = expected_doc_ids & set(retrieved_doc_ids)
        matched_chunks = expected_chunk_ids & set(retrieved_chunk_ids)
        recall = (len(matched_docs) + len(matched_chunks)) / total_expected

    first_rank = _first_relevant_rank(citations[:top_k], expected_doc_ids, expected_chunk_ids)
    rr = 1.0 / first_rank if first_rank else 0.0

    relevances: list[int] = []
    for c in citations[:top_k]:
        rel = int(
            (c.get("doc_id") is not None and str(c.get("doc_id")) in expected_doc_ids)
            or (c.get("chunk_id") is not None and str(c.get("chunk_id")) in expected_chunk_ids)
        )
        relevances.append(rel)
    ideal = sorted(relevances, reverse=True)
    ndcg = (_dcg(relevances) / _dcg(ideal)) if any(ideal) else (1.0 if total_expected == 0 and citations else 0.0)

    return {
        "id": case.get("id"),
        "question": case.get("question"),
        "passed_retrieval": bool(recall > 0),
        "recall_at_k": float(recall),
        "reciprocal_rank": float(rr),
        "ndcg_at_k": float(ndcg),
        "citation_count": len(citations),
        "first_relevant_rank": first_rank,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "missing_expected_doc_ids": sorted(expected_doc_ids - set(retrieved_doc_ids)),
        "missing_expected_chunk_ids": sorted(expected_chunk_ids - set(retrieved_chunk_ids)),
    }


def summarize_eval(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "case_count": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "no_citation_rate": 1.0,
        }
    return {
        "case_count": n,
        "recall_at_k": sum(r["recall_at_k"] for r in results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n,
        "ndcg_at_k": sum(r["ndcg_at_k"] for r in results) / n,
        "no_citation_rate": sum(1 for r in results if r["citation_count"] == 0) / n,
        "retrieval_pass_rate": sum(1 for r in results if r["passed_retrieval"]) / n,
    }


def default_thresholds() -> dict[str, float]:
    return {
        "min_recall_at_k": float(settings.RAG_EVAL_MIN_RECALL_AT_K),
        "min_mrr": float(settings.RAG_EVAL_MIN_MRR),
        "min_ndcg_at_k": float(settings.RAG_EVAL_MIN_NDCG_AT_K),
        "max_no_citation_rate": float(settings.RAG_EVAL_MAX_NO_CITATION_RATE),
    }


def gate_summary(summary: dict, thresholds: Optional[dict[str, float]] = None) -> tuple[bool, dict]:
    t = thresholds or default_thresholds()
    checks = {
        "recall_at_k": summary.get("recall_at_k", 0.0) >= t["min_recall_at_k"],
        "mrr": summary.get("mrr", 0.0) >= t["min_mrr"],
        "ndcg_at_k": summary.get("ndcg_at_k", 0.0) >= t["min_ndcg_at_k"],
        "no_citation_rate": summary.get("no_citation_rate", 1.0) <= t["max_no_citation_rate"],
    }
    return all(checks.values()), checks
