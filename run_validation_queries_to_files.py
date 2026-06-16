from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "8b7d3b2e6b1a4d2f9a3c1c7f0a2e9c44"

OUT_DIR = Path("validation_outputs")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def post_json(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers=HEADERS,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def save_json(name: str, data: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{ts}_{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_answer_txt(name: str, data: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{ts}_{name}_answer.txt"

    answer = data.get("answer", "")
    citations = data.get("citations", [])

    lines = []
    lines.append("ANSWER")
    lines.append("=" * 80)
    lines.append(answer)
    lines.append("")
    lines.append("CITATIONS")
    lines.append("=" * 80)

    for i, c in enumerate(citations, start=1):
        source = c.get("source") or {}
        lines.append(f"\n[{i}]")
        lines.append(f"title: {source.get('title')}")
        lines.append(f"lane: {c.get('rag_lane') or source.get('rag_lane')}")
        lines.append(f"collection: {c.get('collection')}")
        lines.append(f"doc_id: {c.get('doc_id')}")
        lines.append(f"page: {c.get('page')}")
        lines.append(f"score: {c.get('score')}")
        lines.append("chunk_preview:")
        lines.append((c.get("chunk") or "")[:1200])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_query(name: str, payload: dict) -> None:
    print(f"Running {name}...")
    result = post_json("/v1/query", payload)

    json_path = save_json(name, result)
    txt_path = save_answer_txt(name, result)

    print(f"  saved JSON: {json_path}")
    print(f"  saved TXT : {txt_path}")


def main() -> None:
    # Health check
    health = urllib.request.urlopen(f"{BASE_URL}/health", timeout=30).read().decode("utf-8")
    print(f"Health: {health}")

    run_query(
        "quality_control_citrus",
        {
            "query": "What quality-control or safety issues should be considered for citrus fragrance notes and citrus oils in fragrance products?",
            "top_k": 10,
            "retrieval_lanes": ["quality_control"],
        },
    )

    run_query(
        "cross_lane_citrus",
        {
            "query": "Explain citrus notes as phytochemical, physical-chemistry, quality-control, and internal perfumery vocabulary evidence for a fragrance formulation system.",
            "top_k": 16,
            "retrieval_lanes": [
                "phytochemistry_context",
                "physical_chem",
                "quality_control",
                "internal_notes",
            ],
        },
    )

    run_query(
        "physical_chem_citrus_volatility",
        {
            "query": "How do citrus oils, terpene structure, volatility, top-note behavior, and instability affect fragrance performance?",
            "top_k": 10,
            "retrieval_lanes": ["physical_chem"],
        },
    )

    run_query(
        "phytochemistry_terpenes",
        {
            "query": "How are terpenes and terpenoids classified in phytochemistry, and why are monoterpenes and sesquiterpenes relevant to fragrance materials?",
            "top_k": 10,
            "retrieval_lanes": ["phytochemistry_context"],
        },
    )


if __name__ == "__main__":
    main()
