from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases


def post_json(url: str, api_key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ChemRAG Pro retrieval regression gate.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="ChemRAG API base URL")
    parser.add_argument("--api-key", required=True, help="SERVICE_API_KEY")
    parser.add_argument("--golden", default="tests/eval/golden_qa.jsonl", help="Golden QA JSONL path")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--mode", default="hybrid_rerank", choices=["vector", "sparse", "hybrid", "hybrid_rerank"])
    parser.add_argument("--out", default="artifacts/chemrag_eval_report.json")
    args = parser.parse_args()

    cases = load_cases(Path(args.golden))
    payload = {
        "cases": cases,
        "top_k": args.top_k,
        "retrieval_mode": args.mode,
        "generate_answers": False,
        "fail_on_threshold": False,
    }
    result = post_json(args.base_url.rstrip("/") + "/v1/pro/eval/run", args.api_key, payload)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result.get("summary", {}), indent=2))
    if not result.get("ok"):
        print(f"Evaluation gate failed. Full report: {out}", file=sys.stderr)
        return 1
    print(f"Evaluation gate passed. Full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
