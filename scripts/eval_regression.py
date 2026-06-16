from __future__ import annotations

import json
import sys
import httpx

"""
Input: JSONL with lines like:
{"query":"...", "must_contain":["oxidation","terpene"], "min_citations":1}
Run:
python scripts/eval_regression.py http://localhost:8000 change_me tests/eval.jsonl
"""

def main():
    if len(sys.argv) != 4:
        print("usage: eval_regression.py <base_url> <service_api_key> <jsonl_path>")
        sys.exit(2)

    base_url, api_key, path = sys.argv[1], sys.argv[2], sys.argv[3]
    ok = 0
    total = 0
    failures = []

    with open(path, "r", encoding="utf-8") as f, httpx.Client(timeout=60) as client:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            query = item["query"]
            must = item.get("must_contain", [])
            min_cit = int(item.get("min_citations", 0))

            r = client.post(
                f"{base_url.rstrip('/')}/v1/chat/planner",
                headers={"X-API-Key": api_key},
                json={"query": query, "top_k_docs": 8, "top_k_memory": 0},
            )
            r.raise_for_status()
            out = r.json()
            ans = (out.get("answer") or "").lower()
            cits = out.get("citations") or []

            passed = True
            for m in must:
                if m.lower() not in ans:
                    passed = False
            if len(cits) < min_cit:
                passed = False

            if passed:
                ok += 1
            else:
                failures.append({"query": query, "answer_preview": out.get("answer", "")[:300], "citations": len(cits)})

    print(f"PASSED {ok}/{total}")
    if failures:
        print("FAILURES:")
        for f in failures[:20]:
            print("-", f["query"])
            print("  citations:", f["citations"])
            print("  answer:", f["answer_preview"])

if __name__ == "__main__":
    main()
