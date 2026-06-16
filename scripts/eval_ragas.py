from __future__ import annotations

import sys
import json
import httpx
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

"""
Input JSONL lines like:
{"query":"...", "ground_truth":"..."}  # ground_truth optional but recommended
Run:
python scripts/eval_ragas.py http://localhost:8000 change_me tests/ragas.jsonl
"""

def main():
    if len(sys.argv) != 4:
        print("usage: eval_ragas.py <base_url> <service_api_key> <jsonl_path>")
        sys.exit(2)

    base_url, api_key, path = sys.argv[1], sys.argv[2], sys.argv[3]

    rows = []
    with open(path, "r", encoding="utf-8") as f, httpx.Client(timeout=90) as client:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query = item["query"]
            ground_truth = item.get("ground_truth", "")

            r = client.post(
                f"{base_url.rstrip('/')}/v1/chat/planner",
                headers={"X-API-Key": api_key},
                json={"query": query, "top_k_docs": 8, "top_k_memory": 0},
            )
            r.raise_for_status()
            out = r.json()

            answer = out.get("answer") or ""
            contexts = [c.get("chunk", "") for c in (out.get("citations") or []) if c.get("chunk")]

            rows.append({"question": query, "answer": answer, "contexts": contexts, "ground_truth": ground_truth})

    ds = Dataset.from_pandas(pd.DataFrame(rows))

    # Choose a minimal set of metrics (you can expand later)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print(result)

if __name__ == "__main__":
    main()
