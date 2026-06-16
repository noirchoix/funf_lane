from __future__ import annotations

import time
from typing import List, Sequence, Optional, cast

from voyageai.client import Client
from voyageai.error import RateLimitError

from app.settings import settings
from .guards import voyage_key

_client = Client(api_key=voyage_key())

def embed_texts(
    texts: Sequence[str],
    model: str = "voyage-3",
    batch_size: int = 6,       # ✅ small for TPM
    sleep_s: float = 22.0,     # ✅ 3 RPM safety
) -> List[List[float]]:
    """
    Voyage embeddings with strict throttling for free-tier limits:
    - 3 requests/minute -> sleep ~22s between requests
    - 10K tokens/minute -> keep batch small + chunk size reasonable
    """
    all_vecs: List[List[float]] = []
    texts_list = list(texts)

    i = 0
    while i < len(texts_list):
        batch = texts_list[i : i + batch_size]

        try:
            res = _client.embed(batch, model=model)
            raw = cast(List[Sequence[float] | Sequence[int]], res.embeddings)
            vecs = [[float(x) for x in vec] for vec in raw]
            all_vecs.extend(vecs)

            i += batch_size

            # throttle between successful calls
            if i < len(texts_list):
                time.sleep(sleep_s)

        except RateLimitError:
            # ✅ backoff hard on 429
            time.sleep(60.0)
            continue
        except Exception as e:
            # small exponential backoff for transient errors
            for attempt in range(3):
                time.sleep(2.0 * (attempt + 1))
                try:
                    res = _client.embed(batch, model=model)
                    raw = cast(List[Sequence[float] | Sequence[int]], res.embeddings)
                    vecs = [[float(x) for x in vec] for vec in raw]
                    all_vecs.extend(vecs)
                    i += batch_size
                    if i < len(texts_list):
                        time.sleep(sleep_s)
                    break
                except RateLimitError:
                    time.sleep(60.0)
                except Exception:
                    if attempt == 2:
                        raise e

    return all_vecs