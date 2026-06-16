from __future__ import annotations

from typing import Sequence
from app.settings import settings


def _extract_embedding(resp: object) -> list[float]:
    """Extract embedding vector across google-genai response shapes."""
    # google-genai commonly returns .embeddings[0].values for batch embedding
    embeddings = getattr(resp, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        vals = getattr(first, "values", None) or getattr(first, "embedding", None)
        if vals is not None:
            return [float(x) for x in vals]

    # Some versions return .embedding.values for single content embedding
    emb = getattr(resp, "embedding", None)
    if emb is not None:
        vals = getattr(emb, "values", None) or emb
        return [float(x) for x in vals]

    # Defensive dict support for future/client variants
    if isinstance(resp, dict):
        if resp.get("embedding"):
            e = resp["embedding"]
            vals = e.get("values") if isinstance(e, dict) else e
            return [float(x) for x in vals]
        if resp.get("embeddings"):
            e = resp["embeddings"][0]
            vals = e.get("values") if isinstance(e, dict) else e
            return [float(x) for x in vals]

    raise RuntimeError(f"Could not extract Gemini embedding from response type {type(resp)!r}")


def embed_texts(texts: Sequence[str], model: str | None = None) -> list[list[float]]:
    """Gemini embedding adapter. Falls back at router level if this raises."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    model = model or settings.GEMINI_EMBED_MODEL
    vectors: list[list[float]] = []

    # Use per-text calls for compatibility across google-genai releases.
    for text in list(texts):
        if not text:
            text = " "
        # Most current google-genai versions support embed_content.
        if hasattr(client.models, "embed_content"):
            resp = client.models.embed_content(model=model, contents=text)
        else:  # pragma: no cover - defensive fallback for client API variants
            embed_fn = getattr(client.models, "embed", None)
            if embed_fn is None:
                raise RuntimeError("Gemini client does not support embed or embed_content")
            resp = embed_fn(model=model, content=text)
        vectors.append(_extract_embedding(resp))
    return vectors
