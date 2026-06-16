from __future__ import annotations

from typing import Sequence
from app.settings import settings


def _provider_order() -> list[str]:
    primary = (getattr(settings, "EMBED_PROVIDER_PREFER", None) or getattr(settings, "DEFAULT_EMBED_PROVIDER", "gemini")).lower()
    fallback = (getattr(settings, "FALLBACK_EMBED_PROVIDER", "voyage") or "voyage").lower()
    out: list[str] = []
    for p in [primary, fallback, "gemini", "voyage"]:
        if p and p not in out:
            out.append(p)
    return out


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Provider-routed embeddings: Gemini primary, Voyage fallback by default."""
    last_err: Exception | None = None
    for provider in _provider_order():
        try:
            if provider == "gemini":
                if not settings.GEMINI_API_KEY:
                    continue
                from .embed_gemini import embed_texts as gemini_embed
                return gemini_embed(texts)
            if provider == "voyage":
                if not settings.VOYAGEAI_API_KEY:
                    continue
                from .embed_voyage import embed_texts as voyage_embed
                return voyage_embed(texts, model=settings.VOYAGE_EMBED_MODEL)
        except Exception as e:  # keep trying fallbacks
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No embedding provider configured. Set GEMINI_API_KEY or VOYAGEAI_API_KEY.")
