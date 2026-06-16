from __future__ import annotations
from app.settings import settings

def require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in .env or in your process environment."
        )
    return value

def voyage_key() -> str:
    return require_env("VOYAGEAI_API_KEY", settings.VOYAGEAI_API_KEY)

def gemini_key() -> str:
    return require_env("GEMINI_API_KEY", settings.GEMINI_API_KEY)
