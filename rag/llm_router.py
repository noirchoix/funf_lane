from __future__ import annotations
from app.settings import settings


def _try_gemini(prompt: str) -> str:
    from .llm_gemini import generate_answer as gemini_generate
    return gemini_generate(prompt)


def _try_deepseek(prompt: str) -> str:
    from .llm_deepseek import generate_answer as deepseek_generate
    return deepseek_generate(prompt)


def _provider_order() -> list[str]:
    primary = (getattr(settings, "DEFAULT_GENERATION_PROVIDER", None) or getattr(settings, "LLM_PROVIDER_PREFER", "gemini")).lower()
    fallback = (getattr(settings, "FALLBACK_GENERATION_PROVIDER", "deepseek") or "deepseek").lower()
    out: list[str] = []
    for p in [primary, fallback, "gemini", "deepseek"]:
        if p and p not in out:
            out.append(p)
    return out


def generate_answer(prompt: str) -> tuple[str, str]:
    """Provider-routed generation: Gemini primary, DeepSeek fallback by default."""
    last_err: Exception | None = None
    for p in _provider_order():
        if p == "gemini":
            if getattr(settings, "GEMINI_API_KEY", None):
                try:
                    return _try_gemini(prompt), "gemini"
                except Exception as e:
                    last_err = e
        elif p == "deepseek":
            if getattr(settings, "DEEPSEEK_API_KEY", None):
                try:
                    return _try_deepseek(prompt), "deepseek"
                except Exception as e:
                    last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("No generation provider configured. Set GEMINI_API_KEY or DEEPSEEK_API_KEY.")
