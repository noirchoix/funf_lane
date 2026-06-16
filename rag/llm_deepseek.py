from __future__ import annotations

from typing import Optional

from app.settings import settings


def generate_answer(prompt: str, *, model: Optional[str] = None) -> str:
    """
    DeepSeek text generation via OpenAI-compatible API.

    Uses:
      - settings.DEEPSEEK_API_KEY
      - settings.DEEPSEEK_BASE_URL (default https://api.deepseek.com)
      - settings.DEEPSEEK_MODEL (default deepseek-chat)
    """
    api_key = getattr(settings, "DEEPSEEK_API_KEY", None)
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    base_url = getattr(settings, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = model or getattr(settings, "DEEPSEEK_MODEL", "deepseek-chat")
    # Narrow Optional[str] to str for the client call so type checkers know it's not None
    assert model is not None

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'openai'. Install it: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful chemistry assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        # DeepSeek docs note occasional empty content; treat as error so router can degrade.
        raise RuntimeError("DeepSeek returned empty content")
    return content