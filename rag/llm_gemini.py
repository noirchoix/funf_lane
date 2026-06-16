# rag/llm_gemini.py
from __future__ import annotations

from app.settings import settings

def generate_answer(prompt: str) -> str:
    """
    Gemini text generation.
    Uses google-genai (google.genai) instead of deprecated google.generativeai.
    """
    from google import genai

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    model = getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    # resp.text is the common convenience accessor
    text = getattr(resp, "text", None)
    if not text:
        # fallback to string representation
        return str(resp)
    return text