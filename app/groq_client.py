"""
Thin wrapper around the Groq SDK so the rest of the app doesn't import `groq` directly.
Centralizing this also gives us one place to add retry/fallback logic (used by the voice
pipeline's provider-resilience simulation, and reusable for the text side per Part C's
"Groq degrades mid-launch" question).
"""
from __future__ import annotations

from groq import Groq

from app import config

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def chat_completion(messages, model=None, tools=None, tool_choice=None, temperature=0.2, max_retries=2):
    """Single retry wrapper. Real failover-to-another-provider logic is intentionally
    NOT implemented here (see Part C answer: 'Groq degrades mid-launch') -- for this
    assignment we simulate provider failure in the voice pipeline instead, since that's
    what the brief asks to demonstrate."""
    client = get_client()
    model = model or config.GROQ_CHAT_MODEL
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(model=model, messages=messages, temperature=temperature)
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            return client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a network boundary
            last_err = e
            if attempt < max_retries:
                continue
            raise
    raise last_err  # pragma: no cover
