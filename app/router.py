"""
Intent router.

Classifies an incoming (already-guardrail-checked -- see orchestrator.py) user message into
one of ROUTER_INTENTS using a cheap/fast Groq model, with a cache in front of it. The router
NEVER sees raw unchecked input -- orchestrator.py enforces that ordering; this module just
classifies whatever text it's given.

We use a *constrained* classification prompt (explicit label list, forced single-token-ish
output, regex-validated) rather than open-ended generation, specifically so a malformed or
adversarial response can't accidentally route into execution. If the model returns anything
outside ROUTER_INTENTS, we fail closed to escalate_to_human rather than guessing.

Greetings/acks are handled with a local pre-check *before* the classifier cache and LLM.
That avoids both (a) burning a Groq call on "hey", and (b) serving a stale cached
escalate_to_human label if an older prompt once misrouted greetings.
"""
from __future__ import annotations

import re

from app import config
from app.cache import get_cache
from app.groq_client import chat_completion

_VALID_INTENTS = set(config.ROUTER_INTENTS)

# Whole-message greetings / short social acknowledgements -- not support requests.
_GREETING_RE = re.compile(
    r"^(hey|hi|hello|hola|yo|sup|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s*you|thx|ok|okay|bye|goodbye|see\s*ya)"
    r"([!.?\s]*)$",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = f"""You are a strict intent classifier for a customer support system.
Classify the user's message into EXACTLY ONE of these intents:
- order_status: questions about an existing order's status, tracking, delivery, where it is
- product_question: questions about a product's features, compatibility, specs, how to use/troubleshoot it, or policy questions (shipping, returns, warranty, payment)
- greeting: short hellos, thanks, or acknowledgements with no support request (e.g. "hey", "hi", "thanks")
- escalate_to_human: the user explicitly wants a human, expresses strong frustration/anger, or asks for something outside policy (refund amount guarantees, exceptions, discounts, unauthorized overrides). Do NOT use this for ordinary greetings or unclear-but-innocent messages -- prefer product_question when the user seems to be asking about a product/policy but phrasing is vague.

Respond with ONLY the intent label, nothing else. No punctuation, no explanation."""


def _is_greeting(message: str) -> bool:
    return bool(_GREETING_RE.match(message.strip()))


def _call_classifier(message: str) -> str:
    resp = chat_completion(
        model=config.GROQ_ROUTER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip().lower()
    raw = re.sub(r"[^a-z_]", "", raw)
    return raw


def classify(message: str) -> tuple[str, bool]:
    """Returns (intent, was_cache_hit)."""
    # Local greeting gate runs before cache so a stale escalate label can't stick on "hey".
    if _is_greeting(message):
        return "greeting", False

    cache = get_cache()
    cached = cache.get(message)
    if cached is not None:
        if cached in _VALID_INTENTS:
            return cached, True
        # Unknown/stale label from an older schema -- fall through to reclassify.

    intent = _call_classifier(message)
    if intent not in _VALID_INTENTS:
        # Fail closed: anything the model doesn't confidently label goes to a human,
        # rather than guessing and risking a misrouted RAG/tool call.
        # (Greetings never reach here -- they short-circuit above.)
        intent = "escalate_to_human"

    cache.set(message, intent)
    return intent, False
