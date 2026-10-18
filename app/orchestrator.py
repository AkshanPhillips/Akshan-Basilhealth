"""
Single entrypoint for the text support system. `handle_message()` is the ONLY function in
this whole codebase that is allowed to call rag_agent.run() or tool_agent.run() -- this is
enforced both by convention here and verified structurally by
tests/test_no_bypass.py, which statically scans app/*.py and fails the test suite if any
module other than orchestrator.py calls either of those two functions. That's the
"show structurally, not by assuming it" proof the brief asks for: it's not a claim about
runtime behavior that could vary by input, it's a static fact about the codebase that a
reviewer can re-check by running the test.

Call order, unconditionally, for every message:
  1. guardrails.check_input(message)   -- if blocked, return immediately. Router/agents never run.
  2. router.classify(message)          -- only reached if step 1 passed.
  3. dispatch to rag_agent / tool_agent / human-escalation based on intent.
  4. guardrails.check_output(draft_reply, tool_confirmed_facts) -- if blocked, the raw
     agent reply is discarded and SAFE_FALLBACK_REPLY is returned instead.

Reconciling the output guardrail with the tool-calling agent (the brief flags this
explicitly as a design tension): the tool-calling agent legitimately needs to state facts
like "your order was delivered 3 days ago, so it's within the 10-day return window" or
"your refund of ₹1,799 was already processed" -- numbers the output guardrail's naive regex
would otherwise flag as an "unconfirmed promise." We resolve this by having the output
guardrail compare the draft reply against `tool_confirmed_facts`, a plain-text dump of the
RAW tool call results from THIS turn (built in tool_agent.run, never touched by the LLM).
If a number/date in the draft traces back to something present in tool_confirmed_facts,
the LLM-based output check treats it as confirmed and allows it. If a number/date appears
that ISN'T backed by a tool result this turn (e.g. the model inventing "you'll get it by
Friday" with no tool call made), it's blocked. So: tool-confirmed facts are allowed through
verbatim; anything else refund/date-shaped is guilty until proven tool-confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import guardrails, rag_agent, router, tool_agent


@dataclass
class SupportResponse:
    reply: str
    intent: str | None = None
    blocked_at: str | None = None  # "input_guardrail" | "output_guardrail" | None
    cache_hit: bool = False
    tool_calls: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    guardrail_detail: dict = field(default_factory=dict)


def handle_message(
    message: str,
    conversation_context: list[dict] | None = None,
    interrupted_statement: str | None = None,
) -> SupportResponse:
    # --- Step 1: input guardrail, unconditional, first thing that touches the raw message ---
    input_check = guardrails.check_input(message)
    if not input_check.allowed:
        return SupportResponse(
            reply=guardrails.SAFE_FALLBACK_REPLY,
            blocked_at="input_guardrail",
            guardrail_detail={"reason": input_check.reason, "category": input_check.category},
        )

    # --- Step 2: route (only reached post-guardrail) ---
    intent, cache_hit = router.classify(message)

    # Build contextual history, including interruption note if the user barged in mid-speech
    effective_context = list(conversation_context or [])
    if interrupted_statement:
        effective_context.append({
            "role": "system",
            "content": (
                f"[Context: The assistant was speaking: \"{interrupted_statement}\", "
                f"when the user interrupted with: \"{message}\". "
                "Address the user's interruption directly in this context. "
                "After answering, check if their query is satisfied and prompt them if they would like to continue with what was being discussed.]"
            ),
        })

    # --- Step 3: dispatch ---
    resume_phrases = ("continue", "go on", "what were you saying", "carry on", "go ahead", "resume", "repeat that")
    if interrupted_statement and any(p in message.lower() for p in resume_phrases):
        result = {
            "reply": (
                f"Certainly! Picking up from where I was interrupted: {interrupted_statement}. "
                "Does that help, or is there anything else I can check for you?"
            ),
            "tool_confirmed_facts": "(none this turn)",
        }
    elif intent == "order_status":
        result = tool_agent.run(message, effective_context)
    elif intent == "product_question":
        result = rag_agent.run(message, effective_context)
    elif intent == "greeting":
        if interrupted_statement:
            reply = f"No problem at all! As I was mentioning: {interrupted_statement}. Does that resolve what you were looking for?"
        elif conversation_context:
            reply = "You're very welcome! Is there anything else I can assist you with today?"
        else:
            reply = (
                "Hi! I'm the NovaVolt support assistant. I can help with order status, "
                "product questions, shipping and returns, or connect you with a human. "
                "What do you need help with?"
            )
        result = {
            "reply": reply,
            "tool_confirmed_facts": "(none this turn)",
        }
    else:  # escalate_to_human
        result = {
            "reply": (
                "I'll connect you with a human support agent who can help with this. "
                "One moment please."
            ),
            "tool_confirmed_facts": "(none this turn)",
        }

    draft_reply = result["reply"]
    tool_confirmed_facts = result.get("tool_confirmed_facts", "(none this turn)")

    # --- Step 4: output guardrail, unconditional, last thing before returning to the user ---
    output_check = guardrails.check_output(draft_reply, tool_confirmed_facts)
    if not output_check.allowed:
        return SupportResponse(
            reply=guardrails.SAFE_FALLBACK_REPLY,
            intent=intent,
            blocked_at="output_guardrail",
            cache_hit=cache_hit,
            guardrail_detail={"reason": output_check.reason, "category": output_check.category, **output_check.detail},
        )

    return SupportResponse(
        reply=draft_reply,
        intent=intent,
        cache_hit=cache_hit,
        tool_calls=result.get("tool_calls", []),
        retrieved_chunks=result.get("retrieved_chunks", []),
    )
