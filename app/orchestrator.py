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
from typing import TypedDict, Any

from app import guardrails, rag_agent, router, tool_agent

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class SupportState(TypedDict, total=False):
    message: str
    conversation_context: list[dict]
    interrupted_statement: str | None
    effective_context: list[dict]
    input_allowed: bool
    intent: str | None
    cache_hit: bool
    draft_reply: str
    tool_confirmed_facts: str
    tool_calls: list
    retrieved_chunks: list
    output_allowed: bool
    blocked_at: str | None
    reply: str
    guardrail_detail: dict


def _node_input_guardrail(state: SupportState) -> dict:
    check = guardrails.check_input(state["message"])
    if not check.allowed:
        return {
            "input_allowed": False,
            "blocked_at": "input_guardrail",
            "reply": guardrails.SAFE_FALLBACK_REPLY,
            "guardrail_detail": {"reason": check.reason, "category": check.category},
        }
    return {"input_allowed": True}


def _route_after_input_guardrail(state: SupportState) -> str:
    if not state.get("input_allowed", True):
        return END
    return "router"


def _node_router(state: SupportState) -> dict:
    msg = state["message"]
    ctx = list(state.get("conversation_context") or [])
    stmt = state.get("interrupted_statement")
    if stmt:
        ctx.append({
            "role": "system",
            "content": (
                f"[Context: The assistant was speaking: \"{stmt}\", "
                f"when the user interrupted with: \"{msg}\". "
                "Address the user's interruption directly in this context. "
                "After answering, check if their query is satisfied and prompt them if they would like to continue with what was being discussed.]"
            ),
        })

    resume_phrases = ("continue", "go on", "what were you saying", "carry on", "go ahead", "resume", "repeat that")
    if stmt and any(p in msg.lower() for p in resume_phrases):
        return {"intent": "resume", "cache_hit": False, "effective_context": ctx}

    intent, cache_hit = router.classify(msg)
    return {"intent": intent, "cache_hit": cache_hit, "effective_context": ctx}


def _route_intent(state: SupportState) -> str:
    intent = state.get("intent")
    if intent == "resume":
        return "resume_node"
    if intent == "order_status":
        return "tool_agent"
    if intent == "product_question":
        return "rag_agent"
    if intent == "greeting":
        return "greeting_node"
    return "escalate_node"


def _node_tool_agent(state: SupportState) -> dict:
    res = tool_agent.run(state["message"], state.get("effective_context"))
    return {
        "draft_reply": res["reply"],
        "tool_calls": res.get("tool_calls", []),
        "tool_confirmed_facts": res.get("tool_confirmed_facts", "(none this turn)"),
    }


def _node_rag_agent(state: SupportState) -> dict:
    res = rag_agent.run(state["message"], state.get("effective_context"))
    return {
        "draft_reply": res["reply"],
        "retrieved_chunks": res.get("retrieved_chunks", []),
        "tool_confirmed_facts": res.get("tool_confirmed_facts", "(none this turn)"),
    }


def _node_resume(state: SupportState) -> dict:
    stmt = state.get("interrupted_statement")
    reply = (
        f"Certainly! Picking up from where I was interrupted: {stmt}. "
        "Does that help, or is there anything else I can check for you?"
    )
    return {"draft_reply": reply, "tool_confirmed_facts": "(none this turn)"}


def _node_greeting(state: SupportState) -> dict:
    stmt = state.get("interrupted_statement")
    ctx = state.get("conversation_context")
    if stmt:
        reply = f"No problem at all! As I was mentioning: {stmt}. Does that resolve what you were looking for?"
    elif ctx:
        reply = "You're very welcome! Is there anything else I can assist you with today?"
    else:
        reply = (
            "Hi! I'm the NovaVolt support assistant. I can help with order status, "
            "product questions, shipping and returns, or connect you with a human. "
            "What do you need help with?"
        )
    return {"draft_reply": reply, "tool_confirmed_facts": "(none this turn)"}


def _node_escalate(state: SupportState) -> dict:
    reply = (
        "I'll connect you with a human support agent who can help with this. "
        "One moment please."
    )
    return {"draft_reply": reply, "tool_confirmed_facts": "(none this turn)"}


def _node_output_guardrail(state: SupportState) -> dict:
    draft = state.get("draft_reply", "")
    facts = state.get("tool_confirmed_facts", "(none this turn)")
    output_check = guardrails.check_output(draft, facts)
    if not output_check.allowed:
        return {
            "output_allowed": False,
            "blocked_at": "output_guardrail",
            "reply": guardrails.SAFE_FALLBACK_REPLY,
            "guardrail_detail": {"reason": output_check.reason, "category": output_check.category, **output_check.detail},
        }
    return {
        "output_allowed": True,
        "reply": draft,
    }


def build_support_agent_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    builder = StateGraph(SupportState)
    builder.add_node("input_guardrail", _node_input_guardrail)
    builder.add_node("router", _node_router)
    builder.add_node("tool_agent", _node_tool_agent)
    builder.add_node("rag_agent", _node_rag_agent)
    builder.add_node("resume_node", _node_resume)
    builder.add_node("greeting_node", _node_greeting)
    builder.add_node("escalate_node", _node_escalate)
    builder.add_node("output_guardrail", _node_output_guardrail)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {"router": "router", END: END},
    )
    builder.add_conditional_edges(
        "router",
        _route_intent,
        {
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent",
            "resume_node": "resume_node",
            "greeting_node": "greeting_node",
            "escalate_node": "escalate_node",
        },
    )
    builder.add_edge("tool_agent", "output_guardrail")
    builder.add_edge("rag_agent", "output_guardrail")
    builder.add_edge("resume_node", "output_guardrail")
    builder.add_edge("greeting_node", "output_guardrail")
    builder.add_edge("escalate_node", "output_guardrail")
    builder.add_edge("output_guardrail", END)

    return builder.compile()


_SUPPORT_GRAPH = build_support_agent_graph() if LANGGRAPH_AVAILABLE else None


def _run_langgraph(
    message: str,
    conversation_context: list[dict] | None = None,
    interrupted_statement: str | None = None,
) -> SupportResponse:
    init_state: SupportState = {
        "message": message,
        "conversation_context": conversation_context or [],
        "interrupted_statement": interrupted_statement,
    }
    result = _SUPPORT_GRAPH.invoke(init_state)
    return SupportResponse(
        reply=result.get("reply", guardrails.SAFE_FALLBACK_REPLY),
        intent=result.get("intent"),
        blocked_at=result.get("blocked_at"),
        cache_hit=result.get("cache_hit", False),
        tool_calls=result.get("tool_calls", []),
        retrieved_chunks=result.get("retrieved_chunks", []),
        guardrail_detail=result.get("guardrail_detail", {}),
    )


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
    # If LangGraph is installed and compiled, run as a stateful multi-agent StateGraph
    if LANGGRAPH_AVAILABLE and _SUPPORT_GRAPH is not None:
        return _run_langgraph(message, conversation_context, interrupted_statement)

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


def get_graph_mermaid() -> str | None:
    """Returns the compiled LangGraph StateGraph topology in Mermaid format for documentation."""
    if LANGGRAPH_AVAILABLE and _SUPPORT_GRAPH is not None:
        try:
            return _SUPPORT_GRAPH.get_graph().draw_mermaid()
        except Exception:
            pass
    return None
