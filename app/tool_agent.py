"""
Tool-calling agent.

Design choice: Groq's native OpenAI-compatible function-calling, not a hand-rolled ReAct loop.
Justification:
  - With exactly two well-defined tools and a single order_id argument each, the model's job
    is simple argument extraction + tool selection -- native function-calling handles that
    reliably and the Groq API returns structured tool_calls we can execute directly, rather
    than parsing free-text "Thought:/Action:/Action Input:" out of a completion, which is
    fragile and adds a whole extra failure mode (malformed ReAct syntax) for no benefit here.
  - ReAct's real advantage is *multi-step* reasoning where the model needs to see a tool
    result before deciding the next action. Our two tools are independent lookups by
    order_id -- there's no multi-hop reasoning to do, so ReAct's main selling point doesn't
    apply, and native function calling is faster (fewer round trips) and easier to constrain
    (we control the tool schema, not a prompt template).
  - If a future 4th/5th tool introduced real multi-step dependencies (e.g. "cancel order"
    needing a status check -> eligibility check -> cancel sequence), that's still expressible
    as repeated function-calling turns (the model can call tools, see results, call another
    tool) -- Groq's function calling supports that loop too, so we're not boxing ourselves
    out of multi-step flows later. See design_notes.md for the 4th-agent scaling story.
"""
from __future__ import annotations

import json

from app import config
from app.groq_client import chat_completion
from app.mock_orders_db import check_return_eligibility, get_order_status

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current shipping/delivery status of an order by its order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, format NV-XXXXXX"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": "Check whether an order is still within the 10-day return window and eligible for return.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, format NV-XXXXXX"},
                },
                "required": ["order_id"],
            },
        },
    },
]

_TOOL_IMPLS = {
    "get_order_status": get_order_status,
    "check_return_eligibility": check_return_eligibility,
}

_SYSTEM_PROMPT = """You are the order-support agent for NovaVolt Electronics. You have two
tools: get_order_status and check_return_eligibility. Always call the appropriate tool
rather than guessing or inventing order details -- you have no knowledge of real order data
except through these tools. If the user hasn't given an order ID, ask for one instead of
calling a tool with a guessed value. After getting a tool result, answer in plain,
friendly language grounded ONLY in what the tool returned. Never state a refund amount or a
delivery date that isn't explicitly present in the tool result."""


def run(message: str, conversation_context: list[dict] | None = None) -> dict:
    """Returns {"reply": str, "tool_calls": [...], "tool_confirmed_facts": str}
    tool_confirmed_facts is a plain-text summary of raw tool output, handed to the output
    guardrail so it can tell a tool-confirmed number apart from a model-invented one."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages += conversation_context or []
    messages.append({"role": "user", "content": message})

    resp = chat_completion(
        model=config.GROQ_CHAT_MODEL,
        messages=messages,
        tools=_TOOLS,
        tool_choice="auto",
        temperature=0.1,
    )
    choice = resp.choices[0].message
    tool_calls_made = []
    facts_parts = []

    if choice.tool_calls:
        messages.append({"role": "assistant", "content": choice.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in choice.tool_calls
        ]})
        for tc in choice.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            impl = _TOOL_IMPLS.get(fn_name)
            result = impl(**args) if impl else {"error": f"Unknown tool {fn_name}"}
            tool_calls_made.append({"name": fn_name, "args": args, "result": result})
            facts_parts.append(f"{fn_name}({args}) -> {json.dumps(result)}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        final = chat_completion(model=config.GROQ_CHAT_MODEL, messages=messages, temperature=0.1)
        reply = final.choices[0].message.content
    else:
        reply = choice.content

    return {
        "reply": reply,
        "tool_calls": tool_calls_made,
        "tool_confirmed_facts": "\n".join(facts_parts) if facts_parts else "(none this turn)",
    }
