"""
RAG agent for the product_question intent.

Retrieves top-k chunks from the FAISS index (app/retriever.py) and instructs the model to
answer ONLY from those passages -- the system prompt explicitly forbids answering from
general knowledge and requires an explicit "I don't have that information" when the
retrieved passages don't cover the question, rather than letting the model fill the gap
from parametric knowledge. This is the "grounding, not guessing" requirement from the brief.
"""
from __future__ import annotations

from app import config
from app.groq_client import chat_completion
from app.retriever import get_retriever

_SYSTEM_PROMPT = """You are the product-support agent for NovaVolt Electronics. Answer the
user's question using ONLY the retrieved passages below. Do not use any outside knowledge
about electronics or policies in general -- if the passages don't contain the answer, say
you don't have that information and offer to connect them with a human agent. Never invent
a specific refund amount or delivery date; if the question needs that, say it depends on the
specific order and offer to look up the order (the order-status/tool agent handles that).

Retrieved passages:
{context}
"""


def run(message: str) -> dict:
    retriever = get_retriever()
    results = retriever.search(message, top_k=config.TOP_K_RETRIEVAL)

    if not results:
        context = "(no relevant passages found)"
    else:
        context = "\n\n".join(f"[{r['doc_id']}] {r['text']}" for r in results)

    resp = chat_completion(
        model=config.GROQ_CHAT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": message},
        ],
        temperature=0.2,
    )
    reply = resp.choices[0].message.content

    return {
        "reply": reply,
        "retrieved_chunks": results,
        "tool_confirmed_facts": "(RAG agent does not call tools; retrieved KB passages are not order-specific facts)",
    }
