#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate a (mock) RAG agent before it publishes — the LangGraph integration example.

A retrieval-augmented agent like
``chrisipanaque/langchain-runnableparallel-company-research`` retrieves passages,
has an LLM draft an answer, and returns it. This example shows the missing
enforcement step: the draft becomes a Sophia *claim*, the retrieved URLs become its
*sources*, and ``verify_claim`` decides whether it may publish.

Offline and dependency-free: the ``retrieve`` and ``draft`` functions are mocks, but
the gate (``sophia_contract``) is the real thing. Swap the two mocks for a real
LangChain retriever + LLM and the gate node is unchanged.

    python examples/langgraph_rag_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract import SophiaContract
from sophia_contract.langgraph_nodes import run_contract_flow


# --- mock RAG stages (replace with a real LangChain retriever + LLM) ---------- #

def retrieve(question: str) -> "list[str]":
    """Mock retriever: return source ids/URLs the corpus has for this question.

    A real implementation returns the URLs/document ids your vector store found.
    Here, the third query intentionally retrieves nothing (the LLM will still try
    to answer — that is the failure the gate catches)."""
    index = {
        "What year was Acme Corp founded?": ["https://sec.gov/acme/10-K-1998"],
        "Who is Acme Corp's current CEO?": ["https://acme.com/about/leadership"],
    }
    return index.get(question, [])


def draft(question: str, sources: "list[str]") -> str:
    """Mock drafter: an LLM answer. When the retriever found nothing the model
    still 'remembers' an answer — exactly the unsourced assertion to fence out."""
    answers = {
        "What year was Acme Corp founded?": "Acme Corp was founded in 1998.",
        "Who is Acme Corp's current CEO?": "Acme Corp's current CEO is Dana Reyes.",
        "What was Acme Corp's 2019 revenue?": "Acme Corp's 2019 revenue was $4.2B.",
    }
    return answers.get(question, "(no answer)")


# --- the agent: retrieve -> draft -> Sophia gate ------------------------------ #

def answer(contract: SophiaContract, question: str, key: str) -> dict:
    sources = retrieve(question)
    content = draft(question, sources)
    final = run_contract_flow(contract, {
        "idempotency_key": key,
        "content": content,
        "sources": sources,
    })
    return {"question": question, "content": content,
            "sources": sources, "route": final["route"],
            "verdict": (final.get("verdict") or {}).get("verdict")}


def main() -> int:
    contract = SophiaContract()
    queries = [
        ("What year was Acme Corp founded?", "q-founded"),
        ("Who is Acme Corp's current CEO?", "q-ceo"),
        ("What was Acme Corp's 2019 revenue?", "q-revenue"),  # retriever finds nothing
    ]
    print("RAG agent answers, each gated before publish:\n")
    for question, key in queries:
        r = answer(contract, question, key)
        mark = {"publish": "✅ PUBLISH", "review": "⏸  HUMAN REVIEW", "reject": "⛔ REJECT"}[r["route"]]
        print(f"  {mark}  ({r['verdict']})")
        print(f"    Q: {r['question']}")
        print(f"    A: {r['content']}")
        print(f"    sources: {r['sources'] or '(none retrieved)'}\n")
    print("The unsourced answer is held for a human instead of shipped — fail-closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
