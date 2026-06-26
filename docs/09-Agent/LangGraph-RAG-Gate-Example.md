# Worked example: gate a LangGraph RAG agent before it publishes

> Put Sophia's claim gate behind a retrieval-augmented agent — a company-research
> agent like
> [`chrisipanaque/langchain-runnableparallel-company-research`](https://github.com/chrisipanaque/langchain-runnableparallel-company-research)
> is the canonical shape — so a fabricated or unsourced claim never ships.

A RAG agent retrieves passages, has an LLM draft an answer, and returns it. The
failure mode the retrieval *doesn't* catch: the LLM asserts something the
retrieved passages don't actually support, or cites a source it never read. That
is precisely what `verify_claim` is for — the claim only publishes if its
`content` is faithful to the `sources` the retriever found.

## The pattern

The agent's draft becomes a **claim**, and the retrieved passage URLs become its
**sources**. Drop the gate in as the last node before "publish":

```text
question ─▶ retrieve ─▶ draft (LLM) ─▶ [Sophia gate] ─▶ publish | review | reject
                                          record_claim
                                          verify_claim
```

`sophia_contract.langgraph_nodes` already provides the gate as LangGraph nodes
(`record → verify → {publish|review|reject}`), and a dependency-free
`run_contract_flow` for CI / non-LangGraph orchestrators.

## Runnable example

A self-contained, offline-runnable version lives at
[`examples/langgraph_rag_gate.py`](../../examples/langgraph_rag_gate.py):

```bash
python examples/langgraph_rag_gate.py
```

It runs three company-research queries through a mock retriever + drafter and the
real gate:

- two **sourced** claims (the retriever found backing) → `accepted` → publish,
- one **unsourced** claim (LLM "remembered" it; retriever found nothing) → `held` → human review.

The held case is the one that matters: the answer *looked* fine, but nothing
backed it, so it is routed to a human instead of shipped.

Swap the mock `retrieve` and `draft` for your real LangChain retriever and LLM and
the gate node is unchanged — it only cares that the published `content` is backed
by the `sources` the retriever actually returned.

## Why the gate, not just the retriever

Retrieval improves *recall* of relevant context; it does not *enforce* that the
generated answer stays within that context. The gate is the enforcement step:

- **Unsourced assertion** → `held` (`no_source`), routed to a human instead of shipped.
- **Lineage merge** (e.g. crediting one source's idea to another) → rejected — the
  same source-discipline rule Sophia applies to the humanities corpus.
- **Auditable** → every verdict is traced and can be exported to your collector
  ([Observability-OTel.md](./Observability-OTel.md)).

## See also

- [`sophia_contract/langgraph_nodes.py`](../../sophia_contract/langgraph_nodes.py) — the gate nodes.
- [CONTRACT.md](../../CONTRACT.md) — the stable `record_claim`/`verify_claim` interface.
- [Online-RAG.md](./Online-RAG.md) — Sophia's own retrieval surface.
