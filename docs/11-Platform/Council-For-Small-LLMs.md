# Councils + skills + MCP as a small-LLM upgrade

**Thesis.** A small model (7–14B, local, cheap) is a *weak generalist*: shallow
knowledge, weak multi-perspective reasoning, hallucination-prone. Sophia
externalises the three things it is bad at, so the model doesn't have to be good at
them — it just has to follow a scaffold and pass cheap checks.

| Weakness | Sophia externalises it via |
|---|---|
| Knowledge | MCP lookups + RAG + OKF wiki (retrieve, don't recall) |
| Verification | the deterministic gate (citation-exists, arithmetic, provenance) — matters *more* for a weak model, costs ~nothing |
| Breadth of reasoning | **the council** — decompose into narrow seats; a 7B answers one focused role far better than a whole analysis |

> One line: Sophia turns a weak generalist into a *disciplined, tool-checked,
> council-decomposed* reasoner. Many focused gated passes beat one shallow pass.

## The mechanism: council as map-reduce (`agent/council_deliberate.py`)

The old `sophia_sector_council` returned a single big multi-seat *prompt* for one
pass. `deliberate()` instead runs a **map-reduce**:

```
map:    route_council -> a few NARROW substantive seats -> one focused pass each
gate:   each seat output is gate-checked; flagged seats are quarantined
reduce: synthesise the gate-passed seats under the guardian seats + decision contract
```

- **Seat = micro-skill** — the seat's `sourceFrame` + `decisionEmphasis` become a
  tight system prompt; the model answers one role in 2–4 sentences.
- **Gate every seat** (substantive violations only, not style) — weak-model errors
  (fabricated citation, false arithmetic, forbidden attribution) die cheaply before
  they reach the synthesis.
- **Reduce ≠ heavy lifting for the weak model** — synthesis runs only over
  gate-passed seats, under the guardian checklist; if nothing survives, it
  **abstains** rather than guess.

## Surfaces

- **MCP:** `sophia_council_deliberate(query, model, max_seats, gate)` — the one
  high-level tool a small-model host calls instead of orchestrating 21 tools.
- **CLI:** `python tools/council_deliberate.py "<query>" --model ollama:llama3.1:8b`
- **Skill (tool-less hosts):** the portable SKILL.md can encode the same
  decompose→cite→gate→abstain process as a prompt-only playbook for bare Ollama.

## Proving it: the uplift harness (`tools/run_council_uplift.py`)

Runs each task three ways — **alone / +council / +council+gate** — and reports the
gated delta. Scoring is deterministic (no LLM judge): an output is *clean* iff it
has no gate violation.

```bash
python tools/run_council_uplift.py --model mock                 # plumbing
python tools/run_council_uplift.py --model ollama:llama3.1:8b   # real, illustrative
```

Honest framing: a single-model run is **illustrative, never a headline** (same
no-overclaim discipline as the rest of RESULTS.md). On `mock` the delta is 0% (mock
doesn't hallucinate) — that validates the harness; a real small model is where the
uplift shows. To make it a *validated* number, run multiple model families and
report inter-run CIs, as with the faithfulness gate.

## How to improve it (roadmap)

1. **Small-model profile** — fewer seats (3), shorter prompts, **constrained/JSON
   output** per seat (sglang/structured already supported — weak models need schema
   enforcement), retrieval-first to cover weak parametric knowledge.
2. **Council-distillation** — generate seat-structured, gated, cited, abstaining
   traces with a strong model, then LoRA the small model (`train_lora.py`,
   `wiki_to_training.py`) to *internalise* the discipline — so it stays disciplined
   even without MCP.
3. **MCP ergonomics** — surface a curated 3–5 tools, short example-driven
   descriptions, deterministic outputs; prefer one high-level `deliberate` over many
   low-level tools.
4. **Measure where it helps vs hurts** — councils can add noise on easy tasks; the
   uplift harness with per-task breakdown tells you when to route to a council vs a
   direct pass.

## Honest risks

- **Latency/cost** — N passes per query; mitigate with seat caps + cheap models +
  caching.
- **Synthesis ceiling** — a weak reducer can still botch the combine; prefer
  template/guardian synthesis and abstention.
- **Tool-ignoring / malformed JSON** — the classic small-model failure; constrained
  decoding + deterministic fallback.
- **Uplift is not free or guaranteed** — it must be *measured*, not assumed.
