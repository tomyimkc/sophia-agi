# Good First Issues

**Help build the foundation for AI that respects its sources.**

Starter tasks for new contributors. Every attribution or dispute you add makes the gate stronger. Comment on an issue or open a PR referencing the ID.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the required workflow (validate before PR).

## Philosophy (active)

| ID | Task | Status |
|----|------|--------|
| GF-01 | Add attribution for *Mencius* (《孟子》) | ✅ v0.4.2 |
| GF-02 | Add attribution for *Zhuangzi* (《莊子》) | ✅ v0.4.2 |
| GF-03 | Add Plato *Symposium* record | ✅ v0.4.2 |
| GF-04 | Write training example: Socrates vs Plato trap | ✅ `020-socrates-plato-mencius-zhuangzi.json` |
| GF-05 | Add 5 philosophy benchmark traps | ✅ `tests/benchmark-philosophy.json` (9 cases) |

Next philosophy tasks: GF-06+ attributions for *Xunzi*, *Mozi*, *Aristotle Nicomachean Ethics* — open an issue from template.

## Psychology

| ID | Task | Status |
|----|------|--------|
| GF-10 | Add 5 more psychology concepts with subfield tags | ✅ v0.7.38 (`dunning_kruger_effect`, `confirmation_bias`, `ten_percent_brain_myth`, `mozart_effect_myth`, `ptsd_clinical_vs_pop` + 5 benchmark traps) |
| GF-11 | Document pop-psych vs clinical boundary | `docs/08-Domains/Psychology/` (open) |

## History

| ID | Task | Status |
|----|------|--------|
| GF-20 | Add 3 dated events with primary source field | ✅ v0.7.38 (`magna_carta_1215`, `boston_tea_party_1773`, `first_powered_flight_1903` + 3 benchmark traps + dispute note) |
| GF-21 | Write dispute: common mythologized event | ✅ `docs/04-Disputes/Boston-Tea-Party-Tax-Myth.md` |

## Religion

| ID | Task | Status |
|----|------|--------|
| GF-30 | Add scripture attribution record (sect boundaries) | ✅ v0.7.38 (`hadith_canonical_collections` — Sunni/Shia sect boundaries) |
| GF-31 | Document theological vs historical claim types | `docs/08-Domains/Religion/` (open) |

## Tooling

| ID | Task | Status |
|----|------|--------|
| GF-40 | Improve benchmark scorer for multilingual markers | ✅ v0.7.38 (`agent/benchmark_checks.py` 中文 deny/myth/affirm markers + regression test, CI-wired) |
| GF-41 | Add Colab quickstart notebook | `notebooks/` (open) |
| GF-42 | Export gate verdicts as OpenTelemetry (OTLP) spans | ✅ `sophia_contract/otel_export.py` + `tests/test_otel_export.py` ([doc](docs/09-Agent/Observability-OTel.md)) |
| GF-43 | Add an OTLP exporter for the standalone MCP gate (`gate_check`/`check_claim`), mirroring GF-42 | `sophia_mcp/` (open) |
| GF-44 | Worked LangChain/LangGraph RAG-gate integration with a real retriever | `examples/langgraph_rag_gate.py` is the offline skeleton (open) |