# Baseline And Ablation Protocol

The proof question is not whether Sophia can answer visible examples. The proof
question is whether Sophia's method adds value beyond the base model.

## Required Modes

| Mode | Meaning |
|---|---|
| `raw-model` | base model without Sophia prompt, corpus, gate, memory, or tools |
| `raw-model-plus-tools` | base model with tools but without Sophia source discipline |
| `sophia-full` | full Sophia corpus, RAG, gate, agent path, and memory policy |
| `sophia-no-kb` | Sophia instructions without local source records |
| `sophia-no-gate` | Sophia retrieval without post-generation gate |
| `sophia-no-memory` | Sophia without decision/correction memory |
| `sophia-no-council` | Sophia without council-style multi-voice synthesis |

## Required Report Fields

- same hidden question pack;
- same scorer;
- same model family where possible;
- score and pass-rate delta vs `sophia-full`;
- cost and latency;
- all failure cases.

## Passing Signal

Sophia-full should outperform raw and ablated variants on hidden tasks,
especially attribution traps, tradition-boundary tasks, and mixed-domain
questions.

## Runner

`tools/run_ablation_sophia.py` runs every mode over the same pack and the same
scorer (the shared `run_case` pipeline in `tools/run_hidden_eval_sophia.py`), then
publishes per-mode scores and `full-minus-mode` deltas plus a falsification check
against `preregistered-thresholds.md` line 32.

```bash
# Smoke / demo on the visible example pack (needs a working backend):
python3.12 tools/run_ablation_sophia.py agi-proof/baseline-ablation/example-pack.json \
  --backend grok --modes all \
  --out agi-proof/baseline-ablation/ablation-deltas-EXAMPLE.public-report.json
```

A real Level-3 artifact uses a private, reviewer-authored, unspent pack. Deltas
are auto keyword/regex scores until two-pass manual semantic review is done. A
null or negative delta is reported honestly — it is a valid scientific outcome.
Unit tests: `tests/test_ablation_runner.py`.
