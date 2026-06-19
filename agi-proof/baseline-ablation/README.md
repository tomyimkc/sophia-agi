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
