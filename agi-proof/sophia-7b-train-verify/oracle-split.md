# Oracle split — sophia-7b-train-verify

**Registered before any training run.** This experiment treats two disjoint oracle
families. A pass on one is **never** cited as evidence from the other.

## Training oracle (release gate only)

Used to decide whether an adapter may ship inside this repository's local-Sophia
program. **Not third-party evidence.**

| Component | Tool / module | Role |
|---|---|---|
| Moral Gate router | `training/moral_gate_sft.jsonl`, `agent/gate.py` | Routing SFT + runtime filter |
| Conscience / public standard | `moral_corpus/`, `tools/build_moral_gate_sft.py` | Policy-routing supervision |
| Eval ladder (CONTENT channel) | `tools/eval_ladder.py` | Domain benchmark pass gate |
| W2 promotion gate | `tools/promote_adapter.py` | Protected-floor promote/quarantine/reject |
| Invariant suite + Gödel oracle | `agent/invariant_suite.py`, `agent/godel_oracle.py` | Formal invariants, `solverChecked: true` |
| Positive control | `tools/run_positive_control.py` | Known-good must promote on z3 |
| Hard-negative DPO preference | `training/hard_negatives_dpo.jsonl` | Fabrication vs abstain+cite pairs |

**Label in all reports:** `releaseGate: true`, `thirdPartyEvidence: false`.

## Evidence oracle (third-party / disjoint)

Used for falsifiable headline claims. **Never** substituted by training-oracle passes.

| Component | Tool | Pre-registered role |
|---|---|---|
| Vectara HHEM (lead) | external API / manual run | Hallucination rate vs pre-registered ceiling |
| Hidden reviewer pack | `tools/hidden_eval_protocol.py`, `tools/run_hidden_eval_sophia.py` | Calibration Δ on sealed commitments |
| HF Open LLM Leaderboard | `tools/run_external_eval.py` (context) | Broad capability context only |
| Artificial Analysis | external (context) | Cost/latency context only |

**Explicitly not leading with:** LMArena (scoping decision recorded in failure ledger).

## Contamination contract

- Training data is decontaminated via `tools/build_local_sophia_dataset.py` (fail-closed).
- Holdout seal: `agi-proof/sophia-7b-train-verify/heldout-seal.manifest.json` (prompt-only digests).
- Synthetic builders (`wiki_to_training.py`, `mine_hard_negatives.py`, Moral Gate SFT) must **not**
  read `training/lora/holdout.jsonl` assistant gold — benchmark question deleak only.

## canClaimAGI

`False` — always, for this experiment.
