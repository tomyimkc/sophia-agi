# Tool-Use Training Plan — Calibrated, Grounded Tool Calling

**Status:** PLAN (candidate infrastructure). `candidateOnly: true`, `level3Evidence: false`,
`canClaimAGI: false`. Nothing in this plan claims AGI, general agentic capability, or validated
uplift. Every milestone is gated by a sealed, decontaminated, adversarial benchmark.

---

## 1. Goal and the core reframe

**Goal:** train a Sophia model that performs best at **tool calling** — measured, not asserted.

**Reframe (this is the whole plan):** the target is **not "calls tools more / better-looking
tool calls."** It is a **calibrated, grounded tool-use policy**:

> Call the right tool **only when calling helps**, with **schema-valid arguments**, then
> **ground the answer in the tool output** — and **abstain** when the tool fails or returns nothing.

This is forced by our own evidence. Failure-ledger entry
`local-agent-tools-degrade-strong-model-2026-06-21` showed tool access *degraded* a strong model
(gold 90.2% → 51.2%) until selective invocation was added. So **over-calling and mis-grounding are
the failure modes**, not under-calling. The entire reward and eval design below exists to prevent
re-introducing that degradation.

### What this plan does NOT do
- Does NOT make the model "agentic" or generally tool-capable — it produces a calibrated policy
  measured on a sealed benchmark; generalization beyond it is unproven until tested.
- Does NOT learn to emit valid tool-call JSON by imitation — that is enforced **mechanically** at
  inference (grammar-constrained decoding). Schema-compliance must never masquerade as capability.
- Does NOT reward tool use for its own sake — an unnecessary call is penalized.

---

## 2. Decompose "good at tool calling" into verifiable sub-skills

"Good at tools" is unmeasurable as one blob. It is six behaviors, each with a **hard, deterministic
check** (the executor / schema / an entailment check — never the model judging itself):

| # | Sub-skill | Deterministic check |
|---|---|---|
| S1 | **Decision-to-call** (call vs answer-direct vs abstain) | held-out label: should-call / shouldn't-call / unanswerable |
| S2 | **Tool selection** | ground-truth tool id per task |
| S3 | **Argument construction** | the call **parses against the MCP schema AND executes** |
| S4 | **Result grounding** | final answer is **entailed by** the tool output; abstains on tool error/empty |
| S5 | **Multi-step orchestration** | task solved; **stops** when done (no spurious extra calls) |
| S6 | **Error recovery** | tool timeout/error → **fail-closed** (abstain), never parrot/hallucinate |

Each sub-skill gets its own metric (Section 8). A model can be strong on S3 (valid JSON) and fail
S1/S4 (over-calls, ignores output) — and that is exactly the degradation we must measure separately.

---

## 3. Architecture overview

```
 Phase 0  Sealed tool-use benchmark  ───────────────┐  (build FIRST; nothing is measurable without it)
 Phase 1  Sandboxed trace generation → hard-verify  │
 Phase 2  SFT on VERIFIED traces (train_lora)        │
 Phase 3  Over-call-penalized DPO (mine_hard_neg)  ──┼──►  promote_adapter + invariant gate
 Phase 4  (optional) RLVR with executor reward       │       (solverChecked release gate)
 Phase 5  Inference: grammar-constrained decode +    │
          selective-invocation gate (belt+braces)    │
 Phase 6  Adversarial eval vs no-tools & always-tools┘ ──►  held-out evidence (≥3 seeds, CI)
```

Train **content** (when / which / grounding). Enforce **format** (valid call JSON) at decode time.
This FORMAT-vs-CONTENT split is the same lesson that corrected the religion eval in this repo.

---

## 4. Phase 0 — Build the sealed tool-use benchmark FIRST

Without this you cannot measure anything; build it before touching the model.

- **File:** `data/tool_use_benchmark/heldout_v1.jsonl` (sealed; outside `eval/**` globs so it
  never pollutes other sealed sets).
- **Each case:**
  ```json
  {
    "id": "str",
    "prompt": "str",
    "registry": ["tool ids available for this case"],
    "label": {
      "decision": "call | answer_direct | abstain",   // S1
      "tool_id": "str | null",                          // S2
      "gold_answer": "str | null",                      // grounding target
      "gold_citation": "str | null"
    },
    "trap": "none | wrong_tool_output | empty_tool_output | already_known | unanswerable",
    "wouldHelp": true
  }
  ```
- **Composition (balanced, ≥120 cases):**
  - ~⅓ **should-call** (tool genuinely needed).
  - ~⅓ **shouldn't-call / already-known** (base model already answers correctly — calling must
    NOT lower accuracy via over-abstention; this catches over-calling).
  - ~⅓ **traps**: `wrong_tool_output` and `empty_tool_output` (correct behavior is distrust/abstain,
    NOT parrot) + `unanswerable` (correct behavior is abstain).
- **Seal it:** hash + manifest; the trace generator and SFT/DPO data must be **disjoint** from it
  (run `tools/build_local_sophia_dataset.py --check`; record CLEAN + the hash).
- **Owner of scoring:** a deterministic scorer + the executor + an entailment check, all
  **out-of-process** and independent of the training reward (disjoint-oracle rule).

**Acceptance:** sealed, balanced, decontaminated benchmark with traps; scorer is deterministic and
external. Do NOT reuse these cases as training data.

---

## 5. Phase 1 — Sandboxed verified-trace generation

- **Module:** `tools/gen_tool_traces.py`. The model attempts tasks (NOT from the held-out set) with
  the real MCP registry inside the **sandboxed execution environment**; every call is **executed**.
- **Hard-verify each step:** S3 (schema-valid + executes), S4 (answer entailed by output / correct
  abstain), S5 (stops appropriately), S6 (handles errors). Keep **only** fully verified traces.
- **Decontaminate:** run `build_local_sophia_dataset.py --check`; drop any trace overlapping the
  sealed benchmark. Record kept/dropped counts.
- **Output:** `training/tool_use/sft_traces.jsonl` (chat format, `messages` + `metadata`, matching
  `train_lora.py`'s `--train` schema).

**Acceptance:** every stored trace is executor-verified and decontaminated; counts logged.

---

## 6. Phase 2 — SFT on verified traces

- **Base:** the existing 7B QLoRA pipeline (`tools/train_lora.py --4bit --mask-prompt`, ≥3 seeds).
- **Data:** `training/tool_use/sft_traces.jsonl` only (verified). Weights gitignored.
- Trains S1–S6 **competence** (the format/protocol + basic decisions). Calibration comes in Phase 3.

**Acceptance:** SFT completes ≥3 seeds; no held-out leakage; loss logged.

---

## 7. Phase 3 — Over-call-penalized DPO (the core of calibration)

Calibration is learned from **contrastive mistakes**, not positive traces. Extend the existing
hard-negative loop (`tools/mine_hard_negatives.py`, 590 pairs) for tool use.

- **chosen** = correct decision (S1) + correct tool (S2) + valid args (S3) + grounded answer or
  correct abstention (S4/S6).
- **rejected** (mine all of these):
  - **over-call**: called a tool when the model already had the right answer;
  - schema-invalid / non-executing args;
  - **mis-ground**: answer not entailed by the tool output (hallucinated over it);
  - **ignored error**: parroted a failed/empty tool result instead of abstaining;
  - wrong tool selected; spurious extra calls (S5).
- **Output:** `training/tool_use/dpo_pairs.jsonl`. DPO on top of the SFT adapter, ≥3 seeds.

### Reward / preference ordering (the single most important rule)
Correctness + grounding must **dominate**; tool use is instrumental. Enforce this strict ordering:

```
correct + grounded (no needless call)   >   correct + grounded (with needed call)
   >   correct via needed call but weakly grounded
   >   wrong/abstain-when-should-answer
   >   over-call / mis-ground / ignored-error   (explicitly penalized)
```

A **correct direct answer with NO call must outrank a correct answer with a needless call.** Without
this explicit over-call penalty you will re-create the `local-agent-tools-degrade-strong-model`
degradation. This is the tool-use analogue of the "confession reward" trap (reward the behavior, not
the outcome → reward-hacking).

**Acceptance:** DPO completes ≥3 seeds; on the **dev** split, over-call rate drops and grounding rate
rises vs the SFT model. (Dev only — the sealed benchmark is touched once, in Phase 6.)

---

## 8. Phase 4 — (Optional) RLVR with executor reward

Only after Phase 3 looks healthy. Reward = the executor/entailment verdict (verified = positive),
with the same over-call penalty baked into the reward function. Offline reward-wiring invariants must
pass first (mirror the math-RLVR rung). Skip if compute-bound — it is not required for a first result.

---

## 9. Phase 5 — Inference: enforce format, keep the gate

- **Grammar-constrained / structured decoding:** constrain generation to the **valid MCP tool-call
  grammar** so malformed calls are *impossible* (S3 becomes a mechanical guarantee, not a learned
  approximation). Do NOT spend training capacity on JSON validity.
- **Keep the selective-invocation gate** (the fix from `local-agent-tools-degrade-strong-model`):
  tools fire only on low-confidence answers, with rich tool outputs. Training improves the policy;
  the gate bounds the worst case. Belt-and-braces.

**Acceptance:** every emitted tool call is schema-valid by construction; the selective gate remains
active and configurable.

---

## 10. Phase 6 — Adversarial evaluation (so you cannot fool yourself)

Run the trained model on the sealed benchmark, scored by the external executor + entailment check
(disjoint from the training reward), **≥3 seeds, CI must exclude 0**. Compare against **two** strong
baselines: **no-tools** and **always-tools** (our ledger says tools can lose — both bars are real).

| Metric | Definition | Hard-to-game because |
|---|---|---|
| **Tool-decision calibration** (S1) | accuracy of call/direct/abstain, vs no-tools baseline | paired vs baseline; held-out labels |
| **False-call cost** | calls on `shouldn't-call` / `already-known` cases | penalizes over-calling directly |
| **Tool-selection acc** (S2) | right tool id | ground-truth id |
| **Schema-valid rate** (S3) | parses + executes | executor, not self-report |
| **Grounding rate** (S4) | answer entailed by tool output | external entailment check |
| **Trap behavior** (S4/S6) | distrust/abstain on `wrong_/empty_tool_output`; abstain on `unanswerable` | a parrot model fails these |
| **Task pass@1** | end-to-end correct | task scorer |
| **Over-call rate** | needless calls / total | the degradation metric |

**Promotion bar:** must **beat BOTH baselines** on net task pass@1 **without raising false-call or
over-call rates**, CI excluding 0, decontaminated, ≥3 seeds — AND clear the internal invariant gate
(`tools/promote_adapter.py`, `solverChecked: true`, no protected regression). The invariant gate is a
**release gate, not third-party evidence**; the sealed benchmark is the evidence.

A within-noise or negative result is a valid honest outcome — record it; do not tune on the test set.

---

## 11. Failure modes and guards

| Failure mode | Guard |
|---|---|
| **Tool-spamming for diligence** (reward-hack) | over-call penalty in DPO/RLVR ordering (Phase 3); over-call rate is a promotion-blocking metric |
| **Over-abstention on already-known** (false-correction) | `already-known` cases in the benchmark; calling must not lower accuracy there |
| **Format-fitting** (valid JSON ≠ capability) | enforce JSON at decode (Phase 5); score S1/S4 separately from S3 |
| **Parroting bad tool output** | `wrong_/empty_tool_output` traps; grounding/entailment check |
| **Benchmark gaming / contamination** | sealed held-out, fresh, decontaminated; trace gen disjoint; external scorer |
| **Self-graded evidence** | executor + entailment check are the oracle, disjoint from training reward |

---

## 12. Concrete module / file layout

```
data/tool_use_benchmark/heldout_v1.jsonl     # Phase 0 (sealed; hash in manifest)
data/tool_use_benchmark/manifest.json
agent/tool_use/policy.py                      # decision + selection helpers (S1/S2)
agent/tool_use/verifier.py                    # S3 schema/exec, S4 grounding/entailment, S6 error checks
tools/gen_tool_traces.py                      # Phase 1 (sandboxed, executed, verified)
training/tool_use/sft_traces.jsonl            # Phase 1 output
training/tool_use/dpo_pairs.jsonl             # Phase 3 output (gitignored if large)
tools/eval_tool_use.py                        # Phase 6 (≥3 seeds, CI, vs both baselines)
tests/test_tool_use_verifier.py               # hard-checks: valid→accept, malformed→reject, fail-closed
agi-proof/tool-use/eval-tool-use.public-report.json   # candidateOnly artifact
```

Reuse, do not reinvent: `tools/train_lora.py`, `tools/mine_hard_negatives.py`,
`tools/promote_adapter.py` + `agent/invariant_suite.py` (solver-checked gate),
`tools/build_local_sophia_dataset.py --check`, the sandbox, the selective-invocation gate, and
`tools/hidden_eval_protocol.py` (for an optional reviewer-controlled tool-use pack later).

---

## 13. Milestones and acceptance gates

| Phase | Deliverable | Gate to pass |
|---|---|---|
| 0 | Sealed tool-use benchmark | balanced + traps + decontam CLEAN + hash; external scorer |
| 1 | Verified SFT traces | every trace executor-verified; disjoint from benchmark |
| 2 | SFT adapter (≥3 seeds) | trains; no leakage |
| 3 | Over-call-penalized DPO | dev over-call ↓, grounding ↑ vs SFT |
| 4 | (opt) RLVR | offline invariants pass; gated run |
| 5 | Constrained decode + gate | tool calls schema-valid by construction; selective gate live |
| 6 | Adversarial eval | beats no-tools AND always-tools, CI excl. 0, ≥3 seeds; invariant gate promote |

---

## 14. Discipline (applies to every phase)

- **Decontamination + sealed held-out**: benchmark never seen by trace gen / SFT / DPO; `--check`
  CLEAN; hash recorded.
- **Disjoint oracle**: the executor/entailment checker that *labels* training data is NOT the
  evaluator cited as evidence; the sealed benchmark is the evidence.
- **≥3 seeds, CI excludes 0** for any cited number; small-N is "within noise."
- **Fail-closed** everywhere (tool error → abstain; missing evidence → reject/no-promote).
- `python tools/lint_claims.py` OK before every commit; `candidateOnly: true`; `canClaimAGI: false`.
- Record every run (including honest negatives) in `agi-proof/failure-ledger.md`.
- Push after every commit; verify `git ls-remote` SHA == local HEAD.

---

## 15. Reality check (what a successful run does and does NOT prove)

A successful run proves: on a **sealed, adversarial, decontaminated benchmark**, the model calls the
right tool when calling helps, forms schema-valid args, grounds its answer in the tool output or
abstains on failure, and **does not over-call** — beating both a no-tools and an always-tools baseline
with a CI that excludes zero. It does **not** prove general agentic capability, correctness outside
the benchmark, or anything about AGI. Generalization is unproven until tested on an independent pack.
