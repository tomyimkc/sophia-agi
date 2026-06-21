# SPEC — Council distillation (internalising the discipline into a small model)

> **Status: PROPOSAL — awaiting approval. No implementation has started.**
> This document is the plan to be approved before any code is written.

## 1. Goal

Teach a **small student model** to natively emit the council's *discipline* —
decompose → cite → gate-clean → **abstain when unsure** — so it behaves like
`council+gate` **without** the multi-pass scaffold at inference time (cheaper,
lower latency, works offline with no MCP).

The teacher is the existing **map-reduce deliberation + gate**
(`agent/council_deliberate.py`); the student is a small open model trained via the
LoRA tooling already in the repo (`tools/train_lora.py`, `tools/claude_model_lab.py`,
`tools/wiki_to_training.py`).

## 2. Success criteria (pre-registered, honest)

Measured with the existing uplift harness (`tools/run_council_uplift.py`) on a
**held-out** task set, scored by the deterministic gate (no LLM judge):

- **Primary:** distilled student (single pass) `cleanRate` ≥ base student
  `+council+gate` minus a small margin, at a fraction of the token cost. i.e. it
  keeps most of the uplift without the scaffold.
- **Guardrail (must hold):** distilled student `cleanRate` ≥ base student `alone`
  on every task stratum (distillation must not make it *worse*).
- **Abstention calibration:** distilled student abstains on the cases the teacher
  abstained on (no new confident hallucinations); false-abstention rate reported.
- All numbers labelled **illustrative** unless they clear the no-overclaim gate
  (≥2 independent eval judges / runs + CIs), per repo policy. No headline from a
  single run.

If the primary criterion fails, that is a **publishable negative result**, not a
silent drop — distillation may not beat the scaffold, and we report which.

## 3. Approach (3 stages)

```
generate  ──►  filter  ──►  train  ──►  eval
(teacher)     (gate)       (LoRA)      (uplift, held-out)
```

### 3.1 Generate teacher traces
- Run `deliberate(query, client=<strong teacher>, gate=True)` over a task set.
- Capture, per task: the gate-passed seat findings and the synthesised decision.
- Render each into a **student training target** in a fixed disciplined format
  (below). Discard tasks where the teacher itself was gated out / abstained-empty,
  unless the gold answer *is* "abstain" (we keep a calibrated share of abstentions).

### 3.2 Filter (the anti-circularity firewall)
- **Keep only gate-passed traces** — no fabricated citation / false arithmetic /
  forbidden attribution enters the training set (run `check_response`; drop on any
  violation).
- **Teacher ≠ student family** (independence; avoid mode collapse).
- **Hold out** an eval split of tasks the student never trains on; run the existing
  `tests/test_contamination.py` / training-safety guard so eval tasks can't leak
  into train.
- Balance faithful/abstain so the student doesn't learn to always-answer or
  always-abstain.

### 3.3 Train
- LoRA SFT via `tools/train_lora.py` on the existing default (Qwen2.5-3B/7B), or a
  student chosen at approval time. Target = the disciplined-format output.
- Optionally a DPO pass (chosen = gate-passed teacher answer, rejected = a
  base-student hallucinated answer) — **stretch goal, v2**.

### 3.4 Eval
- `run_council_uplift.py` with conditions extended to **base-alone**,
  **base+council+gate**, **distilled-alone**. Report per-condition `cleanRate`,
  `answeredRate`, deltas, and abstention calibration. Held-out tasks only.

## 4. Training-example format (the "disciplined output")

```json
{
  "messages": [
    {"role": "system", "content": "<source-discipline + council system prompt>"},
    {"role": "user", "content": "<task prompt>"},
    {"role": "assistant", "content": "Perspectives:\n- <seat>: <finding> [source]\n...\nDecision: <synthesis>\nConfidence/abstention: <…>\n中文摘要: <…>"}
  ],
  "meta": {"teacher": "<spec>", "gatePassed": true, "councilId": "<id>", "labelStatus": "teacher-trace"}
}
```
The assistant target encodes seat structure + citations + an explicit
abstention/confidence line — the *behaviours* we want internalised.

## 5. Files (planned; ~no churn to existing behaviour)

| File | Purpose |
|---|---|
| `tools/distill_council_traces.py` | stage 3.1–3.2: generate + gate-filter teacher traces → JSONL |
| `training/council/` | generated traces (gitignored if large; a small sample committed) |
| `tools/train_lora.py` | reuse (point `--data` at the new JSONL) |
| `tools/run_council_uplift.py` | extend with a `distilled-alone` condition |
| `tests/test_distill_council.py` | offline: trace generation w/ stub teacher; gate-filter drops a dirty trace; format validity |
| `docs/11-Platform/Council-Distillation.md` | the how-to (after it works) |
| `.github/workflows/ci.yml` | run the offline test + a `--model mock` trace-gen smoke |

No changes to `agent/council_deliberate.py` semantics; this is additive.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Distilling the teacher's hallucinations | gate-filter every trace; teacher≠student; keep only gate-passed |
| Overfitting to the template (parroting structure, not reasoning) | held-out eval on *different* tasks; report base-alone guardrail |
| Teacher errors propagate silently | spot-audit a sample; DPO with negatives (v2) |
| Eval contamination | held-out split + existing contamination/training-safety tests |
| Cost (teacher calls, GPU for LoRA) | small task set for v1 (~100–300 traces); cheap teacher via OpenRouter |
| Over-abstention (safe but useless) | report `answeredRate`; balance the abstention share in training |

## 7. Phases / milestones

1. **v0 (plumbing, offline):** `distill_council_traces.py` with stub teacher +
   gate-filter + format validation + tests + CI. *Deliverable: green, no GPU.*
2. **v1 (real traces):** generate ~100–300 gate-passed traces from a real teacher
   (OpenRouter); LoRA-train the student; run the 3-condition uplift on held-out
   tasks; report **illustrative** numbers + honest caveats.
3. **v2 (stretch):** DPO with hard negatives; scale traces; multi-judge eval to
   pursue a *validated* uplift number.

## 8. Open decisions (need your call at approval)

- **Student model:** default Qwen2.5-3B (repo default) vs 7B vs an Ollama model you
  prefer.
- **Teacher:** OpenRouter (e.g. `deepseek` / `llama-3.3-70b`) — same key as before
  (rotate after) — vs a stronger teacher if you have a key.
- **Task domains for traces:** the sector councils (law/finance/economy) only, or
  include philosophy/provenance traps too.
- **Scope to approve now:** v0 only (offline plumbing, zero cost), or v0+v1 (real
  traces + training, incurs teacher/GPU cost)?
