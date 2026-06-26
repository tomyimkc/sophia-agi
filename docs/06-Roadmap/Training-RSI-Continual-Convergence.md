# Convergence Plan — wire Training × W2-RSI × Continual-Learning into one closed loop

**Context.** Three independent work tracks now exist in this repo and each works on its own,
but they do not talk to each other:

1. **Local training** (`docs/11-Platform/Local-Sophia-Training.md`) — produced one trained
   Qwen2.5-3B MLX adapter + a baseline-vs-adapter eval ladder.
   Status: `local-sophia-v2-mlx-trained-not-promoted-2026-06-24` (NOT promoted).
2. **Bounded RSI / W2** (`docs/06-Roadmap/AGI-Substrate-Plan.md` §W2) —
   `agent/continual_plasticity.py`, the *self-improvement-with-a-proof* promotion gate, plus
   `agent/formal_verifier.py`.
3. **Continual learning** — `agent/memory_consolidation.py`, `agent/gate_feedback.py`,
   `provenance_bench/improvement.py`, and the `agi-proof/learning-under-shift/` protocol.

> Nothing here is an AGI claim. This plan only connects existing, honest pieces and ships
> under the no-overclaim gate (`tools/lint_claims.py`). Every item names its honest bound.

---

## The gap (why this plan exists)

The adapter's promotion decision was written **by hand** as a failure-ledger note. It was
**never run through `agent/continual_plasticity.evaluate_update()`** — which is precisely the
W2 bounded-RSI promotion gate (which only ran in the toy `tools/run_agi_missing_pillars.py`).
Symmetrically, `gate_feedback.py` (gate-vs-judge misses → candidate `doNotAttributeTo`
records) did **not** feed the next training pack.

So: a training output, a promotion gate, and a feedback miner all existed — and none of them
were connected. C1 wired the promotion step (the one place "the gate enforces truth" was not
mechanically true); C4 wired the feedback miner back into the pack. Both were higher-leverage
than training a second adapter.

```
  [train adapter] --metrics--> [W2 plasticity gate + formal proof] --verdict--> ledger/promote
        ^                                                                            |
        |                                                                            v
  [next pack]  <--candidate records--  [gate_feedback miner + learning-under-shift]  (eval)
```

---

## C1 — Wire training output → W2 promotion gate ✅ *(done; no GPU)*

**Why:** turns the hand-written promotion note into a reproducible, machine-checked W2
artifact. This is the highest-leverage item and needs no new training.

**Built:** `tools/promote_adapter.py` reads `eval_ladder_baseline.json` +
`eval_ladder_adapter.json`, builds an `UpdateCandidate` (per-domain suites; `religion`,
`history` marked **protected**), attaches the contamination flag from the dataset manifest +
the on-disk run artifacts, runs an **independent** `agent/formal_verifier` protected-floor
lattice proof, and calls `continual_plasticity.evaluate_update(target_suite="total", ...)`.
The `PromotionDecision` is written to
`agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json`. `--fail-on-reject`
makes it CI-gateable.

**Result:** on the current adapter the gate returns **`reject`** — protected regression on
`religion` (−0.167) — and the formal proof independently agrees (`after_religion(0) >= 157`
violated). This reproduces today's hand decision *automatically*. Gated by
`tests/test_promote_adapter.py` (also asserts a clean improving candidate promotes), wired
into `.github/workflows/ci.yml`. **Honest bound:** a verified *promotion* decision — not a
verified *capability*.

---

## C2 — Fix the religion regression *(the concrete training blocker)*

**Why:** it is the single failing predicate in C1 and the stated reason for non-promotion.

**Build:** add targeted SFT/council traces (per the ledger's next-experiment note) for the
known weak spots — gospel-authorship council, ancestor-veneration vs worship, Dao De Jing
religion/philosophy boundary, hadith sect boundary, Buddha/nirvana pop-myth — then rebuild
the pack (decontam guard must stay CLEAN).

**Acceptance:** Religion ≥ baseline (≥1/6, target ≥3/6) **and** no new protected regression
elsewhere, checked through the C1 gate. **Resource:** none (data) + your hardware (retrain).
**Honest bound:** a less-uneven adapter — not a validated wisdom model.

---

## C3 — Fix the data pipeline defects surfaced by the run ✅ *(done; no GPU)*

**Why:** the v2 run truncated rows >1024 tokens, and SEIB could not load MLX adapters, so the
adapter was never measured on the suite that actually gates promotion.

**Built:**
1. `tools/split_long_training_rows.py` — fits MLX chat rows under a token budget (offline
   heuristic counter by default; `--hf-tokenizer` for the real tokenizer). It keeps fitting
   rows, splits multi-turn rows at turn boundaries, and drops single turns that cannot fit —
   asserting **no emitted row exceeds the budget**. Wired into `build_local_sophia_dataset.py`
   (`MLX_MAX_TOKENS = 1024`); the manifest now records a `mlx.fit` block. On the current pack
   it dropped 11 overlong single-turn rows that the v2 run would have silently truncated.
2. `agent/model.py` gained an `mlx` provider/transport (lazy `mlx_lm`, fails closed off-Mac,
   exempt from the airgap egress block), and `tools/run_seib.py` gained
   `--model mlx:<base> --adapter <path>` (sets `SOPHIA_MLX_ADAPTER`; records `adapterPath`).

**Result:** rebuilt pack has zero >max-token rows (idempotent re-fit drops nothing); the SEIB
`mlx:` path routes correctly and, where `mlx_lm` is absent, writes a clean
"environment artifact, not a score" report instead of crashing. On Apple Silicon the same
command evaluates the real adapter, feeding C1. Gated by `tests/test_split_long_training_rows.py`
and `tests/test_mlx_transport.py` (both in CI). **Honest bound:** measurement hygiene, not
capability.

---

## C4 — Close the continual loop: feedback miner → next pack ✅ *(done; no GPU)*

**Why:** this is what makes the system *continual* rather than a one-shot train. The miners
already existed; they just weren't connected to the trainer.

**Built:** `tools/feedback_to_training.py` provides the return path in three explicit stages —
`mine` (gate MISSES from run/case results → deduped pending queue, all `promoted:false`),
`approve` (the human review step flips specific rids to `promoted:true` with a note), and
`build-sft` (ONLY promoted candidates → `training/feedback/sft_from_feedback.jsonl` +
a promoted gate-records file). `build_local_sophia_dataset.py` ingests that SFT file as a
normal source, so it passes the **same decontamination guard**. Workflow + non-circularity
contract documented in `training/feedback/README.md`.

**Non-circularity (machine-checked):** pending candidates live in a separate file and are
never merged into the frozen runtime records; `build-sft` emits nothing until a human
promotes (default-deny); ingested rows are decontaminated like any source. End-to-end demo:
4 case results → 2 genuine misses mined (non-hallucinated / already-revised skipped) → 0 SFT
rows before approval → 1 well-formed source-discipline SFT row after one approval → ingested
into the pack (`present:true, droppedForDecontamination:0`, guard CLEAN). Gated by
`tests/test_feedback_to_training.py` (in CI).

**Remaining (needs a model backend):** re-run `tools/run_learning_shift.py` with the new pack
as the post-test to show post > pre with protected knowledge stable — runs on your hardware
(`--backend adapter`), not in CI. **Honest bound:** offline selection + manual promotion —
**not** online weight learning.

---

## C5 — Promotion-grade validation *(the binding constraint: hardware)*

**Why:** even a religion-fixed adapter is single-seed/first-party today. The promotion rule
requires real evidence.

**Build:** ≥3 training seeds with CIs; full SEIB-100 ≥3 runs with multi-judge where semantic
scoring is needed; track false-positive / over-abstention cost; only then feed C1 for a real
`promote`. This is also the on-ramp to **W4** (live RLVR, `rlvr-live-run-not-yet-gated`).

**Acceptance:** clears the existing promotion rule (provenance/citation up at acceptable
false-positive cost, no useful-correctness regression) with CIs excluding 0, **and** the
learning-under-shift run returns `passingSignal=true` (old-task retention within tolerance —
see C6).
**Resource:** **GPU / Mac hardware budget — the gate on the whole tier.**
**Honest bound:** evidence the training recipe helps at small scale — not AGI.

---

## C6 — Retention gate: the promotion gate may not reward forgetting ✅ *(done; no GPU)*

**Why:** C1's gate read only the eval ladder + the protected-floor proof — it had **no
old-task retention term**. So the v3 adapter (`local-sophia-v3-mlx-promoted-by-w2-but-not-validated`)
was promoted on a `+0.25` total delta even though its learning-under-shift report showed a
`-50.0pp` old-benchmark regression (`passingSignal=false`). The gate rewarded catastrophic
forgetting — which is the opposite of continual learning. A loop whose gains do not persist
across iterations is a treadmill, not self-improvement.

**Built:** `agent/continual_plasticity.evaluate_update()` takes an optional `RetentionEvidence`
(old-task delta + `passingSignal` + `stabilityEvaluable`) and treats an old-task regression
beyond tolerance (default `5.0pp`, matching the learning-shift stability rule) as a **hard
reject**, on par with a protected-suite regression. Unverifiable retention is not a silent pass:
`require_retention` forces `quarantine` when no verifiable signal is supplied.
`tools/promote_adapter.py` gained `--shift-report` / `--max-retention-regression` /
`--require-retention`, parses the learning-under-shift public report, and records a `retention`
block in the promotion artifact.

**Result:** re-running the gate on v3 with its real shift report flips the verdict to
`reject` — `agi-proof/continual-plasticity/local-sophia-v3-mlx-retention-gated.public-report.json`.
The historical ladder-only `promote` artifact is kept unaltered for provenance. Gated by
`tests/test_promote_adapter.py::test_v3_adapter_rejects_under_retention_gate` and the
`RetentionEvidence` cases in `tests/test_continual_plasticity.py`. **Honest bound:** this stops
the gate from *rewarding* forgetting; it does not *solve* forgetting — that is the open training
work (replay/rehearsal of the old domain or a smaller weight delta, re-run to
`passingSignal=true`), still hardware-bound under C5.

**Multi-goal extension.** `agent/continual_plasticity.evaluate_update_multigoal(goals=...)` (and
`tools/promote_adapter.py --goal SUITE[:MIN_DELTA]`, repeatable) generalize the single
`target_suite` rule to N objectives under a **Pareto rule**: every goal must clear its own floor
and *no* goal or protected suite may regress beyond tolerance, so lifting one goal by sacrificing
another (v3 raised aggregate while religion stayed flat) is a `reject`, not a `promote`. This is
the gate the multi-goal v4 training plan targets; it composes with the retention term above and
with PR #82's CLS consolidation (`agent/cls_consolidation.py`), which routes only stable,
gate-cleared facts into weights while declarative knowledge stays non-parametric in the OKF
graph. Gated by the `multigoal` cases in `tests/test_continual_plasticity.py` and
`tests/test_promote_adapter.py`.

---

## Sequence

| Order | Item | Resource | Ships |
|---|---|---|---|
| 1 | **C1** wire training → W2 gate ✅ | none | machine-checked promotion (auto-reproduces today's decision) |
| 2 | **C3** pipeline fixes (split rows, SEIB-MLX) ✅ | none | honest measurement of the real adapter |
| 3 | **C2** fix religion regression | hardware (retrain) | the one failing predicate cleared |
| 4 | **C4** feedback miner → next pack ✅ | none | the loop becomes *continual* |
| 5 | **C6** retention gate (no rewarding forgetting) ✅ | none | the gate rejects catastrophic forgetting; v3 re-verdicts to `reject` |
| 6 | **C5** multi-seed + SEIB-100 + retain old task + (W4 RLVR) | **GPU** | the first promotable, validated result |

**Prerequisite under all of it (unchanged):** the third-party-validated number still open in
the ledger (`calibration-self-authored-pack-2026-06-22`, `hidden-review-third-party-not-run`).
Without one independent result, a `promote` verdict is still first-party.

**One line:** don't train a second adapter yet — first make the *promotion* decision itself a
verified, closed loop (C1), then fix the one thing that fails it (C2/C3), then let the system
feed its own next pack (C4). That is the honest, achievable convergence of training, bounded
RSI, and continual learning.
