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
W2 bounded-RSI promotion gate. Today that gate only runs in the toy demo
(`tools/run_agi_missing_pillars.py`). Symmetrically, `gate_feedback.py` (gate-vs-judge misses
→ candidate `doNotAttributeTo` records) does **not** feed the next training pack.

So: a training output, a promotion gate, and a feedback miner all exist — and none of them are
connected. The promotion step is the one place where "the gate enforces truth" is currently
**not** mechanically true. Closing that is higher-leverage than training a second adapter.

```
  [train adapter] --metrics--> [W2 plasticity gate + formal proof] --verdict--> ledger/promote
        ^                                                                            |
        |                                                                            v
  [next pack]  <--candidate records--  [gate_feedback miner + learning-under-shift]  (eval)
```

---

## C1 — Wire training output → W2 promotion gate *(do first; no GPU)*

**Why:** turns the hand-written promotion note into a reproducible, machine-checked W2
artifact. This is the highest-leverage item and needs no new training.

**Build:**
1. Add `tools/promote_adapter.py` that reads `eval_ladder_baseline.json` +
   `eval_ladder_adapter.json`, builds an `UpdateCandidate` (per-domain suites; mark
   `Religion`, `History` **protected**), attaches the contamination flag from the dataset
   manifest and the eval-run artifacts, and calls
   `continual_plasticity.evaluate_update(target_suite="Total", ...)`.
2. Emit the `PromotionDecision` to
   `agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json`.
3. Re-run on the current adapter — it **must** return `reject` (Religion −16.7pt protected
   regression), reproducing today's hand decision *automatically*.

**Acceptance:** the v2 adapter is rejected by the gate for the same reason the ledger states;
a test asserts a clean improving candidate would promote and the religion-regressing one
rejects. **Resource:** none. **Honest bound:** a verified *promotion* decision — not a
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

## C3 — Fix the data pipeline defects surfaced by the run *(no GPU)*

**Why:** the v2 run truncated rows >1024 tokens, and SEIB cannot load MLX adapters, so the
adapter was never measured on the suite that actually gates promotion.

**Build:**
1. `tools/split_long_training_rows.py` (or pack-level max-token filtering) so no row is
   silently truncated; record dropped/split counts in the manifest.
2. Add `--model mlx:<base> --adapter <path>` to `tools/run_seib.py` so the trained adapter is
   evaluated on SEIB directly.

**Acceptance:** rebuilt pack has zero >max-token rows; SEIB runs against the MLX adapter and
its result feeds C1. **Resource:** none. **Honest bound:** measurement hygiene, not capability.

---

## C4 — Close the continual loop: feedback miner → next pack *(no GPU)*

**Why:** this is what makes the system *continual* rather than a one-shot train. The miners
already exist; they just aren't connected to the trainer.

**Build:** route `agent/gate_feedback.py` misses and `provenance_bench/improvement.py`
held-out failures into a reviewed candidate queue, and have `build_local_sophia_dataset.py`
optionally ingest **promoted** candidates (manual review step preserved, so it stays
non-circular). Re-run the `learning-under-shift` protocol with the new pack as the post-test.

**Acceptance:** a documented round where mined misses become reviewed training rows and the
learning-under-shift report shows post > pre with protected knowledge stable and contamination
clean. **Resource:** none. **Honest bound:** offline selection + manual promotion — **not**
online weight learning.

---

## C5 — Promotion-grade validation *(the binding constraint: hardware)*

**Why:** even a religion-fixed adapter is single-seed/first-party today. The promotion rule
requires real evidence.

**Build:** ≥3 training seeds with CIs; full SEIB-100 ≥3 runs with multi-judge where semantic
scoring is needed; track false-positive / over-abstention cost; only then feed C1 for a real
`promote`. This is also the on-ramp to **W4** (live RLVR, `rlvr-live-run-not-yet-gated`).

**Acceptance:** clears the existing promotion rule (provenance/citation up at acceptable
false-positive cost, no useful-correctness regression) with CIs excluding 0.
**Resource:** **GPU / Mac hardware budget — the gate on the whole tier.**
**Honest bound:** evidence the training recipe helps at small scale — not AGI.

---

## Sequence

| Order | Item | Resource | Ships |
|---|---|---|---|
| 1 | **C1** wire training → W2 gate | none | machine-checked promotion (auto-reproduces today's decision) |
| 2 | **C3** pipeline fixes (split rows, SEIB-MLX) | none | honest measurement of the real adapter |
| 3 | **C2** fix religion regression | hardware (retrain) | the one failing predicate cleared |
| 4 | **C4** feedback miner → next pack | none | the loop becomes *continual* |
| 5 | **C5** multi-seed + SEIB-100 + (W4 RLVR) | **GPU** | the first promotable, validated result |

**Prerequisite under all of it (unchanged):** the third-party-validated number still open in
the ledger (`calibration-self-authored-pack-2026-06-22`, `hidden-review-third-party-not-run`).
Without one independent result, a `promote` verdict is still first-party.

**One line:** don't train a second adapter yet — first make the *promotion* decision itself a
verified, closed loop (C1), then fix the one thing that fails it (C2/C3), then let the system
feed its own next pack (C4). That is the honest, achievable convergence of training, bounded
RSI, and continual learning.
