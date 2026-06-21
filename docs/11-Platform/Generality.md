# Generality track — verifier-gated reasoning, measured honestly

Four capabilities that extend Sophia's core asset (a *verifier-gated* reasoning
loop: retrieve → reason → machine-checked verifier → repair/abstain) beyond the
provenance niche. Each ships with a **falsifiable metric** and an **honest scope
label** — none of this licenses the word "AGI"; it is engineering toward broader,
*checkable* competence.

## 1. More machine-checked verifiers (`agent/verifiers.py`)

The loop is verifier-agnostic. New deterministic (no-model) verifiers:

| Verifier | Checks | Honest scope |
|---|---|---|
| `citation_faithful(sources)` | cited sentences are lexically supported by their source | catches fabricated / mismatched / out-of-range citations; **not** subtle wrong-predicate (needs an LLM-judge) |
| `code_tests_pass()` | extracts code from the answer and **executes** it; pass iff exit 0 | strongest check (runs the code); `allow_execution`/`SOPHIA_ALLOW_CODE_EXEC` gates untrusted exec, else syntax-only |
| `arithmetic_sound()` | recomputes stated `a OP b = c` equalities | binary arithmetic only; ignores prose without equations |

> **Scope honesty:** more verifier *kinds* is **engineering reuse, not
> generality**. It widens the set of tasks the loop can gate; it does not by
> itself prove broader cognition. Registry: `verifiers.VERIFIERS` + `check_text`.

## 2. Measured self-improvement loop (`provenance_bench/improvement.py`)

A contamination-free continual-learning loop. **Falsifiable claim:** held-out
recall rises cycle-over-cycle as the system learns rules from its own TRAIN
failures, at ~0 false-positive cost.

- TRAIN and HELD-OUT are **disjoint phrasing templates** of the same forbidden
  pairs; rules are learned only from TRAIN. A learned rule must generalize to an
  **unseen surface form** to score — the metric cannot be gamed.
- First run: held-out recall **17% → 98%** over 6 cycles, monotone, **0%**
  false-positive cost. `python tools/run_improvement_loop.py`.
- **Scope honesty:** learns specific do-not-attribute rules and generalizes across
  *phrasing*, **not** across new entities. Deterministic (templated text) to
  isolate the loop mechanics; a model-in-the-loop version is next.

## 3. Long-horizon autonomy curve (`agent/horizon.py`)

Success rate vs task length on chained-dependent tasks (one slip fails the task),
judged by an **external oracle** (independent recomputation), never self-report.
Headline = the **effective horizon** (longest length solved at ≥50%).

- `python tools/run_horizon_curve.py` — perfect solver → horizon = max length
  (sanity); a 10%-per-step-error solver decays to a short horizon (realistic).
- Complements `tools/run_long_horizon.py` (which logs one run's interventions).
- **Scope honesty:** chained arithmetic is a clean proxy; real-agent horizon
  needs the model-in-the-loop runner (`--model`) over richer tasks.

## 4. External-oracle eval (`agent/external_eval.py`)

Correctness scored against **external gold**, independent of the gate.
Dataset-agnostic JSONL (`{question, answer}`).

- `python tools/run_external_eval.py` runs a committed 10-item **style sample**
  (clearly labelled — *not* a benchmark result).
- `--dataset path/to/gsm8k.jsonl --model <spec>` runs the real thing.
- **Scope honesty:** a real external number requires the actual public dataset
  and is only quotable with N, model, and dataset version. The committed sample
  proves the harness, not capability.

## 5. Verifier synthesis — the bridge toward generality (`agent/verifier_synthesis.py`)

The four axes above all *reuse* hand-written verifiers; the loop stays only as
general as the checks it ships with. [Verifier-Synthesis.md](Verifier-Synthesis.md)
is the one piece that attacks that ceiling: the loop **synthesises** candidate
checks for a task it has never seen, **meta-verifies** them against an independent
oracle before trusting any (admit only measured precision + recall), and
**abstains** when none qualify — with calibrated confidence as the fallback for
the genuinely unverifiable (`agent/calibration.py`).

- Falsifiable: WITH meta-verification, in-library precision 1.00 / recall 1.00 and
  100% abstention on out-of-library tasks; WITHOUT, 100% false-admission — the
  ablation proves the validation step earns the trust. `python tools/run_verifier_synthesis.py`.
- **Scope honesty:** a finite template library (a model proposer widens it but
  never confers trust); tasks that don't reduce to a checkable predicate stay out
  of reach, where calibrated abstention is the correct behaviour. Still not AGI —
  but it is the honest direction *toward* it.

## 6. Cross-entity generalization — the measured limit (`provenance_bench/cross_entity.py`)

The self-improvement loop generalizes across *phrasing*; this benchmark asks the
harder question — does it transfer to *unseen entities*? — and answers it honestly
on an **entity-disjoint** split (no author or work shared between train and test):

| regime | recall on UNSEEN entities | false-positive |
|---|---|---|
| memorized rules | **0%** (no transfer) | 0% (precise) |
| structural detector | 100% (transfers) | **100%** (can't tell true from false) |
| *(memorized, on SEEN entities)* | *100%* | *—* |

- `python tools/run_cross_entity.py` — six falsifiable invariants, holds across seeds.
- **The honest conclusion:** neither pattern memorization (precise, no transfer)
  nor structural detection (transfers, imprecise) gives low-false-positive
  cross-entity generalization. That requires **external grounding** (retrieval / a
  knowledge base) — which is exactly why Sophia's answer is the retrieval-grounded
  verifier-gated loop, not a learned per-pair classifier. This benchmark names the
  next real frontier precisely instead of papering over it.

## What this is NOT

- Not "AGI", and not evidence of it. The earned claim is: *a verifier-gated loop,
  reused across several checkable task types, with a measured (small, honest)
  grounding effect and reproducible capability curves.*
- The gate only helps tasks whose correctness reduces to **checkable** claims.
- Every published number must clear the [no-overclaim gate](../../SECURITY.md)
  (multi-judge consensus / external oracle, CIs, runs) — see [RESULTS.md](../../RESULTS.md).
