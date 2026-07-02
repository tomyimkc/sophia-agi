# A-series — Horizon-scaling takeaways from Agents-A1 (arXiv 2606.30616) — 2026-07-02

> Source paper: "Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter
> Performance with a 35B Agent" (Agents-A1 Team, Shanghai AI Laboratory, 29 Jun 2026,
> arXiv 2606.30616v1, 29 pp). Studied by a three-agent team (methods / data+eval /
> repo-mapping); every repo path below was verified against this branch.
> **Honesty boundary:** everything here is a proposal. candidateOnly:true /
> level3Evidence:false / canClaimAGI:false until a pre-registered, decontaminated,
> gated harness (chart 7 discipline) says otherwise.

## The paper in one paragraph (what we verified, not the marketing)

Agents-A1 is a 35B MoE (base Qwen3.5-35B-A3B) that matches or beats 1T-class models on
several long-horizon agent benchmarks. Its stage-decomposition table (their Table 4) shows
**the capability gain lives overwhelmingly in long-horizon trajectory SFT data** — ~100K
trajectories, 45K-token average, built on a knowledge-action graph (KAG) that records
(state, action, observation, **verifier outcome**) per step and keeps failures — while the
stage-3 **multi-teacher domain-routed on-policy distillation with salient vocabulary
alignment (SVA)** functions mainly as *conflict repair + specialist consolidation* (it fixes
the regressions full-domain SFT causes and imports teacher deltas; students do not exceed
teachers). Zero architecture changes: horizon is scaled in data + training + harness.
Measurement caveats we will NOT copy: no decontamination, no CIs on most numbers, no
component ablations (SVA, routing, horizon length are never isolated), tool-augmented model
vs tool-free baselines on the science benchmarks.

## Why this paper fits Sophia unusually well

- Their KAG record `(s, a, o, v)` is what `tools/run_hidden_eval_sophia.py:run_case()`
  already emits per case (`toolLog`, `memoryDiff` with hash proofs, gate verdicts incl.
  fail-closed `held`), and what `tools/run_long_horizon.py` emits per step
  (`verification` events with `passed`, `objectiveGate`). Sophia has the *record*; it has
  never serialized it as *training trajectories*.
- Their per-domain teachers are Sophia's council: `agent/council_registry.py` declares
  **21 discipline seats each with an adapter slot** (`sophia-{id}-3b`) and fail-closed
  verifier bindings — all stubs today ("real trained discipline adapters" is a named open
  gap). `agent/adapter_registry.py` already implements the acceptance-gated per-domain
  adapter registry the paper needs (≥2 judge families, CI excluding 0, κ≥0.40).
- Their five task-acceptance criteria (verifiable / valid / process-informative /
  evidence-covering / no-shortcut) are Sophia's epistemics rendered as a data-quality
  gate — "evidence-covering" IS provenance discipline.
- Their missing stage — multi-teacher on-policy distillation — has **no counterpart in the
  repo** (nearest seams: `agent/closed_loop.py:train_step` injection point,
  `tools/run_rlvr.py`). That is the one genuinely new mechanism to build.

---

## Ranked proposals (A1–A7): leverage = capability gain ÷ effort

### A1 — Sophia Trajectory Packs: serialize run_case/long-horizon records as (s,a,o,v) training data
**Paper:** §2.2 (KAG), §3.2 (loss masks on non-student content), Table 2 (45K-token avg).
**Repo seams:** `run_case()` return fields (`tools/run_hidden_eval_sophia.py:1453-1471`);
`tools/run_long_horizon.py:32-43` event taxonomy; `training/tool_use/sft_traces.jsonl`
(already knowledge→action→observation→verdict shaped); `tools/split_long_training_rows.py`
(token-fit); `agent/context_manager.py` (budgeted packing).
**Build:** a `tools/build_trajectory_pack.py` that converts harness artifacts into chat-format
trajectories carrying per-step verifier outcomes in metadata, loss masks on tool
outputs/observations (the repo's `--mask-prompt` MLX path already supports message-level
masking), and the paper's five acceptance criteria as fail-closed pack gates. Keep failures
as DPO negatives (`agent/trace_distill.py` already builds fail-then-fix pairs).
**Weights?** No (asset) until a training run consumes the pack (then chart-8 re-audit).
**Effort:** ~1 week. **Strongest objection:** our per-case traces are short (hundreds of
tokens, not 45K); the pack only becomes "long-horizon" when A5's task generator exists —
sequence A1 before A5 but expect the first packs to be short-horizon.

### A2 — SVA-lite: top-k truncated reverse-KL distillation step for MLX (the missing stage 3)
**Paper:** §2.3.1 Eq. 4 (renormalized reverse KL on the teacher's top-k support, computed on
the student's own rollouts, loss only on student tokens), Eq. 5 (coverage ρ monitor), §2.3.2
Eq. 6 (average within domain, then across active domains; hard teacher routing).
**Repo seams:** `training/mlx_adapters/` + `tools/train_lora.py` (Qwen2.5-3B);
`agent/adapter_registry.py:resolve()`; `agent/closed_loop.py:train_step` injection point;
`_wrap_collapse_logger` in `tools/run_rlvr.py` (precedent for a training-time diagnostic).
**Build:** `tools/distill_sva_mlx.py`: teachers = N LoRA adapters **on the same Qwen2.5-3B
base** (hot-swap per domain batch group) → zero tokenizer mismatch, so the paper's
"vocabulary alignment" reduces to its clean core (top-k support + renormalization) with none
of the cross-tokenizer pain. Student rollouts generated frozen, teacher scores per-position
top-k (k is unspecified in the paper — pre-register a sweep k ∈ {8, 32, 128}), reverse-KL
loss through the student LoRA only, ρ logged every step. Offline invariants first (loss ≥ 0,
=0 iff distributions match on support, masking correctness, domain-normalization sums), on
synthetic logits, CI-gated like `gate_reward.self_check`.
**Weights?** YES → post-training calibration/abstention re-audit mandatory (chart 8 EVALGATE).
**Effort:** 2–3 weeks incl. Mac-bench validation. **Strongest objection:** at 3B with LoRA
teachers barely stronger than the student, distillation may transfer nothing measurable —
mitigated by A3 (train teachers first, measure their deltas; if teachers show no delta,
A2 is moot and we saved the effort).

### A3 — Make 2–3 council seats REAL teachers using the paper's cheapest teacher recipes
**Paper:** §4.2.2 (two-stage specialist SFT: reasoning-first, then tool-augmented continued
from that checkpoint — their biggest single delta, FS-R 2.5→54.3, no RL), §4.2.1 (search
teacher: SFT then GRPO with mixed-outcome data filtering), §4.2.4 (tool teacher: 64
near-miss samples, heavy reuse, asymmetric advantage).
**Repo seams:** `agent/council_registry.py:52-104` adapter slots; `tools/build_discipline_sft.py`;
`training/council/traces.jsonl` (gate-labelled teacher traces); protected seats
`history`/`religion` are **never RL-tuned** (registry flag) — pick non-protected seats,
e.g. `philosophy` (provenance/search-like) and `coding`.
**Build:** two-stage LoRA SFT per seat on the Mac/Spark iteration tier; register via
`agent/adapter_registry.py` acceptance gate; measure per-seat delta on that seat's verifier
suite (their Tables 5–8 pattern: teacher-vs-base is the transfer upper bound A2 then tries
to consolidate).
**Weights?** YES (per-teacher adapters) → re-audit per adapter before registry acceptance.
**Effort:** ~1–2 weeks per seat, mostly data prep. **Strongest objection:** the council's
1.0-vs-0.27 catch-rate story is on stub seats — real adapters could *reduce* the
prompt-scaffold's catch-rate; that is precisely what the acceptance gate + protected suites
exist to catch, and a negative result closes a named open gap honestly.

### A4 — Frugal-RL flags for run_rlvr: mixed-outcome filtering, dynamic sampling, asymmetric advantage
**Paper:** §4.2.1 (keep only questions where 5 attempts yield BOTH correct and incorrect —
maximal GRPO signal), §4.2.3 (dynamic sampling: drop uniform-reward prompt groups), §4.2.4
(hard-set of 64 near-misses + `A = A_out + 0.5·1[fail]·A_proc` where the process term is
normalized over failures only — no double-counting for successes).
**Repo seams:** `tools/run_rlvr.py` (GRPO config, curriculum flag precedent at
`--curriculum`); the W1 PRM (`provenance_bench/prm_step_reward.py`, landed this branch) is a
ready-made `A_proc` process score; `agent/temptation.py` for difficulty signals.
**Build:** three additive flags: `--mixed-outcome-filter N` (pre-attempt each prompt N
times with the base, keep mixed), `--dynamic-sampling` (drop zero-variance groups — composes
with the existing collapse logger), `--advantage-shaping papo --lambda-neg 0.5` (PRM/rubric
process score applied to failed rollouts only). Offline invariants for each (e.g. shaping
never flips the sign of a success/failure ordering).
**Weights?** Only when a live run is dispatched (existing rlvr-runpod path + re-audit).
**Effort:** 3–5 days. **Strongest objection:** these are optimization-efficiency tricks —
they change nothing unless the live RLVR program (R1) is actually running; land them before
the 3-seed sweep so the sweep benefits, but they are worthless standalone.

### A5 — Proposer–solver–verifier task self-play over Sophia's OWN corpus (KAG-lite generator)
**Paper:** §2.2.2 (self-play game + the five acceptance criteria), §3.1 (wiki entity-graph
random walks → masked-entity multi-hop QA with evidence-path verification), §3.4 (injected
in-context rules + distractors for long-context QA).
**Repo seams:** `data/attributions.json` + the OKF corpus (a provenance-dense entity graph!),
`agent/verifier_synthesis.py` + `agent/governed_rsi.py` (the proposer/verifier loop shells),
`tools/adversarial_gate_selfplay.py` (W4 — already mines fabricate-and-pass negatives),
`tools/make_independent_hidden_pack` (pack sealing), decontamination guard.
**Build:** `tools/selfplay_task_forge.py`: random-walk the attribution/tradition graph
(figure → text → tradition → dispute), mask the terminal entity, require the evidence path
(provenance chain) for acceptance; enforce all five paper criteria fail-closed, with
"evidence-covering" checked against `agent/source_ranking` tiers and "no-shortcut" checked
by requiring the raw model to FAIL the task zero-shot (mixed-outcome filter from A4).
Output feeds BOTH new hidden packs (eval) and A1 trajectory packs (training) with the
decontamination guard keeping them disjoint — the discipline the paper lacks.
**Weights?** No (asset/data engine).
**Effort:** ~2 weeks. **Strongest objection:** self-authored tasks are internally valid
only — the pre-registered thresholds already require independent packs for Level-3 claims,
so A5 products must stay in the iteration tier until third-party review.

### A6 — Long-horizon bench v2: model-driven knowledge-action specs + persistent notes memory
**Paper:** §3.2 tool table (`write_notes`/`read_notes` surviving context compaction;
`analyze` isolated sub-agent; solution-tree ops), §5.3.1 (12-hour run protocol), eval
budget declarations (300-turn cap, seeds, wall-clock per task).
**Repo seams:** `tools/run_long_horizon.py` (event log + objective gate + pre-registered
tiers 30min/2h/1day + demo-honesty labeling — all exists; today's specs are shell commands,
not model actions); `agent/long_horizon.py` TaskLedger; `agent/memory.py` (append-only);
`run_operational_tools` in run_case.
**Build:** (a) add `notes_write`/`notes_read` operational tools (append-only, hash-proofed
like memory) so long runs survive context compaction; (b) a spec type where each step is a
model decision (via `agent/model.py` adapter) instead of a fixed argv, with the gate run
per step and its verdict logged as the `verification` event; (c) a per-pack **resource
manifest** (turns, tool budget, wall clock, seeds, temperature, context cap) enforced by
the runner — the paper declares these per benchmark; we make the runner refuse to score a
run whose manifest was violated.
**Weights?** No (harness).
**Effort:** ~1–2 weeks. **Strongest objection:** tier-2/3 runs (2h/1day) burn real
wall-clock on the shared Spark/Mac; keep tier-1 (30min) as the default gate and reserve
long tiers for milestone runs.

### A7 — Adopt the paper's reporting patterns where they beat ours; keep ours where we beat them
**Paper:** dual-reporting irreproducible baselines (their τ²-Bench 81.2-official vs
32.5-reproduced with community citations), stage-wise checkpoint decomposition as the
primary results table (their Table 4), pinned open-source judges chosen for retirement-
proofness, per-benchmark official judges (no unified judge).
**Repo seams:** `agi-proof/preregistered-thresholds.md`, report schemas in
`agi-proof/benchmark-results/`, `tools/eval_ladder.py`.
**Build:** (a) add `baselineProvenance: {official, reproduced, discrepancyNote}` to the
public-report schema and require the claim gate to use the reproduced number; (b) make
base→SFT→final (or base→adapter→distilled) three-column stage decomposition the standard
eval-ladder report format so stage regressions (their HLE −5.8 pattern) are always surfaced;
(c) pin judge model+version+prompt per pack in the pack manifest. Keep (do NOT import their
gaps): shingle/Jaccard decontamination, ≥3 seeds + bootstrap CIs, abstention/calibration
measurement — the paper has none of these.
**Weights?** No (reporting/harness). **Effort:** 2–4 days.
**Strongest objection:** pure process work with no direct capability gain — its value is
that every A1–A6 result becomes trustworthy and comparable; do it alongside, not instead.

---

## The benchmark experiment the paper never ran (our niche)

The paper contains **no component ablation**: SVA vs sampled-token distillation vs plain
SFT-on-teacher-outputs is never isolated, nor is horizon length. At 3B/LoRA scale on our
hardware this is a tractable, publishable-grade pre-registered experiment:

> **B-SVA (pre-registered):** same teachers (A3), same data, same budgets; arms =
> (i) SFT on teacher outputs, (ii) sampled-token on-policy distillation, (iii) SVA-lite
> top-k reverse-KL (k swept {8,32,128}). Metrics: per-domain verifier suites + protected
> suites + ECE/abstention, ≥3 seeds, 95% bootstrap CI. Promotion decided by
> `evaluate_update_multigoal` (target delta ≥0.03, protected regression ≤0.01, retention).
> Success = arm (iii) beats (i) AND (ii) on multi-domain balance (no-regression count)
> with CI excluding zero; any other outcome is recorded as a negative result against this
> section. Post-training calibration re-audit mandatory for every arm (chart 8 EVALGATE).

This inverts the paper's weakness into our contribution: they showed the recipe works
end-to-end at 35B; we would show (honestly, gated, decontaminated, with CIs) *which part*
works at 3B.

## Suggested sequencing (fits the existing R-series)

1. **A4 flags** (days) — land before the R1 3-seed live sweep so the sweep benefits.
2. **A1 trajectory packs** (week) — starts accumulating (s,a,o,v) data from every harness
   run immediately.
3. **A3 two seats** (2 wks, Mac/Spark) → if teacher deltas are real → **A2 SVA-lite**
   (2–3 wks) → **B-SVA experiment** (the quarter's headline, RunPod registered tier).
4. **A5 + A6** in parallel with training runs (they are asset/harness work).
5. **A7** folded into whichever harness is touched first.

## Verification caveats (flagged by the study team, kept honest)

Paper facts we could NOT verify in the text and must treat as free parameters: SVA k;
OPD LR/batch/budgets; the efficiency-penalty free-round count K; the exact six-domain
enumeration (only four teachers are described); active-parameter count ("A3B" naming implies
~3B active, not stated). Paper-internal inconsistencies we noted: GAIA teacher 85.4 (prose)
vs 95.1 (table); τ²-Bench reproduced baseline reported as 32.5/32.53/33.0 in three places;
the MolBench "leading" claim holds vs the 1T models but not vs GPT-5.5's bolded 62.2.
