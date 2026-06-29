# Measurement Thesis — how to measure properly, and how it fits the Sophia workflow

*Status: methodology (not a model claim). `canClaimAGI` is unaffected by this document.*
*Written 2026-06-26 after the M3 retention episode (Δ −0.118 on N=34 → Δ −0.014 on N=70).*

## Central thesis

> **In a small-corpus, fail-closed, AGI-candidate pipeline, the dominant source of wrong
> conclusions is the measurement instrument, not the model. Therefore evaluation must be
> engineered as a first-class instrument — pre-registered, powered to a minimum detectable
> effect, uncertainty-quantified, valid under repeated looking, and triangulated across
> independent constructs — and no claim may exceed what that instrument can resolve.**

Call it the **Instrumented Evaluation Contract (IEC)**: *measure the measurement before you
trust the result.* The M3 episode is the proof of need — a 34-item probe whose 95% CI is ±0.22
**cannot** resolve a ±0.05 criterion, so its −0.118 point estimate was an instrument artifact,
not a finding. The fix that mattered was not an O-LoRA/meta-RL mitigation; it was widening the
instrument (N=34→70) and attaching a CI. Build the instrument first.

## The eight pillars (each: principle → the failure it prevents → the workflow hook)

| # | Principle | Failure it prevents (ours) | Workflow hook |
|---|---|---|---|
| 1 | **Always quantify uncertainty** — report SE/CI, never a bare point estimate. Use paired/clustered SEs because eval items are a sample from a super-population (Miller 2024). | −0.118 reported as "forgetting" with no CI. | `evaluate_retention` now emits a paired bootstrap 95% CI; extend `evaluate()`/judge to Miller's paired+clustered SEs. `lint_claims` should REJECT any numeric claim with no CI. |
| 2 | **Power before you run** — pre-register the Minimum Detectable Effect (MDE); compute the N needed for ~80% power at that MDE; refuse to run an underpowered probe. | N=34 had MDE ≈ ±0.16 — it physically could not test a ±0.05 criterion. A design bug masquerading as a result. | A `measurement_spec` per experiment with `mde` + `required_n`; a preflight `power_check()` that aborts if `n < required_n`. |
| 3 | **Curate items by discrimination, not count** — item quality dominates raw N; IRT-curated 100 items can estimate within ~2% (tinyBenchmarks). | Some probe items never discriminate (analogy 13/13 = 13/13 → zero information); raw counting hides this. | Track per-item base-vs-adapter flip rate; prune zero-information items; grow toward high-information ones. IRT-lite: weight items by observed discrimination. |
| 4 | **Use anytime-valid inference because you ITERATE** — under optional stopping (peeking, sweeping, re-judging) classical CIs/p-values are invalid; confidence sequences / e-values stay valid at every look and any stopping time. | The −0.118 → re-measure → sweep → re-judge loop is textbook optional stopping; fixed-N CIs silently inflate false positives across the loop. | Replace fixed-N CIs with a **confidence sequence** for any metric monitored across runs; the gate fires only when the *sequence* excludes the threshold. This is what makes the iterate-and-look workflow honest. |
| 5 | **Triangulate constructs — one construct is never a claim** — guard against *construct irrelevance* (measuring the wrong thing) and *construct underrepresentation* (missing part of the target) (construct-validity literature). | Marker metrics measure FORMAT (format-Goodhart = irrelevance); the useful_correctness PROXY missed raw reasoning (underrepresentation). | The VALIDATED bar already needs ≥2 judge families; generalize to **≥2 independent constructs** (deterministic markers + LLM-judge + held-out behavioral transfer). A claim's strength = its *weakest* corroborating construct. Separately report internal vs external validity (does it transfer off the trained structural families?). |
| 6 | **Decontaminate by content + keep a private split** — exact-prompt/n-gram decontam is unreliable; a GLUE-style never-touched private split gives a public-vs-private gap as a live contamination signal. | Train/eval share structural families (decontaminated by exact prompt, not by format/paraphrase). | Strengthen `contamination_report` to content-level (shingle/embedding overlap, not just exact prompt); reserve a **private probe split** never used in any tuning/selection loop. |
| 7 | **Calibrate and de-bias the judge** — LLM judges carry position, verbosity, and self-preference bias; with blinding + calibration they reach >80% human agreement (judge-bias literature). | Mitigated already: we randomize A/B order, use 3 families ≠ subject ≠ gate, and report Gwet AC1 (prevalence-robust) not just κ. Remaining gaps: verbosity, no human anchor. | Add a verbosity covariate/length-control; forbid the subject's own family as judge (self-preference); validate judges against a small **human-graded gold anchor** once. |
| 8 | **Effect size AND practical significance** — "CI excludes 0" is not a headline if the effect is trivially small; pre-register the magnitude that matters. | Risk: over-reading a CI-clean but tiny delta as a "win". | Gates check BOTH significance (CI clear of 0) AND magnitude (≥ pre-registered practical threshold). |

## The instrument is a contract: `measurement_spec.json` (pre-registration)

One artifact per experiment, committed **before** the run, machine-checkable:

```jsonc
{
  "experimentId": "m3-retention",
  "constructs": ["deterministic-markers", "llm-judge-3family", "heldout-generality"],
  "primaryMetric": "generality_accuracy_delta",
  "direction": "adapter >= base - tolerance",
  "tolerance": 0.05,                 // the criterion
  "mde": 0.05,                       // smallest effect that matters
  "requiredN": 64,                   // from a power calc at mde, 80% power  (PILLAR 2)
  "uncertainty": "paired-bootstrap-or-confidence-sequence",   // PILLARS 1,4
  "stoppingRule": "anytime-valid (look as often as you like)", // PILLAR 4
  "decontam": "content-shingle + private-split",               // PILLAR 6
  "judges": ["deepseek","mistral-small","llama-3.3-70b"],      // PILLAR 7  (!= subject, != gate)
  "practicalThreshold": 0.05,        // PILLAR 8
  "claimCeiling": "candidate_only; canClaimAGI:false"
}
```

A claim is admissible only if its spec is satisfied on **every** field. This turns "no-overclaim"
from a vibe into a checklist a linter can enforce.

## How it is enforced in this repo (shipped)

Every pillar below is a deterministic check; the headline ones run in `fast-ci` on every PR, via
`make claim-check`, and (opt-in) the `.githooks/pre-commit` hook:

1. **`tools/claim_gate.py`** — given a `measurement_spec` + a result, returns GO only if: CIs present
   (1) ∧ **anytime-valid CS for any peeked metric** (4) ∧ primary powered, `mde_at_n(N) ≤ MDE` (2) ∧
   ≥2 **distinct-family** constructs agree (5) ∧ decontam clean (6) ∧ magnitude ≥ practicalThreshold
   with no protected regression (8). `--assert-prereg` additionally fails if the spec does not
   provably predate the result (git history). Writes a GO/NO-GO receipt; fail-closed.
2. **`tools/eval_stats.py`** — power/MDE (`mde_at_n`, now paired-aware), bootstrap CI, the Robbins
   anytime-valid CS (`confidence_sequence_mean` / `…_from_summary`), and `verdict_or_underpowered`,
   which refuses to emit a directional verdict word when the probe cannot resolve the effect (3).
3. **`tools/assert_decontam.py`** — independent of the build: re-checks the *committed* training packs
   against the eval surfaces with exact-prompt **and** content-shingle (Jaccard) near-duplicate scans (6).
4. **`tools/lint_training_rows.py`** — asserts every source-discipline / moral-gate target teaches a
   HABIT (route/qualify/refuse), never a bare ground-truth fact (separates truth from behavior).
5. **`tools/lint_claims.py`** — requires a promoted registry entry to carry a passing
   `measurement_receipt`, a *generalizes* claim to carry a passing `transfer_receipt`, and a recipe
   "best" claim to carry a powered superiority receipt; plus the no-overclaim prose scan.
6. **`tools/benchmark_recipes.py --emit-receipt`** + **`build_sophia_wisdom_dataset.py`** — ranking only
   on the powered axis with the simple baseline included (9); the manifest headlines RECORDS (not rows)
   and flags volume inflation when rows/record exceeds the ceiling (8).

## Worked example: "a small distilled model carries a frontier teacher's reasoning"

A recurring external claim is that a ~12B model, distilled from a frontier teacher and served
4-bit on ~8 GB, "matches" or "carries" that teacher's reasoning. Treated as a *claim about the
model*, it is the same shape as the −0.118 retention episode in reverse: an impression that
survives only until the instrument is built. The IEC turns the slogan into a gated measurement
or it stays an anecdote. Concretely, before any such claim could be reported here:

1. **Quantify uncertainty.** "Feels like the teacher" → a measured uplift with a paired CI; a
   bare win rate is rejected by `lint_claims`.
2. **Power first.** Pre-register MDE + required N in `measurement_spec.json`; refuse a verdict
   when MDE(N) exceeds the effect (`eval_stats`).
3. **Triangulate ≥2 constructs.** Deterministic markers + an LLM-judge panel of ≥2 independent
   families (judge ≠ subject), with reported inter-judge agreement (`claim_gate`).
4. **Decontaminate.** The teacher may have memorised public benchmarks, so its distilled traces
   are checked against the held-out eval (exact + shingle near-dup) before they can train the
   student — `tools/assert_decontam.py` now globs the distillation outputs, and
   `tools/distill_export.py` diverts any eval-colliding trace to a decontaminated bucket.
5. **Anytime-valid + private split.** The distil → re-judge → resample loop is optional
   stopping, so the gate fires on a confidence sequence, and the headline rests on a private,
   non-self-authored split.
6. **Bound the low-RAM half separately.** A 4-bit serving result may claim only what
   `serving/lowram_eval.LowRamGate` measures — a BF16-vs-low-bit next-token KL bound (which
   `tools/train_lora.py --qat-consistency` now optimises against directly) — never "frontier
   reasoning on 8 GB". The efficiency framing (`tools/build_efficiency_frontier.py`) reports
   score-per-active-param / per-byte with CIs, not a slogan.

The point is not that the claim is false; it is that, absent this instrument, it is
**unresolved** — `candidate_only; canClaimAGI:false` — exactly like the −0.118 point estimate
on N=34.

## The one-line discipline

> A number without a CI is a rumor; a CI without a pre-registered MDE is theater; a single construct
> is an anecdote; and a CI you computed after peeking ten times is a lie unless it was anytime-valid.
> The Sophia gate already refuses unproven *claims about the world* — the IEC extends the same
> fail-closed refusal to unproven *claims about the model*.

## Sources

- Evan Miller, *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations* — [arXiv:2411.00640](https://arxiv.org/abs/2411.00640)
- Polo et al., *tinyBenchmarks: evaluating LLMs with fewer examples* (Item Response Theory) — [arXiv:2402.14992](https://www.emergentmind.com/papers/2402.14992)
- Ramdas et al., *Game-theoretic statistics and safe anytime-valid inference* (e-values / confidence sequences) — [arXiv:2210.01948](https://arxiv.org/pdf/2210.01948); CWI safe-testing project — [cwi.nl](https://www.cwi.nl/en/research/machine-learning/various-projects-on-e-values-always-valid-confidence-sequences-and-safe-testing/)
- *The Benchmarking Epistemology: Construct Validity for Evaluating ML Models* — [arXiv:2510.23191](https://arxiv.org/abs/2510.23191); *Measuring what Matters: Construct Validity in LLM Benchmarks* — [arXiv:2511.04703](https://arxiv.org/abs/2511.04703); *Are We Learning Yet? A Meta-Review of Evaluation* (internal/external validity taxonomy) — [thomasliao.com](https://thomasliao.com/are_we_learning_yet.pdf)
- *Benchmark Data Contamination of LLMs: A Survey* — [arXiv:2406.04244](https://arxiv.org/html/2406.04244v1); held-out private split as contamination signal (GLUE/SuperGLUE design)
- *Judging the Judges: Position Bias in LLM-as-a-Judge* — [arXiv:2406.07791](https://arxiv.org/abs/2406.07791); *Self-Preference Bias in LLM-as-a-Judge* — [arXiv:2410.21819](https://arxiv.org/html/2410.21819v2)
