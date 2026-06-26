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

## How it bolts onto what already exists (minimal, incremental)

1. **`tools/retention_gate.py`** — already CI-aware (`forgetting_established` = whole CI past the
   threshold). Generalize into a `claim_gate.py` that, given a `measurement_spec` + result, returns
   GO only if: CI present (1) ∧ N ≥ requiredN (2) ∧ ≥2 constructs agree (5) ∧ decontam clean (6) ∧
   magnitude ≥ practicalThreshold (8). Fail-closed, fits the project's gate philosophy.
2. **`evaluate()` / `evaluate_retention()` / `judge_pilot_answers.py`** — already emit CIs / AC1.
   Add: confidence-sequence option (4), per-item discrimination logging (3), verbosity covariate (7).
3. **`build_sophia_wisdom_dataset.py`** — already decontaminates by exact prompt + marks `heldout`.
   Add: content-level overlap check + carve a private split that the builder/selectors never see (6).
4. **`lint_claims.py`** — today it scans prose. Upgrade it to also require that each quantitative
   claim references a `measurement_spec` that passed `claim_gate` (the contract enforcement point).
5. **`failure-ledger.md`** — already the honesty record. Add the IEC fields (MDE, N, CI, constructs,
   stopping rule) as expected columns so every logged result carries its instrument's resolution.

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
