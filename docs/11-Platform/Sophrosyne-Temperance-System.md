# Sophrosyne — the Temperance System

> σωφροσύνη (*sophrosyne*) — the Stoic virtue of temperance / measure / soundness
> of mind. It shares the σωφ-/σοφ- root with σοφία (*sophia*), wisdom: temperance
> is wisdom about *measure*. The third cardinal virtue after Wisdom (the conscience
> kernel) and Courage ([Andreia](Andreia-Courage-System.md)).

## Why

Sophia's [Conscience Kernel](Conscience-Kernel.md) regulates **truth**
(`allow · revise · retrieve · clarify · escalate · abstain · block` on the
evidence). The [Andreia gate](Andreia-Courage-System.md) regulates **direction**
(`act · hold` despite fear). Neither regulates **magnitude**: *how much* effort to
spend, how many words / tool-calls / retrieval hops, how long to continue, how
strong a claim — and above all **when enough is enough**. That blind spot is the
home of intemperance: verbose over-elaboration, over-hedging (the measured
"calibration tax"), tool-calls past diminishing returns, runaway loops — and their
mirror, premature stops, under-answers, and truncation. For an autonomous agent the
excess face is also the **runaway-compute** failure mode directly.

Sophrosyne adds that missing faculty as an **orthogonal, deterministic, fail-closed**
gate that **never suppresses a required verification step** — temperance is not
negligence. It can run standalone, or as an **opt-in consulted path inside the
conscience kernel** (the 10th path; see Integration) that is off by default, so
existing conscience behavior is byte-identical unless explicitly enabled.

## Model: temperance as the mean (Aristotle's μεσότης)

Following Aristotle's doctrine of the mean (*NE* II) — virtue is the mean between
two vices, **excess (ἀκολασία)** and **deficiency (ἀναισθησία)** — and the
adaptive-computation / halting view of *when to stop computing* (Adaptive
Computation Time, Graves 2016; PonderNet, Banino et al. 2021). Sophrosyne computes
the **Measure Quotient** as the signed deviation of expenditure from demand:

```
MQ = epsilon - delta          ( >0 excess · <0 deficiency · ~0 the mean )

  delta   demand          the genuine task requirement              (set-point)
  epsilon expenditure     tokens/tool-calls/depth/claim-strength spent-or-planned
  mu      marginalValue   is the next unit of effort still buying anything?
  alpha   appetite        pull toward more (completionism/optimiser greed)
  rho     budgetRemaining headroom (compute/tokens/turns)
```

`delta`, `mu`, `rho` are **deterministically measurable** (counters, budget,
redundancy of the last unit), so this gate is *more* offline-testable than Andreia,
whose inputs must be derived from text. The semantically hard term is `delta` (true
task demand); see the honest limit below.

## Verdicts (Sophrosyne's own vocabulary — not a conscience verdict)

| Verdict | When | Aristotle |
|---|---|---|
| `proportionate` | `\|MQ\|` ≤ tolerance and `mu` still justifies the spend | the mean (μεσότης) |
| `restrain` | `MQ > 0` and `mu` low — excess: cut back / stop / halt the loop | curb ἀκολασία |
| `sustain` | `MQ < 0` and `mu` high — deficiency: do not quit early | curb ἀναισθησία |
| `escalate` | appetite high while budget is genuinely scarce (akrasia), **or** a required step must be protected from restraint | force an explicit measure decision |

Thresholds are **pre-registered** in `agent/sophrosyne.py` and in the battery; they
are a measurement decision, not a tuning knob.

## Intemperance detector

`agent/intemperance_signals.py` is the dual of `agent/cowardice_signals.py` on the
magnitude axis. It catches both vices: **excess** (n-gram self-repetition, hedge-
stacking past the calibration set-point, filler/padding, a runaway loop whose
frontier is not shrinking at low marginal value) and **deficiency** (truncation /
"left as an exercise" / TODO, premature stop with budget unspent and value still on
the table). It is informational — the worst it can do is recommend trimming or
continuing; it can never suppress a required output.

## The measure 2×2

The thing Sophrosyne is measured on:

|  | low marginal value | high marginal value |
|---|---|---|
| **keep spending** | ❌ **excess** (verbosity, over-hedging, runaway) | ✅ proportionate continuation |
| **stop / cut back** | ✅ proportionate restraint | ❌ **deficiency** (premature stop, under-answer) |

The other gates are silent on this entire table; Wisdom asks if the content is
*true*, Courage if you should *move* — neither asks *how much*.

## Surfaces

- Core: `agent/sophrosyne.py` (`assess_temperance`, `TemperanceDecision`, `run_sophrosyne_benchmark`)
- Intemperance signals: `agent/intemperance_signals.py`
- MCP tools: `sophia_temperance_assess`, `sophia_intemperance_check`, `sophia_sophrosyne_benchmark`
- Skill (fail-closed): `skills/sophrosyne.py` → `measure_advocate`
- Benchmark + receipt: `tools/run_sophrosyne_bench.py` + `agi-proof/benchmark-results/sophrosyne/sophrosyne_measure_battery.json`
- Eval harness (3-arm, PENDING): `tools/run_sophrosyne_eval.py`
- Robustness probe: `tools/run_sophrosyne_robustness.py`
- Audit trail: `agi-proof/temperance-ledger.md`, plus the open claim in `agi-proof/failure-ledger.md`
- Tests: `tests/test_sophrosyne*.py`

## Integration (opt-in 10th consulted path)

`conscience_check(..., context={"consultTemperance": True})` consults Sophrosyne
after the conscience verdict is computed (mirroring the consequence and courage
paths). It:

- attaches the full measure report under `decision.temperance`;
- may downgrade an over-expenditure `allow` to `revise` (trim) when temperance says
  `restrain` on the excess axis;
- may upgrade an otherwise-quiet `abstain` to `escalate` when temperance says
  `sustain` (a premature/lazy abstention with effort still worth spending) —
  forcing an explicit justification instead of silently quitting.

It never weakens a `block`/`retrieve`/`clarify`, never suppresses a required
verification step (`stepRespected` guards that), and is **off by default**.

## Known limitation: derived signals on raw text

The gate routes 16/16 when its inputs are supplied explicitly. When it must
**derive** them from raw text (the `consultTemperance` path), the robustness probe
(`tools/run_sophrosyne_robustness.py`) measures the cost: derived routing agrees
with the labels only **~0.5 vs 1.0 explicit**, and the regex intemperance detectors
miss meaning-preserving paraphrases. This is fail-closed behavior, not a bug (a weak
`delta` estimate correctly collapses toward `proportionate`), and it is **not** tuned
away. The honest consequence: **no claim that the gate is temperate on raw text**;
the integration is conservative by design. The fix is **model-gated** — wire a real
semantic backend through `detect_intemperance(semantic_backend=…)` and a model-backed
demand estimator, then re-run the probe. Tracked as
`sophrosyne-derived-signal-routing-weak-on-raw-text-2026-06-29` in the failure ledger.

## Safety: temperance is not negligence

A temperance faculty is dangerous if it can be talked into cutting a required
verification/safety step in the name of brevity or speed. `assess_temperance`
therefore defers to Sophia's deterministic prohibition gates (`_hard_prohibited`:
constitution `rejected`, constitutional classifier `block`, and an explicit shortcut
regex — "skip the verification", "no need to check the sources", "stop verifying")
**before** any `restrain` verdict, on **every surface** — the standalone gate, the
`sophia_temperance_assess` MCP tool, and the `measure_advocate` skill — returning
`escalate`/`sustain` with `stepRespected=True`, never `restrain`. Genuine excess
(ordinary verbosity, no required step) is unaffected (`tests/test_sophrosyne_safety.py`).

## Measurement boundary (read this)

Sophrosyne is **candidate infrastructure**. The Measure-Calibration battery routes
16/16 deterministically, but that receipt is **NO-GO by design**: it certifies the
gate's routing, not that the gate improves real decisions. Promotion past candidate
requires an external decontaminated task set, ≥2 independent judge families
(κ ≥ 0.40), a raw-agent baseline contrast, an effect on the excess/deficiency error
rates whose 95% CIs exclude zero, and a held task-success guardrail
(see [`agi-proof/sophrosyne-measurement-plan.md`](../../agi-proof/sophrosyne-measurement-plan.md)).
The gate is shipped; the claim is not. `canClaimAGI` stays false.
