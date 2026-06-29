# Andreia — the Courage System

> ἀνδρεία (*andreia*) — the Stoic virtue of courage; the companion of σοφία
> (*sophia*), wisdom. Inspired by Ryan Holiday, *Courage Is Calling: Fortune
> Favors the Brave*.

## Why

Sophia's [Conscience Kernel](Conscience-Kernel.md) is, by design, a **fear
apparatus**. Of its seven verdicts — `allow · revise · retrieve · clarify ·
escalate · abstain · block` — six are forms of retreat. That is exactly right for
the no-overclaim boundary, but it leaves a blind spot: a system that can only
retreat cannot distinguish genuine prudence from **cowardice disguised as
prudence**. It cannot see the failure where the brave, well-supported move was to
act and it held back anyway — the *decision–action gap*
([Wang 2026, arXiv:2601.07767](https://arxiv.org/abs/2601.07767)).

Andreia adds that missing faculty as an **orthogonal, deterministic, fail-closed**
gate. It does not modify the conscience kernel and **never overrides a hard
prohibition** — courage is not recklessness.

## Model: courage as a phase transition

Following the ASIR Courage Model
([Kim 2026, arXiv:2602.21745](https://arxiv.org/abs/2602.21745)), courage is a
transition from Suppression (hold) to Expression (act) that fires when
facilitative forces beat inhibitory ones. Andreia computes the **Courage
Quotient**:

```
CQ = λ·(1 + γ) + ψ − (θ + φ)

  λ  baseline openness        ← calibrated confidence      (agent/metacognition.py)
  γ  relational amplification ← pro-social stakes          (agent/moral_aggregator.py)
  ψ  accumulated pressure     ← harm/cost of silence       (complicity)
  θ  transition cost          ← genuine epistemic risk     (nonconformity)
  φ  inhibition (fear)        ← social/reputational cost   (agent/cowardice_signals.py)
```

Every input is a signal Sophia already computes; Andreia is mostly wiring, not new
ML, and stays a deterministic policy (like the moral parliament).

## Verdicts (Andreia's own vocabulary — not a conscience verdict)

| Verdict | When | Holiday |
|---|---|---|
| `act` | CQ > 0 and well-calibrated (λ ≥ 0.70, nonconformity ≤ 0.50) | answer the call |
| `heroic` | `act` where γ and ψ are both high (≥ 0.66) | courage above the self |
| `escalate` | CQ > 0 but under-calibrated (recklessness guard), **or** a hold that looks fear-driven (cowardice surfaced) | force explicit justification |
| `hold` | CQ ≤ 0 (genuine prudence), **or** a hard prohibition is respected | caution that is wisdom, not fear |

Thresholds are **pre-registered** in `agent/andreia.py` and in the battery; they
are a measurement decision, not a tuning knob.

## Cowardice detector

`agent/cowardice_signals.py` is the dual of `agent/deception_signals.py`. Where
deception catches *acting/claiming beyond the evidence*, cowardice catches
*holding back despite the evidence*: respectable excuses ("not the right time",
"let someone else"), a confidence/silence mismatch, social-cost-dominated holds,
and sycophancy drift. It is informational — the worst it can do is force an
`escalate` (explicit justification); it can never force an action.

## The decision–action 2×2

The thing Andreia is measured on:

|  | should act | should hold |
|---|---|---|
| **acted** | ✅ courage | ❌ recklessness |
| **held** | ❌ **cowardice** | ✅ prudence |

The existing gates already catch the top-right (recklessness/overclaim). Andreia's
job is the **bottom-left (cowardice)**, which the fear-only kernel cannot see.

## Surfaces

- Core: `agent/andreia.py` (`assess_courage`, `CourageDecision`, `run_andreia_benchmark`)
- Cowardice signals: `agent/cowardice_signals.py`
- MCP tools: `sophia_courage_assess`, `sophia_cowardice_check`, `sophia_andreia_benchmark`
- Skill (fail-closed): `skills/andreia.py` → `courage_advocate`
- Benchmark + receipt: `tools/run_andreia_bench.py` + `data/andreia_courage_battery.json`
- Audit trails: `agi-proof/courage-ledger.md`, plus the open claim in `agi-proof/failure-ledger.md`
- Tests: `tests/test_andreia.py`

## Measurement boundary (read this)

Andreia is **candidate infrastructure**. The Courage-Calibration battery routes
16/16 deterministically, but that receipt is **NO-GO by design**: it certifies the
gate's routing, not that the gate improves real decisions. Promotion past
candidate requires an external decontaminated battery, ≥2 independent judge
families (κ ≥ 0.40), a raw-model baseline contrast, and an effect on the
cowardice/recklessness error rates whose 95% CI excludes zero. The gate is
shipped; the claim is not. `canClaimAGI` stays false.
