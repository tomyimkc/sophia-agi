# W5 probe-as-loss — local take-live, methodology on REAL MLX hidden states (2026-07-01)

> **candidateOnly:true · level3Evidence:false · canClaimAGI:false · gateMet:false**
> The `w5-probe-as-loss-not-attempted-goodhart-unproven` ledger row **stays Open.** This is
> the *highest-risk* tool and was deliberately **not rushed**. Result + sha256:
> `w5-probe-live-2026-07-01.candidate.json`.

## What was done

- Reused the real MLX featurizer implemented for W1
  (`agent.activation_probes.build_hidden_state_featurizer(spec="mlx", model, tok)`).
- Ran the W5 **disjoint loss/audit/test Goodhart methodology on REAL hidden states**, over
  genuine honest/deceptive text from the repo's real DPO pairs (chosen = hedged/provenance-
  aware = honest; rejected = confident false attribution = deceptive; 140 rows / 70 pairs).

**Finding:** on real residual streams the base model separates honest vs deceptive text
**perfectly** — loss probe 1.0, audit probe 1.0, `goodhartGap 0.0`. Honesty is linearly
decodable from real hidden states (the precondition for a probe-as-loss). Truncating the
claim text creates non-trivial headroom for a meaningful experiment (35-char: loss 0.80 /
audit 0.89).

## The probe-as-loss coupling WAS then performed — GOODHART DETECTED (valid negative)

A custom MLX LoRA loop (`linear_to_lora_layers` + `nn.value_and_grad` + Adam lr 2e-5 +
grad-clip) actually optimizes the LM **against** a fixed loss-probe on real residual-stream
features, over 4 disjoint splits (probeFit | couplingTrain | auditTrain | test, truncated for
headroom). Result (`w5-probe-as-loss-coupling-2026-07-01.candidate.json`):

| test N=45 | loss probe | disjoint audit probe |
|---|---|---|
| base | 0.867 | 0.756 |
| after probe-as-loss | **0.956** ↑ | **0.711** ↓ |

The coupling loss converged to 0 — the loss-probe accuracy rose (0.867→0.956) — **but the
disjoint audit probe did NOT follow (0.756→0.711)**, `goodhartGap = 0.244 > 0.15`. The tool's
mandatory audit **fires "gaming detected" and refuses to certify.** This is the textbook
probe-as-loss failure, demonstrated empirically on real hidden states with a real
gradient-through-probe coupling: **the probe is gamed (loss-probe up) without genuine honesty
improvement (audit-probe down).** A from-scratch audit still separates at 0.711 (≥0.7), so the
signal was degraded, not fully obfuscated. (An earlier lr=1e-4 run *diverged* — loss→7.06,
both probes dropped; the lr=2e-5 + grad-clip run above is the stable one.)

## Verdict — gate NOT met (row stays Open), and correctly so

The four-part gate requires the disjoint **audit probe to improve** — it did not (it dropped);
`goodhartGap 0.244 > 0.15`. The gate is unmet **because the audit caught the gaming** — which
is the tool working as designed. This validates the W5 safety methodology on a real coupling:
training against a truthfulness probe games it, and the disjoint-audit + goodhartGap gate
catches the gaming that the loss-probe alone would hide. A valid **negative** result.

## To CLOSE it (not just validate the audit)

The coupling above validated the *audit* by triggering it. To close the gate you need a
probe-as-loss variant that improves the **audit** probe too (not just the loss probe):
e.g. multi-probe/ensemble loss directions, adversarial-probe co-training, or activation-steering
toward a held-out honesty target — with `goodhartGap ≤ 0.15` and a from-scratch post-run audit
that *improves*. Until a coupling passes the audit, the direction stays quarantined (row Open).

## v2 (2026-07-02) — ENSEMBLE probe-as-loss REVERSES the gaming (strong signal, still Open)

Hypothesis: optimizing the LM against an **ensemble of K=4 independent probes** (even 30-row
disjoint fits) makes a single-direction Goodhart shortcut insufficient. Result across **3 seeds**
(`w5-ensemble-probe-as-loss-2026-07-02.candidate.json`): the disjoint **audit probe IMPROVES on
all 3 seeds** — 0.90→0.917, 0.75→0.983, 0.917→0.983 (mean Δ **+0.106**) — with goodhartGap ≤ 0.15
every seed and a from-scratch post-run audit separating at 0.95–0.967. This **reverses v1's clean
Goodhart gaming** (v1 single-probe dropped the audit 0.756→0.711).

**Still Open (highest-risk tool, strict bar):** seed0's +0.017 is one test example (noise); there
is no per-seed bootstrap CI; base-audit varies 0.75–0.92 across seeds (regression-to-mean
confound); N=60, single controlled surface. A rigorous close needs per-seed CIs excluding 0 at
matched base-audit on a broader surface. The ensemble is a promising **anti-Goodhart training
method**, not yet a certified one.

## v3 (2026-07-02) — RIGOROUS: 2/3 seeds' audit improvement is statistically significant

Larger N=120 + per-seed **paired bootstrap CI** on the disjoint audit-probe delta
(`w5-ensemble-rigorous-2026-07-02.candidate.json`): **2 of 3 seeds' audit-delta 95% CI excludes 0**
— seed0 0.617→0.90 (CI [+0.20, +0.375]), seed1 0.90→0.95 (CI [+0.017, +0.092]); seed2 0.925→0.975
grazes 0 (CI [-0.008, +0.108]). Every seed: goodhartGap ≤ 0.15, from-scratch audit separates
0.958–0.967. High-base seeds *also* improved → not pure regression-to-mean. This substantiates the
K=4 ensemble as a **genuine anti-Goodhart training method**. Not a full close (seed2 grazes 0);
row stays **Open**.

## v4 (2026-07-02) — larger N: seed2 is a CEILING effect, not low power

Re-ran at larger N (test ~198). 2/3 seeds' audit-delta CI still excludes 0 (seed0 0.889→0.975
[+0.046,+0.126]; seed1 0.753→0.950 [+0.141,+0.258]); **seed2 still grazes 0** (0.919→0.939
[-0.030,+0.071]). Larger N did **not** close seed2 — its non-significance is a **ceiling effect**:
its base audit is already 0.919 (near-saturated), so the room to improve is small (+0.02). The
improvement **magnitude scales with headroom** (seed1's low base 0.753 → +0.197). This substantiates
the ensemble as a genuine anti-Goodhart method (improves the disjoint audit whenever there is
headroom, never degrades it) and honestly explains the seed heterogeneity. A clean 3/3 needs seeds
with matched moderate base-audit. Row stays **Open**.
