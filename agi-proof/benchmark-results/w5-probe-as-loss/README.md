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
