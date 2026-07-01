# Cross-cutting: hidden-state verifier signals are strong within-distribution, domain-bound across it

**Date:** 2026-07-02 · candidateOnly:true · level3Evidence:false · canClaimAGI:false

Three independent tools now share one real MLX hidden-state featurizer
(`agent.activation_probes.build_hidden_state_featurizer`, Qwen2.5-3B). Read together, their
results draw a clean boundary — worth stating explicitly before anyone spends GPU on the
hidden-state introspection direction.

| Tool | What it probes over real hidden states | Setting | Result |
|---|---|---|---|
| **W5** probe-as-loss (peer, 2026-07-01) | honest vs deceptive text (DPO pairs) | **within-distribution** | separates **perfectly** — loss probe 1.0, audit probe 1.0, `goodhartGap 0.0` |
| **O2** energy verifier (this PR) | (answer, evidence) compatibility vs `correct` | **held-out DOMAIN** | out-of-fold AUROC **≈0.49**; robust to a proper logistic probe + more data (0.38–0.43) |
| **W1** distilled PRM (peer, 2026-07-02) | per-step derivation correctness vs symbolic oracle | **held-out DOMAIN** | held-out-domain agreement **≈chance** (math→physics 0.488, physics→math 0.495) |

## The finding

A linear probe over frozen residual streams reads honesty / correctness **very well
in-distribution** (W5: perfect) but **does not transfer across domains** (O2 and W1: chance).
Crucially, the gates that *fail* are exactly the ones that demand cross-domain generalization
(O2's held-out-**domain** goodhartGap; W1's held-out-**domain** agreement), while the one that
*saturates* (W5) is measured in-distribution. This is one phenomenon seen three times, not three
separate negatives.

## Why it matters

- The blocker is **not** the featurizer — that is now genuinely implemented and shared. Making it
  real did not move any cross-domain gate off chance.
- So "wire in real hidden states" is **not** the lever for the O2/W1 gates. Closing them needs
  either (a) **more/broader training domains** (W1 has only 2 verifier domains, so no true 3rd
  held-out domain exists; O2's factcheck pack is 4 small domains, `accepted`⊂`correct`), or
  (b) **GPU-scale coupling** (W1's PRM-as-RLVR-reward, W5's probe-as-loss LM run) — the
  `--run-train` human-escalation lane, not a local CPU/MLX instrument.
- W5's in-distribution saturation (`goodhartGap 0.0`) is itself a caveat: a probe that is trivially
  perfect in-distribution has no headroom to *prove* a probe-as-loss run helped — the peer noted
  this and truncated inputs to recover headroom. In-distribution perfection ≠ a usable training signal.

## Honest status

All affected rows stay **Open**. No gate is met or closed by this synthesis; it reframes the
remaining work (data breadth / GPU coupling), it does not claim progress. W1/W5 were taken live by
a concurrent peer on `feat`; this note only connects those results to O2 — it does not re-run them.
