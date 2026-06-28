# OKF consistency & belief-dynamics evidence (candidate)

Two deterministic, offline reports over the OKF belief graph. Both are **candidate-only**
(`candidateOnly: true`, `level3Evidence: false`) — they are reproducible evidence artifacts
about *how the OKF layer behaves*, not capability or AGI claims, and not validated
weight-level learning.

| Report | What it measures | Reproduce |
| --- | --- | --- |
| `consistency.public-report.json` | Syntactic local-global consistency of the graph (referent partition, declared contradictions, epistemic holes). Escalates disagreements; does not decide truth or auto-generate training facts. | `python tools/eval_okf_consistency.py` |
| `belief-dynamics.public-report.json` | The forgetting layer's honesty properties and falsifiable historical cases. | `python tools/eval_okf_belief_dynamics.py` |

## What `belief-dynamics.public-report.json` covers

The OKF is append-only today: a belief's `authorConfidence` is static once written.
`okf.decay_okf` / `okf.frontier_demotion` / `okf.forgetting_audit` add a *dynamics* layer
without weakening source discipline. This report exercises it on three panels:

1. **Decay honesty (P1–P3).** P1 no-silent-deletion (forgetting is demotion, never
   destruction); P2 provenanced-forgetting (every suppression carries an auditable reason);
   P3 source-discipline-outranks-recency (`consensus` is never time-decayed).

2. **Frontier demotion — the two paradigm cases.**
   - **Newton → Einstein:** decisive evidence (Bayes factor K ≥ 100, ≥ 3 independent
     observation groups, all surprise-gated) → regime-scoped demotion, exactly ONE rank.
     Newton stays consensus in the low-velocity regime.
   - **OPERA faster-than-light neutrino:** one high-surprise event, N = 1 below the
     multiplicity floor → quarantined, consensus untouched. (Historically correct: it was
     an instrumental fault.) This is the rule that would have protected special relativity.

3. **Audit-trail tamper-evidence.** The hash-chained lifecycle ledger verifies clean, and a
   single bit-flip anywhere breaks the chain — non-repudiation of forgetting decisions.

## Honest bound

Pure-stdlib dynamics over supplied/audited inputs. Bayes factors are taken as audited
input (exactly as Sophia's gate takes verdicts as audited input), not computed from raw
data. The forgetting layer is **not wired into the live consolidation loop**; it earns
`level3Evidence: true` only after a real run routes suppress/reinforce/quarantine/demote
decisions through the anti-forgetting plasticity gate.
