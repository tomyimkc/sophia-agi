# Level-3 blocker plan (post-RLVR-null pivot)

**Context.** The RLVR-adapter κ-validation track is closed as a **NULL** (4-judge majority-vote
win-rate 0.532, p=0.65 — see the ledger row `rlvr-adapter-kappa-2family-below-bar-2026-06-26`
and `docs/methodology/llm-judge-validation.md`). A ≥300-case third-party pack would, against a
~0.53 effect, most likely only **confirm** the null, so GPU/pack spend there is not justified.
This doc redirects effort to the genuine Level-3 lanes. `canClaimAGI` stays **False**.

**Gate state (honest).** `run_agi_verification_gate.py --target level3` reports `below-level2`
on this dev box, but `--target level2` lists **no missing blockers** — the level-1→2 confirmation
needs the local-smoke invariants, and `tools/run_cross_entity.py` **exceeds the gate's hardcoded
300s subprocess timeout** on this hardware. So the handover's `level2` stands; the local
`below-level2` is a **timeout/env artifact, not an evidence regression**.

## The five remaining Level-3 blockers, triaged

| Blocker | EV | Autonomy (no human/GPU) | Cheapest credible path | Gating dependency |
|---|---|---|---|---|
| **hidden_full_comparison** | ★★★ — the #1 reviewer-dismissal gap (solo-author independence) | **Low** | harness + seal already exist (`agi-proof/third-party-heldout/PROTOCOL.md`, `tools/seal_third_party_heldout.py`, `caseCount:0`). Commission an **externally-authored** ≥40-case / ≥4-domain pack → run → gate. | **external human authorship** |
| **distribution_shift** | ★★★ — learn-without-forgetting is a real capability | **Med** | author an experiment spec (`run_distribution_shift.py --template`), **≥10 pre / ≥10 fresh post**, no old-knowledge regression. `run_learning_shift.py` is the impl; `--backend adapter` implies a learning run (GPU) — confirm whether a CPU/offline backend qualifies. | spec design + likely GPU |
| **long_horizon_30m** | ★★ — substantive autonomy | **High** (hardware-agnostic, local, no GPU/data) | author a **substantive ≥1800s** goal+plan spec for `run_long_horizon.py --spec`, ≤2 interventions, execution-truth `objectiveGate`; launch as a background run. **Must be genuinely substantive — a padded/sleep run would be gaming.** | spec design only |
| **cross_domain_transfer** | ★ — gate-plumbing (re-runs existing invariants) | High but **slow** | `tools/run_cross_entity.py --json` exceeds the gate's 300s cap on this box | raise gate timeout (≈1-line) or run standalone |
| **verifier_synthesis_integrity** | ★ — gate-plumbing | High | `--run-local-smoke` | (same smoke path) |

## Recommended sequence

1. **long_horizon_30m** — the most autonomously-advanceable *real* lane (no GPU, no human data).
   Author a substantive ≥1800s goal+plan and run it. **Next concrete action.**
2. **distribution_shift** — scaffold the experiment spec; determine if a non-GPU backend
   qualifies. If it needs adapter training, dispatch via the RunPod GitHub Action.
3. **hidden_full_comparison** — highest EV but **blocked on external authorship**. Make the
   pack fully turnkey (seal + run + gate) so an external author can drop in ≥40 cases.
4. **cross_domain_transfer / verifier_synthesis_integrity** — gate-plumbing. Fix the 300s
   gate timeout so `level2` is locally confirmable; these add no new capability evidence.

## Division of labour

**I can do autonomously:** author + launch the long_horizon_30m substantive run; scaffold the
distribution_shift spec; make the hidden-pack harness turnkey; fix the gate's 300s timeout.

**Needs you:** the externally-authored hidden pack (the real bottleneck); a GPU dispatch
(`RUNPOD_API_KEY`) if distribution_shift requires adapter training; fresh API keys (rotate the
OpenRouter + Google keys pasted in chat).

## Honest bottom line

No Level-3 lane is a quick code win. Two (hidden pack, distribution-shift) are gated on
real external data / training; one (long_horizon) is autonomously runnable but must be a
genuine 30-minute task, not padding; two are gate-plumbing. The honest path to a Level-3
*candidate* claim runs through **real evidence on the three substantive lanes**, not more
harness code — exactly as the handover framed it.
