# AutoResearch × Sophia — a live self-improvement loop with brakes and an odometer

> Status: **design + candidate machinery.** The gated controller is shipped and CI-tested
> (`tools/sophia_autoresearch.py`); the live GPU loop is OPEN (needs the RunPod GitHub Action).
> `canClaimAGI` stays false. A "kept" change is CANDIDATE until a multi-seed gate clears.

## Thesis

[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) is an autonomous ML-research
loop: an agent edits `train.py`, trains ~5 minutes on one GPU, keeps the change iff `val_bpb`
dropped, else `git reset`, and repeats overnight. It is the live, weight-updating self-improvement
**engine** Sophia's README says Sophia lacks ("Cannot learn or update its weights … offline
*selection*, not parameter updates").

But autoresearch ships with **no brakes and no odometer**: its `program.md` adds *no* safeguard
against overfitting, validation-set leakage, or cheating, and treats the eval as ground truth the
agent could in principle edit. Sophia is exactly that missing trust layer — unhackable
verification, decontamination, a no-overclaim promotion gate, and a public failure ledger.

**The two are complementary halves of one machine.** Autoresearch gives Sophia a live loop;
Sophia gives autoresearch the governance the loop cannot provide for itself.

## The community already named the gap

The autoresearch generalisation thread (Helix / Greyforge) lists the *actual* hard problems
"beyond the basic loop": **instruction authority, evaluation isolation, review surfaces, bounded
execution, and artifact discipline.** Sophia already implements all five.

| Hard problem beyond the loop | Sophia component |
|---|---|
| Instruction authority | `constitution/constitution.v2.json` + deontic rules + `agent/conscience.py` |
| Evaluation isolation | `tools/assert_decontam.py` + `agent/code_verifier.py` (isolated execution) |
| Review surfaces | `agi-proof/failure-ledger.md` + `tools/claim_gate.py` receipts + `RESULTS.md` |
| Bounded execution | `agent/swarm_router.py` least-privilege specs + `agent/swarm_trust_boundary.py` |
| Artifact discipline | the `ci-artifact-drift` gates + `tools/lint_claims.py` |

## What we keep, and what we replace

**Keep the autoresearch architecture:** one editable surface, a fixed budget, a single metric,
git-as-research-trail, loop-until-interrupt.

**Replace the greedy keep/discard** with the Sophia-gated decision in
`tools/sophia_autoresearch.py`. A change is `keep` only if **all** hold; otherwise `discard` (or
`reject_tamper`), and the reason is logged:

1. **Reward-hacking firewall** — the agent may edit policy / data / hyperparameters, **never** the
   verifier, gate, eval, reward, or constitution. A diff touching a protected path is
   `reject_tamper` even with a spectacular metric (`firewall_violations`,
   `DEFAULT_PROTECTED_PATTERNS`). This mechanises the constitution's "no reward/verifier tampering".
2. **Evaluation isolation** — a result that failed `assert_decontam` (leakage) is discarded.
3. **Power-gated improvement** — kept only if the metric improves with a **95% CI excluding zero**
   on the improving side, measured over paired seeds / held-out items — never one 5-minute number.
4. **Protected-regression block** — religion / history (and any registered protected behaviour)
   must not regress, even for a metric win.
5. **Honest trail** — every discard/reject yields a failure-ledger record; kept changes stay
   CANDIDATE until a real multi-seed run clears the project's κ ≥ 0.40 / CI gate.

## Where to point the loop (not `val_bpb`)

Sophia is not a pretraining lab, so literally minimising nanochat `val_bpb` only fits the toy
`pretraining/` track. The high-value move is to adopt the *method* and aim it at Sophia's own
targets, making the "single editable surface" one of:

- the **verifier mixture / gate thresholds** → maximise the verified hallucination-Δ on a held-out pack;
- the **`trajectory_reward` λ-weights** (`λ_cost, λ_trust, λ_kl` in `provenance_bench/swarm_rl.py`)
  → maximise verified-success / cost;
- the **preference-data mixture** feeding `tools/gen_verifier_dpo.py` → maximise adapter transfer.

Call it **"AutoResearch for the Wisdom Gate."**

## The Claude × Karpathy synthesis

The honest realisation is: **a coding agent (Claude Code) as the autoresearch agent**, **Sophia's
gates as the trust layer**, and **the harness's long-horizon / subagent capability as the overnight
orchestrator** (`docs/09-Agent/Harness-Roadmap.md`). That trio closes Sophia's single biggest OPEN
ledger item — *no live RL weight update / self-improvement loop* — without surrendering the
no-overclaim discipline that is the project's moat. The research sub-agent runs behind the
`GatedSharedState` trust boundary: its outputs enter shared state only if they clear the gate.

## Shipped vs OPEN

| Piece | State |
|---|---|
| `tools/sophia_autoresearch.py` gated keep/discard controller + firewall + ledger trail | shipped, 9 invariants + 9 tests green |
| Wire the GPU training step in as the `experiments` iterator (via RunPod GitHub Action) | OPEN |
| Point the loop at a Sophia metric + run an overnight gated sweep, 3-seed | OPEN |
| Promote any kept change past CANDIDATE (κ ≥ 0.40 / CI gate, ≥2 judge families) | OPEN |

## Honest limits

- **Needs a GPU** for the live loop → RunPod GitHub Action only, never local SSH (repo guardrail).
- **A 5-minute budget lacks statistical power** for most Sophia metrics → "kept" is a *power-gated*
  decision and stays CANDIDATE; an overnight point-estimate must never become a headline.
- **Never let the optimiser near the verifier.** The whole value collapses if the agent can edit
  what scores it; the firewall is not optional.
- This is design + candidate machinery, not a result. Whether the gated loop yields a *transferable*
  improvement is OPEN — the same pre-registered question as the adapter work, not an assumption.

## Sources

- `karpathy/autoresearch` — repository and `program.md`
- autoresearch Discussion #447 — "Generalize to any research problem" (Helix / Greyforge notes)
