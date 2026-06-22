# Self-extension — closing the loop on a held-out domain

The defensible signature of general capability isn't a benchmark score; it's the
**full self-extending loop closing on a domain the system was never built for**:

> abstain → synthesize a verifier → validate it on held-out data → use it as verified
> reward to improve a policy → the gain holds on an independent eval split → competence
> flips abstain→answer — with **no human writing the check** and **no gaming**.

`selfextend/loop.py` (`close_loop`) orchestrates exactly this; `tools/run_selfextend_loop.py`
runs it and `tests/test_selfextend.py` gates it.

## Result (offline, deterministic) — [artifact](closed-loop-2026-06-22.json)

| stage | outcome |
|---|---|
| verifier promoted on held-out | ✅ (held-out acc 1.0) |
| policy accuracy pre → post (eval split) | **0.5 → 1.0 (+0.5)** |
| generalizes on independent eval (not gamed) | ✅ (1.0 ≥ threshold) |
| competence route | **abstain → answer** |
| eval calibration (ECE) | 0.0 |
| **loop closed** | **✅ all 4 invariants** |

Fail-closed check: on unlearnable (random-label) data the verifier fails validation and
the system **stays abstained** (`loop_closed=False`) — it never adopts a checker it
couldn't validate.

## Honest scope (what this is / isn't)

- **Is:** a real, GPU-free closing of the loop. The "improvement via verified reward" is
  **verifier-guided selection** (rejection sampling against a validated verifier) — a
  legitimate form of verified-reward optimization — measured on an eval split disjoint
  from verifier training, with a generalization (anti-gaming) invariant.
- **Isn't:** a live RL weight update (GRPO). That needs a GPU and consumes the same
  reward interface (`selfextend.verified_reward`). It is the one remaining rung.
- **Isn't:** an AGI claim. The domain + labels are self-authored toy data; a headline
  claim needs a third-party held-out domain and the no-overclaim gate. Tracked in the
  failure ledger.

## What would upgrade this to a capability claim
1. Run on a **third-party / unseen** domain (not self-authored).
2. Replace selection with a **live RL update** (GPU) and show held-out gain vs base.
3. Clear the **no-overclaim gate** (≥3 runs, CI excludes 0, multi-judge where judged).
