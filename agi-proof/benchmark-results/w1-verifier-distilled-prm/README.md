# W1 verifier-distilled PRM — local take-live with REAL MLX hidden states (2026-07-01)

> **candidateOnly:true · level3Evidence:false · canClaimAGI:false · gateMet:false**
> The `w1-verifier-distilled-prm-not-trained-live` ledger row **stays Open.** Result +
> sha256: `w1-prm-live-2026-07-01.candidate.json`.

## What was done

Implemented the W1 gate's named seam — `agent.activation_probes.build_hidden_state_featurizer(spec="mlx", model, tok)` — to return real **residual-stream vectors** (mean-pooled final hidden state, 2048-d) from the mlx_lm Qwen2.5-3B trunk, and swapped it in for the degenerate transparent `featurize_text` in the PRM path. Labels remain the **real fail-closed** `agent.step_verifier.verify_derivation` (checkers `math-sympy`, `physics-units`; 188 labeled steps, balanced 94/94).

**Fail-closed preserved:** with no live model handle the featurizer still raises `RuntimeError`, so the offline default never silently degrades and `test_truth_probe` + `test_probe_representation_training` stay green (36 passed).

## Result (held-out agreement with the symbolic oracle)

| | transparent `featurize_text` | **real MLX hidden states** |
|---|---|---|
| in-domain held-out (math), seed 0 | 0.41 (degenerate, FPR 1.0) | **0.795** (FPR 0.30) |
| within-domain 3-seed mean — math | — | 0.727 |
| within-domain 3-seed mean — physics | — | **0.896** |
| **held-out DOMAIN (train math → test physics)** | 0.50 | **0.50 (chance)** |

## Verdict — gate NOT met (row stays Open)

1. **The real featurizer works.** Hidden states lift held-out agreement from the degenerate
   transparent baseline (0.41 math / 0.50 physics) to within-domain **0.727 (math) / 0.896
   (physics)** — physics clears 0.80 within-domain. This retires the degenerate-feature
   concern and validates the implemented seam.
2. **But held-out-DOMAIN agreement is 0.50 = chance.** The accepted/rejected direction
   learned on math does **not** transfer to physics. This is the coverage trap the tool
   itself warned about, now empirically confirmed with real features. The gate requires
   **≥0.80 held-out-DOMAIN**.
3. **The PRM-as-dense-RLVR-reward half is entirely unrun** (`tools/run_rlvr.py` + a symbolic
   reward-hack audit) — needs a GPU RL stack, not available locally.

## To close it

Train a **mixed-domain** PRM (include physics in train) and test on a genuinely held-out
**third** domain; sweep per-layer/per-token features; then wire the PRM as a dense reward in
`tools/run_rlvr.py` on a GPU, keeping the symbolic verifier as a periodic reward-hack audit.

## v2 (2026-07-02) — cross-domain robustly ~chance; gate is infrastructure-bound

3-seed characterization (`w1-char-cross-domain-2026-07-02.candidate.json`): held-out-DOMAIN
agreement is **~chance in BOTH directions** — train-math→test-physics **0.488**, train-physics→
test-math **0.495** (3 seeds each). The accepted/rejected direction is domain-specific; the PRM
does not transfer across domains. Mixed-domain training reaches **0.792** on held-out *instances*
(within trained domains) — but that is not a held-out *domain*.

**W1's ≥0.80 held-out-DOMAIN gate cannot be closed locally:** only two verifier domains exist
(`math-sympy`, `physics-units`) so there is no true third held-out domain, and the
PRM-as-dense-RLVR-reward half needs a GPU RL stack. This is an **infrastructure boundary, not a
method failure** — the featurizer works within-domain (0.73/0.90). Closing needs a 3rd verifier
domain or a GPU RLVR run. Row stays **Open**.
