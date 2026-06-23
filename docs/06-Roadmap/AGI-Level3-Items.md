# Sophia AGI — Level 3 implementation items (machine-verified)

Source: `tools/run_agi_verification_gate.py` (criteria: `agi-proof/agi-verification/criteria.json`).
Level 3 = "Strong AGI-candidate evidence". Mark each item done only when the gate reports `pass`.

> Claim boundary: passing every item below makes Sophia a strong AGI-candidate under a
> non-human verification gate. It does NOT make Sophia "proven AGI"; that requires external
> scientific/social review beyond any self-run script.

## Status snapshot

Deterministic lanes show `open` until you run with `--run-local-smoke`; artifact lanes show
`missing`/`open` until a qualifying artifact exists.

| Item | Lane id | Gate kind | Pass condition (machine) |
|---|---|---|---|
| Provenance validation | `provenance_validation` | deterministic | `tools/validate_attribution.py` exits 0 |
| Validated provenance delta | `published_provenance_delta` | artifact | a `validated` row with all `validatedChecks` true + delta CI excludes 0 |
| Calibration / abstention | `calibration_abstention` | artifact | calibration delta CI excludes 0, or corroboration artifact present |
| Grounded gate N>=40 | `grounded_gate` | deterministic | `run_grounding_gate.py --runs 3 --min-cases 40` prints `ALL INVARIANTS HOLD` |
| Coding eval | `coding_eval` | deterministic | `run_coding_eval.py` all cases pass |
| Memory append-only safety | `memory_eval` | deterministic | `run_memory_eval.py` all checks pass |
| Verifier-synthesis integrity | `verifier_synthesis_integrity` | deterministic | `run_verifier_synthesis.py --json` `ok=true` |
| Cross-domain transfer | `cross_domain_transfer` | deterministic | `run_cross_entity.py --json` `ok=true` |
| Self-extension loop closes | `self_extension_loop` | artifact | closed-loop artifact: loop_closed, invariants true, post>pre |
| Hidden full comparison | `hidden_full_comparison` | artifact | non-smoke `private-hidden` aggregate where sophia_full beats raw |
| Distribution-shift learning | `distribution_shift` | artifact | `passingSignal=true`, postTest >= 10 cases, no old-knowledge regression |
| Long-horizon 30 min | `long_horizon_30m` | artifact | substantive run, durationSec >= 1800, interventions <= 2 |
| RLVR live training | `rlvr_live_training` | artifact | live GRPO training log + report present |

## Remaining work to reach Level 3 (the real gaps)

1. **Hidden full comparison (`hidden_full_comparison`)** — produce a real, non-smoke,
   `private-hidden` pack and run:
   ```bash
   python tools/run_hidden_eval_full.py --pack private/hidden-evals/PACK.json \
     --mode raw=... --mode raw_tools=... --mode rag_only=... \
     --mode gate_only=... --mode sophia_full=... \
     --out agi-proof/hidden-reviewer-packs/results/full-aggregate.json \
     --manual-review-out agi-proof/hidden-reviewer-packs/results/manual-review.md
   ```
   Target: sophia_full beats raw, raw+tools, rag_only, gate_only.

2. **Distribution-shift (`distribution_shift`)** — build a multi-case (>=10 pre / >=10 fresh post)
   new-domain pack with promoted learning records and an old-knowledge stability pack:
   ```bash
   python tools/run_distribution_shift.py EXPERIMENT_SPEC.json --backend adapter
   ```

3. **Long-horizon 30 min (`long_horizon_30m`)** — run a substantive autonomy task:
   ```bash
   python tools/run_long_horizon.py \
     --spec agi-proof/long-horizon-runs/templates/30min-repo-repair.json \
     --out-dir agi-proof/long-horizon-runs/<date>-30min
   ```

4. **Deterministic lanes** — wire into CI so they run on every change:
   ```bash
   python tools/run_agi_verification_gate.py --target level3 --run-local-smoke
   ```
