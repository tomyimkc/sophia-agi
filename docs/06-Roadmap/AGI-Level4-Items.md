# Sophia AGI — Level 4 implementation items (machine-verified)

Source: `tools/run_agi_verification_gate.py` (criteria: `agi-proof/agi-verification/criteria.json`).
Level 4 = "Expert-reviewable AGI evidence". Requires all Level 3 items PLUS the four below.

> Claim boundary: even with every Level 4 item green, the correct public claim is
> "Sophia has expert-reviewable AGI evidence", NOT "Sophia is proven AGI". Reviewer
> independence and signature cannot be self-certified by a repo script.

| Item | Lane id | Gate kind | Pass condition (machine) |
|---|---|---|---|
| RLVR adapter held-out eval | `rlvr_adapter_eval` | artifact | `mode=real`, `passed=true`, no false-positive regression, adapter mean reward > base |
| External benchmarks (Sophia-full) | `external_benchmarks` | artifact | artifact with `system` containing "sophia", non-null `score`, `total` |
| Long-horizon 2h + 1d | `long_horizon_2h_1d` | artifact | substantive runs: one >= 7200s and one >= 86400s, interventions <= 4 |
| Third-party machine replication | `third_party_machine_replication` | artifact | clean-clone checklist artifact: all commands/checks return 0 |

## Remaining work to reach Level 4

1. **RLVR adapter eval (`rlvr_adapter_eval`)** — re-run live RLVR (checkpoint now tar+hashed
   on teardown), then:
   ```bash
   python tools/eval_rlvr_adapter.py --mode real \
     --model zai-org/glm-4-9b-chat-hf \
     --adapter training/rlvr/checkpoints/sophia-rlvr-v1 \
     --out agi-proof/benchmark-results/rlvr.adapter-eval.real.json
   ```

2. **External benchmarks (`external_benchmarks`)** — run Sophia-full (not just base model)
   on ARC-style, GAIA-style, SWE-bench-style, or METR-style suites; write artifacts to
   `agi-proof/external-benchmarks/results/` with `system: sophia-full`.

3. **Long-horizon 2h + 1d (`long_horizon_2h_1d`)** — run:
   ```bash
   python tools/run_long_horizon.py --spec agi-proof/long-horizon-runs/templates/2h-skill-forge.json --out-dir agi-proof/long-horizon-runs/<date>-2h
   python tools/run_long_horizon.py --spec agi-proof/long-horizon-runs/templates/1day-wiki-maintenance.json --out-dir agi-proof/long-horizon-runs/<date>-1day
   ```

4. **Third-party replication (`third_party_machine_replication`)** — have an independent
   reviewer run the clean-clone checklist (`tools/run_replication_check.py --full`); machine
   items can pass automatically, but reviewer identity/signature stays human.

## Hard limit (Level 5)

`level5` ("Proven AGI") is intentionally **not** machine-assertable. The gate keeps
`canClaimAGI=false` permanently; only external scientific/social consensus can move beyond
expert-reviewable evidence.
