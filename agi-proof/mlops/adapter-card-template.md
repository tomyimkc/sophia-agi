# Adapter Card Template

Use this template for a Sophia adapter or model-card draft. Keep unknown fields
as `TBD` until the artifact exists. Do not turn planned runs into claimed
results.

## Identity

- Adapter ID: `TBD`
- Base model: `TBD`
- Adapter format: `LoRA | QLoRA | MLX LoRA | other`
- Registry entry: `agi-proof/mlops/checkpoint-registry.json#TBD`
- Training config hash: `TBD`
- Dataset manifest: `TBD`
- Decontamination report: `TBD`
- Weights checksum: `TBD`

## Claim Boundary

This adapter is an AGI-candidate research artifact. It is not proof of AGI,
not a hallucination guarantee, and not validated external benchmark evidence
unless the listed proof artifacts say so. `candidateOnly: true`.
`canClaimAGI: false`.

## Intended Use

- Intended runtime gate: `TBD`
- Supported domains: `TBD`
- Out-of-scope domains: `TBD`
- Required human oversight: `TBD`

## Training Summary

- Date: `TBD`
- Operator: `TBD`
- Hardware/backend: `TBD`
- Seeds: `TBD`
- Training rows: `TBD`
- Held-out seal: `TBD`
- Known contamination drops: `TBD`

## Evaluation Summary

Report all values as candidate-only unless the proof gate clears.

| Suite | Seeds | Baseline | Adapter | Delta | 95% CI | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Failure Ledger Links

- `TBD`

## Release Checklist

- [ ] Checkpoint registry entry complete.
- [ ] Weights checksum recorded.
- [ ] Dataset decontamination clean or blocker recorded.
- [ ] Per-seed artifacts retained.
- [ ] Aggregate report includes failures beside successes.
- [ ] Promotion verdict recorded.
- [ ] Failure ledger updated.
- [ ] Public wording keeps `canClaimAGI: false`.
