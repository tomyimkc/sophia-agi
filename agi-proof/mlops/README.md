# MLOps Proof Scaffolding

This directory defines production-facing metadata for future Sophia training and
proof runs. It is scaffolding only: no checkpoint in this directory is validated
external evidence, and `canClaimAGI` remains `false`.

## Files

- `checkpoint-registry.schema.json` — machine-readable shape for checkpoint
  registry entries.
- `checkpoint-registry.json` — empty registry scaffold; add entries only after
  a checkpoint has metadata, checksums, eval artifacts, and a failure-ledger ref.
- `experiment-tracking.schema.json` — machine-readable shape for the run and
  aggregation specification.
- `experiment-tracking-spec.json` — required per-run fields, multi-seed
  aggregation rules, judge-family rules, and failure reporting rules.
- `architecture-run-template.json` — scaffold-only run template tying a
  long-context candidate config to context-packing artifacts, architecture bets,
  checkpoints/adapters, evals, ablations, failure ledger refs, and promotion
  verdicts.
- `adapter-card-template.md` — model-card or adapter-card template with explicit
  no-overclaim language.
- `replication-runbook.md` — clean-room replication steps for reviewers.

## Validation

```bash
python tools/validate_proof_scaffolding.py
python tools/run_architecture_ablation.py --json
python -m pytest tests/test_proof_scaffolding.py
```

## Entry Rules

Before a checkpoint is listed in `checkpoint-registry.json`, record:

- the exact base model, adapter path, and training config hash;
- dataset manifest and decontamination report;
- weights checksum or external artifact URI;
- per-seed eval artifacts and aggregate report;
- promotion verdict from the proof gate;
- failure-ledger reference for blockers, negatives, or bounded claims;
- `candidateOnly: true` and `canClaimAGI: false`.

Promotion in this registry means "internally releasable candidate" at most. It
does not mean external benchmark validation, independent replication, or AGI.
