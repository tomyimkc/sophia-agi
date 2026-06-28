# Architecture Bets

This directory pre-registers Sophia's long-context architecture bets before any
training run or benchmark result exists. The files are scaffolding only:
`candidateOnly=true`, `canClaimAGI=false`, and every bet remains `not_run`.

## Files

- `architecture-bet.schema.json` - machine-readable contract for architecture bet
  manifests.
- `manifest.json` - initial scaffold for verifier-gated long context, hybrid
  memory, selective tool-use routing, council orchestration, verifier-as-reward,
  long-context compression/recall, and the architecture-aware eval harness.

## Verdict States

Architecture artifacts use the same explicit states as Sophia proof tooling:

- `accepted` - machine-checkable evidence cleared the relevant gate.
- `rejected` - the gate found a violation or the result failed a threshold.
- `held` - evidence is incomplete or contested; do not promote.
- `abstain` - the correct action is not to answer or not to route the component.

No `accepted` state appears in the initial manifest. It can be recorded only after
offline artifacts, ablations, failure-ledger updates, and promotion checks exist.

## Validation

```bash
python tools/run_architecture_ablation.py --json
python -m pytest tests/test_architecture_scaffolding.py
```

The dry-run tool validates shape and summarizes planned ablations. It does not run
training, launch GPU jobs, or declare model capability.
