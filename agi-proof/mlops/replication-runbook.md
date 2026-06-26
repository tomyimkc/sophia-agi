# MLOps Replication Runbook

This runbook is for repeatable checkpoint and benchmark reproduction. It
complements `agi-proof/third-party-replication/README.md` and does not replace
independent reviewer attestation.

## 1. Prepare Clean Clone

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
git rev-parse HEAD
git status --short
python tools/validate_proof_scaffolding.py
```

Record Python version, OS, hardware, accelerator availability, and whether the
working tree is dirty.

## 2. Select Artifact

Choose one checkpoint registry entry or one external benchmark declaration.
Reject the run if any required field is missing:

- registry entry or benchmark declaration ID;
- base model and adapter identifier;
- dataset manifest and held-out seal;
- decontamination report;
- exact command or official benchmark submission path;
- expected output directory;
- failure-ledger reference.

## 3. Reproduce Locally Where Possible

Run only CPU-safe or local mock validation in normal CI. GPU training and
RunPod-backed jobs must use the repository's GitHub Actions workflow when
required by project policy.

```bash
python tools/validate_proof_scaffolding.py
python -m pytest tests/test_proof_scaffolding.py
```

Do not substitute internal smoke tests for external benchmark evidence.

## 4. Record Per-Seed Runs

For every seed, preserve:

- command and config hash;
- stdout/stderr tails;
- raw outputs and scorer version;
- contamination and prompt-leak checks;
- protected-suite results;
- backend errors and empty responses;
- artifact checksums.

## 5. Aggregate

Use the rules in `experiment-tracking-spec.json`:

- at least three seeds for headline-grade internal aggregation;
- 95% CI must exclude zero for positive delta language;
- protected metrics must not regress beyond the pre-registered threshold;
- semantic/model-judged claims require at least two independent judge families;
- negative, blocked, and within-noise results are publishable evidence.

## 6. Publish Boundaries

Before public wording changes:

- update the failure ledger;
- update or create an adapter card;
- add registry checksums;
- include failures beside successes;
- keep `candidateOnly: true` and `canClaimAGI: false` unless a future,
  separately reviewed proof gate says otherwise.
