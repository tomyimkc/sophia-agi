# Third-Party Replication

Independent reproduction is mandatory for any expert-facing AGI claim.

## Clean Clone Commands

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python tools/validate_attribution.py
python tools/build_agi_proof_package.py
python tools/build_web_data.py
python -m pytest
```

## Reviewer Checklist

- [ ] I used a clean clone.
- [ ] I recorded commit hash and environment.
- [ ] I ran validation and tests.
- [ ] I created hidden tasks not visible to Sophia before evaluation.
- [ ] I ran baselines and ablations on the same hidden pack.
- [ ] I reported failures beside successes.
- [ ] I did not describe pending external benchmarks as achieved.

## Harness

`tools/run_replication_check.py` runs the clean-clone checklist, records the
commit hash + environment + per-command returncodes, and emits a
reviewer-signature template with the machine-checkable items filled and the human
attestation left blank. It **cannot self-certify** — the reviewer identity,
the reviewer-authored hidden tasks, and the signature must be completed by an
independent human.

```bash
# Machine-checkable items (read-only; skips mutating builders):
python3.12 tools/run_replication_check.py

# In a real clean clone, a reviewer also runs the builders:
python3.12 tools/run_replication_check.py --full
```

Reviewer:

Date:

Commit:

Environment:

Signature:
