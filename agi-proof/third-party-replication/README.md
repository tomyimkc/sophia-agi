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

---

## Claim-replication pack (reproduce the effect on YOUR model)

The checklist above replicates the **process** (clean clone builds + tests + reviewer independence).
`tools/replication_pack.py` replicates the **capability claim** — that a filter + abstention gate turns
*fabrication on unknown-answer traps* into *refusal* at low false-positive cost — **on a model of your
choosing**, with no Sophia infrastructure (stdlib-only; ships its own decontaminated traps + controls).

```bash
python tools/replication_pack.py --selftest                                        # offline sanity
python tools/replication_pack.py --endpoint http://HOST:PORT --model M --out raw-run.json          # raw
python tools/replication_pack.py --endpoint http://HOST:PORT --model M-gated --out gated-run.json  # gated
# replicable number = raw.fabrication_rate - gated.fabrication_rate, at matched control_over_abstain_rate
```

**Supports** the claim if gated `fabrication_rate` is materially below raw without a large rise in
`control_over_abstain_rate`. **Falsifies** it if the delta vanishes or is bought only by over-abstaining
on knowable controls — **a null is a valid, publishable result.** Pre-registration:
[`PRE-REGISTRATION.md`](./PRE-REGISTRATION.md). File your result (supporting or refuting) as an issue;
refutations update the failure ledger. `canClaimAGI` false.
