# Third-party held-out pack

**Status:** OPEN / EMPTY (`caseCount: 0` in
[`third-party.commitments.json`](third-party.commitments.json)).

This is the **only path to a clean external generalization claim**. The repo-authored
held-out sets (style-samples + synthetic RLVR packs) are useful, suggestive evidence,
but they inherit two honesty gaps a third-party pack closes:

1. **Pretraining contamination** — public-benchmark-style items may have been seen by the
   base model during pretraining.
2. **Label provenance** — the same party writes the training data and the held-out
   labels, so an "uplift" could be circular.

A third-party pack — authored independently, with machine-checkable (sympy/exec) oracles,
sealed with a salted commitment **before** any model run, and revealed only after the
gated eval — removes both gaps.

- **Protocol:** [PROTOCOL.md](PROTOCOL.md)
- **Sealing tool:** [`tools/seal_third_party_heldout.py`](../../tools/seal_third_party_heldout.py)
- **Committed manifest:** [`third-party.commitments.json`](third-party.commitments.json) (empty)
- **Failure-ledger item:** `third-party-heldout-pack-empty` (OPEN)

`canClaimAGI = false` regardless of outcome.
