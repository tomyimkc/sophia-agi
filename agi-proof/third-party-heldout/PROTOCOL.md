# Third-party held-out authoring protocol

**Status:** OPEN — no third-party pack exists yet. The committed
[`third-party.commitments.json`](third-party.commitments.json) is **EMPTY by design**
(`caseCount: 0`). A clean "external generalization" claim MUST cite a non-empty
manifest produced by [`tools/seal_third_party_heldout.py`](../../tools/seal_third_party_heldout.py)
from a private pack authored under this protocol. `canClaimAGI = false` regardless.

## Why this exists

The repo's own held-out sets — `eval/external/*-style-sample.jsonl`,
`provenance_bench/data/{math_problems,code_tasks}.json`,
`benchmark/code_tasks.json` — are **repo-authored**. Even the public-benchmark-style
samples inherit the field's pretraining-contamination problem (a model may have seen
similar items), and the synthetic packs are authored by the same party that writes the
training data. A held-out gain on any of them is therefore **suggestive, not proof** of
contamination-free external generalization. This protocol closes that gap.

## The protocol (what makes a pack "third-party")

1. **Independence.** The author must have **no access** to Sophia's training data,
   training prompts, the code RLVR / math RLVR packs, or any sealed held-out item. The
   author composes novel problems from their own head or from sources Sophia was not
   trained on.
2. **Machine-checkable answers.** Every case must have an objective, ungameable oracle:
   - `math` cases carry a `scoring.gold` checked by sympy canonical equivalence.
   - `code` cases carry a `scoring.test` (hidden asserts) checked by execution.
   No LLM-judged cases in the headline pack (a judge is a separate, labelled family).
3. **Sealed before any model run.** The author writes a private pack JSON
   (`{packId, salt, cases:[...]}`) with a fresh 256-bit salt, then runs
   `python tools/seal_third_party_heldout.py --private-pack <path>`. Only the salted
   per-case SHA-256 commitments are committed (`saltStatus: "withheld until reveal"`).
   The salt + unsealed prompts stay under gitignored `private/` and are revealed only
   after the gated eval completes.
4. **Decontamination.** The committed prompts must be disjoint from Sophia's eval/training
   prompt sets (`python tools/build_local_sophia_dataset.py --check` must remain CLEAN
   after the prompts are registered). The author warrants they did not reuse Sophia text.
5. **Pre-registered thresholds.** The gated eval must clear the no-overclaim bar
   (≥3 seeds, 95% CI excludes 0, multi-judge where judged) before any headline wording.
6. **Reveal + audit.** After the eval, the salt is revealed so reviewers can verify each
   committed hash against the actual cases and re-run the oracle.

## What this is / isn't

- **Is:** the only path to a clean external generalization claim for Sophia's
  verifier-as-reward runs.
- **Isn't:** satisfied by the repo-authored style-samples or synthetic packs, however
  well-sealed. It is also **not** an AGI claim — `canClaimAGI` stays false.

## How to fill it (maintainer)

1. Commission an independent author (the `good-faith` requirement is independence, not
   affiliation — a colleague, a contractor, or a public reviewer who has not seen the
   training data is sufficient).
2. Author `private/third-party/pack.json` under the schema in
   [`tools/seal_third_party_heldout.py`](../../tools/seal_third_party_heldout.py).
3. `python tools/seal_third_party_heldout.py --private-pack private/third-party/pack.json`
   → replaces the empty manifest with the salted commitments.
4. Update the failure-ledger `third-party-heldout-pack-empty` item from OPEN → progress.
