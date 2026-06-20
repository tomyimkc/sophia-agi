# Sophia AGI-Candidate Proof Package

Sophia AGI is not presented as proven AGI. This folder is the public evidence
package for a narrower and testable claim:

> Sophia is an AGI-candidate provenance system with reproducible source-discipline
> benchmarks, RAG/local-model baselines, and a pre-registered path toward hidden
> evaluation, ablation, long-horizon autonomy, learning-under-shift, and
> third-party replication.

## Current Evidence

- 518 training examples across philosophy, psychology, history, and religion.
- 23 visible benchmark cases across four domains.
- `sophia-v1` local LoRA baseline: 20/23 visible benchmark cases.
- `rag-claude` curated RAG baseline: 22/23 visible benchmark cases.
- Claude Sonnet/reference runs are tracked in per-domain leaderboards.
- Public failure boundary: external benchmarks and independent replication are
  still required before any stronger AGI claim.

Machine-readable summary:

```bash
python tools/build_agi_proof_package.py
```

Output: `agi-proof/evidence-manifest.json`

## Package Structure

| Area | File |
|---|---|
| Operational definition | `definition.md` |
| Pre-registered thresholds | `preregistered-thresholds.md` |
| Benchmark result artifacts | `benchmark-results/README.md` |
| External benchmark plan | `external-benchmarks/README.md` |
| Baseline and ablation protocol | `baseline-ablation/README.md` |
| Hidden reviewer packs | `hidden-reviewer-packs/README.md`, `hidden-reviewer-packs/schema.json` |
| Long-horizon autonomy logs | `long-horizon-runs/README.md` |
| Learning under distribution shift | `learning-under-shift/README.md` |
| Failure ledger | `failure-ledger.md` |
| Third-party replication | `third-party-replication/README.md` |
| Religion figure council | `../docs/08-Domains/Religion-Figure-Council.md` |
| Coding council | `../docs/08-Domains/Coding-Council.md` |
| Manual semantic review | `hidden-reviewer-packs/MANUAL-SEMANTIC-REVIEW.md` |

## Claim Ladder

1. **Implemented**: corpus, schema, source records, dispute notes.
2. **Implemented**: visible local benchmarks and leaderboards.
3. **Implemented**: RAG/local-model baselines.
4. **Protocol-ready**: ablation comparisons.
5. **Protocol-ready**: hidden reviewer tasks.
6. **Not yet run**: ARC-AGI/GAIA/SWE-bench/METR-style external evaluations.
7. **Not yet run**: independent clean-clone reproduction.

## Public Wording

Use:

> Sophia AGI is an AGI-candidate proof package for provenance-aware reasoning.

Do not use:

> Sophia is proven AGI.
