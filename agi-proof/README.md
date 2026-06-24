# Sophia AGI-Candidate Proof Package

**This is not a claim of AGI.**

This folder is the public, auditable evidence package for a precise claim:

> Sophia is an AGI-candidate provenance system with reproducible source-discipline benchmarks, fail-closed gates, and a pre-registered path to third-party replication.

## Current Evidence (gated)

- 528 bilingual training examples (philosophy, psychology, history, religion)
- Per-domain leaderboards + validated provenance deltas (see main [RESULTS.md](../RESULTS.md))
- Strong baselines (local LoRA and curated RAG)
- Full self-extension flywheel closing on held-out domains
- Pre-registered thresholds, hidden-reviewer packs, failure ledger, and replication checklist

External benchmarks and clean third-party replication remain the next required rung.

Machine-readable summary:

```bash
python tools/build_agi_proof_package.py
```

Output: `agi-proof/evidence-manifest.json`

## Package Structure

| Area | File |
|---|---|
| Consolidated proof TODO | `TODO.md` |
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
