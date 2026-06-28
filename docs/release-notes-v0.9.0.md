# Release notes — v0.9.0 (draft for the GitHub Release / Zenodo DOI)

> Paste this into the GitHub Release body when you tag `v0.9.0`. With Zenodo enabled for
> the repo, publishing this release auto-archives the snapshot and mints a permanent DOI.
> After the DOI is minted, add it to `CITATION.cff` (`doi:` field) and the README badge.

---

## Sophia — the Wisdom Gate · v0.9.0

**Wisdom before intelligence.** A provenance-aware, verifier-gated reasoning layer for
LLMs that **abstains instead of fabricating**:

```
claim  →  verify against sources  →  accept · abstain · block
```

This release is also a **defensive publication**: it fixes a citable authorship + date
record for the method. Cite via [`CITATION.cff`](../CITATION.cff); the
[whitepaper](../paper/sophia-whitepaper.md) describes the method in full.

> **Scope, stated plainly.** A research program *toward* grounded, machine-checked
> reasoning — **not a claim of AGI**. Pre-registered thresholds are stated; where not yet
> met, said so. The deliverable is the honest machinery and the measured data.

### Validated results (clear the no-overclaim gate)
- **Provenance delta:** on a local 8B model, hallucinated attributions **36.1% → 23.6%**
  (Δ **12.5%**, 95% CI [5.6%, 19.4%]) at **0% false-positive cost**.
- **Calibration / abstention:** full Sophia fabricates **0%** on "I don't know" traps
  (deterministic scorer, 3 runs) vs 16.7–25% raw; corroborated by two independent judge
  families (κ = 0.74).
- Every public number requires ≥2 judge families, κ ≥ 0.40, ≥3 runs, and confidence
  intervals. See [RESULTS.md](../RESULTS.md).

### What's in this release
- Provenance-aware accept/abstain/block gate; belief/provenance graph with retraction +
  counterfactual analysis; deterministic and model-based verifiers; calibrated abstention;
  fail-closed governance contract.
- Bilingual humanities corpus (528 examples) — also on
  [Hugging Face](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus).
- IP / provenance package: whitepaper, prior-art survey, and IP-protection strategy
  (`docs/IP-PROTECTION.md`).

### Honest limitations
Several benchmarks are small and self-authored (third-party replication pending).
Provenance-derived confidence predicts source quality, not answer correctness. Strict
grounding trades recall for trap-safety. No claim of AGI, sentience, or consciousness.

### Reproduce
```bash
python scripts/demo_gate.py                          # offline gate demo, no keys
python tools/run_provenance_delta.py --models mock   # offline plumbing
```

### How to cite
See [`CITATION.cff`](../CITATION.cff). Once this release is archived on Zenodo, cite the
minted DOI.

**License:** Apache-2.0 (code, tools, benchmarks, corpus). Brand/trademarks reserved — see
[NOTICE.md](../NOTICE.md) and [TRADEMARK-POLICY.md](../TRADEMARK-POLICY.md).
