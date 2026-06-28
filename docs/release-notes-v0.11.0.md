# Release notes — v0.11.0 (draft for the GitHub Release / Zenodo DOI)

> Paste this into the GitHub Release body when you tag `v0.11.0`. With Zenodo enabled for
> the repo, publishing this release auto-archives the snapshot and mints a permanent version
> DOI under the author of record (Yim Kin Cheong, ORCID 0009-0005-9520-0033). After the DOI is
> minted, update `CITATION.cff` (`doi:` field) and the README badge.

---

## Sophia — the Wisdom Gate · v0.11.0

**The independent-verification toolkit release.** A provenance-aware, verifier-gated
reasoning layer for LLMs that **abstains instead of fabricating**:

```
claim  →  verify against independent sources  →  accept · abstain · block
```

This release is also a **defensive publication**: it fixes a citable authorship + date
record for the method as extended here. Cite via [`CITATION.cff`](../CITATION.cff); the
[whitepaper](../paper/sophia-whitepaper.md) (§3.6, §5, §8) describes it in full.

> **Scope, stated plainly.** A research program *toward* grounded, machine-checked reasoning
> — **not a claim of AGI** (`canClaimAGI = false`). Where a threshold is not met, we say so.
> The deliverable is the honest machinery, the measured data, and the recorded negatives.

### What's new — a layered, independence-labelled verification toolkit
Each verifier is routed to the fabrication mode it covers and **every verdict is tagged with
its independence tier**; the toolkit is **fail-open on ignorance** (it never fabricates a
contradiction) and reports its coverage honestly.

- **Citation existence** (`agent/citation_existence_verifier.py`) — a cited study/DOI that
  Crossref cannot confirm is flagged (the *Mata v. Avianca* mode). HIGH independence; **0% over-block**.
- **Attribution swap** (`agent/attribution_swap_verifier.py`) — a real work credited to the wrong
  creator, checked against the Wikidata record. HIGH independence. Live, 3-run bootstrap CI:
  **caught 10.8% [9.3%, 11.6%] / 0.0% [0,0] over-block** (≈⅔ of the swap cases it targets).
- **Source faithfulness** (`agent/source_faithfulness_verifier.py`) — a real source whose *finding*
  is misstated, judged by a multi-judge entailment panel over an independent source, strict-majority
  consensus, fail-open on insufficiency. MEDIUM independence (flagged). Demo 4/4.
- **Supporting layers** — LLM/NLI debunk detector via a meta-labeler (detection **0→100%** live),
  core-claim verification, Google+Wikidata layered routing, Wikipedia retrieval.
- **Bench** — `tools/run_source_contamination_bench.py --verifier {atomic,core,hybrid,citation,attribution,faithfulness}`,
  `--answer-spec/--judge-spec` separation, `--runs N` bootstrap CIs, `--retrieve`.
- **MCP + skill** — `sophia_source_verify` tool and the `source_verify_audit` skill surface the
  keyless high-independence checks.
- **Replication** — `agi-proof/verification-replication/` (REPRODUCE.md, EXPECTED-RESULTS.json,
  DECONTAMINATION-CHECKLIST.md) + `tools/verify_replication_manifest.py` (54 checks);
  `paper/THESIS-OUTLINE.md`; an arXiv-ready source under `paper/arxiv/`.

### Method & honesty
- **No free lunch:** catching open-world contamination needs an oracle that covers the claim
  (sparse), fail-closed strictness (which over-blocks), or model knowledge (low independence).
- **Self-correction recorded:** an earlier **70.6%** curated over-block figure was found to be a
  stale-report artifact and **withdrawn** (corrected to **5.9%**; the true driver is open-world
  retrieval). See `agi-proof/THEORY-ISSUES-RESOLUTION-2026-06-28.md` and the failure ledger.

### Carried forward (validated, from prior releases)
- Provenance delta **36.1% → 23.6%** (Δ **12.5%** [5.6%, 19.4%]) at **0%** false-positive cost.
- Calibration/abstention: **0%** fabrication on "I don't know" traps vs 16.7–25% raw (κ = 0.74).
- Self-consistency selective prediction validated on **external** public data (SimpleQA).

### Outstanding (why `canClaimAGI` stays false)
An independent third-party run of the replication pack on an **independently-authored** pack.

---

### How to tag & publish (triggers the Zenodo DOI)
```bash
git tag -a v0.11.0 -m "Sophia v0.11.0 — independent-verification toolkit"
git push origin v0.11.0
# then publish the GitHub Release from the tag with this body; Zenodo mints the version DOI.
```
