# Verification Toolkit — Reproduction Runbook (2026-06-28)

This pack lets an independent party clone the repo, set their own API keys, and re-run the
2026-06-28 independent-verification toolkit. It has two tiers:

- **Tier A — KEYLESS / deterministic.** No keys, no network. Reproduces the *architecture and the
  fail-closed contracts* via `python3 tests/test_*.py`. This is the floor: it must pass on a clean
  clone before any live run is meaningful.
- **Tier B — LIVE.** Re-runs the bench tools against real models/oracles with **your own** keys,
  and compares the output against the committed report JSONs listed here.

Every number in this runbook is a **candidate**, not a validated headline: each source report
carries `candidateOnly:true` / `validated:false`. `canClaimAGI` stays **false** throughout.
See `DECONTAMINATION-CHECKLIST.md` for why these are candidates and what a true third-party
replication additionally requires. The machine-readable expected values are in
`EXPECTED-RESULTS.json`; `tools/verify_replication_manifest.py` confirms every file below exists.

---

## 0. Clean clone + manifest check (no keys)

```bash
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python3 tools/verify_replication_manifest.py        # PASS = every module/test/report present, all canClaimAGI=false
python3 tests/test_verify_replication_manifest.py   # asserts the above
```

`pytest` is **not** required (and may be absent). Every test below is a standalone script that
prints `ok <name>` per test and `all passed`, and is run directly with `python3`.

---

## Tier A — KEYLESS / deterministic (no keys, no network)

Run each. Each exits non-zero on failure and prints `all passed` on success. What each locks:

| Command | Asserts (the fail-closed contract it locks) |
|---|---|
| `python3 tests/test_llm_debunk_detector.py` | LLM/NLI debunk detector classifies debunk vs affirm vs abstain; fail-closes without keys. |
| `python3 tests/test_meta_labeler.py` | Abstaining meta-labeler abstains rather than guessing when below confidence. |
| `python3 tests/test_source_verifier.py` | Independent source-verification channel: truth_refs independent of the contaminated source. |
| `python3 tests/test_core_claim_verifier.py` | Core-claim verification routes a claim to an oracle and tiers independence (high vs low). |
| `python3 tests/test_layered_verifier.py` | Layered verifier routes each claim to the most independent oracle that covers it; abstains otherwise. |
| `python3 tests/test_core_claim_source_verifier.py` | Pass-unless-contradicted (fail-open) core-claim verifier on verbose grounded answers. |
| `python3 tests/test_hybrid_source_verifier.py` | Hybrid (core-claim direction + authoritative oracles); 0% over-block by construction. |
| `python3 tests/test_citation_existence_verifier.py` | Never vouch for a citation it cannot confirm exists (Crossref existence check); fail-closed. |
| `python3 tests/test_attribution_swap_verifier.py` | Real work credited to wrong creator caught via Wikidata; fail-OPEN on entity ambiguity. |
| `python3 tests/test_wiki_truth_refs.py` | Open-world truth-ref retrieval shape and independence. |
| `python3 tests/test_proof_novelty.py` | Semantic proof-novelty assessor (Lean cluster support). |
| `python3 tests/test_retrieval_transition_model.py` | Retrieval-augmented transition predictor (world-model FLOOR baseline). |
| `python3 tests/test_llm_world_model.py` | LLM-as-world-model (world-model CONTENDER) shape and fallback. |
| `python3 tests/test_verify_replication_manifest.py` | This pack's manifest checker reports PASS. |

Run all in one shot:

```bash
for t in llm_debunk_detector meta_labeler source_verifier core_claim_verifier \
         layered_verifier core_claim_source_verifier hybrid_source_verifier \
         citation_existence_verifier attribution_swap_verifier wiki_truth_refs \
         proof_novelty retrieval_transition_model llm_world_model \
         verify_replication_manifest; do
  python3 "tests/test_${t}.py" >/dev/null && echo "ok ${t}" || echo "FAIL ${t}"
done
```

You can also exercise the bench *plumbing* keyless with the deterministic `--fake` path
(no model, no network) before spending any keys:

```bash
python3 tools/run_source_contamination_bench.py --fake --no-write
python3 tools/run_debunk_gate_bench.py --fake --detector llm
```

---

## Tier B — LIVE (your own keys)

### Keys / environment variables

Provide your **own** keys; the original numbers were produced by a single operator's keys.

| Env var | Used by | For |
|---|---|---|
| `LLMHUB_API_KEY` | `llmhub:<model>` answer/judge specs | the answer model (e.g. `llmhub:claude-sonnet-4-6`) |
| `OPENROUTER_API_KEY` | `openrouter:<vendor/model>` specs | the separated judge (e.g. `openrouter:deepseek/deepseek-chat`) |
| `GOOGLE_FACTCHECK_API_KEY` | Google Fact Check oracle (core-claim / layered paths) | high-independence professional ClaimReview lookups |
| `OPENAI_API_KEY` | `tools/run_debunk_gate_bench.py --relay` | the relay key the debunk bench reads for its live LLM detector |

Models change over time; if your provider no longer serves `claude-sonnet-4-6` or
`deepseek/deepseek-chat`, substitute comparable strong models via `--answer-spec` / `--judge-spec`
and expect the numbers to shift. The **shape** of the findings (no-free-lunch, fail-closed
abstention, coverage-bounded oracles) is what is being reproduced, not exact percentages.

### Protocol rigor — which runs are which

- **3-run + answer≠judge + bootstrap CI** (the rigorous protocol the capstone recommends):
  only **attribution-swap** in this set meets it (`--runs 3`, separated specs).
- **Single-run candidates**: everything else below. To raise any of them toward "validated",
  re-run with `--runs >=3` and `--answer-spec != --judge-spec`, and report the bootstrap CI.

### B1. Source-contamination bench (`tools/run_source_contamination_bench.py`)

Verifier modes: `--verifier {atomic,core,hybrid,citation,attribution}`. Flags:
`--answer-spec` / `--judge-spec` (separate the answer model from the entailment judge),
`--runs N` (N>1 reports a bootstrap 95% CI), `--retrieve` (open-world Wikipedia refs instead of
curated), `--fake` (no network), `--relay` (live).

| Run | Command | Expected headline (committed report) | Runs |
|---|---|---|---|
| Citation-existence | `python3 tools/run_source_contamination_bench.py --verifier citation --relay --answer-spec llmhub:claude-sonnet-4-6` | **9.3% caught / 0.0% over-block**, HIGH independence — `agi-proof/source-verifier/citation-existence-2026-06-28.json` | 1 (candidate) |
| Attribution-swap | `python3 tools/run_source_contamination_bench.py --verifier attribution --relay --answer-spec llmhub:claude-sonnet-4-6 --judge-spec openrouter:deepseek/deepseek-chat --runs 3` | **10.8% caught [9.3, 11.6] / 0.0% over-block [0.0, 0.0]**, HIGH independence — `agi-proof/source-verifier/attribution-swap-2026-06-28.json` | 3 + CI (most rigorous) |
| Atomic vs core × curated vs open-world | `python3 tools/run_source_contamination_bench.py --verifier atomic --relay --answer-spec llmhub:claude-sonnet-4-6 --judge-spec openrouter:deepseek/deepseek-chat` (then `--verifier core`; add `--retrieve` for open-world) | Cluster C matrix — curated_atomic **97.7%/5.9%**, curated_core **95.3%/0.0%**, openworld_atomic **97.7%/52.9%**, openworld_core **58.1%/0.0%** — `agi-proof/source-verifier/verifier-compare-2026-06-28.json` | 1 (candidate) |
| Hybrid | `python3 tools/run_source_contamination_bench.py --verifier hybrid --relay --answer-spec llmhub:claude-sonnet-4-6` | authoritative-only catch **0/43** on this pack; hybrid catch comes from the flagged LLM tail (**58.1% / 0.0%**) — `agi-proof/source-verifier/hybrid-verifier-2026-06-28.json` | 1 (candidate) |
| Multi-family (self-judged) | `python3 tools/run_source_contamination_bench.py --verifier atomic --relay --answer-spec llmhub:claude-sonnet-4-6` and again with `--answer-spec openrouter:deepseek/deepseek-chat` | both families **97.67% caught / 5.88% over-block** (answer==judge) — `agi-proof/source-verifier/live-multifamily-2026-06-28.json` | 1 (candidate) |
| Hardened (open-world retrieval) | `python3 tools/run_source_contamination_bench.py --verifier atomic --relay --retrieve --answer-spec llmhub:claude-sonnet-4-6 --judge-spec openrouter:deepseek/deepseek-chat` | open-world over-block ~**64.7%** — `agi-proof/source-verifier/live-hardened-2026-06-28.json` | 1 (candidate) |

**Correction to honor when reproducing:** `live-hardened-2026-06-28.json` reports a 70.6% *curated*
answer≠judge over-block. `verifier-compare-2026-06-28.json` **withdraws** that figure as a stale,
mis-saved retrieve report; the corrected curated answer≠judge over-block is **5.9%**, and the real
over-block driver is **open-world retrieval (52.9%)**, not answer≠judge separation. Reproduce the
corrected matrix, not the withdrawn number.

### B2. Debunk-gate bench (`tools/run_debunk_gate_bench.py`)

Detectors: `--detector {keyword,llm}`. Modes: `--fake` (no network), `--relay` (live).
The live relay path reads `OPENAI_API_KEY`; the core-claim/layered verification paths additionally
use `GOOGLE_FACTCHECK_API_KEY` for the high-independence layer.

| Run | Command | Expected headline (committed report) | Runs |
|---|---|---|---|
| LLM detector (detection vs verification) | `OPENAI_API_KEY=<relay-key> python3 tools/run_debunk_gate_bench.py --relay --detector llm` | detection **100%** (21/21), verified-debunk **0/21** — detection solved, verification is the new bottleneck — `agi-proof/debunk-gate/live-llm-detector-2026-06-28.json` | 1 (candidate) |
| Keyword detector (the broken baseline) | `OPENAI_API_KEY=<relay-key> python3 tools/run_debunk_gate_bench.py --relay --detector keyword` | per-family **debunk_recall 0.0** — models DO debunk but the keyword detector tags all as 'affirm' — `agi-proof/debunk-gate/live-multifamily-2026-06-28.json` | 1 (candidate) |
| Core-claim verification | `GOOGLE_FACTCHECK_API_KEY=... OPENAI_API_KEY=<relay-key> python3 tools/run_debunk_gate_bench.py --relay --detector llm` (core-claim path) | **18/21 verified** (4 high-independence Google + 14 flagged-LLM; 3 correct fail-closes) — `agi-proof/debunk-gate/core-claim-verification-2026-06-28.json` | 1 (candidate) |
| Layered verification | `GOOGLE_FACTCHECK_API_KEY=... OPENAI_API_KEY=<relay-key> python3 tools/run_debunk_gate_bench.py --relay --detector llm` (layered path) | **19/21 verified** (4 high + 15 flagged-LLM); provenance layer fires 0× on this pack but is 4/4 on a Dickens/Shakespeare/Tolstoy/Twain authorship demo — `agi-proof/debunk-gate/layered-verification-2026-06-28.json` | 1 (candidate) |

Google Fact Check covered only **4/21** of the debunk pack (viral myths it has professional
ClaimReviews for); the remaining verified claims rely on the flagged LOW-independence LLM tail.
That coverage bound is the point, not a bug — see the decontamination checklist.

### B3. Other bench tools (architecture present; secondary to the verifier results)

- `tools/run_meta_labeler_bench.py` — abstaining meta-labeler bench.
- `tools/run_world_model_baselines.py` — world-model FLOOR (retrieval) vs CONTENDER (LLM) baselines.
- `tools/run_lean_expert_iteration.py` — Lean expert-iteration loop (separate proof-search cluster;
  the Lean kernel + novelty bet is **not** closed; see `agi-proof/proof-search/`).

---

## What reproduction does and does not establish

Reproducing Tier A re-establishes the **fail-closed architecture and contracts**. Reproducing
Tier B re-establishes the **candidate** live numbers under *your* keys. Neither makes any AGI claim:
`canClaimAGI` stays **false**. A genuine third-party replication of the *gate as a whole* — with an
independently-authored contamination pack and reviewer-authored hidden tasks — is the standing
top-level gap recorded in the capstone. The honest takeaway the toolkit supports is composition:
layered independent verifiers, each labelled with its independence and coverage, that **abstain
rather than vouch** for what they cannot independently confirm.
