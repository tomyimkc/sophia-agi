# Provenance Delta — public results

<!-- GENERATED from agi-proof/benchmark-results/published-results.json by
     tools/build_results_page.py — do not edit by hand. -->

_Last updated: 2026-06-21_

**No-overclaim gate.** A number is **VALIDATED** only with ≥2 independent judges in consensus (judge ≠ subject), reported inter-judge agreement, ≥3 runs, and confidence intervals. Everything else is **illustrative** and labelled. Hidden-eval prompts are never published — only aggregates. See [SECURITY.md](SECURITY.md) and [methodology](docs/11-Platform/Provenance-Delta.md).

## Validated results

_None yet._ No run has cleared the gate (multi-judge consensus + agreement + ≥3 runs + CIs). This is intentional and honest: see the illustrative section and the audit below for why a single judge is not enough.

## Illustrative only (not headline-grade)

### `ollama:dolphin-llama3:8b` — uncensored 8B local tune (the population the gate is for)

- Judge: **single LLM-judge: deepseek** · runs: 3 · false cases: 46
- Hallucination alone **42.8%** → gated **33.3%** · Δ **9.4% [4.3%, 15.2%]**
- False-positive cost 0.0% · gate coverage 23.7%
- ⚠ Single judge. An independent Claude audit panel re-judged all 46 cases and found this judge OVER-counted (76% agreement, 10 false positives), so the validated alone-rate is ~21.7%, not 42.8%. This row is illustrative ONLY and must not be quoted as a headline.

## External-oracle evals (base-model accuracy via the harness)

Scored by **exact-match against external gold** (no LLM judge). These report the **base model's** accuracy through Sophia's external-eval harness and validate the harness end-to-end — they are **not** claims about Sophia's provenance gate or any Sophia-specific capability.

| Dataset | Model | N | Accuracy | Date |
|---|---|---|---|---|
| GSM8K test (openai/grade-school-math, MIT) | `deepseek-chat` | 100 | 98.0% | 2026-06-21 |

- _GSM8K test (openai/grade-school-math, MIT):_ Objective exact-match against external gold. This reports the BASE MODEL's accuracy via Sophia's external-eval harness and validates the harness end-to-end; it is NOT a claim about Sophia's provenance gate or any Sophia-specific capability. N=100 of the 1319-item test split.

## Verifier evals (objective accuracy of a Sophia verifier)

Scored by **exact-match against ground-truth labels** with a **deterministic verifier** (no LLM judge). Unlike the provenance-delta rows, these measure a **machine-checked gate's** accuracy directly, so they need no multi-judge consensus — but they are honestly bounded by small, constructed benchmarks and are **not** headline capability claims.

| Verifier | Benchmark | N | Accuracy | Fabrication recall | False-alarm | Date |
|---|---|---|---|---|---|---|
| `legal_citation_exists` | legal_citations (real-vs-fabricated HK/UK/US common-law citations) | 14 | 100.0% | 100.0% | 0.0% | 2026-06-21 |

- _legal_citation_exists:_ Measures the VERIFIER's accuracy at catching fabricated legal citations (the Mata v. Avianca failure mode) across federated sources (HK e-Legislation/HKLII, UK National Archives, US CourtListener), not any model's. Includes the actual Mata fabrication, Varghese v. China Southern Airlines 925 F.3d 1339, which is flagged. Honest bounds: tiny, constructed benchmark (N=14) and the result is capped by the bundled register's completeness, so this validates the extraction + fail-closed gate logic end-to-end — it is NOT a headline capability claim. Reproduce: python tools/run_legal_citation_bench.py.

## Judge audit (why the gate matters)

Independent Claude panel vs the DeepSeek judge on 46 false cases: 76% agreement; validated alone-rate 21.7% (10/46) vs 41.3% single-judge. Judge choice dominates the absolute number.

Robust regardless of judge:
- 0% false-positive cost (the gate never broke a correct answer)
- the delta is positive and real
- the effect tracks a model's propensity-to-assert, not its size

## Reproduce

```bash
python tools/run_provenance_delta.py --models mock            # offline plumbing
python tools/run_provenance_delta.py --models <subject> \
    --judges <judgeA>,<judgeB> --runs 3                       # validated-grade run
```

Offline tests run in CI on every commit. Real-model numbers are produced locally by the maintainer and curated into `agi-proof/benchmark-results/published-results.json`.
