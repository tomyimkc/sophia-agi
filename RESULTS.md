# Provenance Delta — public results

<!-- GENERATED from agi-proof/benchmark-results/published-results.json by
     tools/build_results_page.py — do not edit by hand. -->

_Last updated: 2026-06-27_

**No-overclaim gate.** A number is **VALIDATED** only with ≥2 independent judges in consensus (judge ≠ subject), reported inter-judge agreement, ≥3 runs, and confidence intervals. Everything else is **illustrative** and labelled. Hidden-eval prompts are never published — only aggregates. See [SECURITY.md](SECURITY.md) and [methodology](docs/11-Platform/Provenance-Delta.md).

## Validated results

| Model | Judges | Agreement | Runs | Halluc. alone | Halluc. gated | Δ (95% CI) | FP cost | Coverage |
|---|---|---|---|---|---|---|---|---|
| ollama:dolphin-llama3:8b | consensus: openrouter:deepseek/deepseek-chat + openrouter:meta-llama/llama-3.3-70b-instruct (2 families) | — | 3 | 36.1% | 23.6% | 12.5% [5.6%, 19.4%] | 0.0% | 34.6% |
| ollama:dolphin-llama3:8b | consensus: llmhub:gpt-4o (openai) + llmhub:claude-sonnet-4-6 (anthropic) (2 families) | 88.2% | 3 | 42.4% | 33.3% | 9.0% [4.2%, 14.6%] | 0.0% | 22.9% |

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

## Calibration evals (abstention vs fabrication, deterministic)

Scored by a **deterministic marker-based scorer** (no LLM judge) that rewards honest abstention on genuinely-unknown questions and scores a confident fabricated specific 0. Validated by **≥3 runs with a 95% CI excluding zero**. Honestly bounded: the scorer and pack are **self-authored** (internally valid cross-mode deltas; a third-party audit of the labels/markers — and human semantic review — would harden these to headline grade).

| Method | Baseline | Pack (runs) | Calibration Δ (95% CI) | Fabrication reduction (95% CI) | Method fab-rate | Date |
|---|---|---|---|---|---|---|
| sophia-full | raw-model (deepseek-chat) | abstain-calibration-2026-06-22 (18 cases: 12 abstain / 6 definite) (3) | 22.0% [14.5%, 29.6%] | 19.4% [14.0%, 24.9%] | 0.0% | 2026-06-22 |

- _sophia-full vs raw-model (deepseek-chat):_ DeepSeek subject. sophia-full fabricates 0% (deterministic scorer) on the unknown-author/quote cases in all 3 runs; raw-model 16.7-25%. Keyword/regex scoring is blind to this; the calibration scorer reveals it. vs raw-model-plus-tools the gap is larger (calibration Δ 28.3% [24.5%, 32.2%]). CORROBORATED by two INDEPENDENT judge families (gpt-4o + claude-sonnet, distinct from the deepseek subject): all three methods rank sophia-full lowest fabrication; inter-judge κ=0.74, scorer-vs-judge κ=0.48/0.40 (both ≥0.40). Meets the multi-judge bar. Residual caveat: the pack is self-authored — a third-party pack + human review remain for full independence. (A 3rd family, qwen3.7-max, was tested and found non-discriminating — 0% fabrication everywhere, κ=0 — so it is excluded as an uninformative judge; the 2-family OpenAI+Anthropic corroboration stands.)

## External-benchmark calibration (selective prediction) — **VALIDATED**

On a **public, human-authored, external** benchmark (SimpleQA / SimpleQA Verified (OpenAI + Google DeepMind) — public, human-authored, external), graded by consensus: llmhub:claude-sonnet-4-6 + llmhub:gemini-2.5-pro (2 independent families). The first Sophia calibration result validated on **non-self-authored** data — the selective-accuracy lift's 95% CI excludes zero on two independent subject models. A **calibration / selective-prediction** result, **not** an AGI claim.

| Subject | Dataset | N (attempted) | Signal | AUROC | Selective-acc lift @20% cov (95% CI) | Inter-grader κ | Date |
|---|---|---|---|---|---|---|---|
| deepseek-chat | SimpleQA Verified | 1000 (940) | self-consistency | 0.649 | +15.8% [9.8%, 22.1%] | 0.974 | 2026-06-26 |
| qwen-2.5-72b-instruct | SimpleQA (original) | 2000 (786) | self-consistency | 0.636 | +7.8% [2.3%, 13.5%] | 0.995 | 2026-06-26 |

- CROSS-MODEL VALIDATED on external public data. Self-consistency selective prediction lifts selective accuracy on BOTH independent subject families, each graded by 2 independent families (Cohen kappa 0.97/0.99) with the lift's 95% CI excluding zero. The effect is base-model-dependent: larger for the overconfident DeepSeek (+15.8pts; self-abstains 6%) than the cautious Qwen (+7.8pts; self-abstains 61%). Of three confidence signals only self-consistency works; stated confidence and token-logprob are non-significant on both. This is a calibration / selective-prediction result, NOT an AGI claim; canClaimAGI stays false. Detail: agi-proof/benchmark-results/real-model/simpleqa/.

## Semantic evals (model-judged, gated)

Judging whether a holding *supports* a proposition is a model call, so these are held to the no-overclaim gate (multi-judge + agreement + runs + CIs). A single judge is illustrative, never a headline.

- Benchmark: legal_holding_faithful (does a real authority's holding SUPPORT the cited proposition? — the Ayinde misstated-authority failure)
- Gate: validated = >=2 independent judges (>=2 provider families, no mock) + mean pairwise Cohen's kappa >= 0.40 + >=3 runs + bootstrap 95% CI lower bound above chance (0.5).
- Validated result: **VALIDATED** — consensus accuracy **100.0%** (CI [1.0, 1.0]), mean pairwise κ **1.0**, N=8, 3 runs, families deepseek, meta-llama, qwen (2026-06-21).
- Clears the pre-registered no-overclaim gate (3 independent provider families via OpenRouter, kappa=1.0, 3 runs, CI above chance). HONEST CAVEAT: N=8 and the cases are clear-cut (a clearly-supported vs a clearly-misstated proposition for each real authority), so unanimous agreement (kappa=1.0) and the degenerate CI [1.0,1.0] are expected — this validates the tier + harness end-to-end and shows three frontier-class judges reliably catch blatant misstatement, but it does NOT measure subtle ratio-vs-obiter discrimination or performance on a large, adversarial set. Reproduce: OPENROUTER_API_KEY set; python tools/run_legal_faithfulness_bench.py --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct,openrouter:qwen/qwen-2.5-72b-instruct --runs 3.

## Judge audit (why the gate matters)

Independent Claude panel vs the DeepSeek judge on 46 false cases: 76% agreement; validated alone-rate 21.7% (10/46) vs 41.3% single-judge. Judge choice dominates the absolute number.

Robust regardless of judge:
- 0% false-positive cost (the gate never broke a correct answer)
- the delta is positive and real
- the effect tracks a model's propensity-to-assert, not its size

## Continual / grounded-answering (CANDIDATE — not a headline)

Continual Provenance QA (CPQA): a frozen LLM answers either from the retrieved OKF/wiki source (`grounded`) or from parametric memory (`raw`), and a cross-provider judge panel scores both. Held to the no-overclaim gate and **candidate, not validated** — self-authored benchmark, keys held by one operator, no external replication.

- Benchmark: Continual Provenance QA (CPQA) — grounded vs raw answers over the 92-page wiki corpus
- Answers: gpt-4o-mini (OpenAI, via LLMHub) · Judges: openrouter:deepseek/deepseek-chat (DeepSeek), openrouter:meta-llama/llama-3.3-70b-instruct (Meta) (judges via OpenRouter; answers via LLMHub — independent gateways) · 3 runs · N=92
- Overall consensus pass: grounded **52.9%** [47.1%, 58.7%] vs raw **88.4%** [84.8%, 92.0%]
- By expectation — **abstain/attribution-traps: grounded 100.0% vs raw 0.0%**; recall: grounded 50.2% vs raw 93.5% (a strong raw model already knows well-known facts; grounding's win is fail-closed abstention on traps)
- Inter-judge κ 0.94 · percent-agreement 97.0%
- **Recall fix (hybrid (typed gate) + graph-neighborhood retrieval, attribution-safe fallback):** overall **68.5%** [63.0%, 73.9%], recall **67.8%** (up from strict 50.2%), traps 80.0%; policy {'abstain_no_source': 15, 'grounded_strict': 120, 'grounded_fallback': 141}
  - Steps 1+2+4 combined. Recall recovered 0.50 -> 0.68 (+35% rel), overall 0.53 -> 0.69. Behavioral trap-safety preserved: all 15 trap evaluations hit the hard-abstain policy (abstain_no_source=15) and 0 traps reached the parametric fallback (fallback fired only on the 141 grounded-but-thin facts). The trap SCORE 0.80 (vs strict 1.0) is judge measurement noise on the fixed abstention string, not gate leakage. raw still wins overall (0.91) but the gap narrowed from 0.38 to 0.22.
- **Corpus enrichment (strict grounded on Step-5-enriched corpus (pure grounding, no parametric fallback)):** overall **62.3%** [56.5%, 67.8%], recall **60.2%** (up from strict 50.2%), traps 100.0% — pure grounding, no fallback
  - Step 5 surfaces existing sourced provenance fields as answer-bearing prose (thin-source 58%->12%). Pure-grounded recall rose 0.502 -> 0.602 (+20% rel), overall 0.529 -> 0.623, with traps a clean 1.0 and ZERO parametric reliance. It did not reach the ~0.88 prose-ceiling because the summaries are terse provenance-derived sentences (who/when/what-domain), which judges credit on focused questions but not fully on open-ended ones. Closing the rest needs genuinely richer authored+sourced content (a maintainer task), not field reformatting.
- **3-family validation (deepseek/deepseek-chat (DeepSeek), meta-llama/llama-3.3-70b-instruct (Meta), qwen/qwen-2.5-72b-instruct (Qwen)):** grounded **67.8%** vs raw **89.5%**; traps grounded **93.3%** vs raw 0.0%; mean pairwise κ 0.88/0.81; policy {'abstain_no_source': 15, 'grounded_strict': 237, 'grounded_fallback': 24}
  - 3-FAMILY VALIDATION (DeepSeek+Meta+Qwen, full-92, 3 runs). The grounded-vs-raw finding is robust across three independent families: mean pairwise kappa 0.88 (grounded) / 0.81 (raw), per-judge spread tight. Key behavioral result: vs the thin-corpus hybrid (141 parametric fallbacks), enrichment shifts the hybrid to 237 strict / 24 fallback — SAME recall (~0.67) with ~6x LESS parametric reliance (more grounded, not just more accurate). Traps behaviorally 1.0 (all 15 took the hard-abstain policy; the 0.933 score is judge noise on the fixed abstention string). raw still wins overall (0.895) — the residual gap is corpus coverage (terse field-derived summaries), not method.
- ⚠ CANDIDATE, not validated (self-authored benchmark, keys held by one operator, no external replication). FULL 92-query cross-gateway run, 3 runs. Honest headline: on attribution-traps/retractions grounded scores 1.0 vs raw 0.0 (perfect fail-closed abstention), but on plain recall grounded collapses to 0.50 vs raw 0.93 because answers are constrained to the retrieved wiki page and many pages are thin provenance stubs that don't contain the answer. Net, the raw model wins OVERALL (0.88 vs 0.53) on this recall-heavy, thin-source corpus. This is the predicted coverage-vs-fabrication tradeoff: grounding buys trap-safety at a recall cost, not a blanket win. Inter-judge kappa is now healthy (0.94/0.67; the earlier degeneracy was a small-subset artifact). UPDATE: a Step 1+2 hybrid (graph-neighborhood + typed gate with attribution-safe fallback) recovers recall to 0.68 and overall to 0.685 while keeping all traps on the hard-abstain path (see continualGroundedEvals.hybrid).

## Systems benchmarks (performance — candidate, host-dependent)

Throughput/latency micro-benchmarks for the systems components. These are **candidate** engineering numbers (single host, vary by machine), not no-overclaim accuracy results — reproduce with the command in each note.

### kvcache (Phase 1) (`storage/kvcache`)

Sharded async in-memory KV cache (Rust/Tokio). kvcache-bench measures the full loopback TCP round trip (client→server→client), not the bare in-memory map. Numbers vary by host.

_Representative run — 32 clients × 30,000 GETs, 100,000 keys, 256-byte values, 16 shards, 100% read, no evictions (all hits):_

| pipeline depth | throughput | latency p50 | latency p99 |
|---|---|---|---|
| 1 (no pipelining) | ~186k ops/sec | ~168 µs (per-op) | ~300 µs (per-op) |
| 16 | ~1.60M ops/sec | ~306 µs (per-batch) | ~546 µs (per-batch) |
| 64 | ~2.13M ops/sec | ~920 µs (per-batch) | ~1602 µs (per-batch) |

- Phase 1b request pipelining packs a batch into one flush and coalesces responses — the textbook throughput-for-latency trade (8.6× throughput at depth 16 for higher per-batch latency; depth-1 is the honest single-request baseline). Single node, in-memory only; persistence (io_uring) + replication (Raft) are Phases 2–3. Reproduce: `cd storage && cargo run --release --bin kvcache-bench -- --clients 32 --ops 30000 --pipeline 16`

## Reproduce

```bash
python tools/run_provenance_delta.py --models mock            # offline plumbing
python tools/run_provenance_delta.py --models <subject> \
    --judges <judgeA>,<judgeB> --runs 3                       # validated-grade run
```

Offline tests run in CI on every commit. Real-model numbers are produced locally by the maintainer and curated into `agi-proof/benchmark-results/published-results.json`.
