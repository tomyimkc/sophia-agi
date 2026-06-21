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
