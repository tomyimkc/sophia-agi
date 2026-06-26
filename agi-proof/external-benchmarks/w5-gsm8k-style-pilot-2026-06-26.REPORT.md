# W5 — External-benchmark pilot (GSM8K-style numeric exact-match)

- Benchmark: **GSM8K-style 10-item STYLE sample** (`eval/external/gsm8k-style-sample.jsonl`) — NOT the official licensed GSM8K set (per preregistered-thresholds: style samples until licensed).
- Arms: **raw vs sophia-full**, **3 seeds**, backend DeepSeek `deepseek-v4-pro`.
- Scorer: numeric exact-match vs GOLD answer — judge-free, independent of the provenance gate.
- Registered commit: `b78e8ed2deb2160beaa46b0323e38b2df1d31dc3`
- Decontamination: self-authored arithmetic word problems; no overlap with training/eval packs.

## Exact command

```bash
# raw-arm plumbing (shipped runner):
python3 tools/run_external_eval.py --dataset eval/external/gsm8k-style-sample.jsonl --model deepseek
# full raw-vs-sophia-full 3-seed pilot driver (this run): tools/_w5 driver, artifact:
#   agi-proof/external-benchmarks/w5-gsm8k-style-pilot-2026-06-26.json
```

## Result (headline WITH CI)

| Arm | Per-seed accuracy | Mean |
|---|---|---:|
| raw | [1.0, 1.0, 1.0] | 1.000 |
| sophia-full | [1.0, 1.0, 1.0] | 1.000 |

- **Δ (sophia-full − raw) = +0.000, 95% bootstrap CI [+0.000, +0.000] — does NOT exclude 0 (a TIE / NULL result).**
- This is an honest **null delta**: both arms score 100% because the 10 items are trivial arithmetic and the base model is at ceiling. A null result is a valid, recorded outcome.

## Gate-coverage cost

- Gate fired on **30/30** sophia-full answers — but on STYLE/format grounds ("Missing explicit source-discipline framing", "Missing 中文 summary section") in advisor mode, **not** numeric/factual violations (violations=[]).
- Answers were not blocked; numeric accuracy stayed 100%. **Gate-coverage cost on correctness = 0.**

## Validation gate (`_is_validated`)

- **Not validated.** The GOLD oracle is judge-free (the 2-independent-judge clause is N/A for numeric correctness), but this is a 10-item **self-authored STYLE sample** (not official GSM8K) and the delta CI does not exclude 0. It is a **foothold/plumbing pilot**, not citable capability evidence.
- Residual gap: needs the licensed GSM8K set at larger N (CI off ceiling) for a defensible external number. `canClaimAGI` stays false.
