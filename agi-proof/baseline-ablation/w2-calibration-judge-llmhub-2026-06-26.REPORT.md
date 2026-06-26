# W2 calibration — independent multi-judge re-judge (2026-06-26) — NEGATIVE

Re-judged the W2 (2026-06-26) abstain ablation dumps with two judge families
distinct from the DeepSeek subject — `llmhub:claude-opus-4-8` and
`llmhub:gpt-4o-mini` — to test whether the deterministic calibration scorer's
fabrication labels survive independent corroboration. **They do not.** This is
an honest negative; it does NOT clear `_is_validated`. `canClaimAGI` stays false.

## Result (108 paired abstain answers, 3 seeds)

Per-mode fabrication rate:

| Method | sophia-full | raw-model | raw+tools |
|---|---|---|---|
| deterministic scorer | 0.000 | 0.111 | 0.167 |
| claude-opus-4-8 | 0.083 | 0.139 | 0.139 |
| gpt-4o-mini | 0.056 | 0.139 | 0.111 |

Inter-method Cohen's κ:

| pair | κ |
|---|---|
| scorer vs claude-opus-4-8 | **−0.020** |
| scorer vs gpt-4o-mini | **−0.107** |
| scorer vs consensus | **−0.096** |
| claude vs gpt-4o-mini | 0.719 |

## Honest reading

- **The deterministic scorer is NOT corroborated.** scorer-vs-judge κ is ~0 /
  slightly negative — far below the **≥0.40** bar the repo requires. So the W2
  (2026-06-26) calibration result **does not clear `_is_validated`**. The two
  independent judge families agree with *each other* (κ=0.719) but not with the
  scorer, i.e. the scorer's per-answer "fabricated" labels diverge from
  independent semantic judgment on this pack.
- **Aggregate direction survives, per-answer labels do not.** Both judges still
  rank sophia-full BELOW raw on fabrication (claude 0.083<0.139, gpt
  0.056<0.139), so the *direction* of the W2 effect is preserved — but corroboration
  requires per-answer agreement, and that is absent.
- **Contrast with the validated 2026-06-22 result.** The June-22 run cleared the
  bar (scorer-vs-judge κ 0.48/0.40 with gpt-4o + claude-sonnet via direct keys).
  This June-26 re-judge does not. This REINFORCES the earlier ledger note: the
  June-26 W2 run is weaker, and the 2026-06-22 multi-judge result remains the
  only headline-grade calibration evidence.
- **Caveats.** (1) Both judges share the llmhub gateway, so their mutual κ=0.719
  is not independent infrastructure. (2) Fabrication is a rare event here
  (sophia-full ~0–8%), and Cohen's κ is fragile on rare positives with small N —
  a near-zero κ partly reflects that. Neither caveat rescues the result: the bar
  is unmet either way.

## Conclusion

Independent judges do **not** validate the W2 (2026-06-26) deterministic
calibration scorer. The W2 ledger status stays **Partial / NOT `_is_validated`**;
the 2026-06-22 direct-key multi-judge result remains the high-water mark. Honest
negative recorded per the repo's failure-as-evidence discipline.

Reproduce: `LLMHUB_API_KEY=… python tools/run_calibration_judge.py
agi-proof/baseline-ablation/abstain-pack-2026-06-22.json
agi-proof/baseline-ablation/w2-ablation-2026-06-26.seed{0,1,2}.private.json
--judge llmhub:claude-opus-4-8 --judge llmhub:gpt-4o-mini --json`.
