# Failure Ledger

Failures are claim evidence. They show where the system is not AGI.

| Failure ID | Status | Claim impact | Required response |
|---|---|---|---|
| external-benchmarks-not-run | Open | Blocks expert AGI claim | Keep wording at AGI-candidate |
| hidden-review-third-party-not-run | Open | Blocks independent hidden generalization claim | Run third-party packs |
| hidden-prepared-pack-grok-cli-2026-06-19 | Open | Preliminary hidden run only: 28.75/40 auto score, 2/8 strict pass | Improve strict pass rate; run fresh third-party hidden pack |
| hidden-fresh-pack-sophia-grok-2026-06-19 | Open | Full hidden-run artifact exists, but backend produced 0/8 nonempty answers; not valid evidence of reasoning competence | Fix Grok/session/network execution and run a new unspent hidden pack |
| hidden-fresh-pack-sophia-deepseek-2026-06-19 | Open | Diagnostic spent-pack run reached 27.5/40 auto score, 8/8 nonempty answers, and 0 backend failures, but 0/8 strict pass; not independent proof evidence | Complete manual semantic review, improve missed rubric/coding/tool-use behavior, then run a new unspent reviewer-controlled pack |
| hidden-fresh-pack-sophia-deepseek-coding-council-repair3-2026-06-20 | Open | Diagnostic spent-pack rerun improved to 31.9/40 auto score with 8/8 nonempty answers; strict pass remains 0/8 because manual semantic review is still pending and tool-use dropped to 50% | Complete two-pass manual review, strengthen tool-use log-grounding prompts, then run a new unspent reviewer-controlled pack |
| hidden-full-sophia-valid-run-not-yet-run | Open | Blocks claim that the full Sophia pipeline beats direct-model answering on hidden tasks | Run `tools/run_hidden_eval_sophia.py` on an unspent reviewer-controlled pack with working backend |
| hidden-manual-review-not-complete | Open | Blocks semantic-quality claims from automatic keyword/regex scoring alone | Complete manual judge review templates |
| baseline-ablation-missing | Closed | RAN (DeepSeek, 2026-06-22, 3 runs each; agi-proof/baseline-ablation/TRACK2-DEEPSEEK-2026-06-22.md). Keyword/regex scoring shows NO net method advantage on a capable base model: 5-case full−raw +1.33 [0.48,2.18] (beats bare model) but full−raw+tools ties; 17-case hard full−raw −1.11 [−2.23,0.01] (raw slightly beats). Superseded by the calibration finding below. | — |
| ablation-method-value-needs-calibration-scoring-2026-06-22 | Closed | VALIDATED (deterministic scorer): on the 18-case abstain pack, 3 runs, DeepSeek — sophia-full vs raw-model calibration Δ +22.0% [14.5%, 29.6%] and fabrication reduction +19.4% [14.0%, 24.9%], both CI exclude 0; sophia-full fabricates 0% in all 3 runs (raw 16.7–25%). vs raw+tools larger (Δ 28.3%). Recorded in RESULTS.md (Calibration evals). Residual caveat: deterministic scorer + self-authored pack → not multi-judge headline grade. | Third-party audit of the scorer labels/markers + human semantic review to harden to headline grade (new item below) |
| calibration-multijudge-corroborated-2026-06-22 | Closed | MULTI-JUDGE CORROBORATED: two INDEPENDENT judge families (openai:gpt-4o + openai:claude-sonnet-4-6, distinct from the deepseek subject) re-scored the 108 abstain answers across 3 runs (tools/run_calibration_judge.py). All three methods rank sophia-full LOWEST fabrication; inter-judge κ=0.74 (substantial), scorer-vs-judge κ=0.48/0.40 (both ≥0.40). Meets the multi-judge + inter-judge-agreement + ≥3-runs bar. Artifact: agi-proof/baseline-ablation/calibration-2family-judge-2026-06-22.json. | — |
| calibration-self-authored-pack-2026-06-22 | Open | Residual independence gap: the abstain pack + epistemic labels are self-authored. Multi-judge corroboration is done, but a fully independent claim needs a third-party-authored pack + human semantic review. | Commission a third-party abstain pack + a human review pass |
| self-extending-loop-closes-offline-2026-06-22 | Partial | The full loop CLOSES offline on a held-out domain (selfextend/loop.py): abstain→synthesize verifier→validate on held-out→verified-reward SELECTION→policy acc 0.5→1.0 on an independent eval split→competence flips abstain→answer; all 4 invariants pass; fail-closed on unlearnable data. agi-proof/self-extension/. | (a) run on a THIRD-PARTY domain, (b) replace selection with a live RL update (GPU), (c) clear the no-overclaim gate — then it becomes a capability claim |
| long-horizon-not-run | Open | Blocks autonomy claim. Effective-horizon CURVE measured (DeepSeek 16 steps, 8 trials, noisy) but that is the chained-arithmetic metric, NOT a long-horizon autonomy run | Publish timed long-horizon autonomy run logs |
| distribution-shift-not-run | Open | RAN the mechanism (DeepSeek, 2026-06-22): promotion gate 1/2, contamination clean, protected knowledge unchanged — but the 1-case demo pack shows 0% improvement (no signal). Mechanism sound, evidence insufficient | Build a real multi-case pre/post shift pack and re-run |
| rlvr-live-run-not-yet-gated-2026-06-21 | Open | Blocks RLVR capability claim (held-out pass@1 rise vs base) | Run a gated live GRPO run clearing `aggregate._is_validated` (≥2 judge families, κ≥0.40, ≥3 runs, CI excludes 0) on an entity-disjoint held-out split + manual semantic review; offline reward-wiring invariants pass in CI but are not capability evidence |
| local-agent-tools-degrade-strong-model-2026-06-21 | Closed | FIXED: selective invocation (tools fire only on low-confidence answers) + richer tool outputs (wiki_search snippets, belief wiki fallback) eliminated the degradation — on qwen3:30b-a3b `+mcp-tools` now *beats* alone (gold 90.2%→92.7%, false-positive 9.8%→7.3%), was 90.2%→51.2% before | — |
| local-agent-delta-strong-model-headroom-2026-06-21 | Superseded | Single-LEXICAL-judge run on dolphin-llama3:8b showed alone 15.2% → +gate 4.3%. This did NOT survive validation — see below. `+mcp-tools` 0.0% was re-generation, NOT tool-use (`toolsUsed: []`). | Superseded by `local-agent-delta-not-validated-2026-06-21` |
| local-agent-delta-not-validated-2026-06-21 | Closed | RESOLVED by the benchmark expansion (#6, 87→290 cases) + the unified harness (#1). The earlier N=46 run's CI straddled zero; on the expanded set a validated run (3 runs, 2 judge families = openrouter:deepseek + openrouter:meta-llama) gives the +gate lever halluc alone 36.1% → gated 23.6%, **Δ12.5%, 95% CI [+5.6%, +19.4%] EXCLUDES zero**, 0% FP-cost → `validated=True`. Recorded in RESULTS.md / published-results.json. | — |

| grounded-gate-not-yet-validated-2026-06-22 | Open | The retrieval-grounded gate (check_claim ground=True) is verified bug-fixed (no pen-name false positives; catches known-author misattributions for out-of-corpus works) but a 3-run/2-family N=24 run gave +gate Δ8.3%, 95% CI [0.000, +16.7%] — lower bound touches zero, so illustrative not validated (vs the prior non-grounded validated Δ12.5%). Sampling variance at small N. | Re-run grounded at larger N (>=40 cases / more runs) to push the CI off zero |

## Template

```text
Failure ID:
Date:
Task or benchmark:
Expected behavior:
Observed behavior:
Likely cause:
Fix or next experiment:
Claim impact:
```

## steering-live-run-not-yet-gated-2026-06-23

**Status:** OPEN. The Spec B activation-steering engine is built and its machinery
invariants pass offline (`python tools/run_steering.py --model mock --dry-run` →
`STEERING WIRING VERIFIED ✓`; `tests/test_steering.py` green in CI). The live SSA
claim — that Level-3 steering beats Spec A's Level-1 persona baseline,
behavior-corroborated and capability-preserving — is **pre-registered and OPEN**:
it requires a gated real run (Phi-3.5 on MPS + the Ollama-judged battery at
N≥8/K≥20). `SSA = 0/N` would be a legitimate honest result. Thresholds are fixed
in `agent/steering/stats.py:SSA_THRESHOLDS` before any run.
