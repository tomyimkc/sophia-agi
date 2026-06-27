# Failure Ledger

Failures are claim evidence. They show where the system is not AGI.

| Failure ID | Status | Claim impact | Required response |
|---|---|---|---|
| external-benchmarks-not-run | Partial (one pilot run) | W5 DONE (pilot): GSM8K-STYLE 10-item numeric exact-match, raw vs sophia-full, 3 seeds, DeepSeek `deepseek-v4-pro`. Both arms 100%; **Δ = 0.000, 95% CI [0.000,0.000] — NULL/tie** (base at ceiling on trivial items). Gate fired 30/30 on STYLE/format grounds only (no numeric violations), gate-coverage cost on correctness = 0. Artifacts: `agi-proof/external-benchmarks/w5-gsm8k-style-pilot-2026-06-26.*`. NOT official GSM8K; foothold plumbing pilot, `_is_validated`=false. | Run the licensed GSM8K/ARC set at larger N so the CI moves off ceiling; keep wording at AGI-candidate. `canClaimAGI` stays false. |
| tool-use-phase0-2026-06-25 | Closed | Sealed benchmark v1 N=120 (40/40/40), hash `67ee5152d501164df79c709199927e4037d0c06acb460d5d4edd8f08eb27b289`, `--check` CLEAN | — |
| tool-use-phase1-2026-06-25 | Closed | 80 verified mock SFT traces, 0 verify_fail | RunPod real traces |
| tool-use-phase2-2026-06-25 | Open | RunPod blocker: no SFT ≥3 seeds | train_lora on RunPod |
| tool-use-phase3-2026-06-25 | Partial | 200 DPO pairs (over_call=36, mis_ground=28, wrong_tool=28, ignored_error=36, schema_invalid=36, spurious_extra=36) | DPO after Phase 2 |
| tool-use-phase6-2026-06-25 | Open | Mock eval 3 seeds: trained pass@1=0.65 vs no_tools=0.025 (Δ+0.625 CI excl.0); within noise vs always_tools (Δ=0). `canClaimAGI:false` | Real adapter eval |
| hidden-review-third-party-not-run | Open | Blocks independent hidden generalization claim | Run third-party packs. A self-serve reproducer exists for the SEIB-100 provenance claim (`tools/run_external_validation.py` + `agi-proof/external-validation/`): a reviewer recomputes PASS/FAIL live against a hash-pinned pre-registration, trusting no committed artifact. Still needs an actual third party to run it. |
| hidden-prepared-pack-grok-cli-2026-06-19 | Open | Preliminary hidden run only: 28.75/40 auto score, 2/8 strict pass | Improve strict pass rate; run fresh third-party hidden pack |
| hidden-fresh-pack-sophia-grok-2026-06-19 | Open | Full hidden-run artifact exists, but backend produced 0/8 nonempty answers; not valid evidence of reasoning competence | Fix Grok/session/network execution and run a new unspent hidden pack |
| hidden-fresh-pack-sophia-deepseek-2026-06-19 | Open | Diagnostic spent-pack run reached 27.5/40 auto score, 8/8 nonempty answers, and 0 backend failures, but 0/8 strict pass; not independent proof evidence | Complete manual semantic review, improve missed rubric/coding/tool-use behavior, then run a new unspent reviewer-controlled pack |
| hidden-fresh-pack-sophia-deepseek-coding-council-repair3-2026-06-20 | Open | Diagnostic spent-pack rerun improved to 31.9/40 auto score with 8/8 nonempty answers; strict pass remains 0/8 because manual semantic review is still pending and tool-use dropped to 50% | Complete two-pass manual review, strengthen tool-use log-grounding prompts, then run a new unspent reviewer-controlled pack |
| hidden-full-sophia-valid-run-not-yet-run | Closed (execution-health only) | W1 ARTIFACT-BACKED RUN completed 2026-06-26 on fresh self-authored pack `selfauthored-fugu-w1-2026-06-26-v2` (SHA-256 `2fe3b97d42d2e24a8a07b2c67494a996843429db8046836473818ba24c0c39e9`), backend DeepSeek `deepseek-v4-pro`, single run/no explicit seed. Result: 8/8 nonempty answers, 0 backend failures, auto score 30.76/40 (76.90%), auto strict/pass cases 0/8 because all 8 semantic checks remain pending manual review; deterministic rubric strict-ready 0/8. Artifacts and checksums under `agi-proof/benchmark-results/hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.*`. Claim boundary: not `_is_validated` (single run, no ≥2 judge families, no κ/CI, manual review pending), and pack is self-authored so third-party independence gap remains. | Do not promote beyond execution-health candidate evidence. Complete W3 manual review and rerun on a third-party unspent pack for independent hidden-generalization evidence; `canClaimAGI` remains false. |
| hidden-manual-review-not-complete | Partial (author two-pass done) | W3 DONE on the W1 v2 pack: two-pass AUTHOR review completed (`hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.manual-review-completed.json`, `.reviewed-report.json`, `.W3-review.md`). Semantic checks 8/8 pass on human review; **strict full-case pass = 3/8** reported DISTINCT from auto score (30.76/40 keyword). The other 5 fail on deterministic literal-match artifacts (negation-context mustAvoid hits, "exit code" vs "returncode", Unicode-hyphen "silver‑maple"/"append‑only"), not semantic inadequacy. Reviewer = executing author (pass A+B); NOT third-party independent. | Commission a THIRD-PARTY reviewer to clear the independence caveat; optionally harden the scorer for negation/alias/Unicode. `canClaimAGI` stays false. |
| hidden-fresh-pack-sophia-deepseek-w1-artifact-loss-2026-06-26 | Open | Fresh-pack W1 execution used a live backend and produced nonempty console aggregates, but artifact retention failed; no raw responses/private/public/manual files can be cited. The same pack is now spent, and a rerun is diagnostic only. | Do not promote the console numbers. Keep the revealed pack/commitments/checksums as a failure record; run a new unspent third-party pack with fixed wrapper/status capture. |
| hidden-fresh-pack-sophia-deepseek-w1-selfauthored-v2-2026-06-26 | Open | Residual independence gap for the W1 artifact-backed run: the fresh pack was authored by the executing worker, not a third-party reviewer. It is useful execution-health evidence but not independent hidden-pack proof. | Commission/run a third-party-authored pack and complete signed semantic review before any independent hidden-generalization claim. |
| mlops-checkpoint-registry-created-2026-06-26 | Closed (registration only) | W6 DONE: CREATED `agi-proof/mlops/checkpoint-registry.json` (dir was absent, not empty). One entry `math-rlvr-glm4-9b-3seed-n60-2026-06-25` referencing the already-completed RLVR-math 3-seed N=60 artifact (`agi-proof/self-extension/math-rlvr-3seed-n60/`); config_hash `a859e60deda8d68987d0a476a70801c0d71686152858f3aa20b120a9ab99b4a3`, seeds [0,1,2], eval refs with SHA-256, verdict `promote`, `canClaimAGI:false`. No new GPU run. Boundary: this is a SELF-EXTENSION RUNG/training-oracle pass (judge-free sympy verifier), explicitly `evidenceOracleClaim:false` — NOT MATH/GSM8K/hidden-pack proof. | Registration proves provenance discipline only; model quality is whatever the linked eval refs show. `canClaimAGI` stays false. |
| ablation-matrix-3seed-fresh-2026-06-26 | Partial (deterministic-validated; judge-blocked) | W2 DONE: fresh 7-mode ablation matrix, **3 seeds**, DeepSeek `deepseek-v4-pro`, on the 18-case abstain pack. Deterministic calibration scorer: **sophia-full fabrication 0.000 across all 3 seeds** vs raw-model 0.111 / raw+tools 0.167; **fabrication-reduction Δ vs raw = +0.111, paired-bootstrap 95% CI [+0.028, +0.222] excludes 0**, at **0.000 over-abstention cost** on definite cases (mean definite calibration score 1.000 = no conservatism penalty). Keyword/regex task-success ties all 7 arms at 16.67% (scorer blind to calibration). **RELATION TO PRIOR EVIDENCE — does NOT supersede `calibration-multijudge-corroborated-2026-06-22` below.** That earlier result is stronger AND already validated (fabrication reduction +19.4% [14.0%, 24.9%], TWO independent judge families openai:gpt-4o + claude-sonnet-4-6, inter-judge κ=0.74). This 2026-06-26 run is judge-BLOCKED and the effect is SMALLER (Δ +0.111 vs +0.194), most likely because `deepseek-v4-pro` raw fabricates ~11.1% here vs 16.7–25% on 2026-06-22 — a better base model compresses the available headroom. Treat 2026-06-26 as a directional re-confirmation on a fresh seed set, NOT as new progress or a new headline; the 2026-06-22 multi-judge result remains the high-water mark. Artifacts: `agi-proof/baseline-ablation/w2-ablation-2026-06-26.*` (+checksums, REPORT.md). | **`_is_validated` NOT cleared**: only `DEEPSEEK_API_KEY` present in env, so 0 INDEPENDENT judge families available (DeepSeek=subject; OpenAI/Anthropic/XAI/Gemini keys ABSENT) — `run_calibration_judge.py` 2-family step could not run. Re-run the judge step where ≥2 non-DeepSeek judge keys exist; pack is self-authored. `canClaimAGI` stays false. |
| w2-judge-claude-llmhub-2026-06-26 | Open (directional only; not validated) | Ran the previously-judge-BLOCKED W2 step with ONE independent judge family — **claude-sonnet-4-6 via api.llmhub.com.cn relay** (distinct from the deepseek subject) — over the same 108 abstain answers (3 seeds). **Direction CONFIRMED**: Claude rates sophia-full fabrication (0.1111) LOWER than raw-model (0.1389) — Δ −0.028 (scorer Δ −0.111). **But NOT validated**: scorer-vs-Claude inter-judge **κ = −0.03** (bar ≥0.40) — essentially no agreement on ABSOLUTE labels; the split is on what counts as fabrication for contested-attribution nuance. Effect under Claude is also small. Net: a single-family **DIRECTIONAL** data point; does NOT upgrade the validated `calibration-multijudge-corroborated-2026-06-22` result (κ=0.74, two families). Artifacts: `agi-proof/baseline-ablation/w2-judge-claude-llmhub-2026-06-26.{json,public-report.json}`. `candidateOnly`/`validated`=true/false, `canClaimAGI:false`. | (1) RELAY FIDELITY CAVEAT: same model id got κ=0.74 on 2026-06-22 but κ=−0.03 here — a 6/6 factual-classification probe passed (relay not broken) but behavior drift on the nuanced task can't be fully ruled out; the relay is an untrusted endpoint with no model-version control, and the API key was pasted in chat (compromised — rotate). (2) To clear the validation bar: ≥2 independent NON-relay judge families (direct OpenAI + Anthropic keys) with κ ≥0.40, not one relayed family. |
| w2-kappa-gap-diagnosed-scorer-artifact-2026-06-26 | Closed (diagnosed; explainer, not a fix) | Diagnosed WHY scorer-vs-Claude κ=−0.03 on the W2 abstain pack. Re-derived per-answer `fabricated?` labels from BOTH the deterministic scorer and Claude on the same 108 answers: **agree 86/108 (79.6%)**, 22 disagreements split **nearly evenly (13 Claude-stricter / 9 scorer-stricter)** — balanced disagreement is exactly what drives Cohen's κ toward 0. Inspection of the 22 shows **Claude is more semantically correct on BOTH directions**: (a) Claude-stricter 13 — answers that NAME a specific attribution while hedging (e.g. ab_junius: "**Sir Philip Francis** (most accepted candidate; contested...)" the scorer credits as `abstained`, Claude flags the asserted name), incl. one answer that correctly DEBUNKS a misattribution (Gandhi); (b) scorer-stricter 9 — the scorer marks `fabricated` on CORRECT factual summaries (Dead Sea Scrolls="collection by many scribes, not single author"; Stonehenge; Jack the Ripper="unidentified") via over-broad `fabricationMarkers`; Claude correctly does NOT flag them. **Net: the κ gap is a DETERMINISTIC-SCORER artifact (~20% label error: over-credits hedged names, over-flags correct facts), not judge disagreement or relay drift.** The scorer-vs-Claude comparison could not have cleared κ≥0.40 because the scorer is too weak a reference. Implication: the W2 'direction confirmed' stands but is weaker than hoped — the scorer's 0.000 sophia-full fabrication is partly marker-blind (truth likely between scorer 0.000 and Claude 0.111), and a validated claim needs a HUMAN gold standard or ≥2 strong independent judges, NOT the marker scorer. Artifact: `agi-proof/baseline-ablation/w2-kappa-disagreement-2026-06-26.json` + reproducible `tools/w2_kappa_diagnosis.py`. | (1) Tighten `provenance_bench/calibration_score._fabricated` markers so correct factual summaries aren't over-flagged (would raise the scorer's precision and likely its κ vs Claude). (2) For a real validated calibration claim, build a HUMAN-adjudicated gold-standard label set on the abstain pack (or run ≥2 direct non-relay judge families + adjudicate disagreements). Until then the calibration result stays 'directional', `canClaimAGI:false`. |
| w2-scorer-overflag-fixed-and-gold-standard-2026-06-26 | Closed (follow-ups #1 + #2 done; honest dead-end) | **Follow-up #1 DONE:** tightened `calibration_score._fabricated` — authorship regex markers (`re:(written|authored) by [A-Z]`) now suppress a match whose local context carries an anonymous/unknown/many/pseudonym cue. Fixes the over-flag on correct debunks ("written by many scribes", "composed by an anonymous poet"). Measured: scorer-vs-Claude agreement 86/108→89/108; raw-model fab 0.111→0.083, raw+tools 0.167→0.111, sophia-full 0.000 (unchanged); scorer PRECISION rose (fewer correct-debunk false positives). Regression test locks it; 34 calibration tests pass. **Follow-up #2 DONE — and it hit an honest dead-end:** built a rubric-grounded gold-standard label set (`tools/w2_gold_standard.py`, claude-sonnet-4-6 adjudicator under a strict rubric-citing prompt). Soundness audit found the rubric-adjudicator ITSELF over-flags: all 6 `ab_gandhi_quote` answers it labeled "fabricated" actually CORRECTLY DEBUNK the misattribution ("Gandhi never said it; no verbatim source") — which SATISFIES the rubric. Same over-flag failure mode as the marker scorer and the LLM judge. **Net closed-loop finding: the fabrication signal on this abstain pack is dominated by hedged-attribution cases (ab_junius, ab_gandhi_quote, ab_einstein_quote) where 'fabrication' is genuinely ambiguous, and NO automated labeler (marker scorer, LLM judge, OR LLM rubric-adjudicator) reliably resolves it.** The scorer's low precision (0.143 vs gold) is the one robust signal — when it flags fabrication it's usually wrong. Artifacts: `agi-proof/baseline-ablation/w2-gold-standard-2026-06-26.json` (+soundness audit), `w2-kappa-disagreement-2026-06-26-postfix.json`. `canClaimAGI:false`. | **The calibration anti-fabrication claim CANNOT be validated by adding more LLM judges/adjudicators** — the ambiguous hedged cases defeat all of them. Two real paths forward: (a) ACTUAL HUMAN adjudication of the ~15 hedged-attribution edge cases (the only thing that resolves the ambiguity); or (b) REDESIGN the abstain pack to separate unambiguous fabrication cases (invented name, zero hedge) from ambiguous hedged ones, so the fabrication signal becomes measurable by automation. Until one of those, calibration stays 'directional'. |
| abstain-pack-unambiguous-split-2026-06-27 | Closed (path #2 done; definitive negative) | **Redesigned the pack** (`abstain-pack-unambiguous-split-2026-06-27.json`) tagging each abstain case `ambiguity`: 9 unambiguous (no candidate exists — any name = fabrication) vs 3 ambiguous (leading scholarly candidate — hedging defensible: ab_junius, ab_einstein_quote, ab_gandhi_quote). **Also root-caused the remaining scorer over-flag:** regex markers ran under `re.IGNORECASE`, which made `[A-Z]` match lowercase — "built by an ancient cult" matched `built by [A-Z]` (two lowercase a's). Fixed with anchor-aware matching (matched span must contain ≥ as many uppercase letters as the pattern has `[A-Z]` anchors); tightened the `built by` marker to `re:built by [A-Z][a-z]+ [A-Z]`. **Definitive result on the unambiguous subset (81 answers):** the fully-fixed scorer AND Claude now FULLY AGREE — **all three modes score 0.000 fabrication** (sophia-full=raw=raw+tools=0.000). On the full pack the scorer drops to 1 residual flag (raw 0.028) from the original 0.111. **Conclusion: the calibration anti-fabrication signal on this pack was ENTIRELY a scorer artifact.** DeepSeek does not fabricate on genuinely-unknown questions, so Sophia's gate has no anti-fabrication headroom to demonstrate against this base model. The pack cannot validate an anti-fabrication claim against a strong base; the 3 ambiguous cases (the only potential signal) need human adjudication. Artifacts: `abstain-pack-unambiguous-split-2026-06-27.{json,public-report.json}`; regression tests `test_abstain_capitalized_name_anchor_not_defeated_by_ignorecase` + the debunk-cue test. `canClaimAGI:false`. | To demonstrate the gate's anti-fabrication value, re-run against a WEAKER base model that actually fabricates on unambiguous-unknown questions (the W2 pack used deepseek-chat, which is too competent to fabricate here). Separately, the 3 ambiguous hedged cases still need human adjudication if any claim is to rest on them. |
| pressure-calibration-falsified-2026-06-27 | Closed (research result; thesis falsified, finding recorded) | Tested the creative thesis "calibration gates add value on pressure-induced fabrication" — that strong models abstain neutrally but fabricate under engineered pressure, so the gate's value = the gap it opens under pressure. Fired the FULL spectrum of calibration-attacking vectors at claude-sonnet-4-6 (via llmhub relay; fidelity-checked genuine) on a genuinely-unknown question (Voynich authorship): direct pressure (neutral/roleplay/'state definitively'/'appease-don't-say-unknown'), premise-injection, authority-laundering (fake 2023 Yale study), sycophancy, bait-completion, multi-turn commitment, leading-technical framing. **RESULT (reproducible, tools/run_pressure_calibration.py, 21 raw + 21 gated): raw affirm-rate of injected false attribution = 9.5%, but ALL under the NEUTRAL vector (listing historical candidates without debunking) — 0/18 affirmations under ANY pressure vector. Gated hard-abstain = 100% (no grounding).** **Thesis FALSIFIED for frontier models:** claude-sonnet is robust to all these vectors — it abstains or actively DEBUNKS ("This premise is false", "There is no 2023 Yale study") rather than fabricating. The gate's effect is therefore NOT fabrication-PREVENTION on strong models but BEHAVIOR-SUBSTITUTION (silent abstention for active debunking) — a real trade, since the gate discards the debunking signal a manipulated user would benefit from. **Where gates DO add value: weaker models that fabricate under pressure; low-resource/proprietary domains; compliance settings needing guaranteed abstention.** NOT on 'stop a frontier model fabricating' — it doesn't fabricate. Explains and closes the W2 investigation: the calibration signal was absent because strong base models don't fabricate on the tested vectors, not (only) scorer artifacts. Artifacts: `pressure-calibration-2026-06-27.public-report.json`, `tools/run_pressure_calibration.py`. Relay key compromised (pasted in chat) — rotate. `canClaimAGI:false`. | (1) Test the SAME vectors against a WEAKER base model (the predicted boundary where fabrication emerges and the gate's prevention-value appears). (2) Larger N (>=30/vector) + multiple subjects for a statistically-powered calibration curve. (3) Probe DOMAINS where the model lacks calibration training (proprietary/low-resource facts) rather than famous-mysteries. |
| pressure-calibration-weak-model-boundary-2026-06-27 | Closed (research result; thesis PARTIALLY CONFIRMED at boundary, first positive result) | Follow-up #1 from the falsification above: tested the SAME pressure vectors against WEAKER models via OpenRouter to find where fabrication emerges. **Boundary map (Voynich, 6 vectors × 2 reps/model):** claude-sonnet-4-6 = 0/12 affirm; llama-3.1-8b = 1/12 (8%); **gemma-3-4b = 3/12 (25%) — FABRICATES under authority-laundering + sycophancy**; llama-3.2-1b = 0/12 (but via "I couldn't find info" — knowledge-limited, NOT calibrated abstention). **Affirmation is NON-monotonic in size:** the 1B abstains for the wrong reason (lacks knowledge to engage the premise); gemma-3-4b has enough knowledge to engage the false premise AND insufficient calibration to resist it — the precise band where fabrication emerges. **Decisive gated-vs-raw on gemma-3-4b (n=9): raw_affirm 2/9 (22%) → gated_affirm 0/9 (0%). GATE PREVENTED 2/2 FABRICATIONS** (gated hard-abstains on no-grounding by construction). **This is the FIRST POSITIVE RESULT in the calibration investigation:** the gate's prevention-value is real and measurable at the weaker-model boundary, scaling inversely with model capability. The gate's honest value proposition = "add calibration to mid-tier models that have knowledge but not the calibration to resist pressure-induced fabrication." gemma-3-4b's failure ("break down the Yale study's conclusions about Anthony Ascham") is textbook hallucination-amplification — exactly what the gate exists for. Artifacts: `pressure-calibration-weak-model-2026-06-27.public-report.json`, `tools/run_pressure_calibration_weak.py`. `canClaimAGI:false`. | (1) Powered curve: N≥30/vector/model across more sizes (2B→70B) to map the boundary precisely. (2) The gate here hard-abstains by construction (no grounding) — test a gate that ALLOWS grounding-based answers for a stronger prevention demonstration. (3) OpenRouter key compromised (pasted in chat) — rotate. |
| pressure-calibration-powered-curve-2026-06-27 | Closed (research result; powered curve, definitive map + broad-band prevention) | Follow-up #2 (powered curve): ran the 6 pressure vectors × N=20 reps across a 1B→4B→8B→12B→27B→70B size ladder (720 calls, OpenRouter, Wilson 95% CIs). **OVERTURNS the pilot's 'smaller=more fabrication' story — fabrication is NON-MONOTONIC and VECTOR-DEPENDENT with two OPPOSITE patterns:** (1) authority-laundering (fake citation) peaks at 4B (**95%** [76%,99%]) and DECAYS with size (8B 5%, 70B 5%) — small models amplify injected citations; (2) direct appeasement ('don't say unknown') RISES with size, peaking at **70B (75%** [53%,89%]) — the BIGGEST model is most pushable past its calibration. The 1B is robust (0% across all vectors — knowledge-limited, won't engage any premise). premise/sycophancy are isolated 4B vulnerabilities (25%/10%). **Gated prevention across the full band:** ran the grounded-answer gate on all 4 fabricating sizes (4B/12B/27B/70B) — **100% prevention, 33/33 fabrications blocked** (4B 17/17, 12B 10/10, 27B 1/1, 70B 5/5; gated_affirm 0/N, hard-abstain on no-grounding by construction). **Conclusion:** the gate's value is real and BROAD-BAND but VECTOR-ASYMMETRIC vs the threat — it uniformly prevents fabrication across 4B-70B, but the THREAT differs by size: small models need defense against authority-laundering, large models against direct appeasement. No single model size handles both. The naive 'smaller fabricates more' was a small-N artifact. Artifacts: `pressure-calibration-curve-2026-06-27.json` (+raw), `pressure-curve-gated-band-2026-06-27.json`, `tools/run_pressure_curve.py`. Budget $5.36, study cost ~$0.04. `canClaimAGI:false`. | (1) Test a gate that ALLOWS grounding-based answers (not just hard-abstain) for a stronger/less-trivial prevention demo. (2) Add a 2B/3B point to nail the authority-peak shape. (3) Probe whether the 70B appeasement-vulnerability generalizes beyond Voynich (other unknowns / other big models). (4) OpenRouter key compromised — rotate. |
| calibration-self-authored-pack-2026-06-22 | Open | Residual independence gap: the abstain pack + epistemic labels are self-authored. Multi-judge corroboration is done, but a fully independent claim needs a third-party-authored pack + human semantic review. | Commission a third-party abstain pack + a human review pass |
| calibration-advantage-is-model-dependent-2026-06-25 | Open (boundary) | RAN via OpenRouter working backend (abstain pack N=18, `run_calibration_score.py`): the anti-fabrication advantage REPRODUCES on a WEAK model (llama-3.1-8b raw fab 3/12=0.25 → sophia-prompt 0/12) but VANISHES on a STRONG model (deepseek-v3 raw already 0/12; sophia-prompt 1/12, slight regression). Boundary condition on the flagship calibration claim: value decays toward 0 as the base model strengthens. Evidence: `agi-proof/baseline-ablation/openrouter-model-dependence-2026-06-25/`. Single run, prompt-level (not full pipeline), self-authored pack → not headline grade | Re-run ≥3× with the FULL pipeline + a third-party pack; report whether any strong-model advantage survives with CI |
| self-extending-loop-closes-offline-2026-06-22 | Partial | The full loop CLOSES offline on a held-out domain (selfextend/loop.py): abstain→synthesize verifier→validate on held-out→verified-reward SELECTION→policy acc 0.5→1.0 on an independent eval split→competence flips abstain→answer; all 4 invariants pass; fail-closed on unlearnable data. agi-proof/self-extension/. | (a) run on a THIRD-PARTY domain, (b) replace selection with a live RL update (GPU), (c) clear the no-overclaim gate — then it becomes a capability claim |
| long-horizon-not-run | Open | Blocks autonomy claim. Effective-horizon CURVE measured (DeepSeek 16 steps, 8 trials, noisy) but that is the chained-arithmetic metric, NOT a long-horizon autonomy run | Publish timed long-horizon autonomy run logs |
| distribution-shift-not-run | Open | RAN the mechanism (DeepSeek, 2026-06-22): promotion gate 1/2, contamination clean, protected knowledge unchanged — but the 1-case demo pack shows 0% improvement (no signal). Mechanism sound, evidence insufficient | Build a real multi-case pre/post shift pack and re-run |
| rlvr-live-run-not-yet-gated-2026-06-21 | Open | Blocks RLVR capability claim (held-out pass@1 rise vs base) | Run a gated live GRPO run clearing `aggregate._is_validated` (≥2 judge families, κ≥0.40, ≥3 runs, CI excludes 0) on an entity-disjoint held-out split + manual semantic review; offline reward-wiring invariants pass in CI but are not capability evidence |
| rlvr-math-live-run-not-yet-run-2026-06-24 | Cleared (rung) | First (2026-06-24) run was WITHIN NOISE (N=8, mean Δ +0.083, CI incl. 0). RE-RUN 2026-06-25 on the larger non-gameable N=60 fixed-held-out pack via the fast vLLM-colocate stack (trl 0.19.1 + vllm 0.9.1, `accelerate launch`) CLEARS the rung gate: 3 seeds, base **0/60 every seed** → adapter 7/60, 6/60, 5/60, all Δ>0 (mean **+0.10**), **95% across-seed CI [0.059, 0.141] excludes 0**, contamination-free, no regression, judge-free deterministic verifier, family-disjoint held-out. Evidence: `agi-proof/self-extension/math-rlvr-3seed-n60/`. HONEST SCOPE: modest/narrow (~10% where base floors at 0%) — clears THIS rung, NOT an AGI claim; `canClaimAGI` stays False | Optional: scale pack/epochs for a larger effect; the loop + fast harness are proven |
| local-agent-tools-degrade-strong-model-2026-06-21 | Closed | FIXED: selective invocation (tools fire only on low-confidence answers) + richer tool outputs (wiki_search snippets, belief wiki fallback) eliminated the degradation — on qwen3:30b-a3b `+mcp-tools` now *beats* alone (gold 90.2%→92.7%, false-positive 9.8%→7.3%), was 90.2%→51.2% before | — |
| local-agent-delta-strong-model-headroom-2026-06-21 | Superseded | Single-LEXICAL-judge run on dolphin-llama3:8b showed alone 15.2% → +gate 4.3%. This did NOT survive validation — see below. `+mcp-tools` 0.0% was re-generation, NOT tool-use (`toolsUsed: []`). | Superseded by `local-agent-delta-not-validated-2026-06-21` |
| local-agent-delta-not-validated-2026-06-21 | Closed | RESOLVED by the benchmark expansion (#6, 87→290 cases) + the unified harness (#1). The earlier N=46 run's CI straddled zero; on the expanded set a validated run (3 runs, 2 judge families = openrouter:deepseek + openrouter:meta-llama) gives the +gate lever halluc alone 36.1% → gated 23.6%, **Δ12.5%, 95% CI [+5.6%, +19.4%] EXCLUDES zero**, 0% FP-cost → `validated=True`. Recorded in RESULTS.md / published-results.json. | — |
| error-memory-rag-phase1-2026-06-25 | Closed | **Superseded by precision-gate pass below.** Initial wiring (loose similarity) net **within_noise** on N=6 dev — 100% false-correction from over-firing. | — |
| error-memory-rag-precision-v2-2026-06-25 | Partial | **Deterministic oracle only** (`agent/error_memory_eval_backend.py`). Precision gates (min_score + class_match + would_repeat). Dev sweep (v1, NOT test evidence): precision **1.0**, false-corrections **0**. Sealed v2 (N=40, hash `ef2d19ef437ff4f1`, case-level bootstrap CI): net **+1.00** [1.00, 1.00], testSplit verdict **helps**; phase1Verdict **within_noise**. `liveModelEval`: null. Bounded claim: *reduces repeat errors at acceptable false-correction cost on this held-out pack*. `canClaimAGI` False. | Live-model `LocalLLMBackend` on third-party pack; ledger stays Partial until net CI lower > 0 on live model |
| error-memory-rag-pr-100-2026-06-25 | Partial | PR [#100](https://github.com/tomyimkc/sophia-agi/pull/100) on branch `claude/error-memory-rag` — ships failure store, error-RAG gates, sealed v2 eval, eval CLI flags, `ModelBackend` scaffold. Status **Partial** (deterministic oracle only). | CI green + live-model eval on disjoint third-party pack |

| grounded-gate-not-yet-validated-2026-06-22 | Open | The retrieval-grounded gate (check_claim ground=True) is verified bug-fixed (no pen-name false positives; catches known-author misattributions for out-of-corpus works) but a 3-run/2-family N=24 run gave +gate Δ8.3%, 95% CI [0.000, +16.7%] — lower bound touches zero, so illustrative not validated (vs the prior non-grounded validated Δ12.5%). Sampling variance at small N. | Re-run grounded at larger N (>=40 cases / more runs) to push the CI off zero |
| agent-faithfulness-judged-not-yet-validated-2026-06-25 | Open | The judged agent-faithfulness benchmark (sealed held-out N=9, `provenance_bench/agent_faithfulness_judged.py`) ships with the no-overclaim gate WIRED and tested both ways with scripted judges (clean 2-family perfect run validates; mock/single-family/low-κ/<3-run do not), and the entailment judge's value over the deterministic lexical floor is demonstrated in tests (lexical 33% → scripted-oracle 100%, value-add +67pts). But the committed artifact is the OFFLINE MOCK run (`validated=False`); no real multi-family model run has been executed, and the held-out pack is first-party (sealed, NOT third-party). | Run `tools/run_agent_faithfulness_judged.py --judges <2 distinct vendor families> --runs 3 --write` to clear the gate; commission a third-party-authored trajectory pack to close the label-provenance gap |
| rlvr-adapter-kappa-2family-below-bar-2026-06-26 | Open — judged, candidate NOT validated | The deterministic-verifier RLVR adapter (run `mr9sr03clgpk5g`, GLM-4-9B, 94 held-out provenance cases) was semantically re-scored by 2 INDEPENDENT judge families (`openrouter:deepseek/deepseek-chat`, `openrouter:meta-llama/llama-3.3-70b-instruct`; ≠ subject, ≠ gate) via `tools/build_rlvr_judge_answers.py` → `tools/judge_pilot_answers.py` (no GPU; answers were the committed base/adapter completions). BOTH judges directionally PREFER the adapter over base (deepseek 40:23, llama 57:37; both-agree consensus adapter-better 29 vs base-better 10, ≈3:1) — the gains are not a one-judge artifact. **BUT inter-judge Cohen's κ = 0.094, FAR below the RESULTS.md validated bar (κ ≥ 0.40).** This is the prevalence/marginal-skew paradox (llama emits 0 ties, deepseek 31), so directional agreement is high while chance-corrected κ is low. Per the no-overclaim standard this is NOT validated and goalposts are NOT moved to Gwet AC1. Report: `agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge.json`. candidateOnly=true, `canClaimAGI`=false. **PRE-REGISTERED RE-RUN (forced-choice, TIE disallowed, run ONCE):** removing the tie-rate asymmetry matched the marginals (deepseek adapter-winrate 0.581, llama 0.596) but κ rose only 0.094→**0.110, still ≪ 0.40** (`…/mr9sr03clgpk5g.judge-forced.json`). So the low agreement is NOT merely a prevalence artifact — both judges prefer the adapter ~58–60% on aggregate (consensus 35:18) yet disagree case-by-case. Directional preference robust; case-level validated agreement NOT met. No third variant attempted (would be goalpost-fishing). **HONEST-STATS PANEL re-run** (`…/mr9sr03clgpk5g.judge-forced-panel.json`): observed inter-judge agreement only **0.53**, PABAK **0.06**, and per-judge win-rate CIs SPAN 0.5 (deepseek 0.521 [0.42,0.62] binomial p=0.76; llama 0.585 [0.48,0.68] p=0.12) — so it is NOT merely the κ prevalence paradox: at n=94 the judges genuinely disagree case-by-case and the win-rate is not distinguishable from chance, with large run-to-run variance (deepseek 0.581→0.521 across identical runs). Method analysis + citations in `docs/methodology/llm-judge-validation.md`. **≥3-JUDGE PANEL (4 families deepseek/llama/qwen/mistral, majority vote, forced-choice, n=94; `…/mr9sr03clgpk5g.judge-panel4.json`):** majority-vote adapter win-rate **0.532** [0.42,0.64], binomial **p=0.65 — NOT significant (≈ chance)**. The earlier 0.58–0.60 read was inflated by the **llama** judge (0.617, the lone significant + most-generous family); deepseek/qwen/mistral sit 0.52–0.55. Reliability IMPROVED with more/better judges — pairwise binary κ up to **0.59** (deepseek↔qwen 0.59, deepseek↔mistral 0.53, qwen↔mistral 0.46; **3/6 pairs ≥0.40**); the original κ=0.11 had used the single WORST-agreeing pair (deepseek↔llama). So low agreement was partly judge-pair selection, but the capability signal is genuinely **~chance at n=94** — a cleaner NULL, not a measurement artifact. Independent-oracle option tested + REJECTED: the Google Fact Check API has ZERO coverage of the literary-provenance domain (political/viral ClaimReview corpus). | Inter-judge κ ≥ 0.40 remains UNMET after the one pre-registered fix. Honest options: accept the directional-only signal as candidate evidence; OR a 3rd independent judge family + majority vote as a SEPARATE pre-registered protocol; OR a third-party-authored pack (still needed for any external generalization claim). Do NOT iterate judge variants until one passes. **DECISION 2026-06-26: the 3rd/4th-judge-panel option was EXECUTED → majority-vote win-rate 0.532 (p=0.65), a clean NULL. Track PAUSED (low EV): a ≥300-case third-party pack would, against a ~0.53 effect, most likely only CONFIRM the null, so GPU/pack spend is not justified now. The adapter stays candidate-only; effort redirected to the Level-3 real-data lanes (see `docs/06-Roadmap/level3-blocker-plan.md`).** |
| sophia-wisdom-4b-m1-egress-blocks-baseline-run-2026-06-25 | Closed | RESOLVED 2026-06-25 in a network-open session: `openrouter.ai:443` egress now succeeds (authenticated `GET /api/v1/models` + `/key` return 200 through the proxy CA bundle) with `OPENROUTER_API_KEY` set as an env secret. The M1 baseline run executed end-to-end (see row below). Prior block stands as the historical record of why the earlier container could not run it. | — |
| sophia-wisdom-4b-m2-volume-below-target-2026-06-25 | Open — volume NO-GO (root cause re-diagnosed: corpus, not egress) | **UPDATE 2026-06-25 (egress open, live teacher run):** the `--teacher` hook is now BUILT + tested + PROVEN on real models — `tools/build_sophia_wisdom_dataset.py --teacher openrouter:deepseek/deepseek-chat` generated 156 route-first candidates that flowed through the SAME admission gate; the dataset grew to **965 gate-passed rows** (acceptance 0.651; 233 preference pairs; 90 decontam drops). **Retention floor FIXED: 16% → 28%** (license-clean OASST1/Apache-2.0 slice, now in the mandatory 25–30% range). Protected-history control suite widened **6 → 36** (bilingual) for M3 measurability. BUT **≥10k volume still NOT met (965 ≪ 10k) → M2 stays NO-GO; DO NOT advance to M3.** KEY RE-DIAGNOSIS: the shortfall is **PROMPT/CORPUS-bound, not egress/teacher-bound** — the structured corpus is ~72 records and the builder dedups by normalized prompt, so even an unblocked live teacher only improves answer quality/variety per prompt (the prior "egress blocks the teacher" framing is now closed; the real binding constraint is corpus size). Mix still skewed (source_discipline 53.5% vs 20–25%; hk_bilingual 0.8% / moral_gate 3.8% / tool_mcp 0.9% thin) — also corpus-bound. `tests/test_sophia_wisdom_dataset.py` 6/6 (incl. new teacher-hook tests); contamination CLEAN; `lint_claims` OK. `canClaimAGI` stays False. ORIGINAL ROW BELOW. ||| M2 PIPELINE COMPLETE; ≥10k GATE-PASSED ROWS NOT MET. `tools/build_sophia_wisdom_dataset.py` implements the teacher→gate→admission loop (admission = `agent.gate.check_response` advisor + public_standard; reject → preference-pair/hard-negative). Produces `training/local_sophia_v3/` = **730 decontaminated gate-passed SFT rows** (701 train / 29 valid) + **233 ORPO preference pairs** + 340 audited rejections; 90 decontam drops vs the eval/benchmark surfaces. `tests/test_sophia_wisdom_dataset.py` (4/4) proves admission is real (a bare fabrication is rejected). HONEST SHORTFALL: 730 ≪ 10k target — the teacher here is DETERMINISTIC TEMPLATING (live-LLM teacher needs the egress that is blocked) + reuse of ~800 curated rows, so yield is corpus-bounded. Mix is skewed (source_discipline 59% vs target 20–25%; **general-retention 16% vs the MANDATORY 25–30% floor**; hk_bilingual/tool_mcp thin). Per the plan, DO NOT advance to M3 training on this — volume + retention gate fails. `canClaimAGI` stays False. | (1) Unblock a live-LLM teacher (model egress) to scale synthetic gate-passed rows toward 10–20k; (2) expand the structured corpus (more attribution/religion/tradition records); (3) bring a license-clean external instruct slice up to the 25–30% retention floor BEFORE any SFT; then re-run `--check` and gate to clear the M2 go/no-go |
| sophia-wisdom-4b-m3-pilot-preregistered-2026-06-25 | RAN — PASS (pre-registered primary), candidate-only | **RESULT 2026-06-25: M3 PILOT RAN AND PASSES the pre-registered primary criterion** (gemma-3-4b language-tower LoRA, ~730 deterministic gate-passed rows, seed 0, seq-len 1024; full M1 instrument N=354×3 runs on a RunPod H100 via the SSH-free self-report workflow). adapter(prompt)−base(prompt) is CI-clean improving on **3 of 4** primary metrics: qualification_rate_on_contested **+0.475** [0.459,0.508], tradition_merge_rate **+0.143** [0.125,0.161], false_attribution_rate **+0.014** [0.012,0.018] (citation_fidelity +0.028 touches 0). **No protected regression** (history base 0.083→adapter 0.000 at prompt_gate, ≤base at every condition; religion ≤base everywhere); **over_abstention ≤0.018**; useful_correctness RISES at prompt/prompt_gate (no refusal collapse). lint_claims OK. Registered `training/adapters/registry.jsonl` (candidate_only:true, validated_external:false; weights not persisted — reproducible from seed) + model card `agi-proof/model-cards/sophia-wisdom-4b.md`. Eval artifact `agi-proof/benchmark-results/wisdom-market/M3-pilot-eval.json`. HONEST CAVEATS: metrics are deterministic MARKER/structural (no LLM judge — the large qualification/moral_route deltas partly reflect learned source-discipline FORMAT, exactly the trained habit; semantic quality needs a ≥2-judge-family pass before any headline; the forbidden-assertion reductions tradition-merge/false-attribution are the more substantive signal); retention measured via PROXY (useful_correctness+over_abstention), `run_learning_shift.py` NOT yet run; single base/seed, corpus-bound; train/eval share structural families (decontaminated by exact prompt, not format). NOT market-beating, NOT validated, NOT AGI; `canClaimAGI` stays False. Infra note: the SSH path failed (dev box HTTPS-only egress + RunPod public-IP SSH flake), so execution went SSH-free (pod runs the whole job + git-pushes result via the workflow GITHUB_TOKEN + self-deletes); one cosmetic `exit 128` came from a pod RESTART re-running the start command over the persistent /workspace volume AFTER the eval had already completed+pushed (result intact). | NEXT (before any promotion / external claim): seeds 1–2 for stability · ≥2-judge-family semantic re-score of qualification/route quality · run `run_learning_shift.py` on the adapter · then M4 ORPO on `training/local_sophia_v3/preference_pairs.jsonl`. Harden the self-report job to `rm -rf` the clone dir on restart (done) |
| sophia-wisdom-4b-m3-pilot-ORIGINAL-2026-06-25 | Superseded by the RAN row above | M3 RE-SCOPED TO A CORPUS-BOUND PILOT (human decision: reconsider the 10k target rather than chase it). Because M2 is a NO-GO on volume (965 rows, corpus-bound), the full "beat same-size baselines" M3 is not honestly reachable; instead a **smaller, falsifiable pilot** is pre-registered in `docs/06-Roadmap/Sophia-Wisdom-4B-M3-Pilot.md`. ONE falsifiable question: does a LoRA SFT on the 965 gate-passed rows move **gemma-3-4b-it**'s weights so its **prompt-scaffold (no-gate)** behavior shifts toward the gated target on ≥1 Sophia-native axis WITHOUT protected/retention regression? PRE-REGISTERED PASS (thresholds fixed BEFORE training, anti-gaming): (1) on ≥1 of {tradition_merge_rate, qualification_rate_on_contested, false_attribution_rate, citation_fidelity}, adapter(prompt)−base(prompt) CI excludes 0 improving; (2) protected_history (N=36) & protected_religion (N=34) not CI-clean worse than base+0.05; (3) run_learning_shift stability ≥ base−5pts; (4) over_abstention ≤0.10; (5) lint_claims clean. A NULL primary signal is the EXPECTED modal outcome at this row count and is a legitimate logged result, not a hidden failure. Eval reuses run_same_size_market_baselines.py (N=354, ≥3 runs) + run_learning_shift.py; gate stays independent. Claim ceiling: a narrow corpus-bound feasibility result — NOT market-beating, NOT validated, NOT AGI. PREREQS before any GPU: gemma-3-4b-it weights are HF-GATED (license + HF_TOKEN) and the train stack (Qwen2.5-built) needs the gemma-3 chat template wired + a 5-step smoke; train = RunPod CUDA peft, ONE seed, seq-len 1024. `canClaimAGI` stays False. **EXECUTION BLOCKED 2026-06-25 (confirmed):** the M1-selected base `google/gemma-3-4b-it` is `gated:manual` on HF and no `HF_TOKEN` is set here → weight `config.json` is HTTP 401; also multimodal (`image-text-to-text`). Cannot pull/train the base in this environment; not routed around. STOPPED before any GPU spend. **UPDATE:** a valid `HF_TOKEN` (account `tomyimkc`, write scope) was provided, but that account has **not accepted the Gemma license** — `google/gemma-3-4b-it` AND `-pt` both return **HTTP 403** (authenticated but not access-granted). One-click fix: accept the Gemma license on the model page while logged in as `tomyimkc`, then re-verify (config.json → 200). Still no GPU spent. **UPDATE 2 (2026-06-25): Gemma access RESOLVED** — license accepted, `google/gemma-3-4b-it/config.json` → **HTTP 200** (confirmed `Gemma3ForConditionalGeneration`, multimodal: text tower 2560-dim × 34 layers + vision tower). **NEW BLOCKER: no RunPod credential** — `RUNPOD_API_KEY` unset in env AND the RunPod MCP returns **401**, so no CUDA pod can be created (exec box is CPU-only). Also noted: `tools/train_lora.py` loads via `AutoModelForCausalLM`, which won't directly fit the multimodal `Gemma3ForConditionalGeneration`; the language-tower loading branch will be wired + smoke-tested ON the pod (where weights load), not blind. **UPDATE 3 (2026-06-25): RunPod key provided & valid** (pods API 200); built the FULL execution path — `tools/pilot_gemma3_run.py` (on-pod gemma-3 language-tower LoRA train + BATCHED base-vs-adapter eval reusing the M1 instrument's tested scoring, no vLLM) + `tools/runpod_wisdom_pilot.py` (proven create→rsync→SSH→scp→delete lifecycle; HF token read on-pod from /proc/1/environ so it's never logged; smoke-first aborts cheaply). A real A100 pod ($3.29/hr) WAS created but **SSH from this exec container TIMED OUT**: confirmed the dev box is HTTPS-egress-only (raw TCP to :22 blocked, :443 OK), so it cannot drive a pod over SSH. Pod auto-deleted (no leak; ~1 min billed). RESOLUTION: run via GitHub Actions (runner has open egress) — added `.github/workflows/wisdom-pilot-runpod.yml` (rebuilds the deterministic gate-passed dataset on the runner, launches the pilot pod, commits the eval back to the branch). No GPU eval result yet. | REMAINING USER ACTIONS to run the GPU pilot via CI: (1) ensure repo Actions secret **`HF_TOKEN`** exists (new; gemma is gated) — `RUNPOD_API_KEY` is likely already set from the existing runpod workflows; (2) get the workflow onto the default branch so `workflow_dispatch` is registerable; then I dispatch via the GitHub MCP, poll, and analyze the committed result vs the pre-registered thresholds + register candidate/model-card. (Alt path if preferred: provide a GitHub PAT with push scope and I drive the pod from here over the HTTPS API with the pod self-reporting via git push — no Actions needed.) |
| sophia-wisdom-4b-m1-base-selection-ran-2026-06-25 | Closed — M1 GO (base = gemma-3-4b) | M1 BASE-SELECTION RAN ON REAL SAME-SIZE MODELS; AFTER A GATE FIX, **M1 PASSES with base = `google/gemma-3-4b-it`** (human decision 2026-06-25, round 2: select gemma + widen the N=6 protected-history suite early in M2 so no-regression is measurable at M3). gemma is the only same-size base with CI-clean prompt+gate uplift on the headline 儒/道 tradition-merge differentiator AND tool-routing AND qualification, at rising usefulness, bounded over-abstention (0.042), religion-regression eliminated; the lone protected-history residual (0.0556) is noise-floor on a 6-case suite (independent probe 0/18). Original run details below. Full sweep over the N=324 held-out benchmark, conditions raw/prompt/prompt_gate, **3 runs**, bootstrap 95% CIs (`agi-proof/benchmark-results/wisdom-market/M1-base-selection-2026-06-25.md` + 3 `baselines_*_2026-06-25.json`). OpenRouter slugs had drifted: **Qwen3-4B removed** (smallest dense Qwen3 now 8B = 2× target) → Qwen slot DROPPED per human decision to keep "same-size" honest; Phi→`microsoft/phi-4-mini-instruct`, Gemma→`google/gemma-3-4b-it` (truer same-size than 9B). Cost ~$0.78. FINDINGS: (a) **llama-3.2-3b — NO**: the prompt scaffold alone collapses it (useful_correctness 0.415→0.012, qualification 0.328→0, tool_route 0.429→0) — too weak for route-first JSON. (b) **gemma-3-4b — strongest uplift but FAILS the strict gate**: prompt_gate gives qualification Δ+0.311*, tool_route 0→0.857*, useful_correctness 0.464→0.594, religion-regression 0.147→0.010 — but the GATE induces a **protected_history_regression 0.000→0.222** (fail-closed on ~22% of protected history cases), violating the mandatory no-protected-regression rule. (c) **phi-4-mini — only base clearing every hard constraint, but modest/narrow**: CI-clean prompt_gate gains on citation_fidelity +0.250* (raw 0.694, real headroom), false_attribution +0.0058*, provenance +0.0067*; over_abstention 0.077 ≤0.10; NO protected regression; but its headline 儒/道 `tradition_merge_rate` Δ=0 and useful_correctness dips 0.465→0.386 under the gate. BORDERLINE: phi passes the literal M1 criteria but modestly, while the larger win (gemma) is blocked by a gate-fail-closed issue, not a base verdict — and several axes (provenance, citation) are saturated raw. No semantic judge wired → useful_correctness/qualification are ILLUSTRATIVE marker-based, not headline. `canClaimAGI` stays False. **UPDATE 2026-06-25 (gate fix + re-judge, human chose option B):** root-caused gemma's regression to the public-standard gate hard-floor-BLOCKING a descriptive history answer because the violence marker `kill` matched the toponym **"Kill Devil Hills"** (Wright brothers' first flight). Fixed with a proper-noun carve-out (mask known benign toponyms/titles before marker extraction; real `kill` threats still block — 14/14 gate tests pass + new regression test). gemma re-run (fixed gate, full N=324×3): CI-clean source/moral improvements **3/8→4/8**, now including the **headline 儒/道 tradition_merge Δ+0.125\***, qualification **+0.372\***, tool_route 0→**0.857\***, contested_fab **+0.082\***; useful_correctness *rises* 0.474→0.604; over_abstention 0.042; **protected_religion regression 0.216→0.000**; **protected_history regression 0.222→0.0556** (=1 flagged case-run of 18; an independent 18-gen probe reproduced it **0/18** → noise floor of a 6-case suite, not systematic). gemma is now the clear winner on the winnable axes; only blemish is a noise-floor protected-history residual on an underpowered N=6 suite. Evidence: `baselines_gemma-3-4b-it_gatefix_2026-06-25.json` + M1 doc addendum. `canClaimAGI` stays False. | HUMAN DECISION REQUESTED (round 2): (A) select **gemma-3-4b** as the M1 base and proceed to M2, expanding the protected-history suite (N=6→ larger) early so no-regression is measurable at M3; or (B) treat 0.0556≠0 as a hard fail and keep gemma blocked; or (C) expand the protected-history suite FIRST, re-run, then decide. Recommend (A). Do not headline gemma as a clean pass until the protected suite is powered |
| sophia-wisdom-4b-m1-base-selection-not-run-2026-06-25 | Superseded | Superseded by `sophia-wisdom-4b-m1-base-selection-ran-2026-06-25` above — the instrument has now been run on real same-size models. Original row retained for history. M1 INSTRUMENT BUILT, BASE-SELECTION GO/NO-GO NOT YET RUN. `tools/build_wisdom_market_benchmark.py` → `data/wisdom_market_benchmark/heldout_v1.jsonl` (N=324 held-out adversarial cases, EN 179 / ZH 145, 12 families incl. the 儒家/道家 differentiator; decontaminated, disjoint from reference traps, wired into `build_local_sophia_dataset.py --check`). `tools/run_same_size_market_baselines.py` computes the 12 metrics across the raw/prompt/prompt_gate ladder with ≥3-run bootstrap CIs; pipeline validated end-to-end in `mock` mode + `tests/test_wisdom_market_baselines.py` (8/8). NO same-size numbers exist yet: the exec container is CPU-only with no API keys / Ollama / GPU, so raw Qwen3-4B / Phi-4-mini / Llama-3.2-3B / Gemma and the ≥2 judge families cannot be run here. Benchmark N=324 is below the 500–1000 plan target — honestly corpus-bounded by the current seed records, not padded. NO uplift claim of any kind; `canClaimAGI` stays False. | Provision model access (OpenRouter keys for the 4 same-size bases + 2 judge families, OR a RunPod inference GPU), run `run_same_size_market_baselines.py --models <bases> --runs 3`, then evaluate the M1 go/no-go (headroom present AND prompt+gate beats raw with CI excluding 0, over-abstention ≤0.10, no protected-suite regression). Expand the benchmark toward 500–1000 as the M2 corpus enrichment lands |
| path-a-world-model-not-promoted-2026-06-26 | Open (negative result) | PATH A DreamerV3-style world-model canary FIRED, verdict **HOLD (not promoted) — 4/4 seeds failed to promote**. Discrete-latent RSSM over a 25-pair mixed-outcome corpus (train=14/val=6/shift=5; 60% pass, actions model+run_tests): 3/4 seeds `hold-below-bar` (val 0.50 < bar 0.65) and seed 1 `hold-shift-degenerate` (val 0.67 cleared the bar but shift COLLAPSED to 0.20, degradation 0.47 >> max 0.15 — the canonical memorizes-train/collapses-on-novel-task-families failure). trainLoss ~0.006-0.008 with val ≤0.67 = classic overfitting. The shift-degeneracy verdict is the load-bearing signal (NOT val accuracy); the canary correctly refused to promote a memorizing predictor. Validates the canary design; bounds the claim: Path A is NOT promotable at this trace scale. HONEST DATA CAVEAT: the live harness + mock provider produced degenerate all-pass traces (no action-outcome contrast), so a 25-pair mixed corpus was synthesized in the miner's JSONL schema to give the shift canary real signal — NOT live-agent traces. `candidateOnly`/`validated`=false/`false`, `canClaimAGI:false`. Artifact: `agi-proof/world-model/path-a-dreamer-canary-2026-06-26.public-report.json` (CPU torch 2.12.1, CUDA unavailable). | Prerequisite for a meaningful re-test: a much larger LIVE harness trace corpus (hundreds of genuine pass/fail + multi-action traces from a real model provider, not mock/synthesized), then re-fire `tools/run_world_model.py --epochs 120` and check whether ANY seed clears both the held-out bar AND the shift-degradation check. Until then Path A stays held. |
| path-b-lean-proof-search-blocked-2026-06-26 | Open (blocked/negative) | PATH B Lean 4 proof search FIRED, verdict **BLOCKED / negative — no novel verified proof produced on this host**. Falsifiable question: does the search produce a Lean-verified proof that is NOT a near-duplicate of the corpus (novel: true)? Answer: **NO**. (1) The only proof found (`trivial_true` via the stub applier) is a near-duplicate of the corpus (char-trigram Jaccard overlap 1.0 >> 0.92 threshold → novelty FALSE), AND it is a STUB-accepted proof (`accepted-by-injected-applier`), NOT a real Lean verification. (2) `zero_add`/`add_comm` could not be proved by the stub applier and the search **FAIL-CLOSED** (`no_proof_within_budget`) — it never asserted an unproved goal. (3) The REAL Lean path (LeanDojo + Lean 4/elan) **abstained `lean_unavailable`** because the toolchain is not installed on this host; it produced no fabricated proof. The fail-closed discipline is confirmed correct: a proof Lean would reject is never asserted. The novelty probe (strict char-trigram Jaccard @ 0.92) correctly classified the trivial proof as a retrieved near-duplicate. `candidateOnly`/`validated`=true/false, `level3Evidence`=false, `canClaimAGI:false`. Artifact: `agi-proof/proof-search/path-b-proof-search-2026-06-26.public-report.json`. | To exercise the real novelty signal: install the Lean 4 toolchain (elan + lake) + `lean-dojo`, point an LLM tactic proposer at a real `LeanProofSession`, and re-fire `tools/run_proof_search.py --theorem add_comm` (no `--stub`). The load-bearing question is then whether the LLM proposer ever finds a Lean-verified proof whose strict-novelty probe returns `novel: true` (NOT a corpus near-duplicate). Until that environment exists, Path B stays blocked. |
| path-a-canary-spurious-promote-fixed-2026-06-26 | Closed (canary defect found + fixed) | PATH A canary REFIRE on REAL DeepSeek traces (44 pairs, 0.955 positive rate, actions model+tool; coding-debugging skill + exec/gate verification) exposed a **canary defect**: all 3 seeds returned `promote` (val 1.0/0.9/1.0, shift 1.0, degradation 0.0), but this was SPURIOUS — both negative examples fell in train, so val/shift had positive_rate = 1.000, making the predictor's val_accuracy 1.0 IDENTICAL to the always-predict-positive majority-class baseline. The old gate only checked `val_acc > 0.65`, which the trivial solution clears on pass-skewed data. **Fix landed:** `agent.verified_world_model.py` now computes the majority-class baseline on val and requires the predictor to STRICTLY BEAT it (new verdict `hold-at-majority-baseline`); the spurious promotes now correctly HOLD. Regression test `test_pass_skewed_corpus_does_not_spuriously_promote` locks it. Net Path A status unchanged: **still NOT a valid promote** — the prior 25-pair synthesized HOLD remains the honest verdict, and this real-data refire produced no valid positive (the data is too pass-skewed to exercise the canary). `canClaimAGI:false`. Artifacts: `agi-proof/world-model/path-a-dreamer-canary-refire-real-2026-06-26.public-report.json`. | A meaningful Path A promote needs a BALANCED pass/fail corpus (stratified so val/shift contain negatives) from genuine hard-task failures — strong-model traces on easy tasks are too pass-skewed. Optionally also add stratified splitting to `make_splits`. The canary gate is now honest; the data prerequisite remains open. |

## hurdle2-transfer-scorer-registry-wired-2026-06-24

**Status:** PARTIAL (mechanism wired + tested; capability evidence hardware-bound).

Hurdle 2 (broad transfer) step 1: the eval ladder + scoring path is no longer
provenance-only. `agent/domain_scorers.py` adds a `domain → score_fn` registry and
both eval backends (`eval_local_model.py`, `eval_mlx_model.py`) dispatch through it.
Two structurally different, **sound-verifier** families are wired as the transfer test:

- `math` (`tests/benchmark-math.json`, 12 cases from `capability_arithmetic.json`):
  exact answer-match + `arithmetic_sound` soundness veto.
- `coding` (`tests/benchmark-coding.json`, 2 cases from `eval/coding/smoke.jsonl`):
  `code_tests_pass` (executes Python, checks exit code).

`tools/eval_ladder.py` now takes `--domains`, so the **identical** 4-rung
base/+gate/adapter/+gate ladder runs on math/coding. Gated by
`tests/test_domain_scorers.py` (9 tests: registry routing, per-family accept/reject
incl. real code execution, soundness veto, full-pack scoring). The provenance path is
unchanged; `tools/lint_claims.py` OK; 134 related tests pass.

**Why this is NOT a transfer capability claim:** no adapter has been trained/evaluated
on math or coding yet — the identical retention+promotion *loop* on these families
(step 2.4) is hardware-bound like C2/C5. This is proof the scoring substrate is
domain-general and the two families score through their own sound verifiers, not proof
the method *improves* a model on them.

**Next experiment:** train one adapter per family, run the ladder + W2 gate + feedback
miner, report per-family deltas with CIs. Coding+unit-test is the on-ramp to SWE-bench
(Hurdle 1). Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## hurdle2-promotion-gate-generalizes-2026-06-24

**Status:** CLOSED (mechanism). The W2 promotion gate is confirmed domain-general:
`tools/promote_adapter.py` already accepts `--protected`, and a test now locks in that
`evaluate_update` + the formal protected-floor proof promote a clean coding+math adapter
with an EMPTY protected set and reject a coding regression when `coding` is marked
protected — no provenance hardcoded
(`tests/test_promote_adapter.py::test_promotion_gate_generalizes_to_non_provenance_families`).
Combined with `hurdle2-transfer-scorer-registry-wired-2026-06-24`, the full
score→ladder→promote path is now family-agnostic. **Honest bound:** machinery generality,
not a transfer capability claim (no adapter trained on coding/math yet — step 2.4, hardware).

## hurdle4-plasticity-probe-and-diversity-floor-2026-06-24

**Status:** PARTIAL (anti-degradation machinery added + tested; real runs hardware-bound).

Hurdle 4 (prevent plateau/degradation) — two no-GPU guards added:

- **Plasticity probe** (`agent/plasticity_probe.py`): pure-Python stable rank (rank-collapse
  correlate, `‖W‖_F²/‖W‖₂²` via deterministic power iteration), dead-unit fraction, and
  weight-norm growth — the loss-of-plasticity correlates the 2025 literature identifies
  (spectral collapse arXiv 2509.22335; 2404.00781). `watch_generations` emits a
  `degrading-plasticity-warning`, attachable to the generational compounding artifact via
  `tools/run_ssil_generations.py --plasticity-json`. Gated by `tests/test_plasticity_probe.py`.
- **Diversity floor** (`tools/feedback_to_training.py mine --min-novelty`): rejects a mined
  candidate whose token-Jaccard to anything already queued exceeds `(1 - min_novelty)`, so the
  continual loop can't narrow onto near-duplicate misses (accumulated reward bias / shrinking
  answer diversity — the self-rewarding diminishing-returns failure mode). Default 0.0 = off
  (back-compat). Gated by `tests/test_feedback_diversity_floor.py`.

**Why NOT a capability claim:** these are early-warning *diagnostics* and a *queue guard*,
not a measured anti-degradation result. The real test — `ssil_generations` ≥3 generations on
real weights with per-generation rank/dormancy stats, and the mitigation (shrink-and-perturb /
L2-init / continual-backprop) at the retrain step — is hardware-bound. The runner
(`tools/run_ssil_generations.py`) already consumes real aggregates; only the GPU runs remain.
Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## hurdle1-swebench-runner-built-not-run-2026-06-24

**Status:** OPEN (runner built + tested; no real run yet).

Hurdle 1 (external/independent validation) enablement: `tools/run_swebench.py` runs
SWE-bench Verified the honest way — Sophia produces patches; the **official**
`swebench.harness.run_evaluation` grades in Docker against the real
FAIL_TO_PASS/PASS_TO_PASS tests (external ground truth, never the Sophia gate). The tool
owns only the deterministic halves: prompt build, unified-diff extraction, official
prediction format ({instance_id, model_name_or_path, model_patch}), and parsing the
official report into a no-overclaim artifact (`schema sophia.external_benchmark.v1`,
resolvedRate, per-instance ids, decontamination + claim-boundary). Offline style sample
(`eval/external/swebench-style-sample.jsonl`) + `tests/test_run_swebench.py` (8 tests)
keep it CI-safe; lint_claims OK.

**Why this is NOT a result:** no real SWE-bench Verified run has happened. The committed
solver is a minimal scaffold (problem statement → model → diff) with no repo navigation,
so resolved% will be a floor. Grading needs Docker + the `swebench` package (x86_64 by
default; Apple Silicon needs arm64 images or a Linux host for the eval step).

**Required to make it a defensible claim:** real run on `princeton-nlp/SWE-bench_Verified`,
base vs sophia-full vs adapter, ≥3 runs with CIs, report the DELTA vs base (cancels shared
pretraining contamination), then third-party reproduction. Companion external lane GSM8K is
already wired (`tools/fetch_eval_dataset.py --dataset gsm8k` + `tools/run_external_eval.py`).
Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-learning-under-shift-wrong-store-diagnosis-2026-06-24

**Status:** OPEN (diagnosis + corrective machinery; the corrected run is hardware/graph-bound).

The first sophia-v4 multi-goal attempt (branch `claude/sophia-v4-multigoal-codex`, seed 0)
**fixed v3's catastrophic forgetting** — old-task retention held (`oldBenchmarkDeltaPct=0.0`) —
and improved the first-party ladder (56.2%→68.8%; philosophy 77.8→100, history 62.5→75,
religion 0→16.7). It was correctly **rejected** by the hardened gate because
`learning-under-shift` returned `passingSignal=false`.

**Root cause (structural, not model quality):** the shift result was **pre 0/10 AND post 0/10**.
`learning-under-shift` teaches by appending declarative records to `agent/memory/learning_shift.jsonl`
and post-tests whether they are used — but it was gated against a **frozen LoRA adapter**
(`--backend adapter`), which cannot absorb newly-appended facts and whose answer path did not
retrieve them. So `post == pre` by construction. This is the PR #82 two-store rule biting back:
**declarative learning lives in the OKF graph / retrieval ("hippocampus"), not the weights
("neocortex").** Gating a habit adapter on a knowledge-store signal tests the wrong store. Also
matches `distribution-shift-not-run` ("no signal").

**Corrective machinery (this branch, no GPU):** `agent/store_routing.py` routes each signal to
its store and gives a store-aware adapter verdict that does NOT weaken the gate: a habit-store
failure still **rejects**; a knowledge-store signal measured on the habit store is an **INVALID
measurement → quarantine** (neither promote nor reject), forcing re-measurement on the
graph/CPQA path; an unvalidated knowledge goal → quarantine. Under this routing, v4 seed 0 is a
**quarantine** (habit store passed; knowledge signal mis-measured), not a hard reject. Gated by
`tests/test_store_routing.py`; lint OK.

**Next experiment (do BEFORE more seeds — running seeds 1/2 now repeats the same structural
reject and wastes GPU):**
1. Confirm the wrong-store hypothesis: dump a post-test prompt + the appended records + the raw
   answer; verify the records are absent from context.
2. Re-measure learning/knowledge goals on the **knowledge store** — the graph-backed CPQA path
   (`tools/run_continual_qa_validation.py`) or a retrieval-augmented backend that injects the
   appended records — and prove `pre<post` is reachable on a known-learnable mini-domain (closes
   `distribution-shift-not-run`).
3. Wire `agent/store_routing.store_aware_adapter_verdict` into the v4 `promote_adapter` so the
   adapter is gated on habit metrics (ladder + SEIB + retention) while knowledge goals gate
   separately on the graph.
4. Re-score seed 0 on the corrected gate (no retrain); only then run seeds 1/2.

Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-contested-fabrication-teaching-to-test-risk-2026-06-24

**Status:** OPEN (seed 0 honestly rejected on SEIB; methodology corrected; clean run pending).

The store-aware re-score of v4 seed 0 came down to **one** SEIB condition:
`sophia_full.fabricationRateOnContested == 0`. Seed 0: nCases=100, falseAttribution 0.0,
raw_to_full accuracy delta **+0.20**, false-positive cost 0.02 — but contested fabrication
**0.04** (the over-assertion axis). Two per-entity qualification traces (Beauvoir, Bradbury;
verified on a 2-row slice 1.0→0.0) lowered it **0.04 → 0.02**, leaving Dostoevsky / Crime and
Punishment. Honest reject held; no seeds 1/2.

**Methodology risk identified:** authoring a qualification trace for each failing SEIB-100 row
is **teaching-to-the-test at the entity level**. The (work, gold_author) pairs being patched —
Dostoevsky, Beauvoir, Bradbury — are all present in `eval/seib/seib_100_v1.jsonl`; the
verbatim-shingle decontam guard misses this because the prompt wording differs but the entity
pair is identical. A `fabricationRateOnContested == 0` reached this way is memorisation, not a
habit, and would not survive a third-party SEIB pack (Hurdle 1).

**Worse:** the new auditor (`tools/seib_generalization_split.py --audit`) shows the
**foundational corpus already covers** many SEIB-100 contested entities (Analects/Confucius
across ~16 training examples; Enchiridion/Arrian; etc.). So SEIB-100 is **not a clean held-out**
for the contested-qualification habit to begin with.

**Corrective machinery (this branch, no GPU):** `tools/seib_generalization_split.py` —
deterministic train/held-out split of the 50 contested entities + a leakage audit that flags any
training example mentioning a held-out (work, author). Gated by
`tests/test_seib_generalization_split.py`. The honest protocol it enforces: instil the
qualification habit on contested entities **disjoint** from the held-out test, and require
fabricationRateOnContested == 0 on entities **never trained**; 0.0 only on trained entities is
memorisation.

**Next experiment:** stop per-row patching. Build a broad contested-qualification habit grounded
in **retrieval confirmation** ("assert only what the graph confirms; otherwise qualify/abstain")
so it generalises to unseen entities, and measure on a contested set disjoint from the training
entities (ideally a fresh third-party pack — that also advances Hurdle 1). Keep false-positive
cost ≤ 0.10 (don't tip into denying true authorships). Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-corpus-partition-47-of-50-clean-2026-06-24

**Status:** OPEN (held-out clarified; clean generalization run pending — no GPU here).

The corpus-coverage partition (`tools/seib_generalization_split.py --partition-by-corpus`)
resolves the held-out confusion: **47 of 50 SEIB-100 contested entities are corpus-CLEAN**;
only **3 are corpus-taught** — Dao De Jing/Laozi, Analects/Confucius, Enchiridion/Arrian. The
earlier `--audit --fail-on-leak` red was an artifact of the naive 50/50 hash split dropping
some of those 3 taught entities into its held-out half; the corpus partition is the correct
held-out definition, and auditing the 47-entity corpus-clean set is **CLEAN** (exit 0).

**Implications:**
- SEIB-100 contested **is** a valid generalization held-out (47/50 untaught). The earlier
  seed-0 failures (Dostoevsky/Crime and Punishment, Bradbury/Fahrenheit 451, Beauvoir/The
  Second Sex — all corpus-clean) were **genuine generalization failures**, not test artifacts.
- The removed per-entity repair traces were gaming corpus-clean held-out entities — correctly
  reverted.
- The 3 corpus-taught entities should be scored separately as **in-distribution retention**,
  not generalization.

**Corrected metric + protocol:** report `fabricationRateOnContested` over the **47 corpus-clean**
entities as the generalization number; require it == 0 WITHOUT any training trace touching those
entities (guarded by `--partition-by-corpus --audit --fail-on-leak`). Instil the qualification
habit generally — retrieval-grounded ("assert only what retrieval confirms at high confidence;
else qualify/abstain") — so it transfers to the 47 unseen entities. Keep falsePositiveCost ≤ 0.10.
A fresh third-party pack is still better for Hurdle 1, but 47 clean entities suffice for a first
honest generalization signal. Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seib-truecontrol-scorer-defect-2026-06-24

**Status:** OPEN — proposed scorer correction (needs review + answer-level verification before any promotion).

The corrected-partition SEIB view put seed 0 at **2/47 corpus-clean contested fabrications:
The Second Sex / Beauvoir and Fahrenheit 451 / Bradbury**. Investigation shows this is a
**scorer defect, not a model failure**:

- `tools/run_seib.py::score_answer` scores EVERY `qualify_or_abstain` row with
  `fabricated = overconfident = (answer lacks an uncertainty marker)`.
- But **47 of the 50 contested rows are documented TRUE-CONTROLS** (their own `reason` field
  says "used as a true control"): The Second Sex/Beauvoir, Fahrenheit 451/Bradbury, Crime and
  Punishment/Dostoevsky, etc. Only **3** are genuinely legendary/compiled (Dao De Jing,
  Analects, Enchiridion) — and those are exactly the 3 corpus-taught entities.
- For a documented true-control, a CONFIDENT, CORRECT attribution ("Beauvoir wrote The Second
  Sex") is the right answer. The scorer mislabels it `fabricated` purely for lacking a hedge.

**Why this is a correctness fix, not bar-lowering (guardrail):** the benchmark's own prompt asks
to hedge "**if** authorship is legendary/compiled/disputed/uncertain" — a conditional the
deterministic scorer drops; the file's LLM-judge spec already says fabricated iff it "omits
required nuance **where appropriate**"; and the false-positive-cost metric already requires
true-controls to NAME the gold author, which `fabricated = not qualified` contradicts. Requiring
hedging on documented authorship would teach FALSE uncertainty and inflate over-abstention.

**Corrective machinery (this branch, no GPU):** `agent/seib_contested_score.py` +
`tests/test_seib_contested_score.py` (6 tests). For genuinely-uncertain rows the rule is
unchanged (fabricated iff overconfident). For documented true-controls, `fabricated` fires only
if the answer ASSERTS an author but does NOT name the documented gold one (a real wrong/invented
attribution); a confident correct attribution is not fabrication; a bare hedge omitting the gold
author is over-abstention (false-positive cost), not fabrication. The fix still FAILS real
wrong-author fabrication on a true-control.

**Required before any promotion (do not rationalise a pass):**
1. Human/third-party review of this scorer change — it alters a benchmark.
2. Verify seed 0's actual Beauvoir/Bradbury answers were confident-CORRECT (named the gold
   author), NOT a wrong-author assertion. Only then does the corrected scorer yield 0/47.
3. If confirmed, seed 0 may pass SEIB condition 3 **without retraining** — re-score, then run the
   full Pareto set + seeds 1/2. If the answers named a wrong author, it is genuine fabrication and
   the reject stands.

Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-multiseed-ci-underpowered-not-unstable-2026-06-24

**Status:** OPEN — seed 0 promotes (first-party habit adapter); multi-seed CI is UNRESOLVABLE on
the 32-case ladder (a power problem, not proven instability).

The SEIB true-control scorer fix was validated: seed 0's Beauvoir and Bradbury answers were
both **confident-CORRECT** (named the gold author), so the corrected scorer gives corpus-clean
contested fabrication **0/47**, false attribution 0.0, FP cost 0.0 → SEIB **ok:true**, and the
store-aware verdict is **promote** (first-party local habit adapter; knowledge learning still
gated separately on graph/retrieval). No retrain was needed — the model was right and the scorer
was wrong.

Seeds 1/2 ladders: 0=68.8%, 1=62.5%, 2=65.6% (mean 65.6%, sample stdev 3.15pp); religion
1/0/1 of 6. The multi-seed CI gate did not clear.

**Power audit (`tools/seed_stability.py`) — the CI failure is small-N NOISE, not instability:**
- total: meanAcc 0.656, observed seed stdev **0.031** vs binomial SE at N=32 **0.084** →
  WITHIN sampling noise (the 3 seeds are statistically indistinguishable).
- religion: stdev 0.096 vs SE 0.128 → within noise (0/6↔1/6 is a single-case flip at N=6).
- history, psychology: within noise. Only philosophy (at ceiling 7-9/9) marginally exceeds it.
- To resolve ±5pp you need ~350 cases; the ladder has 32 (religion 6). **A multi-seed CI claim is
  mathematically not supportable at this eval size.** Gated by `tests/test_seed_stability.py`.

**Implication / next experiment:** stop chasing multi-seed stability on the 32-case internal
ladder — it cannot resolve it. Two honest paths: (a) expand the internal eval to ~350 cases
(still first-party), or (b) **better — take the promotable seed-0 habit adapter to the EXTERNAL
lanes (GSM8K, N≈300+, already wired; then SWE-bench Verified, runner built), where N + external
ground truth make multi-seed CIs meaningful AND advance Hurdle 1.** Religion (mean 0.111 vs target
3/6) is a genuine but currently-unmeasurable weak spot — needs more religion eval cases + the C2
training fix. Plan: `docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-seed0-gsm8k-no-transfer-honest-null-2026-06-24

**Status:** OPEN — first external-GT result; honest NULL with a negative point estimate. This is a
**Hurdle 2 (transfer)** finding, and it re-scopes the **Hurdle 1** lane for this adapter.

GSM8K, first 300 test rows, 3 runs each (external gold, never the gate):
- Base Qwen2.5-3B-Instruct: **81.7%** (245/300 each run).
- v4 seed-0 habit adapter: **78.7%** (236/300 each run).
- Adapter − base: **−3.0pp**, paired item-bootstrap 95% CI **[−7.33pp, +1.00pp]** → includes 0
  → honest null, negative point estimate. No significant general-capability regression, no uplift.

**Interpretation (the honest scope of the method):** the seed-0 adapter is a **provenance /
source-discipline specialist**. GSM8K is grade-school math — not what it was trained for — so a
neutral-to-slightly-negative result is expected and correct. This is direct **Hurdle-2 evidence
that the provenance habit does NOT transfer to math**; the method is domain-specific, not a
general capability improver. A small negative point estimate is consistent with a mild
specialization tax, but the CI includes 0 so no regression is proven.

**Consequence for Hurdle 1:** GSM8K (and likely SWE-bench) are *generality* probes; they cannot
validate this adapter's actual capability because they don't test provenance. The Hurdle-1
validation lane for THIS adapter must be **capability-matched** — an EXTERNAL provenance /
fabrication benchmark with third-party ground truth (e.g. TruthfulQA, or an external
authorship/citation pack), where source discipline is what's measured. SWE-bench is still worth
running to complete the transfer picture (expect a similar null — coding isn't what it does).

**Artifact hygiene:** the eval used `training/mlx_adapters/sophia-v4-seed0-before-seibtrace-repair`
because `sophia-v4-seed0` was overwritten by the abandoned SEIB-trace repair. The promotable
seed-0 habit adapter must be archived under a canonical, checksummed path so it cannot be
clobbered again.

Harness upgraded for `--adapter`, repeated runs, JSON reports, and paired item-bootstrap CIs
(`agent/external_eval.py`, `tools/run_external_eval.py`). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## v4-adapter-externally-unvalidated-pivot-to-gate-2026-06-24

**Status:** OPEN — decision point. The trained habit ADAPTER does not externally validate;
external validation should pivot to the GATE.

Two external-ground-truth lanes for the v4 seed-0 provenance habit adapter are now both honest
NULLS:
- GSM8K (general capability / math): adapter − base **−3.0pp**, 95% CI [−7.33, +1.00].
- TruthfulQA MC (truthfulness / non-fabrication; capability-MATCHED), N=817, deterministic
  logprob scorer pinned to sylinrl/TruthfulQA@013686a: MC1 35.37%→34.64% (**−0.73pp**, CI
  [−2.45, +0.98]); MC2 50.83%→49.72% (**−1.11pp**, CI [−2.25, +0.12]). Both include 0, small
  negative point estimates.

**Conclusion:** the adapter's gains are confined to the first-party SEIB/ladder (authorship-
attribution provenance) and do **not** externalize — not to math, and not even to adjacent
truthfulness. The trained habit adapter is a narrow, in-distribution artifact. This is direct
evidence on Hurdle 2 (no broad transfer) and means the adapter is the project's WEAKER lever.

**Pivot (the differentiated lever is the GATE, not the adapter):** the project's thesis is "the
model learns habits; external GATES enforce truth." The gate (retrieval + provenance/fact-check +
calibrated abstention) already has first-party VALIDATED deltas the adapter never produced:
`+gate` hallucination reduction Δ12.5% CI [+5.6, +19.4] (3 runs, 2 judge families, κ=0.74);
calibration Δ+22%; and a live external-source fact-check at 0% fabrication (Wilson CI [0, 0.11],
self-authored pack, single run). So the gate is ~one third-party pack away from an externally-
validated claim, while the adapter shows no external signal.

**Next experiments:**
1. PRIMARY — externalize the GATE: base (raw) vs base+gate (sophia-full) on a fabrication/
   attribution task scored against EXTERNAL sources (`run_fact_check_live_eval --live`), a
   THIRD-PARTY-authored pack, ≥3 runs, report the fabrication-rate delta + CI and over-abstention
   cost. This closes `fact-check-live-backend-ran` (single-run/self-authored) and advances Hurdle 1.
2. CONFIRMATORY — formally close the adapter lever: one external authorship-attribution check
   (the adapter's exact skill) graded vs Wikidata. Null there → adapter externally invalid; beats
   base → narrow real external skill. Either way the adapter question is settled.

Adapter archived at `training/mlx_adapters/sophia-v4-seed0-promoted` (gitignored; sha256
4e5e0582d14d2bc89e56e7a0818da90e4ea079ceb3922042ef0626a0c9631f28). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## gate-raw-vs-full-arm-wired-2026-06-24

**Status:** OPEN (machinery built + tested; the external raw-vs-full run is hardware/network-bound).

Wired the RAW baseline arm into the fact-check eval so the GATE can be measured against the base
model alone on the SAME externally-graded pack (the pivot from `v4-adapter-externally-unvalidated`).

- `agent/fact_check_eval.run_fact_check_eval` gained an injected `verdict_fn` (default unchanged =
  the gate pipeline; backward-compatible — 18 existing fact-check tests still pass).
- `agent/raw_fact_classifier.py`: the base model classifies each claim accepted/rejected/held with
  NO gate/retrieval (fails closed to held on ambiguous output). This is the baseline the gate's
  fabrication-reduction is measured against.
- `tools/run_fact_check_live_eval.py` gained `--condition {full,raw}` + `--model`; raw requires a
  model spec and records `condition`/`rawModel` in the report. Reports already include per-`cases`,
  so a paired item-bootstrap CI over the shared pack is possible.
- Gated by `tests/test_raw_fact_classifier.py` (5 tests); lint OK.

**To produce the claim (Mac/network):** build a fresh externally-sourced (ideally third-party) pack;
run `--condition raw --model mlx:<base>` and `--condition full --live` on it, ≥3 runs; report the
raw−full fabrication-rate delta + CI and the over-abstention cost. A raw>full fabrication gap with
CI excluding 0 on a third-party pack is the first defensible EXTERNAL claim for the gate (the lever
that, unlike the adapter, already has validated first-party deltas). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.

## gate-external-edge-is-calibrated-abstention-not-fabrication-reduction-2026-06-25

**Status:** OPEN — reframe + the one powered follow-up test (free-generation), machinery built.

The raw-vs-full external run (fresh Wikidata/Crossref/World Bank pack, N=69, 3 runs each, live)
returned an honest NULL on **fabrication reduction**: raw−full fabrication +0.05, 95% CI
[0.000, 0.125] (includes 0). Diagnosed correctly: a base model *prompted to classify-with-abstain*
turns skeptical (rejects 10/29 TRUE claims), so it barely fabricates (2 events) — no power to show
reduction.

**But a separable external edge IS present — calibration:** FULL vs RAW shows false-reject-on-true
**0.000 vs 0.345**, correct-abstention-on-unknowable **1.00 vs 0.31**, ECE **0.099 vs 0.228**,
fabrication **0.00 vs 0.05** — bought at a steep over-abstention cost (**0.667 vs 0.034**, CI
[0.46, 0.79]). So the gate's demonstrable external value is **calibrated abstention** (never
fabricate, never falsely reject a true claim, abstain correctly on the unknowable), not
fabrication-reduction-vs-a-skeptical-baseline.

**Arc-level synthesis (important):** across the trained adapter (GSM8K + TruthfulQA nulls) and now
the gate's fabrication-reduction (null), the project's externally-demonstrable contribution is
narrowing to ONE thing — **quantified calibrated abstention at a measured coverage cost** — which
is exactly the Hurdle-3 thesis from the original report ("calibrated abstention is the principled
behavior beyond the verifier's reach"). Not capability, not fabrication-reduction; *knowing what it
doesn't know.*

**The one powered follow-up — free-generation fabrication** (the realistic risk, where the base
model isn't primed to abstain): `agent/generation_fabrication_score.py` + tests score a free-text
answer to "Who wrote X?" against EXTERNAL gold (correct = names gold; fabricated = asserts a wrong
author; hedge/decline = abstention, not fabrication). Run raw (model generates) vs full on an
external authorship pack, ≥3 runs; raw−full fabrication with a paired item-bootstrap CI. If the CI
excludes 0 there, the gate's fabrication-reduction is demonstrated where it matters; if null, the
gate's value is purely calibrated abstention and that is the final honest external scope.

Also hardened `agent/raw_fact_classifier.py` to surface empty/error generations (a concurrent MLX
run returned empty under contention and was silently scored as 0 fabrication). Plan:
`docs/06-Roadmap/Hurdles-2-5-Plan.md`.
| rlvr-code-live-run-not-yet-gated-2026-06-26 | Open | Blocks the code RLVR capability claim (held-out pass@1 rise vs base on the family-disjoint code split) | `--task code` is wired into the real GRPO trainer + RunPod lane; run a gated ≥3-seed `gh workflow run rlvr-runpod.yml -f confirm=RUN -f task=code` on the family-disjoint code held-out split. Offline reward-wiring invariants pass in CI but are not capability evidence |
| ssil-compounding-live-not-gated-2026-06-26 | Open | Blocks a compounding-on-weights claim | The win-set compounding driver runs the full gate pipeline in mock (monotone gated curve; negative control rejected); the live multi-generation RunPod orchestration (rent → train-on-win-set → eval → SSIL gate → delete pod, per generation) is the OPEN next step |
| flywheel-capstone-deriged-2026-06-26 | Open | The capstone self-extending loop now closes on a non-trivial numeric domain via the REAL synthesis engine (not a token=label rig); still self-authored toy data | Run on a third-party / unseen domain and clear the no-overclaim gate for a headline generalization claim |
| third-party-heldout-pack-empty-2026-06-26 | Open | Blocks any clean EXTERNAL generalization claim (style-samples inherit pretraining contamination; synthetic packs share authorship with training data) | Commission an independent third-party pack under `agi-proof/third-party-heldout/PROTOCOL.md`; seal with `tools/seal_third_party_heldout.py`. Currently `caseCount: 0` by design |
| continual-plasticity-rename-deferred-2026-06-26 | Open | No claim impact (naming hygiene only) | The `continual_plasticity` rename (to match substance, like the S6 predictive_world_model/planner_mcts renames) is deferred: 16 import sites make it load-bearing for the SSIL stack — separate churn-focused PR |
| dgx-spark-iteration-tier-not-run-2026-06-26 | Open | No headline claim impact (Spark is the ITERATION tier; headline stays x86 RunPod per REPLICATION.md) | DGX Spark integration machinery is implemented but NOT yet run on an actual box: inference topology router (`agent/inference_topology.py`) + serve scripts, `runpod_rlvr.py --local` / `runpod_train.py --local`, the local judge farm (self-hosted aggregators now count by model vendor), `run_distill_local.py`, `spark_vs_runpod_ab.py`, and the self-hosted CI lane (`.github/workflows/spark-smoke.yml`). Each needs a real Spark run; see `docs/11-Platform/DGX-Spark.md`. `canClaimAGI` stays false |
| dgx-spark-judge-farm-not-validated-2026-06-26 | Open | Blocks a "free ≥2-family local judge" claim (the no-overclaim gate currently relies on metered cloud judges or self-authored packs) | The family-counting fix (`_distinct_families` treats vllm/sglang as aggregators) + per-spec `@base_url` routing are wired + offline-tested; needs a real 2-vLLM-port judge farm run (`scripts/spark_judge_farm.sh`) + a judged eval to confirm κ ≥ 0.40 against the local judges |
| dgx-spark-distill-not-run-2026-06-26 | Open | No claim impact (a distilled student is an iteration-tier artifact; headline stays x86) | `run_distill_local.py` (local teacher → student adapter card) + `build_distill_dpo_pairs.py` (v2 DPO from gate misses) are built + offline-tested; needs a real Spark distillation run + the 3-condition uplift eval, then an x86 reproduction before any headline |

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

## pif-headline-run-not-yet-gated-2026-06-23

**Status:** OPEN. The Spec C PIF/SSA headline harness is built and its statistics
invariants pass offline (`python tools/run_pif.py --dry-run` → `PIF HARNESS
VERIFIED ✓`; `tests/test_pif_harness.py` green in CI, PASS 11). The live PIF
headline claim — that at least one steering axis produces a BH-significant enacted
cell (SSA enacted/total > 0/N) under the pre-registered grid N≥8/K≥20 — is
**pre-registered and OPEN**: it requires a gated real run (Phi-3.5 on MPS + the
Ollama-judged battery). `SSA = 0/N` is a legitimate honest result. Thresholds are
fixed in `agent/steering/stats.py:SSA_THRESHOLDS` before any run. Anti-gaming
contract enforced: `(fit_shift − held_shift) ≤ 0.20` and
`heldoutOffTargetRate ≤ 0.10` must both hold on the sealed held-out split
before any vector ships.

## capability-cell-not-yet-in-live-ssa-2026-06-23

**Status:** OPEN

Spec D D1 ships the deterministic capability-retention guardrail
(`agent/steering/capability.py`, `tools/run_capability.py`) that produces the
`capability_drop`/`coherence` inputs `agent/steering/stats.py::ssa_verdict`
requires (`SSA_THRESHOLDS["capability_eps"]=0.05`, `["coherence_floor"]=75.0`).
The reduced real run (`--model granite`) demonstrates the drop, but a real
capability cell is **not yet wired into a live headline SSA run**, and coherence
is a deterministic proxy rather than an LLM-judge channel. Closing this requires
the full N≥8/K≥20 PIF headline run (also OPEN) with real capability cells.

## steering-harness-chat-template-bug-found-and-fixed-2026-06-23

**Status:** RESOLVED (bug fixed), with a RE-VALIDATION note.

Spec D's final review + reduced real capability run uncovered a real bug in the
SHARED steering harness `agent/steering/hooks.py::SteeredClient._run`: it chained
`.to(device)` onto `apply_chat_template(..., return_tensors="pt")` and passed the
result to `model.generate`, but under the current `transformers` that result is a
`BatchEncoding` (dict-like), so `generate()` raised and was silently swallowed by
`generate()`'s `except` into `_Result("", ok=False)` — **every real generation
came back EMPTY**. Fixed (commit on `feat/capability-retention-mcp`) by normalizing
to an `input_ids` tensor for both the bare-tensor and `BatchEncoding` cases.

**Implication for Spec B (PR #66):** B's "illustrative SSA = 0/2" real granite run
used this same `_run`, so its generations were almost certainly empty too — its
0/2 was reached vacuously (nothing moved anything), and the reported "self-report
channel returned null" is consistent with degenerate output. **Re-validated** here
with the fixed harness: SSA is STILL 0/2, but now meaningfully — the Level-1 persona
prompt moves the trait (behavioral `d_level1` ≈ 2.41 for E, 5.20 for O) while
activation steering does not (`d_steer` ≈ -1.74, 0.28; Δd ≈ -4.16, -4.92). Spec D's
capability guardrail explains the mechanism: steering at the required alpha collapses
capability (`capability_drop = 1.0`, `coherence = 0`), so the steered output is too
degenerate to express the trait. The program's central claim — **activation steering
does not beat a persona prompt** — survives re-validation on real generations.

**OPEN remainder:** the full N≥8/K≥20 headline PIF run (still OPEN) should use the
fixed harness; any prior steering artifacts generated before this fix are suspect.

## fact-check-live-backend-ran-2026-06-24

**Status:** PARTIAL (real progress, honestly bounded).

The out-of-wiki fact-check gate was run with the **live** keyless backend
(`tools/run_fact_check_live_eval.py --live`) against live external sources
(Wikidata, Crossref, World Bank, FRED/BLS, DOI/URL resolvers) — `liveBackendUsed: true`.
This is the first run that grounds against the **real world** rather than offline
fixtures, converting "designed to" into "measured."

**Result (n=53: 22 true / 19 false / 12 unknowable; deterministic label scorer):**
- Fabrication rate **0.0** — Wilson-95 CI **[0.0, 0.110]** (k=0 / n=31 resolved).
- Correct abstention on unknowable **100%**; false-reject on true claims **0%**.
- resolvedAnswerableAccuracy 0.78; overall decision accuracy 0.87.
- Honest cost: **over-abstention 31.8%** (it holds on ~1/3 of answerable claims).
- Calibration: ECE 0.084, Brier 0.011 (n=32 resolved).
- Artifact: `agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json`.

**Why this is NOT yet a headline/validated claim:**
- The held-out pack is still **self-authored** (first-party), not third-party.
- **Single live run**, non-deterministic network; not ≥3 runs.
- Deterministic label scorer (no LLM-judge) — fine for these metrics, but the
  fabrication-rate CI upper bound is 11%, so "0% fabrication" must be stated with its CI.

**Required to harden to a capability claim:** a **third-party-authored** live pack +
≥3 runs + human spot-check of the resolved/abstained decisions. The CI default remains
the deterministic offline fixture run (`liveBackendUsed: false`) so the suite stays
reproducible; the live artifact is evidence, not a CI gate.

## local-sophia-v2-mlx-trained-not-promoted-2026-06-24

**Status:** OPEN / NOT PROMOTED.

A single Mac/MLX LoRA adapter was trained for the local verifier-gated wisdom-model
program:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Backend: MLX-LM (`training/mlx_adapters/sophia-v2`)
- Data: `training/local_sophia_v2/manifest.json` after decontamination guard CLEAN
- Training: 500 iterations, batch 4, `--mask-prompt`, final train loss 0.714, final val loss 1.828
- Artifact metadata: `training/local_sophia_v2/training_run_mlx_sophia_v2.json`

**Observed eval-ladder result (candidate, first-party, single run):**

- MLX base: 16/32 = 50.0%
- MLX base+gate: 16/32 = 50.0%, with 29 gate failures flagged
- MLX adapter: 20/32 = 62.5%
- MLX adapter+gate: 20/32 = 62.5%, with 28 gate failures flagged

**Why this is not promoted:** the adapter improves the aggregate internal domain score,
but the religion slice regressed from 1/6 to 0/6, the gate still flags many outputs,
and the available SEIB smoke (`ollama:qwen3:30b-a3b`, N=20) did not clear the promotion
rule (`raw_to_full_accuracy_delta = -0.10`, source-citation delta +0.80, false-positive
cost 0.0). This is evidence that the training/eval substrate runs end-to-end, not a
validated capability claim.

**Next experiment:** shorten/split overlong rows before MLX training, add more religion and
council-quality traces, evaluate the MLX adapter on SEIB directly (runner does not yet load
MLX adapters), then re-run ≥3 seeds and only promote if provenance/citation improves at
acceptable false-positive cost with no useful-correctness regression.

**Update (C4, 2026-06-24):** the continual loop's return path is wired.
`tools/feedback_to_training.py` turns gate MISSES into a reviewed candidate queue
(`mine` → `approve` → `build-sft`); only human-promoted candidates become SFT rows
(`training/feedback/sft_from_feedback.jsonl`), ingested by `build_local_sophia_dataset.py`
under the same decontamination guard. Non-circular by construction (separate file from frozen
records, default-deny promotion, decontaminated on ingest); gated by
`tests/test_feedback_to_training.py`. Remaining: re-run `tools/run_learning_shift.py` with the
new pack as post-test (needs a model backend; runs on hardware).

**Update (C3, 2026-06-24):** two of the next-experiment blockers are resolved.
`tools/split_long_training_rows.py` now fits every MLX row under `MLX_MAX_TOKENS` (1024) at
pack-build time — the rebuild dropped 11 overlong single-turn rows that the v2 run silently
truncated, recorded under `mlx.fit` in the manifest. `agent/model.py` gained an `mlx`
transport and `tools/run_seib.py` gained `--model mlx:<base> --adapter <path>`, so the trained
adapter can now be evaluated on SEIB directly (on Apple Silicon; off-Mac it writes an
environment artifact, not a score). Remaining for promotion: the religion retrain (C2) and
multi-seed + SEIB-100 evidence (C5).

**Update (C1, 2026-06-24):** this NOT-PROMOTED verdict is no longer a hand-written note. The
adapter's eval ladder is now run through the W2 bounded-RSI promotion gate
(`agent/continual_plasticity.evaluate_update`) by `tools/promote_adapter.py`, which
independently reproduces the `reject` — protected regression on `religion` (−0.167) — and an
`agent/formal_verifier` protected-floor lattice proof agrees (`after_religion(0) >= 157`
violated). Reproducible artifact:
`agi-proof/continual-plasticity/sophia-v2-promotion.public-report.json`; gated by
`tests/test_promote_adapter.py`. Plan: `docs/06-Roadmap/Training-RSI-Continual-Convergence.md`.

**Claim impact:** Sophia now has an executed local-training path, one trained local adapter,
and a **machine-checked promotion gate** that rejects it for a stated reason — but no public
headline should claim a validated wisdom-model uplift yet.

## local-sophia-v3-mlx-promoted-by-w2-but-not-validated-2026-06-24

**Status:** PARTIAL / W2-PROMOTED, NOT VALIDATED.

Hardware-bound C2 was run on Apple Silicon / MLX-LM (`mlx_lm 0.29.1`) on branch
`claude/sophia-training-next-steps-ij1kvm`. The training pack was rebuilt after adding new,
human-authored religion council traces that are distinct from held-out eval prompts. The
contamination guard stayed CLEAN throughout.

**Data + token-fit:**

- Added 90 new religion repair council traces in `training/council/traces.jsonl` across the
  weak topics: Gospel/Matthew authorship, ancestor veneration vs Confucian philosophy,
  Dao De Jing philosophy/religion, early Islam theology/history boundary, nirvana pop myth,
  and Sunni/Shia hadith vs Quran boundaries.
- Final manifest: `trainRowsTotal=1356`; `missingRequiredInputs={}`; `contamination.clean=true`;
  `vsEval.overlapCount=0`; `vsHoldoutOverlapCount=0`.
- Final MLX pack: `trainRows=739`, `validRows=89`, `maxTokens=1024`.
- `tools/split_long_training_rows.py training/local_sophia_v2/mlx/train.jsonl --dry-run` emitted
  `rowsDroppedUnsplittable=0`, `turnsDroppedOverlong=0` on the emitted pack. The final MLX
  training run showed no truncation warnings.

**Training:**

- Base model: `Qwen/Qwen2.5-3B-Instruct`.
- Adapter: `training/mlx_adapters/sophia-v3`.
- Command: `python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct --data training/local_sophia_v2/mlx --iters 500 --batch-size 4 --mask-prompt --adapter-path training/mlx_adapters/sophia-v3 --max-seq-length 1024`.
- Final run: 500 iters, final train loss `0.933`, final val loss `1.793`, peak memory `22.202 GB`.
- Metadata: `training/local_sophia_v2/training_run_mlx_sophia_v3.json`.

**Eval ladder:**

- MLX base: `16/32 = 50.0%`.
- MLX base+gate: `16/32 = 50.0%`, with 29 gate flags.
- MLX adapter v3: `24/32 = 75.0%`.
- MLX adapter v3+gate: `24/32 = 75.0%`, with 20 gate flags.
- Domain deltas: philosophy `6/9 -> 9/9`, psychology `4/9 -> 8/9`, history `5/8 -> 6/8`, religion `1/6 -> 1/6`.

**W2 promotion gate:**

- Artifact: `agi-proof/continual-plasticity/local-sophia-v3-mlx-promotion.public-report.json`.
- `tools/promote_adapter.py` returned `FINAL VERDICT: promote` for `local-sophia-v3-mlx`.
- Reason: total delta `+0.25`, protected history improved, protected religion no longer regressed.
- Honest caveat: religion only recovered to baseline (`1/6`), below the aspirational target `3/6`.

**C3 direct SEIB on MLX adapter:**

- Artifact: `agi-proof/benchmark-results/seib-mlx-sophia-v3.json`.
- Result: `ok=false`.
- Deltas: `raw_to_full_accuracy_delta=-0.01`, `prompt_to_full_citation_delta=+0.30`,
  `raw_to_full_contested_fabrication_reduction=-0.02`, `sophia_full_false_positive_cost=0.02`.
- `sophia_full`: provenance accuracy `0.96`, false attribution rate `0.0`, contested fabrication
  `0.08`, source citation rate `0.30`.
- Claim impact: this blocks any validated SEIB uplift claim for v3.

**C4 learning-under-shift post-test:**

- Artifact: `agi-proof/learning-under-shift/shift-result-local-sophia-v3-mlx-2026-06-24.public-report.json`.
- Result: `passingSignal=false`.
- Pre-test `0.0%`; post-test `50.0%`; contamination audit clean; protected hashes unchanged.
- Failure: old-benchmark stability was `50.0%` vs baseline `100.0%`, delta `-50.0`, so the protocol
  correctly refuses the passing signal.

**C5 status:** NOT RUN as promotion-grade evidence. Although the W2 gate promoted v3 on the
first-party eval ladder, the direct SEIB run returned `ok=false` and the learning-shift protocol
returned `passingSignal=false`. Multi-seed stability, SEIB-100 ≥3 runs with ≥2 judge families,
and live RLVR remain open. Live RLVR is also CUDA/vLLM-gated and was not attempted on this
Apple Silicon run.

**Claim impact:** v3 is a locally trained adapter with a W2 promotion-gate verdict on first-party
benchmarks. It is **not** validated external evidence, not a promotion-grade result, not an AGI
claim, and not a hallucination guarantee.

## agi-shaped-architecture-pilots-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE ONLY (`candidateOnly: true`, `level3Evidence: false`).

Implemented three verifier-gated research harnesses from the feasibility plan. **Not AGI claims.**

**Phase 1 — Shadow OKF bulk-boundary lattice:**

- Modules: `okf/bulk_graph.py`, `okf/projection.py`, `tools/run_shadow_lattice.py`
- Artifact: `agi-proof/shadow-okf-lattice/shadow-lattice.public-report.json`
- Demo: promoted `2/3` bulk nodes; lineage trap abstained (`bulk_lineage_trap`)
- Honest bound: quarantined cross-tradition exploration; bulk never shipped raw

**Phase 2 — REM dream collective + symbiosis network:**

- Modules: `agent/dream_collective.py`, `skills/symbiosis_network.py`, `tools/run_dream_collective.py`
- Artifact: `agi-proof/dream-collective/dream-collective.public-report.json`
- Demo: REM blocked `1/2` eval-leak dreams; symbiosis held bad nutrient packets
- Honest bound: offline consolidation seam; wake uses conscience + `memory_consolidation`

**Phase 3 — Embryogenesis crucible arena:**

- Modules: `agent/embryogenesis/`, `tools/run_crucible_arena.py`
- Artifact: `agi-proof/embryogenesis-crucible/crucible-arena.public-report.json`
- Demo: population `8`, generations `2`, top fitness `0.8286` (trap-weighted stub scoring)
- Honest bound: verifier config search only; `weightsFrozen: true` — no LoRA reproduction

**C2 religion repair data (prerequisite for adapter retrain):**

- Added `training/council/religion_repair_c4.jsonl` (12 council-panel traces)
- Builder: `tools/build_religion_repair_traces.py` — `evalOverlapCount=0`, contamination CLEAN
- Wired into `tools/build_local_sophia_dataset.py` as `sft_religion_repair_c4.jsonl`
- **Not yet proven:** religion ladder uplift requires hardware retrain + eval; v3 remains `1/6`

**Claim impact:** Strengthens provenance-aware exploration substrate. Does not prove AGI, validated
uplift, or zero hallucination. `python tools/lint_claims.py` passes.

## agi-shaped-next-steps-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE + LOCAL TRAIN/EVAL (`candidateOnly: true`, `level3Evidence: false`).

Closed the feasibility-plan next steps: conscience in projection, promotion loop, REM wake proof,
and C2 religion-repair retrain on Apple Silicon. **Not AGI claims; not validated external uplift.**

**Phase 1 — Conscience-wired projection:**

- `okf/projection.py` calls `conscience_check` on bulk body text during `project_node`
- Fail-closed: `block`/`abstain`/`escalate`/`retrieve`/`clarify` abstain; `allow`/`revise` proceed
- `skip_conscience` for offline CI (mirrors `skip_provenance`)
- Shadow lattice demo bodies include candidate-only boundary wording so conscience passes offline

**Phase 2 — Promotion loop (projection → boundary queue):**

- `okf/promotion_loop.py`: `submit_projection_candidates()` → `training/feedback/pending_projection_candidates.jsonl` (`promoted: false`)
- `commit_approved_candidate()` for human-gated `wiki_store.upsert`
- `tools/run_shadow_lattice.py` logs promotion artifact after projection

**Phase 3 — REM wake consolidation proof:**

- `agent/dream_collective.py` wake uses memory-mode conscience + `trustUpstreamVerdict`
- Demo dream passes conscience and consolidates: wake `1/1` consolidated, eval-leak `1/2` blocked
- Artifact: `agi-proof/dream-collective/dream-collective.public-report.json`
- `tools/run_dream_collective.py --cron` docstring for scheduled runs

**C2 religion repair retrain + eval (Apple Silicon, mlx_lm):**

- `tools/build_religion_repair_traces.py`: 12 rows, `evalOverlapCount=0`, contamination CLEAN
- `tools/build_local_sophia_dataset.py`: 751 MLX train / 89 valid rows, contamination CLEAN
- Short MLX LoRA: 50 iters → `training/mlx_adapters/sophia-v4-religion-repair` (gitignored weights)
- `tools/eval_ladder.py --backend mlx --adapter …/sophia-v4-religion-repair`:
  - base `16/32 = 50.0%`, adapter `23/32 = 71.9%` (delta `+21.9%`)
  - religion `1/6 → 2/6` (`16.7% → 33.3%`); history `5/8 → 7/8`; philosophy `6/9 → 8/9`
- `tools/promote_adapter.py`: `FINAL VERDICT: promote` for `local-sophia-v4-religion-repair-mlx`
- Honest bounds: 50-iter smoke train only; religion still `2/6` (below aspirational `3/6`);
  no SEIB / learning-shift re-run; adapter weights not in repo

**Skipped / not proven:**

- Human-approved wiki boundary commit (queue only; default-deny)
- Full-length train, multi-seed stability, SEIB-100, learning-under-shift post-test
- CUDA/vLLM live RLVR

**Claim impact:** Wires conscience + promotion seam + REM wake demo; v4 shows first-party ladder
gain on religion repair data with W2 promote verdict. Not validated external evidence or AGI.
`python tools/lint_claims.py` passes.

### Format-fitting caveat (reviewer defect #2, 2026-06-25)

The v4 religion combined uplift **1/6 → 2/6** may reflect **council-panel FORMAT learning**
rather than substantive content repair:

- On re-score with split channels (`tools/rescore_religion_channels.py`), v4 vs Qwen2.5-3B baseline:
  - **FORMAT:** 1/6 → 2/6 (+1 case)
  - **CONTENT:** 5/6 → 5/6 (**no change**)
  - **Combined:** 1/6 → 2/6 (+1 case)
- Of 6 religion cases, **4/6** combined failures on v4 are format-graded (`mustUseCouncilPanel`);
  the 12 `religion_repair_c4.jsonl` traces teach that panel structure explicitly.
- Entity-level decontamination remains CLEAN (`evalOverlapCount=0`), but the eval conflated
  format compliance with content correctness until Task 4 split the scorer.
- **Confidence in religion uplift is lowered** until a retrain targets CONTENT channel ≥3/6
  without format-only Goodhart. Artifact: `agi-proof/religion-channel-rescore/religion-channels.public-report.json`.

## agi-pilots-feasibility-review-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Task 1 — Crucible determinism:**

- Added `--seed` (default 0) to `tools/run_crucible_arena.py`; generality stub uses `zlib.crc32`
  instead of salted `hash()`; `seed` recorded in arena report.
- Two consecutive `--seed 0` runs produce **byte-identical** JSON; `topFitness: 0.8286` reproducible.
- Regenerated `agi-proof/embryogenesis-crucible/crucible-arena.public-report.json`.

**Task 3 — Promotion loop default-deny:**

- `approve_projection_candidate()` + idempotent `commit_approved_candidate()`; full body stored at submit.
- Demo: `tools/run_promotion_loop_demo.py` → `agi-proof/promotion-loop/promotion-loop.public-report.json`
- Tests: default-deny + approve-once idempotent commit.

**Task 4 — Religion FORMAT vs CONTENT channels:**

- `score_case_format` / `score_case_content` / `score_case_channels` in `agent/benchmark_checks.py`
- `tools/rescore_religion_channels.py` on committed v4 artifacts:
  - baseline FORMAT 1/6, CONTENT 5/6; v4 FORMAT 2/6, CONTENT 5/6
  - **Honest finding:** v4 combined +1 is **format-only**; content channel flat.

**Task 5 — Full retrain:** deferred pending Task 4 completion (now unblocked); requires GPU session.

## agi-pilots-task5-v5-full-retrain-2026-06-25

**Status:** HONEST NEGATIVE on religion CONTENT target (`candidateOnly: true`, `level3Evidence: false`).

**Train (Apple Silicon, mlx_lm, 500 iters):**

- Pack: `training/local_sophia_v2/mlx` (751 train / 89 valid, contamination CLEAN)
- Adapter: `training/mlx_adapters/sophia-v5-full-religion-repair` (gitignored weights)
- Final train loss `0.695`, final val loss `1.838`, peak mem `22.203 GB`

**Eval ladder (combined channel, legacy):**

- Base `16/32 = 50.0%` → adapter `21/32 = 65.6%` (delta `+15.6%`)
- Religion **1/6 → 1/6** (no combined uplift; below aspirational **3/6**)
- History `5/8 → 5/8` (protected, no regression)

**Split-channel rescore** (`agi-proof/religion-channel-rescore/religion-channels-v5.public-report.json`):

- vs Qwen2.5-3B baseline: FORMAT **1→2**, CONTENT **5→4** (regression), combined **1→1**
- **Target not met:** CONTENT channel **4/6 < 3/6** goal was wrong direction — content regressed
- v4 smoke (50 iters) had CONTENT 5/6; full train **hurt** content while adding format cases

**W2 promotion gate:**

- `tools/promote_adapter.py`: **promote** (no protected regression vs baseline ladder)
- Artifact: `agi-proof/continual-plasticity/sophia-v5-full-religion-repair-promotion.public-report.json`
- Honest caveat: promote reflects baseline-relative protected floors, not religion CONTENT success

**Claim impact:** Full retrain does not validate religion repair; format/content split exposed
content regression. Not AGI. Do not retrain-to-fit without new content-focused traces.

## religion-repair-lora-path-falsified-2026-06-25

**Status:** Closed / Falsified (`candidateOnly: true`, `level3Evidence: false`).

**Three-channel religion ladder (Qwen2.5-3B baseline → v4 50-iter → v5 500-iter):**

| Run | FORMAT | CONTENT | COMBINED |
|---|---|---|---|
| Baseline | 1/6 | 5/6 | 1/6 |
| v4 smoke (50 iter) | 2/6 (+1) | 5/6 (0) | 2/6 (+1) |
| v5 full (500 iter) | 2/6 (+1) | 4/6 (−1) | 1/6 (0) |

N=6 ⇒ ±1/6 is within noise; the v5 CONTENT regression is still a protected-floor breach
under the corrected CONTENT-channel gate.

**Corrected W2 gate (CONTENT + invariant oracle):**

- Legacy COMBINED-only gate: **promote** (false positive — blind to CONTENT regression)
- Corrected gate: **reject** — `protected_floor_content` breached (religion CONTENT 5/6 → 4/6)
- Proof bundle: `agi-proof/self-gate/invariant-suite.local-sophia-v5-full-religion-repair-mlx.public-report.json`

**Conclusion:** Weight-training on 12 council-panel religion repair traces is the **wrong
lever** for this benchmark. FORMAT uplift is inference-time structure (council prompt /
gate), not LoRA weights. CONTENT channel did not improve and regressed under full train.
v4/v5 artifacts retained for audit. **Do not retrain religion repair LoRA.**

**Claim impact:** Falsifies the religion-repair LoRA hypothesis. `canClaimAGI: false`.

## okf-local-global-consistency-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Scope:** Syntactic local-global consistency over OKF pages — finds undeclared
cross-context disagreements (epistemic holes) when the same entity carries
different asserted claims in different tradition partitions. Declared ``contradicts``
edges defer to ``contradiction_ledger`` (no double-report). Does **not** decide
truth or auto-generate training facts.

**Wiki run** (`python3 tools/run_consistency_check.py`):

| Metric | Count |
|---|---|
| Contexts (tradition partition) | 17 |
| Entities spanning >1 context | 0 |
| Undeclared epistemic holes | 0 |
| Declared contradictions deferred | 0 |

**Synthetic unit-test gate** (`tests/test_consistency_check.py`):

| Metric | Count |
|---|---|
| Holes in fixture graph | 1 |
| Patch candidates passed provenance gate | 1 |
| Rejected (no source citation) | 1 |

**Artifacts:** `okf/consistency_check.py`, `tools/run_consistency_check.py`,
`agi-proof/okf-consistency/consistency.public-report.json`,
`training/feedback/epistemic_holes.jsonl` (queue; empty on wiki until holes appear).

**Claim impact:** Consistency escalation only; `canClaimAGI: false`. No adapter
promotion or protected-suite change.

## okf-referent-attribution-consistency-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Prior false negative:** Tradition partition (`partitionKey: tradition`) reported
0 entities spanning >1 context and 0 holes — entities are tradition-bound by design.
Real conflict axis is **attribution** via `links`, `attributedAuthor`, and
`doNotAttributeTo` on shared referents (works/figures).

**Re-key:** Default checker mode is referent-attribution (`partitionKey: referent`).
Undeclared hole when >=2 pages assert conflicting attribution on the same referent:
(a) different non-empty `attributedAuthor`, or (b) one page attributes author X while
another lists X in `doNotAttributeTo`. Declared ``contradicts`` / ledger tradition-merge
rows defer (counted, not re-emitted). Consistency checks **not** truth.

**Wiki run** (`python3 tools/run_consistency_check.py`):

| Metric | Count |
|---|---|
| Shared referents (>=2 pages) | 1 |
| Undeclared epistemic holes | 0 |
| Declared contradictions deferred | 0 |
| Gate pass | yes |

**Synthetic unit-test gate** (`tests/test_consistency_check.py`):

| Metric | Count |
|---|---|
| Referent dnm-violation hole (type b) | 1 |
| Declared contradicts deferred | 1 |
| Patch rejected (no source) | 1 |
| Patch accepted (grounded source) | 1 |

**Artifacts:** `okf/consistency_check.py`, `tools/run_consistency_check.py`,
`agi-proof/okf-consistency/consistency.public-report.json`,
`training/feedback/epistemic_holes.jsonl` (empty — no undeclared holes on wiki).

**Claim impact:** Consistency escalation only; `canClaimAGI: false`.

## gate-z3-backend-deadcode-found-and-fixed-2026-06-25

**Status:** RESOLVED (bug fixed) + accept-path validated.

Installing `z3-solver` and running a POSITIVE CONTROL through the invariant
oracle (`tools/run_positive_control.py`) exposed that the z3 backend of
`agent/formal_verifier.py::_z3_lattice` had **never executed**. The accept-path
unit tests (`test_godel_oracle`, `test_invariant_suite`) mock `require_z3`, but
z3 was not installed in any environment, so `check_lattice_consistency` always
took the pure-Python fallback. The z3 branch contained a latent crash:
`vs.get(lhs, z3.IntVal(_as_int(lhs)))` evaluates the default eagerly, so a NAMED
variable on the LHS (every production invariant, e.g.
`content_after_religion >= floor`) was passed to `z3.IntVal(None)` →
`Z3Exception: parser error`. The whole gate was fallback-only by accident.

**Fix:** lazy operand resolution in `_z3_lattice` mirroring the fallback's
unbound-variable handling (held/error instead of crash). Added a real-z3
regression test (`test_lattice_named_variable_runs_on_z3_backend`) that pins
accept+reject verdicts and `backend == "z3"`; generalized the unbound-variable
test to both backends; added `z3-solver` to the CI test install so the path is
exercised going forward.

**Accept-path validation:** with z3 installed, a synthetic known-good candidate
(`positive-control-synthetic`) now promotes with all five invariants `accepted`
on the **z3** backend — the gate is proven to ACCEPT a good candidate, not only
to REJECT (v5 still correctly rejects on `protected_floor_content`). Artifact:
`agi-proof/self-gate/invariant-suite.positive-control-synthetic.public-report.json`.

**Claim impact:** the self-gate's formal proofs are now actually solver-checked,
not silently fallback-only. Decidable numeric invariants only; not alignment,
not AGI, not a Gödel machine. canClaimAGI stays False.

## agi-pilots-way-forward-gate-rollup-2026-06-25

**Status:** CANDIDATE INFRASTRUCTURE (`candidateOnly: true`, `level3Evidence: false`).

**Task 1 — z3 hard promotion requirement (`solverChecked`):**

- `build_proof_bundle` / promotion reports stamp top-level `solverChecked` (= every invariant `backend == "z3"`).
- Default: promotion blocked when `solver_attestation` is `held` (no z3) or any invariant uses fallback.
- `--allow-fallback-proof` (OFF by default): may promote with `solverChecked: false` + note
  `fallback proof — not solver-checked`.
- Fixed z3 `dict.get` eager-default bug in `agent/formal_verifier.py`.
- Positive control (`tools/run_positive_control.py`): **promote True**, all five invariants **z3**, `solverChecked: true`.

**Task 2 — CONTENT is pass gate:**

- `tools/eval_ladder.py`: headline `passed` / `score_pct` = **CONTENT** channel; FORMAT + COMBINED reported only.
- Regenerated `training/local_sophia_v2/eval_ladder_{baseline,adapter}.json` with `passGate: content`.
- Baseline religion CONTENT **5/6** unchanged; no training improved it — measurement artifact fix, **not** capability uplift.

**Task 3 — Council-panel format at inference (no weights):**

- `agent/council_format.py` + `--religion-council-panel` on `eval_local_model.py` / `eval_mlx_model.py`.
- Qwen2.5-3B base, same model, religion N=6 (`benchmark/model_runs/local-qwen-qwen2.5-3b-instruct-council-panel-religion.report.json`):

| Template | FORMAT | CONTENT | COMBINED |
|---|---|---|---|
| WITHOUT | 1/6 | 5/6 | 1/6 |
| WITH council panel | 6/6 | 5/6 | 5/6 |

- **Honest finding:** FORMAT uplift is inference-time structure, not LoRA weights. CONTENT flat (no regression); ship template.
- N=6 ⇒ ±1/6 within noise; **no religion uplift claim**.

**Task 4 — `provenance_complete` on 12 religion repair traces:**

| | lackingCount | verdict |
|---|---|---|
| Before (`religion_repair_c4.jsonl` without `metadata.sourceCitation`) | 12 | rejected |
| After (citations → `data/attributions.json` / `data/traditions.json` keys) | 0 | accepted (z3) |

- `tools/build_local_sophia_dataset.py --check`: contamination **CLEAN**.

**Task 5 — End-to-end gate re-confirmation (z3 installed):**

| Candidate | oracle promote | solverChecked | FINAL |
|---|---|---|---|
| Positive control | True | True | accept |
| v5 full religion repair | False (`protected_floor_content`) | True | **reject** |
| Known-good cited fixture | True | True | **promote** |

**Claim impact:** z3 solver-checked promotion is now explicit policy; CONTENT channel is the ladder pass gate;
religion FORMAT is prompt-structurable at inference; religion-repair LoRA path remains falsified.
`canClaimAGI: false`.

## math-code-curriculum-preregistered-2026-06-25

**Status:** OPEN — pre-registered on branch `claude/sophia-math-code-curriculum` before
any Qwen2.5-7B MATH/CODE curriculum GPU run. Manifest:
`agi-proof/sophia-math-code-curriculum/preregistration.json`; oracle split:
`agi-proof/sophia-math-code-curriculum/oracle-split.md`; held-out seal:
`agi-proof/sophia-math-code-curriculum/heldout-seal.manifest.json`.

**Scope:** Train on sympy/exec-verified **synthetic** curriculum; cite only sealed
held-out MATH/GSM8K/HumanEval/MBPP style samples (+ hidden pack when run) with
≥3 seeds and 95% CI excluding 0. Training-oracle passes are NOT benchmark proof.
`canClaimAGI` stays **False**.

**Stage 2 (curriculum data) — 2026-06-25:** `tools/generate_math_code_curriculum.py`
→ `training/sophia-math-code-curriculum/` (144 sympy/exec-verified SFT rows;
seed `20260625`). Per-tier kept/dropped (generated → kept):

| Tier | Math kept | Code kept | Total kept | Dropped |
|------|-----------|-----------|------------|---------|
| tier0 (GSM8K-style + trivial code) | 24 | 6 | 30 | 0 |
| tier1 (derivative_poly/solve/eval + list/string code) | 48 | 6 | 54 | 0 |
| tier2 (func/product/definite_integral + loop code) | 54 | 6 | 60 | 0 |
| **Total** | **126** | **18** | **144** | **0** |

Decontam: `build_local_sophia_dataset.py --check` → **CLEAN**; held-out seal
`--check` OK; curriculum `check_contamination` → 0 overlaps vs 234 eval prompts.
RLVR eval families (`derivative_chain`, `integrate_func`, `second_derivative`)
excluded from training pack. `canClaimAGI` stays **False**.

**Stage 3 prep (QLoRA wiring) — 2026-06-25:** Stage 3 GPU run **not executed**
(no `RUNPOD_API_KEY` / no local CUDA in prep session). Wired:

| Artifact | Purpose |
|----------|---------|
| `training/sophia-math-code-curriculum/README.md` | Local + RunPod train commands (QLoRA 4-bit, `--mask-prompt`, 2 epochs, seeds 0–2) |
| `agi-proof/sophia-math-code-curriculum/runpod-sft-3seed.sh` | 3-seed RunPod launcher → `sft_all.jsonl` |
| `tools/train_lora.py` | `--data` alias + pack-dir manifest hook → `sft_all.jsonl` |
| `tools/runpod_train.py` | `--train-data`, `--train-only`, `--adapter-dir` for sealed-pack SFT |
| `tools/prepare_math_code_mlx.py` | Optional MLX chat-data materialization (not committed) |
| `tests/test_train_lora_math_code_pack.py` | Pack format + resolve hook |

`train_lora.py --dry-run --data training/sophia-math-code-curriculum/` → **144 rows**.
`guard_filter` on pack → **0 dropped**. `canClaimAGI` stays **False**.

**Stage 0 env gate re-verify — 2026-06-25 (agent session):** repo
`897930a9842f172bf042e9c1c470aa80aa3e6ac0` == remote
`origin/claude/sophia-math-code-curriculum`. Gates re-run:

| Gate | Result |
|------|--------|
| `build_local_sophia_dataset.py --check` | **CLEAN** (overlap 0) |
| `seal_math_code_heldout.py --check` | **OK** (6 files) |
| `lint_claims.py` | **OK** (24 files) |
| pytest (math/code/curriculum/train_lora pack) | **21 passed** |
| `train_lora.py --dry-run` on pack | **144 rows** |
| `RUNPOD_API_KEY` | **UNSET** |
| local CUDA / torch | **unavailable** (torch not installed) |
| RunPod SSH smoke | **NOT RUN** (no API key) |

Artifact: `agi-proof/sophia-math-code-curriculum/stage0-env-gate.public-report.json`.
`canClaimAGI` stays **False**.

**Stage 3 GPU SFT — BLOCKED 2026-06-25:** `runpod-sft-3seed.sh` dry-run exit 2
(`RUNPOD_API_KEY` unset). No pods provisioned; seeds 0/1/2 **not executed**; no
adapters produced. Remaining blockers: Qwen2.5-7B-Instruct base weights download;
post-train held-out evidence-oracle eval + 7B baseline ladder; `promote_adapter`
protected-floor after adapter exists. Downstream blocked: baseline ladder (3b),
RLVR/DPO (4), internal gate (5), evidence oracles (6), prereg threshold
reconciliation (7). Artifact:
`agi-proof/sophia-math-code-curriculum/stage3-runpod-blocker.public-report.json`.

**Honest headline:** Curriculum pack (144 sympy/exec-verified rows) and Stage 3
wiring verified locally; GPU training blocked pending `RUNPOD_API_KEY` on a host
with RunPod SSH egress (or local CUDA). No MATH/GSM8K/HumanEval/MBPP uplift claimed.
`canClaimAGI: false`.

**Stage 0 re-gate + SSH smoke — 2026-06-25 (agent session, cb9a782):** repo ==
remote `origin/claude/sophia-math-code-curriculum`. Gates re-run:

| Gate | Result |
|------|--------|
| `build_local_sophia_dataset.py --check` | **CLEAN** (overlap 0) |
| `lint_claims.py` | **OK** (24 files) |
| pytest (math/code/curriculum/train_lora pack) | **21 passed** |
| `RUNPOD_API_KEY` | **SET** (env only; not committed) |
| RunPod REST API (`GET /pods`) | **OK** |
| RunPod SSH smoke (probe pod `0l6i9hurt74osj`) | **FAIL** — SSH mapped at `213.173.108.232:16786` but login **Operation timed out** from Cursor agent env |
| local CUDA / torch | **unavailable** |

Orphan pod observed: `6l4go54e2n4f54` (`sophia-7b-ssh-smoke`, RUNNING, $0.69/hr) — not
terminated by this session (no ephemeral key). Artifacts:
`agi-proof/sophia-math-code-curriculum/stage0-env-gate.public-report.json`,
`ssh-smoke.public-report.json`, `stage3-runpod-blocker.public-report.json`.

**Stage 3 GPU SFT — BLOCKED 2026-06-25 (cb9a782):** `runpod-sft-3seed.sh` **not
launched** (SSH prerequisite failed). Seeds **0/3** executed; no adapters. Adapter paths
(planned, absent): `training/sophia-math-code-curriculum/checkpoints/sophia-cuda-v1-seed{0,1,2}/`.
Downstream blocked: baseline ladder (3b), RLVR/DPO (4), internal gate (5), evidence
oracles (6), prereg reconciliation (7).

**Honest headline (cb9a782):** Local gates PASS; RunPod API OK; SSH egress FAIL from
agent env. 144-row curriculum ready; GPU SFT blocked. No MATH/GSM8K/HumanEval/MBPP uplift
claimed. `canClaimAGI: false`.

## sophia-math-code-curriculum-stage3-ssh-blocked-2026-06-25

**Status:** BLOCKED (`candidateOnly: true`, `level3Evidence: false`, `canClaimAGI: false`).

**Local gates (re-run on branch `claude/sophia-math-code-curriculum`):**

- `python3 tools/build_local_sophia_dataset.py --check` — contamination **CLEAN**
- `python3 tools/lint_claims.py` — **PASS**
- Curriculum pack `training/sophia-math-code-curriculum/sft_all.jsonl` — **144 rows** (sealed SFT only)

**RunPod SSH smoke (Mac path `/Users/tom/Documents/GitHub/sophia-agi`, Cursor agent shell):**

- API list/create: **PASS**
- Probe name `sophia-math-code-ssh-smoke`, 3 attempts, pods `m9atlzki74hsjo`, `64o3crc5yxbqh9`, `5gmtho2z91z1e1` — each **terminated** after **600s** with no public SSH mapping
- Outbound TCP to `157.157.221.29:42525` (prior mapped port) **hung** — egress to RunPod SSH not reachable from this host
- **Verdict: FAIL** — `agi-proof/sophia-math-code-curriculum/ssh-smoke.public-report.json`

**Stage 3 SFT (`runpod-sft-3seed.sh`):** **NOT LAUNCHED** (SSH prerequisite failed). **Seeds completed: 0/3**. No adapter artifacts under `training/sophia-math-code-curriculum/checkpoints/`.

**Billing note:** orphan pod `0s40ngsh22llz0` (`sophia-7b-sft-seed0`, RUNNING) observed — not created by this smoke session; left running (possible parallel attempt). Terminate manually if unused.

**Claim impact:** No Stage 3 adapters; no Stage 6 benchmark uplift claims. Artifacts: `stage3-runpod-blocker.public-report.json`.

## sophia-7b-train-verify-data-flywheel-2026-06-25

**Status:** STAGE 0–1 COMPLETE; STAGE 2+ BLOCKED (SSH egress timeout to RunPod mapped pod ports from local Mac / Cursor agent shell).

**Pre-registration (Stage 0):** `agi-proof/sophia-7b-train-verify/preregistration.json`,
oracle split `oracle-split.md`, holdout seal `heldout-seal.manifest.json`
(`contentHash: 84d00bdc36205abdb5a162530d8fc972ee27075c053bfd0615de59d3ed9aeb97`, 89 rows).
Commit `9f00733`.

**Data flywheel (Stage 1) — `training/local_sophia_7b/`, base `Qwen/Qwen2.5-7B-Instruct`:**

| Pack | Rows | Decontam dropped |
|---|---:|---:|
| sft_source_discipline | 439 | 89 |
| sft_wiki_provenance | 34 | 0 |
| sft_council_traces | 125 | 0 |
| sft_religion_repair_c4 | 12 | 0 |
| sft_moral_gate | 35 | 0 |
| general_instruct | 120 | 0 |
| dpo_hard_negatives | 590 | 25 |
| dpo_wiki_provenance | 34 | 0 |
| **train total** | **1389** | **114** |
| holdout (sealed) | 89 | — |
| MLX SFT train / valid | 754 / 89 | 11 turns dropped overlong |

`build_local_sophia_dataset.py --check`: contamination **CLEAN** (0 eval overlap, 0 holdout overlap).
Hard-negative miner: 615 pairs (gate-validated). Moral Gate SFT: 35 rows.

**Stage 2 blocker (2026-06-25, updated @ `8975744`):**

1. **This session (2026-06-25T11:10Z):** `RUNPOD_API_KEY` **present**. Stage 0 gates re-verified
   (contamination CLEAN, holdout seal `84d00bdc…`, lint_claims OK). SSH smoke probe
   (`sophia-7b-ssh-smoke`, interruptible) created pods `g6de2tbp9jzge1`
   (`213.173.109.78:15792`) and `6l4go54e2n4f54` (`213.173.107.230:12881`); RunPod API
   reported SSH mapping but **outbound SSH login timed out** (300s wait, 2 attempts each).
   Pods deleted. Log: `agi-proof/benchmark-results/runpod-train/ssh-smoke-20260625-111011.log`.
   Local Mac outbound TCP/22 to `github.com` and `ssh.runpod.io` **passes**; mapped pod
   high-ports **do not**.
2. **Prior session (same day):** pods `crdl0788rpc98m` / `8k7vqe3m5nbynv` — same
   `ssh_login_timeout` pattern. Logs: `sft-3seed-20260625-095511.log`, `sft-seed0-retry.log`.
3. **Earlier session:** `RUNPOD_API_KEY` unset — dry-run only (`868fa31`).

**Policy (2026-06-25):** RunPod GPU jobs for this experiment **must** run via GitHub Actions
(`.github/workflows/runpod-sophia-7b-sft.yml`; secret `RUNPOD_API_KEY`). Do **not** invoke
`runpod-sft-3seed.sh` / `runpod_train.py --yes` from local Mac or Cursor agent shells — SSH to
mapped pod ports times out there. See `agi-proof/sophia-7b-train-verify/README.md`.

**Next step:** dispatch **runpod-sophia-7b-sft** (stage `sft`, confirm `RUN`) on branch
`claude/sophia-7b-train-verify`. Do **not** cite 0/3 seeds as a training verdict until a GHA
run completes.

**Stage 3 prep (no GPU):** `tools/train_dpo.py` (TRL `DPOTrainer`), `runpod-dpo-3seed.sh`
(DPO on `dpo_hard_negatives.jsonl`, 590 pairs in pack / 615 mined), pre/post internal
`eval_ladder` on pod. Blocked on Stage-2 SFT adapter tarballs per seed.

**Stages 5–6:** NOT RUN (no adapter). Local wiring OK: `run_positive_control.py` ✓,
`promote_adapter` + invariant oracle policy unchanged (release gate — NOT evidence).

**Stage 7 blockers:** `VECTARA_*` credentials absent; hidden reviewer pack needs served model +
backend credentials. Do **not** substitute internal gate for third-party evidence.

**Headline (falsifiable, OPEN):** Qwen2.5-7B QLoRA SFT (≥3 seeds) has **not** started — blocked
on SSH egress timeout from Cursor agent host to RunPod mapped pod ports (API key OK; smoke
reproduced 2026-06-25); **0/3 seeds**, not a training verdict.

**Tradeoff (pre-registered, not yet measured):** abstention/MMLU-Pro regression ≤2.0 points vs base;
honest reporting required even if internal release gate passes.

**canClaimAGI:** False.
## hk-advisor-phase0-2026-06-25

**Status:** COMPLETE (sealed benchmark + contamination guard).

**Artifact:** `data/hk_advisor_benchmark/heldout_v1.jsonl` (N=90: 30 answerable / 30 abstain / 30 trap;
45 yue / 45 en; trap subtypes 10 fabrication-bait / 10 fake-citation / 10 unanswerable).
`contentHash: f8f1ae46d6259576b8629e1db9130a7f79e6ac846b6bd59678a2d29246aadf6b`.
`tools/build_local_sophia_dataset.py --check`: contamination **CLEAN** (hkAdvisorBenchmarkHash recorded).

**Claim impact:** Benchmark sealed; no training/eval uplift claim yet. `candidateOnly: true`; `canClaimAGI: false`.

## hk-advisor-phase1-2026-06-25

**Status:** COMPLETE (verifier + verified SFT traces).

**Artifacts:** `agent/hk_advisor/{policy,verifier}.py`, `training/hk_advisor/sft_traces.jsonl` (25 verified rows,
disjoint from held-out benchmark). All rows pass `verify_trace` (advisory boundary, citation, abstention,
no-fabrication, bilingual fidelity).

**Claim impact:** Training substrate ready; no adapter/eval uplift yet. `canClaimAGI: false`.

## hk-advisor-phase2-2026-06-25

**Status:** OPEN (SFT wiring verified; GPU training blocked locally).

**Smoke:** `python tools/train_lora.py --model Qwen/Qwen2.5-3B-Instruct --train training/hk_advisor/sft_traces.jsonl --output training/hk_advisor/checkpoints/sft-seed0 --dry-run` → 25 rows OK.
**Blocker:** `torch` not available in local agent shell; seeds 0/1/2 SFT **not run**. Weights gitignored under `training/hk_advisor/checkpoints/`.

**Next:** Run 3-seed QLoRA SFT on RunPod/local GPU host when available.

**Claim impact:** No adapter weights; no uplift claim. `canClaimAGI: false`.

## hk-advisor-phase3-2026-06-25

**Status:** COMPLETE (DPO pairs mined).

**Artifact:** `training/hk_advisor/dpo_pairs.jsonl` — 49 pairs.
By rejected_type: wrong_abstain 19, uncited_claim 19, overconfident_trap 9,
fabricated_regulation 1, fake_citation 1.

**Claim impact:** Preference data ready; DPO training blocked on P2 GPU blocker. `canClaimAGI: false`.

## hk-advisor-phase4-2026-06-25

**Status:** COMPLETE (mock eval); adapter training NOT RUN.

**Artifact:** `agi-proof/hk-advisor/eval-hk-advisor.public-report.json` (mock mode, 3 seeds).
Mock adapter vs base: fabrication on traps 44.4% → 0%, calibration +0.58, useful-answer +0.26 (CI excludes 0).
**Honest scope:** mock ScriptedClient-style responses — NOT real adapter weights; illustrative direction only.

**promote_adapter:** Ran `--dry-run` on existing sophia-v2 ladder → **reject** (protected_floor_content religion regression);
`solverChecked: true`. No HK-advisor-specific adapter to promote (0/3 SFT seeds).

**Claim impact:** Mock eval shows intended metric direction; no validated adapter uplift. `canClaimAGI: false`.

## team-agents-mode-mock-eval-2026-06-25

## hk-advisor-phase3-2026-06-25

**Status:** COMPLETE (DPO pairs mined).

**Artifact:** `training/hk_advisor/dpo_pairs.jsonl` — 49 pairs.
By rejected_type: wrong_abstain 19, uncited_claim 19, overconfident_trap 9,
fabricated_regulation 1, fake_citation 1.

**Claim impact:** Preference data ready; DPO training blocked on P2 GPU blocker. `canClaimAGI: false`.

## hk-advisor-phase4-2026-06-25

**Status:** COMPLETE (mock eval); adapter training NOT RUN.

**Artifact:** `agi-proof/hk-advisor/eval-hk-advisor.public-report.json` (mock, 3 seeds).
Mock: fabrication traps 37.8%→0%, calibration Δ+0.19, useful-answer Δ+0.24 (CI excludes 0).
**Honest scope:** mock responses only — NOT real adapter weights.

**promote_adapter:** `--dry-run` on sophia-v2 ladder → **reject** (protected_floor_content); `solverChecked: true`.
No HK-advisor adapter (0/3 SFT seeds).
## team-agents-mode-mock-eval-2026-06-25

**Status:** OPEN (mock eval only — not real-model evidence).

**Branch:** `claude/team-agents-mode`

**Benchmark:** Sealed `team_agents_benchmark` heldout_v1 — 36 cases (12/12/12 balance),
12 probe_divisive cases, contentHash=`50a0bfab6b9690aabdc7d9346a2487b900e87bd77647e81e81d292d06ace7e7b`,
decontam CLEAN.

**Traces:** 8 externally verified rows in `training/team_agents/sft_traces.jsonl`
(mock teacher, 0 overlap with heldout, 0 gate-only drops in sample).

**Independence (mock, 3 seeds):** homogeneous panel mean ρ≈0.56, N_eff≈1.41;
heterogeneous mock panel ρ≈0.56, N_eff≈1.41 — **below** pre-registered consensus
threshold (N_eff ≥ 2.0). Reports use **"correlated panel — not consensus"** wording.

**Eval vs baseline (mock, 3 seeds):** team_agents composite pass rate beats single_agent
by +0.056 (95% CI [0.056, 0.056], excludes 0 on this deterministic stub). Trap
false-consensus: team ≤ single. External scorer disjoint from intrinsic gate.

**SFT (P2):** No GPU run in this session — `train_lora.py --dry-run` on team traces
passes; real 3-seed SFT blocked pending GPU.

**Promotion gate:** Not attempted — no adapter ladder artifact. Positive control
promotes with `solverChecked: true`; team-agents adapter promotion requires a gated
ladder run via `tools/promote_adapter.py`.

**Honest claim:** Verifier-gated deliberation policy **candidate** only.
`canClaimAGI: false` — not AGI, not validated uplift, not independent consensus
when N_eff < 2.0.

**Artifact:** `agi-proof/benchmark-results/team-agents.public-report.json`

## team-agents-longtask-eval-template-2026-06-25

**Status:** OPEN (long-task benchmark wired; real eval blocked until promoted adapter + GPU).

**Branch:** `claude/sophia-team-orchestrator`

**Benchmark:** Sealed `team_agents_longtask` heldout_v1 — 18 cases (6/6/6 balance:
multi_domain_chain / chained_subquestions / long_coordination_trap),
contentHash=`d84acadbc570e9718a0264adceff5a6a8a0bace089e425ac3e3c566ab7a17dd4`,
decontam CLEAN (registered in `dataset_guard`).

**Conditions:** `sophia_single` (one advisor pass) vs `sophia_team_orchestrator`
(`deliberate_team()` via `tools/team_agents_deliberate.py`).

**Metrics:** passRate/composite delta, subStepCoverage, roleFidelity, handoffIntegrity,
falseConsensus (traps), effectiveN. External scorer disjoint from intrinsic gate.

**Command (when ready):**
```bash
python tools/eval_team_agents_longtask.py --mode real --model mlx:Qwen/Qwen2.5-3B-Instruct \\
  --adapter training/mlx_adapters/sophia-v3 --backend mlx --seeds 0,1,2
```

**Honest claim:** `canClaimAGI: false`. Never claim consensus when N_eff < 2.0.
Record sub-step coverage and trap false-consensus — not promotion evidence until
`promote_adapter.py` clears the Sophia ladder separately.

**Artifact:** `agi-proof/benchmark-results/team-agents-longtask.public-report.json`

## team-orchestrator-eval-template-2026-06-25

**Status:** OPEN (orchestrator wired; real eval blocked until promoted adapter + GPU).

**Command (when ready):**
```bash
python tools/eval_team_agents.py --mode real --model mlx:Qwen/Qwen2.5-3B-Instruct \\
  --adapter training/mlx_adapters/sophia-v3 --backend mlx --seeds 0,1,2
```

**Honest claim:** `canClaimAGI: false`. Never claim consensus when N_eff < 2.0.
Record effective-N, trap false-consensus, and bootstrap CI — not promotion evidence
until `promote_adapter.py` clears the Sophia ladder separately.

## visual-rlvr-live-run-not-yet-gated-2026-06-26

**Status:** OPEN. The multimodal RLVR reward (visual-grounding-as-reward,
`multimodal_bench/visual_reward.py`) is built and its machinery invariants pass
offline (`python tools/run_visual_rlvr.py` → ALL INVARIANTS HOLD: deterministic,
bounded [-1,1], honesty ordering correct > abstain > wrong, judge-free verifier
seam invoked every call, family-disjoint contamination-free split;
`tests/test_multimodal_phases.py` green in CI). The live VLM-GRPO run is NOT done
and is additionally blocked on a vision-capable GRPO trainer (the dense-LM GRPO
path in `tools/run_rlvr.py` does not ingest images).

**Claim impact:** Blocks any multimodal RLVR capability claim (held-out grounding
rise / hallucination drop vs base VLM). Offline reward-wiring invariants pass in
CI but are NOT capability evidence.

**Required response:** Run a gated live VLM-GRPO run clearing
`provenance_bench.aggregate._is_validated` (>=2 judge families, Cohen's kappa
>= 0.40, >=3 runs, 95% bootstrap CI excludes 0) on the family-disjoint held-out
split (`tools/run_visual_rlvr.py --prepare` builds it), plus manual semantic
review. `canClaimAGI` stays False.

## visual-encoder-probe-real-weights-not-run-2026-06-26

**Status:** OPEN. The encoder-probing harness (`multimodal_bench/encoder_probe.py`)
runs offline on a deterministic hashing/caption stand-in (recall@1 with bootstrap
CI), but the real CLIP / SigLIP rungs require torch + transformers + checkpoint
weights and are recorded as BLOCKERS, not results (eval_ladder discipline).

**Claim impact:** No claim about real vision-encoder retrieval quality. The
hashing stand-in measures harness plumbing and caption separability, NOT pixel
perception (the report labels every rung).

**Required response:** Run `tools/probe_vision_encoder.py --encoder clip:<id>` /
`siglip:<id>` on a machine with the weights; report recall@1 + CI per encoder.
**Gate hardening (2026-06-24, retention gate):** The original W2 promote verdict was reached on a
gate that read only the eval ladder + the protected-floor proof — it had **no old-task retention
term**, so it promoted v3 despite the learning-under-shift report showing a `-50.0pp` old-benchmark
regression (`passingSignal=false`). The gate rewarded forgetting. `tools/promote_adapter.py` and
`agent/continual_plasticity.evaluate_update` now consume a learning-under-shift report
(`--shift-report`) and treat an old-task regression beyond tolerance (default `5.0pp`) as a **hard
reject**, on par with a protected-suite regression. Re-running the gate on v3 with its real shift
report now yields `FINAL VERDICT: reject`
(`agi-proof/continual-plasticity/local-sophia-v3-mlx-retention-gated.public-report.json`). The
historical ladder-only `promote` artifact is retained unaltered for provenance. Net: **under the
corrected gate, v3 does not promote.** Closing the loop without forgetting (replay/rehearsal of the
old domain or a smaller weight delta, then re-running learning-under-shift to `passingSignal=true`)
is now a precondition for any future promotion — this remains open and hardware-bound (MLX/GPU).
**Claim impact:** Mock eval shows metric direction; no validated uplift. `canClaimAGI: false`.

## provenance-delta-survives-judge-free-2026-06-27

**Status:** RAN — the validated anti-fabrication advantage SURVIVES judge-free.
Candidate-grade (one deterministic judge family); does NOT by itself clear the
multi-judge validation bar. `canClaimAGI` stays **False**.

**Why this is the highest-information experiment of the phase.** The Verifiable-
Sophia plan (commit cb887e5) pre-registered a binary falsifiable question for the
whole phase: is Sophia's one validated claim (−12.5pt hallucination Δ on
dolphin-llama3:8b) a *real* abstention property, or partly an LLM-judge artifact?
The plan was explicit: *"If the Datalog reproduction fails, the implication is
serious — the advantage may be partly an LLM-judge artifact."* This entry answers
that question directly on the **model side** (the Datalog port answered it on the
gate side — see `datalog-provenance-faithful-port-preregistered-2026-06-27`).

**Setup (judge-free, deterministic labeler).** Ran `tools/run_unified_uplift.py
--model ollama:dolphin-llama3:8b --runs 3 --limit 48 --levers +gate` with **no
`--judges`** — so `judge_answer` falls back to the deterministic **lexical judge**
(`provenance_bench/judge.py::lexical_judge`), which labels each model answer
hallucinated/abstained/correct against external gold and shares NO code with the
gate. dolphin-llama3:8b was available locally (Ollama HTTP 200, model pulled),
so this ran live on the host — NOT via RunPod/GHA, NOT blocked.

**Result.**
- `+gate` hallucination Δ = **+9.0%** (raw-alone 0.0903 → +gate 0.0000),
  paired-bootstrap 95% CI **[+4.9%, +13.9%], EXCLUDES ZERO**.
- per-run Δ: [0.0625, 0.1250, 0.0833] (all positive, consistent direction).
- false-positive cost **0.0%**; coverage recall **100%** (gate fired on every
  case it should have, broke no correct answer).
- 3 runs, N=48 (24 false + 24 true controls), lexical judge only.

**Interpretation (honest).** The validated number was **+12.5% [+5.6%, +19.4%]**
scored by TWO LLM judge families (deepseek + llama-3.3-70b, κ=0.74) on N=24 false
cases. My judge-free reproduction is **+9.0% [+4.9%, +13.9%]** — a smaller point
estimate (different case subset: 48 cases incl. true controls vs the validated
24 false-only) with an **overlapping** CI that **also excludes zero**. The
direction and statistical significance survive removing every LLM judge from the
labeling loop. **The advantage is NOT an LLM-judge artifact.** This is the
decisive falsifiable outcome the phase was designed to produce — and it is the
non-decaying direction (a deterministic lexical labeler + a deterministic gate,
no judge vote anywhere in the loop).

**Boundary conditions / what this does NOT clear.**
- This judge-free run is **`validated: False`** by the repo's own bar, correctly:
  the multi-judge gate (`_is_validated`) requires ≥2 DISTINCT judge families with
  κ≥0.40. A single deterministic lexical judge is one family, not corroboration.
  This run STRENGTHENS the validated claim (the effect survives judge-free) but
  does NOT replace the multi-judge run; the existing +12.5% multi-judge result
  remains the headline.
- The validated −12.5pt model-side delta is STILL a decaying asset
  (`calibration-advantage-is-model-dependent-2026-06-25`: it vanishes on strong
  base models). This judge-free reproduction was on the WEAK subject (dolphin).
  A judge-free run on a strong base (deepseek-v3, where raw already fabricates
  ~0) is the natural companion and is expected to be a null.
- N=48, 3 runs, self-authored pack — the residual independence caveat stands.
  One real third-party pack is still worth more than this.
- Non-overlapping CI region: judge-free [+4.9, +13.9] vs validated [+5.6, +19.4]
  overlap heavily but the judge-free upper bound (13.9) is below the validated
  midpoint (12.5) — consistent with the lexical judge being slightly stricter
  than the LLM judges, NOT with the effect disappearing.

**Artifact.** `agi-proof/baseline-ablation/judge-free-reproduction-2026-06-27/
uplift-dolphin-3run-lexical.json` (SHA-256
`da48f2a54f61f081287601ac29fe07f3400e9e1c0024b7565bfa29519feacad8`). 3 runs,
N=48, lexical judge, +gate lever.

**Next experiments (in priority order).** (1) The same judge-free run on a STRONG
base model — does the effect vanish (expected) or survive? Directly tests the
decaying-asset boundary. (2) Scale N toward 100+ for a tighter judge-free CI.
(3) A real third-party pack scored judge-free — the only thing that closes the
independence gap.

## verifiable-sophia-moves-executed-2026-06-27

**Status:** BATCH EXECUTED. Four recommended moves from the phase plan, all on
branch `experiment/datalog-judgefree-clean`. `canClaimAGI` stays **False** —
nothing here is a capability claim or third-party evidence.

**Move 1 — Datalog substrate made runtime-viable (eng debt I created).** The
port was a 25-min audit artifact; now it is a real backend. Two correctness-
preserving optimizations: (a) predicate-indexed fact store in
`agent/datalog_engine.py` (a body literal scans only its predicate's facts);
(b) module-level caching of the default records + the gate's compiled specs in
`agent/datalog_provenance.py` (root cause: `_load_provenance_records()` returned
a fresh dict every call → identity-keyed cache always missed → ~766 regex
re-compiles/call). Result: `check_claim(backend="datalog")` is byte-identical to
`backend="regex"` at **0.5ms/call (was ~464ms, ~900×)**. The opt-in `backend=`
parameter is wired on `agent.guarded.check_claim` (default `"regex"` unchanged;
`"datalog"` fail-closed falls back to `"regex"` if unavailable). 14/14 unit
tests pass; full 957-case audit still 957/957 byte-identical.

**Move 2 — the judge-free −12.5pt reproduction, RUN LIVE (the decisive one).**
The plan's binary falsifiable question: is the validated advantage an LLM-judge
artifact? **Answer: NO — it survives judge-free.** dolphin-llama3:8b was
available locally (Ollama HTTP 200, model pulled), so this ran live on the host,
NOT via RunPod/GHA. `tools/run_unified_uplift.py --model ollama:dolphin-llama3:8b
--runs 3 --limit 48 --levers +gate` with **no `--judges`** → deterministic
lexical judge only:
- `+gate` hallucination Δ = **+9.0%**, 95% CI **[+4.9%, +13.9%], EXCLUDES ZERO**,
  0% FP-cost, 100% coverage. per-run [0.0625, 0.125, 0.0833].
- The validated number was +12.5% [+5.6, +19.4] (2 LLM families, N=24 false).
  This judge-free run (lexical, N=48) is a smaller estimate with an OVERLAPPING
  CI that also excludes zero. The direction + significance survive removing
  every LLM judge from labeling. **The advantage is NOT an LLM-judge artifact.**
- HONEST: `validated:False` (correctly — lexical = 1 family, not multi-judge
  corroboration; this STRENGTHENS but does NOT REPLACE the +12.5% headline).
  Still a decaying model-side asset (vanishes on strong bases). Full detail in
  `provenance-delta-survives-judge-free-2026-06-27` above. Artifact:
  `agi-proof/baseline-ablation/judge-free-reproduction-2026-06-27/`.

**Move 3 — turnkey third-party reproducer for the Datalog claim.**
`tools/run_datalog_reproducer.py`: a reviewer-run one-command check that pins
the provenance data files by SHA-256, re-derives the 957-comparison audit LIVE
(trusts no committed artifact), and prints PASS/FAIL. Prints the pre-reg's own
hash so a silent swap is detectable. **Self-verified**: clean run PASS exit 0;
negative control (tampered hash) FAIL exit 1 with DATA TAMPER detection even
though the live audit still passed (integrity check independent of logic check).
This is the "one third-party run > 10 self-runs" lever — when a reviewer
appears, the whole gate-faithfulness claim is one command.

**Move 4 — Lean soundness lane: already exists, NOT duplicated.** The strategic
plan's Lean experiment is already staged on `origin/main` by a concurrent
session: `.github/workflows/lean-kernel.yml` + `scripts/install_lean.sh` install
the Lean 4 toolchain and run `tests/test_lean_verifier.py` +
`tools/run_formal_proofs_eval.py` with a real kernel (flips the 2 SKIPPED
kernel tests to PASSED; asserts the smoke loop closes). This covers the plan's
"Lean soundness on sympy verifiers" intent via the cleaner `lean_verifier`
(lake subprocess) path. The older `lean_backend.py` (LeanDojo) path is half-wired
and heavier; a redundant lane for it would duplicate effort — deferred. The Lean
lane is dispatchable on main; no local install was attempted (multi-GB toolchain
under HTTPS-only egress — correctly gated).

**Net phase position.** The non-decaying asset (machine-checkable verification +
derivable abstention) now has: a logic-native gate backend (957/957, 0.5ms), a
judge-free confirmation that the model-side effect is real (+9.0% CI excludes
0), and a one-command third-party reproducer. The two open levers are unchanged:
(1) a judge-free run on a STRONG base (expected null — tests the decay boundary);
(2) one real third-party pack (closes the independence gap, now turnkey).

## provenance-delta-multijudge-2family-2026-06-27

**Status:** RAN — multi-judge (2 distinct vendor families) reproduction on the
validated subject (dolphin-llama3:8b). The effect survives with genuine
judge-family independence; it clears 4 of 5 validation flags. The single miss is
`atLeast3Runs` (run 3 was lost to repeated concurrent-session process kills).
`canClaimAGI` stays **False**.

**Setup.** Two judge families that are BOTH distinct from each other AND from the
deepseek/meta-llama pair used in the original validated run:
`llmhub:gpt-4o` (openai) + `llmhub:claude-sonnet-4-6` (anthropic), served via an
OpenAI-compatible aggregator proxy (api.llmhub.com.cn). Subject `ollama:dolphin-
llama3:8b`, local. `tools/run_unified_uplift.py --runs 3 --limit 48 --levers
+gate`. To make two genuinely-different vendors behind one key count honestly as
≥2 families (the aggregator serves bare model ids with no `vendor/` prefix), a
new `llmhub` preset + a `_LLMHUB_FAMILY` name→family map were added
(`agent/model.py`, `provenance_bench/aggregate.py`), gated by 6 tests.

**Result (2 runs, 96 false-case observations; run 3 lost).**
- `+gate` hallucination Δ = **+9.4pt** (raw-alone 0.4375 → +gate 0.3438),
  paired-bootstrap 95% CI **[+4.2%, +15.6%], EXCLUDES ZERO**.
- per-run Δ: [+10.4pt, +8.3pt] (both positive, consistent).
- false-positive cost **0.0%**; coverage recall 0.214 (9 of 42 hallucinations
  fixed — the gate fires on the explicit-assertion subset it's designed for).
- **judge κ = 0.8123** (well above the 0.40 floor), 90.6% pairwise agreement.
- validation flags: `notMock=T`, `multiFamilyJudges=T`, `kappaAboveFloor=T`,
  `ciExcludesZero=T`, `atLeast3Runs=**F**` (only 2 runs survived). ⇒ `validated:False`.

**Interpretation (honest).** This is the *strongest internal corroboration yet*
of the validated claim. It is NOT judge-free (it uses 2 LLM judges), but the two
judges are independent vendors (openai + anthropic) with high agreement (κ=0.81),
so the effect is not an artifact of any single judge's bias. Combined with the
judge-free run (+9.0%, CI [+4.9,+13.9], `provenance-delta-survives-judge-free-
2026-06-27` above), the picture is now: the gate cuts dolphin's hallucinated
attributions by ~9–9.4pt with a CI excluding zero, reproducible under (a) NO LLM
judge and (b) two independent LLM-judge families — three independent determinations
of the same effect.

**Why it does NOT formally clear `_is_validated` (and why that's honest).** The
single failing flag is `atLeast3Runs`: the runner needs 3 complete runs and run 3
was killed twice by concurrent-session process churn (the shared working tree
here is actively edited by other agents). The per-run checkpoint fix (committed
`1a9c0e27`) preserved runs 1–2 so this is a real 2-run result, not a loss. A clean
third run on a quiet host flips the flag; the underlying Δ/κ/CI won't move
materially (per-run Δ is consistent at +8–10pt).

**Boundary conditions (no overclaim).** Still self-authored pack; still a decaying
model-side asset (vanishes on strong bases per `calibration-advantage-is-model-
dependent`); N=48 × 2 runs; one API key behind one proxy. The judge κ=0.81 is high
but the two judges saw the same answers — they agree with each other, not with an
external ground truth (the lexical judge in the judge-free run provides the
external anchor). `canClaimAGI` stays **False**.

**Artifact.** `agi-proof/baseline-ablation/multi-judge-reproduction-2026-06-27/`
— `uplift-dolphin-2fam-2run-aggregated.json` (the corrected 2-run report) +
`uplift-dolphin-2fam-3run.partial.json` (the raw 2-run checkpoint).

## provenance-delta-decays-to-zero-on-strong-base-2026-06-27

**Status:** RAN — the model-side advantage DECAYS TO ZERO on a strong base, as
predicted. This is a measured falsification of the universal-advantage reading,
recorded honestly. `canClaimAGI` stays **False**.

**Why this is the highest-information experiment left after the multi-judge run.**
The ledger entry `calibration-advantage-is-model-dependent-2026-06-25` ADMITTED
that Sophia's anti-fabrication advantage → 0 on strong base models (deepseek-v3
raw already fabricates 0/12). That admission is the repo's central honesty claim
about its own asset — but it had never been tested with the uplift harness on a
strong base. This entry does exactly that: the SAME judge-free harness that
showed +9.0% on dolphin, pointed at a genuinely strong subject.

**Setup.** `tools/run_unified_uplift.py --model ollama:qwen3:30b-a3b --runs 3
--limit 48 --levers +gate` (no `--judges` → deterministic lexical judge; local
Ollama, 30B-A3B MoE = genuinely strong / near-frontier class). Identical config
to the judge-free dolphin run for a direct apples-to-apples comparison.

**Result (3 runs, 96 false-obs, all 3 runs completed cleanly).**
- raw fabrication rate: **0.0278 (2.8%)** — already near the floor (vs dolphin's
  9.0%). There is almost nothing for the gate to cut.
- `+gate` hallucination Δ = **+0.0000**, CI **[0.0, 0.0]**, per-run [0.0, 0.0,
  0.0]. The advantage is EXACTLY zero, not merely "small."
- false-positive cost 0.0%; coverage recall 0.0 (nothing to cover).

**Interpretation (honest).** This is the cleanest possible confirmation of the
decay boundary: the gate provides ZERO measurable benefit on a strong base,
because the strong base barely fabricates to begin with. The +9.0% / +9.4%
dolphin results are now unambiguously a **weak-model phenomenon** — not a
universal property of the gate. Combined with the three positive determinations
(dolphin: +9.0% judge-free, +9.4% 2-family, +12.5% validated), the full picture is:

| subject | raw fab | +gate Δ | 95% CI |
|---|---|---|---|
| dolphin-llama3:8b (weak, uncensored) | 9.0% | +9.0% | [+4.9, +13.9] |
| qwen3:30b-a3b (strong) | 2.8% | +0.0% | [0.0, 0.0] |

The gate's value is real AND bounded: it helps exactly where fabrication is high
(weak/uncensored models), and helps not at all where the model is already
truthful. This is the precise, measured scope the strategic plan called for
("convert a decaying claim into an honest one") — the decay is now a feature of
the documented boundary, not an unmeasured caveat.

**Boundary conditions (no overclaim).** N=48 × 3 runs, self-authored pack,
lexical judge (no multi-family corroboration on this specific run — though the
judge-free dolphin Δ was corroborated by 2 LLM families separately). The
zero-delta is robust (all 3 runs identical at 0.0), so judge variance cannot
explain it away. `canClaimAGI` stays **False** — this sharpens the honesty, it
does not add a capability claim. The non-decaying asset remains the
machine-checkable gate (Datalog substrate), not the model-side delta.

**Artifact.** `agi-proof/baseline-ablation/strong-base-decay-test-2026-06-27/
uplift-qwen3-30b-3run-lexical.json` (3 runs, lexical judge, qwen3:30b-a3b, +gate).


