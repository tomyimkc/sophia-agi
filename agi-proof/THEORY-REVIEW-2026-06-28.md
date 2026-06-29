# Theory Review — Tested / Untested / Doesn't Work

_Generated 2026-06-28. Sources of truth: `agi-proof/failure-ledger.md`,
`agi-proof/architecture-bets.json`, `agi-proof/long-context-bets.json`,
`agi-proof/evidence-manifest.json`, `RESULTS.md`. Classifications use the repo's
own status tags (no-overclaim gate, ledger verdicts, bet honest-status)._

**Reading note.** "Works" here means *validated by the project's own no-overclaim
bar* (≥2 independent judge families, κ≥0.40, ≥3 runs, CI excludes zero) **or** a
deterministic verifier validated on a constructed set. "Doesn't work" means a
hypothesis the project itself recorded as falsified / negative / dead-end.
`canClaimAGI` is still `false`; nothing here is an AGI claim.

---

## 1. TESTED — and works (validated / positive result)

### Core theses with external or multi-judge validation
- **Provenance / verifier gate reduces hallucination.** dolphin-llama3:8b: 36.1%→23.6% (Δ 12.5% [5.6, 19.4]) and 42.4%→33.3% (Δ 9.0% [4.2, 14.6]), two judge families, 0% false-positive cost. *Validated, narrow.*
- **Fail-closed abstention beats fabrication on "I don't know" traps.** sophia-full 0% fabrication vs raw model 16.7–25%; calibration Δ 22.0% [14.5, 29.6], corroborated by gpt-4o + claude-sonnet (κ=0.74). *Validated (pack self-authored — residual caveat).*
- **Self-consistency selective prediction (calibration) — the first externally validated result.** SimpleQA Verified, public/human-authored: DeepSeek +15.8% [9.8, 22.1], Qwen-72B +7.8% [2.3, 13.5] selective-accuracy lift @20% coverage, two judge families each (κ 0.97/0.99). Cross-model confirmed. (`simpleqa-external-validation`, `simpleqa-crossmodel-qwen-validated`)
- **Legal-citation-existence verifier** (Mata v. Avianca failure mode): 100% / 100% fab-recall on N=14 deterministic set. *Validated (tiny, constructed).*
- **Legal-holding-faithful** (Ayinde misstated-authority): consensus 100%, κ=1.0, N=8, 3 families. *Clears the gate; clear-cut cases only.*
- **External-eval harness end-to-end**, GSM8K 98% (N=100) — validates the *harness*, not a Sophia capability.

### Validated mechanisms / findings
- **Measured-improvement loop:** held-out recall 17%→98% over 6 cycles, monotone, 0% false-positive — learned do-not-attribute rules generalize across *phrasing*. *Validated (deterministic; not across new entities).*
- **Verifier synthesis + meta-verification** on toy arithmetic: in-library 1.0 precision/recall, out-of-library 100% abstention; ablation proves the *meta-verification* step is what earns trust. *Validated on toy scope.*
- **Cross-entity generalization is impossible without external grounding** (a validated *negative*): memorized rules → 0% transfer; structural detectors → 100% false-positive. Motivates the retrieval-grounded design.
- **SSIL Layer-0 (skill-level) self-improvement closes offline:** 0.525→0.825→0.875 on held-out splits with per-round corrigibility checks. *Validated offline only.*
- **Independent live verifier resolves source-contamination.** `grounded-gate-independent-verifier` (POSITIVE), `live-wikipedia-verifier-resolved` — an independent Wikipedia/source channel catches contaminated grounded answers without false-blocking clean ones.
- **Pressure-calibration map (powered curve):** fabrication-under-pressure is non-monotonic in model size and vector-dependent; gate prevents fabrication across 4B–70B on ungrounded prompts. Weak-model boundary partially confirmed.
- **Deterministic calibration gate, 3-seed fresh ablation:** drives fabrication to 0 across all 7 architectures with no abstention cost on definite cases (deterministic-validated; judge arm blocked).
- **Wisdom-4B base selection (M1):** gemma-3-4b chosen by pre-registered criterion = GO. **M3 pilot:** RAN and PASSED the pre-registered primary (candidate-only).
- **Promotion-gate canary fix:** spurious-promote defect found and fixed (`path-a-canary-spurious-promote-fixed`).
- **Google Fact Check live integration** works for general/viral claims (6/6) — documented *boundary*: 0/6 on literary-provenance, so it complements but can't replace Wikidata/Crossref.
- **Tool-use scaffolding** phase0 (N=120 sealed benchmark) + phase1 (80 verified mock traces) built and closed; **error-memory-rag phase1** closed.

### Implemented infrastructure (deterministic, in the live path — "works" as plumbing, not capability claims)
OKF belief graph + provenance verifier; min-over-chain confidence with counterfactual removal; Conscience Kernel (7-path fail-closed control); Public Moral Standard hard floor + 8-theory parliament; fail-closed governance contract (BLP + taint-labeling + output re-verify); calibrated confidence as a **weak source-quality prior** (weak sources downgraded 100%, default `hi=0.7` near-optimal — explicitly *not* a correctness signal); two-class verifier gate (deterministic + model-judged).

---

## 2. NOT TESTED (built/wired but no validating run; protocol-ready; toy-only)

### Architecture bets — 8 of 9 are `scaffold` (wired, no ablation CI yet)
`claim_router`, `graded_decision`, `layered_memory`, `planner_mcts`,
`predictive_world_model`, `selective_tool_router`, `hybrid_memory`,
`ontology_concept_discipline`. Each needs its closing experiment: wire into
`run_case` behind a flag and show a delta whose CI excludes 0. (The 9th,
`verifier_synthesis_over_proof_kernel`, is also scaffold pending the Lean eval.)

### Long-context bets — mostly `partial`, one `scaffold`
selective-tool-router (`scaffold`); verifier-gated-long-context, hybrid-memory,
council-of-small-models, verifier-as-reward, compression/recall (`partial` —
blocked on independent corpus + live non-mock run + third-party review).

### "Toy reference" theories (implemented, but not real implementations)
Program induction (template-matching, `level3Evidence:false`); MCTS planning
(scripted simulator); predictive world model (lookup table over a few traces);
layered memory (permission-gated dict); active inference (interface; no live
fact-check hookup). Marked `candidateOnly:true` / `depth:"toy-reference"`.

### Run-but-not-yet-validated, or never run (from the ledger)
- **RLVR / verifier-as-reward** live runs: math, code, adapter — `rlvr-live-run-not-yet-gated`, `rlvr-code-live-run-not-yet-gated`, `rlvr-math` cleared the rung but no live capability gate. RLVR adapter was *judged but failed the agreement bar* (κ too low, capability at chance — `rlvr-adapter-kappa-2family-below-bar`).
- **Grounded-answer gate** improves selective accuracy only at marginal significance — `grounded-gate-not-yet-validated`.
- **Agent-faithfulness benchmark** wired, no multi-family live validation / third-party trajectory pack.
- **SSIL compounding (live)** and **flywheel capstone** close only on self-authored toy data — `ssil-compounding-live-not-gated`, `flywheel-capstone-deriged`.
- **Safe parametric plasticity Layer-1** (LoRA / weight deltas): interface wired, **no GPU training run**.
- **Conformal abstention gate, truth/deception probe, CoT-faithfulness bench, abstention-aware scoring, prover-verifier self-play** — all real-but-**synthetic-only / single-run / offline**, not yet on real model output.
- **Hidden-reviewer packs (Level 4):** protocol-ready, never independently completed (deepseek/grok packs all incomplete, backend-broken, or self-authored).
- **External public benchmarks (Level 5)** and **third-party clean-clone replication (Level 6):** *not run*. Third-party held-out pack is empty.
- **DGX-Spark iteration / judge-farm / distill tiers:** not run / not validated.
- **Wisdom-4B M2** synthetic-data egress: see §3 (NO-GO).

---

## 3. DOESN'T WORK (falsified / negative / dead-end — recorded as evidence)

- **Anti-fabrication gate value on strong models — FALSIFIED.** `pressure-calibration-falsified`: frontier models abstain or debunk pressure vectors on their own; the gate's effect is *behavior substitution*, not fabrication *prevention*. Re-confirmed by `abstain-pack-unambiguous-split`: on no-fabrication-tendency models the measured anti-fab signal was **entirely a deterministic-scorer artifact**. (The gate's real value lives at the weak-model boundary — see §1.)
- **Grounded gate vs source contamination — NEGATIVE (original).** `grounded-gate-source-contamination`: a grounded gate trusts and repeats contaminated sources → **zero** behavioral prevention. (Later *resolved* by adding an independent verifier channel — that fix is in §1.)
- **No automated labeler handles ambiguous hedged-attribution.** `w2-scorer-overflag-fixed-and-gold-standard` + `w2-kappa-gap-diagnosed`: marker scorer, LLM judge, and rubric-adjudicator all fail on hedged cases; the low inter-judge κ was a scorer label artifact, not real disagreement. Honest dead-end.
- **DreamerV3-style world model — does not learn.** `path-a-world-model-not-promoted`: discrete-latent world model fails to learn a predictive representation on mixed-outcome reasoning traces; not promoted.
- **Lean-4 proof search — blocked/negative.** `path-b-lean-proof-search-blocked`: produces no novel verified proofs under this architecture / compute.
- **Wisdom-4B M2 synthetic-data volume — NO-GO.** `sophia-wisdom-4b-m2-volume-below-target`: teacher-egress synthetic generation fell below target; root cause re-diagnosed as the corpus, not egress.

---

## Bottom line
- **Strongest validated theses:** provenance/verifier gate + fail-closed abstention reduce hallucination; self-consistency calibration is the one result validated on *external* public data; deterministic verifiers (legal-citation, holding-faithful) and the measured-improvement loop hold up.
- **Biggest gaps (untested):** every "AGI-shaped" module (planning, world model, hybrid/layered memory, tool routing) is scaffold or toy; all weight-level learning (RLVR/LoRA) is ungated; **no external benchmark and no third-party replication have ever run** (Levels 5–6).
- **Honest negatives:** the anti-fabrication gate does *not* help strong models (only weak ones); grounded gating alone is contamination-vulnerable; a learned world model and Lean proof search both failed; one synthetic-data pipeline was a NO-GO.
