# Hardening program: continual learning toward general AI

**Scope, stated plainly (per Sophia discipline).** This is a program to make the continual-
learning architecture **more general, robust, and safe** — *not* a claim of AGI, and not a
finish line. "Toward general AI" is the direction; "becomes AGI" is an overclaim this repo's
own gate rejects. Every capability below ships `candidateOnly` until it clears the
no-overclaim gate.

## Locked decisions (owner)
1. **Sequencing:** foundations-first (#7 + #3 before all else).
2. **Weights:** **strictly non-parametric** — weights frozen forever; all learning lives in
   the OKF graph + skill library. 0 catastrophic forgetting by construction. (Gated-LoRA
   skills **dropped**.)
3. **Skills substrate:** executable-code library **+** declarative verifier specs (no fine-tuning).
4. **Self-modification (#6):** **full autonomous RSI inside an inviolable cage** — the loop
   self-improves with no human in the loop, but **cannot modify** its safety invariants;
   verifier-based (never classifier) gates; automatic rollback on breach; append-only audit;
   hard kill-switch.

## Non-negotiable invariants (the cage — apply to every phase)
- Fail-closed: unverifiable ⇒ abstain/reject, never assert.
- Weights never change (non-parametric).
- Provenance discipline holds (0 forbidden attributions) under all self-modification.
- The anti-forgetting tripwire (retention regression ⇒ reject) is inviolable.
- Self-improvement only where **outcomes are verifiable** ([RSI-workshop finding](https://iclr.cc/virtual/2026/workshop/10000796); classifier gates fail at scale — use verifiers).
- The invariant set itself is **out of the loop's reach**; violating any ⇒ auto-rollback + halt.

---

## Phase 0 — Foundations  ✅ delivered (candidate)
**#7 Symbol grounding / stable identity** — canonical entity+sense layer with **versioned IDs**,
so "the same fact" is stably addressable across contradictions and time. Pairs with the
existing temporal-validity layer. *Done when:* entity resolution is deterministic + tested, and
a fact's identity survives revision/retraction/restore round-trips.
→ **`agent/symbol_identity.py`** (`canonical_id`/`resolve_all`/`is_ambiguous`, `build_sense_index`,
`lineage`/`current_version`/`stable_identity`/`version_tag`, `identity_round_trip_report`);
tests in `tests/test_symbol_identity.py` (CI `validate-core`). Resolution is deterministic
(sorted, hash-seed-independent); `stable_identity` is shared across a supersession chain and
proven invariant under a `forget`+`restore` round-trip; `version_tag` is content-addressable and
reproducible across processes.

**#3 Calibrated metacognition** — a per-domain **empirically-calibrated** competence model over
the graph (build on `semantic_entropy`, `conformal_gate`, `calibration`) feeding the gap-loop.
*Done when:* abstention is calibrated (reliability diagram), and "what to learn next" is driven
by measured weakness, not heuristics.
→ **`agent/competence_model.py`** (`reliability_diagram`, `build_competence_model` →
`CompetenceModel.competence`/`threshold`, `learning_priorities`, `competence_gap_worklist`);
tests in `tests/test_competence_model.py` (CI `validate-core`). Per-domain ECE/AURC/selective-vs-base
risk reuse `agent.calibration`; the conformal answer threshold reuses `agent.conformal_gate`;
unseen domains fail closed (competence 0.0, threshold 0.0); `learning_priorities` ranks
weakest-first by **measured** deficit, bridging into `agent.knowledge_gap_log`.

## Phase 1 — Trust  ✅ delivered (candidate)
**#2 Verified compositional reasoning** — adopt the **[VeriCoT](https://arxiv.org/abs/2511.04662)**
pattern: autoformalize each reasoning step, tag each premise by source type (grounded-fact /
commonsense / prior-step), solver-verify the chain → **proof-carrying answers** over the grounded
graph. Closes the ripple-effect / synthesis-hallucination gap the gate currently misses.
*Done when:* multi-hop answers carry a verified premise chain; unverifiable steps ⇒ abstain.
→ **`agent/proof_carrying_reasoning.py`** (`autoformalize_claims`, `verify_step`, `verify_chain`,
`proof_carrying_answer`); tests in `tests/test_proof_carrying_reasoning.py` (CI `validate-core`).
Grounded-fact premises must be `okf.is_grounded` and clear an effective-confidence floor; prior-step
premises must cite an *earlier verified* step (forward/self-reference earns no warrant); commonsense
premises abstain by default. The whole autoformalized claim set is run through
`agent.formal_verifier.check_no_contradiction` (z3 optional, fail-closed) → `verified` /
`abstain` / `rejected`; verified answers carry `agent.symbol_identity` stable-identity citations.

**#4 Poisoning-robust ingestion** — the threat is real and active
([PoisonedRAG](https://hf.co/papers/2402.07867), [RAGDefender](https://hf.co/papers/2511.01268)):
**k-independent corroboration + source-trust modeling + post-retrieval adversarial filtering**,
with retraction/counterfactual as remediation. *Done when:* a single well-sourced falsehood
cannot be admitted without k independent corroborations; poisoned-stream benchmark passes.
→ **`agent/poison_resistant_ingestion.py`** (`SourceTrust`, `assess_item`, `adversarial_filter`,
`ingest_stream`, `run_poison_benchmark`); tests in `tests/test_poison_resistant_ingestion.py`
(CI `validate-core`). Admission requires ≥k *distinct* independence groups above a trust floor **and**
pooled confidence (trust-weighted, deduped by group via `agent.corroboration`) above a floor — so a
single source, or a Sybil sharing one group, can never alone meet k≥2 however confident; low-trust
sources are downweighted; values conflicting with an established consensus are flagged as suspected
poison and quarantined; proven-malicious sources are remediated via `agent.unlearning.Unlearner`
(cascade un-grounds dependents). Seeded poison-stream benchmark passes deterministically.

## Phase 2 — Scale + generality  ✅ delivered (candidate)
**#5 Incremental belief revision** — recompute only the affected subgraph; shard + index; the
scaling harness already measures the cost curve. *Done when:* revision is sub-linear as facts
scale; the harness shows the new curve.
→ **`agent/incremental_revision.py`** (`IncrementalReviser.add_fact`/`belief_state`/`kept_ids`,
`incremental_sweep`, `compare_to_full`). Maintains contradicts-adjacency + reverse-`derivesFrom`
indices built per-add (O(edges of the new page), never O(N)); `add_fact` re-resolves only the
affected closure (new node + its `derivesFrom` ancestors + contradiction neighbours + transitive
dependents). Proven **belief-state-identical** to the full `agent.belief_revision_policy.resolve_conflicts`
oracle at several N and order-independently; `maxAffectedNodes` stays **bounded (=2)** across
N∈{30,60,120} while full recompute touches ~N — `subLinear=True`. Timing reported, never asserted.
Tests in `tests/test_incremental_revision.py` (CI `validate-core`).

**#1 Skills (generality leap)** — a **[Voyager](https://arxiv.org/abs/2305.16291)/AutoSkill**-style
**executable-code skill library** (versioned, gated, compositional, forgetting-resistant, *no
fine-tuning*) **+ declarative verifier specs** (preconditions/effects/verifier) in the OKF graph.
Skills promote through the same anti-forgetting gate as facts. *Done when:* a skill learned in
one episode is reused later with 0 regression of prior skills (skill-retention benchmark).
→ **`agent/skill_library.py`** (`Skill`, `SkillLibrary.learn`/`invoke`/`version_tag`/`to_update_candidate`,
`verify_skill`, `skill_retention_benchmark`). Skill code is sandboxed through `agent.program_induction`'s
safe-AST compiler (imports / `eval` / dunders / attribute escapes rejected — all four spot-checked);
composites resolve only already-verified deps. `learn` admits iff the code is safe, passes its own
verifier, **and** the anti-forgetting tripwire holds (no dependent skill's verifier regresses) — so a
breaking upgrade is **rejected** and the dependent stays passing; `to_update_candidate` renders the
promotion through `agent.continual_plasticity` for parity. `skill_retention_benchmark` proves
`forgottenSkills == 0` with a skill reused across later episodes (deterministic).
Tests in `tests/test_skill_library.py` (CI `validate-core`).

## Phase 3 — Capstone: governed autonomous RSI (gated by all above)  ✅ delivered (candidate)
**#6 Full autonomous RSI inside the inviolable cage.** The loop proposes + commits improvements
(new facts, skills, verifiers, corpus enrichment) autonomously, **but**: only in verifiable
domains (Judge-Code/RLVR); every commit passes verifier-based gates + the anti-forgetting
tripwire; any invariant breach ⇒ automatic rollback + halt; full append-only audit; kill-switch.
Substrate: `constitution/`, `conscience.py`, `verifier_synthesis.py`, `continual_plasticity.py`.
*Done when:* the loop runs unattended over a stream, improves a measured metric, and a red-team
proves no invariant can be driven false (rollback fires every time).
→ **`agent/governed_rsi.py`** (`CAGE_INVARIANTS` (frozen tuple), `Proposal`, `GovernedRSI`
(`step`/`run`/`check_invariants`/`kill`/`audit_log`), `red_team_report`); tests in
`tests/test_governed_rsi.py` (CI `validate-core`). The `step()` pipeline is fail-closed in order:
halted-noop → **tamper⇒reject+halt** → weight-update⇒reject (`weights_frozen`) → **verifiability via
`verifier_synthesis.synthesize`** (abstain⇒reject — verifier-based, never a classifier, per the
RSI-workshop finding) → shadow-apply + verifier `.gate` + poison check (`poison_resistant_ingestion`)
+ provenance (`run_attribution_checks`) + anti-forgetting pre-check → **commit then re-audit all
invariants; any breach ⇒ rollback + halt** (the inviolable backstop). `CAGE_INVARIANTS` is an
immutable tuple with no mutation path; the audit is append-only on a monotonic `seq`; a hard
kill-switch makes `step()` a no-op. **A 4-agent red-team independently attacked each invariant and
could not breach the cage** (`red_team_report().ok == True`, `anyInvariantDrivenFalse == False`,
reject/rollback fired every time).

---

## Status: all 7 factors delivered (candidate)
Every phase is implemented, offline-tested, CI-wired, and built through a build → adversarial-verify
→ (red-team) → cage-compliance multi-agent pipeline. Per Sophia discipline these ship `candidateOnly`
until they clear the candidate → validated gate (≥2–3 distinct judge families, κ ≥ 0.40, ≥3 runs,
CIs, independent replication). "Toward general AI" remains the direction; this is honest, measured,
gated machinery — not an AGI claim.

## Next milestone: closed-loop lifelong accumulation  ✅ delivered (candidate)
The seven factors are isolated modules until they are shown to **compound**. This milestone wires all
seven into one loop and measures, honestly, whether the system **net-accumulates capability over a long
stream without catastrophic forgetting**.
→ **`agent/lifelong_accumulation.py`** (`make_lifelong_stream`, `run_accumulation`, `accumulates_cleanly`)
+ `tools/run_lifelong_accumulation.py`; tests in `tests/test_lifelong_accumulation.py` (CI `validate-core`).
The headline metric is a **net-capability-accumulation curve** measured over a *fixed/growing held-out
query set* (re-asked every episode, so a rising count means genuine retained+new capability, not a gamed
just-taught measure). Observed (seed 0, 12 episodes): graph-backed cumulative-correct **2→24 monotone**;
the frozen `ParametricBaseline` stays flat at the t0 count and **drops** on the deliberate-retraction
episode (a weight model can't unlearn) — `finalGraphCorrect ≫ finalBaselineCorrect`. **`unintendedForgetting == 0`**
across the whole stream (catastrophic forgetting separated from `deliberateUnlearning`). The **governed-RSI
cage is genuinely in the loop**: every fact is admitted via `GovernedRSI.step` (verifiable + poison-clean +
provenance-clean); the seeded poisoned and forbidden/parametric proposals are **rejected and never enter
the graph**; `cageBreaches == 0`. Competence-model `learning_priorities` + the gap-log worklist supply the
measured "what to learn next" signal; `symbol_identity` attaches version-tag citations; the control-flow
gap is measured separately with a real lexical router. A dedicated anti-gaming honesty audit (run the
report, check the curve/baseline/forgetting/cage are real) passed with zero violations.

**Open follow-on:** the live multi-judge grading seam (`LLM_JUDGE_HOOK`) is left deliberately uncalled —
running the accumulation curve under ≥3 independent judge families is the step that moves this from
`candidate` to `validated`. The control-flow gap is the next capability bottleneck to attack once the
loop is judged (it is 0 on the clean synthetic stream but tracks router quality — a crippled router shows
gap ≈ 0.98).

---

## Cross-cutting discipline
- Every capability: offline deterministic tests + CI; CPQA-style measurement; cross-family
  judged validation where outputs are graded; honest failure-ledger updates.
- candidate → validated gate: ≥2 (ideally ≥3) distinct judge families, κ ≥ 0.40, ≥3 runs, CIs,
  independent replication. Nothing self-certifies.

## Dependency rationale
#6 (autonomous RSI) is last by necessity: safe self-modification presupposes calibration (#3),
verified reasoning (#2), poisoning defense (#4), stable identity (#7), and scale (#5). Building
#6 before them is the configuration the field shows is unsafe.

## Explicit non-goals
Not an AGI claim. No weight training (non-parametric). RSI confined to verifiable domains. The
deliverable is honest, measured, gated machinery — same as everything else in this repo.
