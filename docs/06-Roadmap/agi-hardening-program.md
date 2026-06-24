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

## Phase 1 — Trust
**#2 Verified compositional reasoning** — adopt the **[VeriCoT](https://arxiv.org/abs/2511.04662)**
pattern: autoformalize each reasoning step, tag each premise by source type (grounded-fact /
commonsense / prior-step), solver-verify the chain → **proof-carrying answers** over the grounded
graph. Closes the ripple-effect / synthesis-hallucination gap the gate currently misses.
*Done when:* multi-hop answers carry a verified premise chain; unverifiable steps ⇒ abstain.

**#4 Poisoning-robust ingestion** — the threat is real and active
([PoisonedRAG](https://hf.co/papers/2402.07867), [RAGDefender](https://hf.co/papers/2511.01268)):
**k-independent corroboration + source-trust modeling + post-retrieval adversarial filtering**,
with retraction/counterfactual as remediation. *Done when:* a single well-sourced falsehood
cannot be admitted without k independent corroborations; poisoned-stream benchmark passes.

## Phase 2 — Scale + generality
**#5 Incremental belief revision** — recompute only the affected subgraph; shard + index; the
scaling harness already measures the cost curve. *Done when:* revision is sub-linear as facts
scale; the harness shows the new curve.

**#1 Skills (generality leap)** — a **[Voyager](https://arxiv.org/abs/2305.16291)/AutoSkill**-style
**executable-code skill library** (versioned, gated, compositional, forgetting-resistant, *no
fine-tuning*) **+ declarative verifier specs** (preconditions/effects/verifier) in the OKF graph.
Skills promote through the same anti-forgetting gate as facts. *Done when:* a skill learned in
one episode is reused later with 0 regression of prior skills (skill-retention benchmark).

## Phase 3 — Capstone: governed autonomous RSI (gated by all above)
**#6 Full autonomous RSI inside the inviolable cage.** The loop proposes + commits improvements
(new facts, skills, verifiers, corpus enrichment) autonomously, **but**: only in verifiable
domains (Judge-Code/RLVR); every commit passes verifier-based gates + the anti-forgetting
tripwire; any invariant breach ⇒ automatic rollback + halt; full append-only audit; kill-switch.
Substrate: `constitution/`, `conscience.py`, `verifier_synthesis.py`, `continual_plasticity.py`.
*Done when:* the loop runs unattended over a stream, improves a measured metric, and a red-team
proves no invariant can be driven false (rollback fires every time).

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
