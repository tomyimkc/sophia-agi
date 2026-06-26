# Verifiable Sophia — Strategic Development Plan

> **Status:** Strategic recommendation (judge + strategist pass). Not a validated capability claim.
> **Author of this doc:** advisor session 2026-06-26. Grounded in repo state at v0.9.1, commit `bd22499`.
> **No-overclaim boundary:** This document is *planning*. Nothing here clears `canClaimAGI`. `canClaimAGI` stays **false** until the criteria in §4 are met by a third-party-authored, entity-disjoint pack.

---

## 0. The hard truth (read first)

Sophia's **single fully-validated headline claim** is a **−12.5pt hallucination delta on `dolphin-llama3:8b`** (an *uncensored 8B local tune*), plus a self-authored, deterministic-scored calibration pack (+19.4% fabrication reduction, corroborated by two judge families, κ=0.74). Everything else in `RESULTS.md` is explicitly labelled "harness evidence, not a capability claim" (e.g. `coding_eval` is `2+2==4`, `memory_eval` labels itself "not a long-horizon memory capability claim"), "illustrative," or `canClaimAGI: false`. The failure ledger (`agi-proof/failure-ledger.md`) is the most honest document in the repo, and it is correct to keep `canClaimAGI` false.

The **load-bearing structural fact** comes from the project's own evidence, ledger entry `calibration-advantage-is-model-dependent-2026-06-25`:

> *"the anti-fabrication advantage… VANISHES on a STRONG model (deepseek-v3 raw already 0/12)… value decays toward 0 as the base model strengthens."*

In plain terms: Sophia's validated value is currently **anti-fabrication on weak/uncensored models**, and that delta shrinks as base models improve. The "wisdom before intelligence" layer is, today, a *correction for models that are bad at being honest* — a decaying asset.

**Every option below is judged against one question: does it build a capability that does not decay as language models improve?**

A soundness proof, an exact-match legal-citation verifier, an exec-pass test, and a derivable `abstain` rule do **not** vanish when GPT-6 stops hallucinating. That is the only non-decaying axis, and the plan must anchor on it.

### Corrections to the prior framing (provenance discipline applied to the meta-question)
- **No polyglot exists yet.** `find` returns **0** `.pl` / `.lisp` / `.jl` / `.hs` / `.ex` / `.zig` files in the repo. The codebase is Python + ~50 Rust files (one HNSW workspace). "Polyglot hybrid" is currently aspirational, not actual.
- **Branches:** only **4 local**. The "80" are remote `claude/*` automation branches. This is a solo dev *orchestrating many AI coding agents* — a process fact, not just a headcount fact.
- **`CONTRACT.md` and `AGENTS.md` are git-crypt encrypted** (binary). The governance contract itself could not be read; judgment is from code + plaintext failure-ledger + `RESULTS.md`.
- **The flywheel is real but small:** `selfextend/` ≈ 1010 LOC of Python with a genuine promote-only-on-held-out invariant — still 100% Python with LLM-as-scorer callables.

---

## 1. The three options

**Option A — Deepened Python + Rust (status quo, compounding rungs).**
Merge #129 (code-RLVR, de-rigged flywheel), #131 (NVFP4/Spark), close more self-extension rungs, keep conscience/flywheel in Python, expand verifier coverage. The inertial path.

**Option B — Full polyglot neuro-symbolic.**
Prolog conscience + Lisp flywheel + dependent-type invariants + Julia compute + Elixir agents + Zig kernels. The strongest *theoretical* fit for the identity. **5 new language ecosystems for one person is a complexity-death sentence.** Judged as written, this fails the feasibility test.

**Option C — Pivot: Sophia as a *verification engine*, not an LLM-augmentation layer.**
Re-anchor on the one asset that does **not** decay with base models: **verifier synthesis and machine-checkable gates.** "Can a claim be machine-verified, and can we *synthesize* the verifier?" becomes the product; the conscience becomes a provable (logic-engine-backed) fail-closed controller over it.

---

## 2. Scored comparison (1 = bad, 5 = excellent; bold = decisive)

| Criterion | A. Deepen Py+Rust | B. Full polyglot | C. Verification pivot |
|---|---|---|---|
| **Advances toward real AGI/ASI (not a better wrapper)** | 2 | 3 (only if it ships) | **4 — only non-decaying vector** |
| **Fidelity to provenance/conscience identity** | 4 | **5** | **5** |
| **Technical depth (symbolic, verifiable self-improve, epistemic, neuro-symbolic)** | 2 | **5 (if delivered)** | **4** |
| **Feasibility for solo Tom** | **5** | **1** | **3 (scoped right, feasible)** |
| **Risk (complexity, ecosystem loss, safety regression, overclaiming)** | low complexity / **high overclaim drift** | **very high** | medium |
| **Out-of-the-box / addresses the un-considered** | 1 | 3 (right idea, wrong dose) | **5 — reframes the decaying-asset problem** |
| **Weighted total** | weak | theoretically strongest, practically fatal | **winner** |

**Verdict:** reject full-B. Reject pure-A as the *strategy* (fine as the *tactics*). The answer is **a constrained B-substrate inside C's reframing** — "Verifiable Sophia."

---

## 3. Recommended integrated plan: "Verifiable Sophia"

Keep Python as the spine. Add **exactly one** symbolic substrate under the two components where "LLM-judged" is currently load-bearing, and **exactly one** dependently-typed core for the fail-closed invariants. Everything else stays Python/Rust.

**Discipline:** *a new language is admitted only if it makes a currently-LLM-judged safety property machine-checkable.*

### 3.1 Language / substrate mapping

| Component | Current | Target | Why |
|---|---|---|---|
| **Provenance Gate** | Python, LLM-judge consensus | **Datalog/Prolog rules (embedded)** for the closed-world fail-closed abstention core; LLM-judge retained only for the semantic-support sub-question | Abstention on "no verifier covers this claim" must be **provable & exhaustive**, not a 0.74-κ judge vote. This is where "wisdom before intelligence" actually lives. |
| **Conscience Kernel / Moral Gate** | Python, 7-path, frozen-dataclass verdict | **Dependent types (F\* or Agda) for the ~3–5 invariants that must be fail-closed**; Python stays the orchestrator | Make *illegal states unrepresentable* for the safety-critical paths only. Don't type the whole kernel — type the invariants that, if wrong, mean a fabricated attribution ships. |
| **Self-Extending Flywheel / verifier synthesis** | Python, LLM-as-scorer | **Python + Lean 4 backend** for verifier-validation (half-present: `agent/lean_backend.py`, `agent/proof_search.py`); Python keeps proposal/scoring | Verifier synthesis is Sophia's only **non-decaying** asset. Lean-checking makes "the flywheel learned a sound rule" a *theorem*, not a held-out accuracy number. |
| **OKF Wiki / belief graph** | Python graph + `consistency_check` | **Datalog** as the query/contradiction/retraction engine | Contradiction detection is *literally* a logic-programming problem; makes "is this belief graph consistent" decidable & auditable. |
| **Agent Harness / council** | Python, three-path | **Stay Python.** Optionally an actor model *in Python* (`asyncio` + supervision) | Actor concurrency is nice-to-have, not load-bearing for AGI. Elixir isn't worth the tax. |
| **RAG / indexing / HNSW** | Rust HNSW + Python | **Stay Rust + Python.** No Julia | Numerical-compute tax not justified; Rust covers the hot path. |
| **MoE / quant / serving** | Python (`moe/quant.py`) | **Stay Python.** No Zig | Premature; the bottleneck is not a kernel. |

**Net new substrates: 2** (embedded Datalog/Prolog; Lean 4 + a tiny F\*/Agda invariant core). Not 5.

### 3.2 Integration without breaking Docker / MCP / harness
- **No new long-running services in phase 1.** Embed the logic engine **in-process** via `janus` (SWI-Prolog↔Python), `problog`, or `soufflé`. Conscience & provenance gate call it as a library — same process, same Docker image, MCP surface unchanged.
- **Lean 4 as a subprocess verifier**, not a service. `lean_backend.py` already shells out to `lean`; formalize it as the *validator* for synthesized verifiers, invoked at the flywheel's promote step. CI installs Lean via existing toolchain scripts.
- **Dependent-type invariants as a compiled checker.** Write the 3–5 invariants in F\*/Agda, *extract* them to a Python-callable checker (or emit runtime guards), assert in `conscience_runtime.py`. Source of truth = the formal spec; Python is generated. This is the partial-evaluation / compilation pattern: write the invariant once, formally; it becomes both a proof and a runtime check.
- **Split behind a service boundary only where it earns its keep** (Unix-socket/gRPC with a typed IDL when the logic engine or Lean validation becomes a bottleneck). Don't pre-split.

### 3.3 Concrete mechanisms to add (the "out-of-the-box" content)
1. **Native abstention as a logic program.** Replace the closed-world portion of abstention with `can_answer(Claim) :- verified_source(Claim, Src), supports(Src, Claim).` Failure → `abstain` is *derivable*, not voted. Highest-leverage move; literal embodiment of "wisdom before intelligence."
2. **Partial evaluation / staging in the flywheel.** `selfextend/evolve.py` already has the right invariant (promote only on held-out). Add **partial evaluation**: specialize a verifier against the claim family it was synthesized for → faster, narrower, Lean-checkable specialized rule. The historic PL-for-AGI move (Futamura projections) applied to verifier synthesis — *not yet in the repo*.
3. **Provable moral invariants (dependent types).** Type the verdict-returning function so `allow` is *impossible* when `hard_rule_violations ≠ ∅`. (Unrepresentable, not unit-tested.)
4. **Modal/epistemic logic layer over the OKF graph.** "Tom *believes* P" vs "P is *verifiably true*" vs "P is *asserted by source S*" are different modalities currently collapsed into one graph. Datalog + modal operators make `metacognition.py` *derive* uncertainty from structure, not heuristic.
5. **Verifiable self-modification.** Every promoted verifier ships with a **Lean-checked soundness proof** on its declared domain. Promotion requires held-out Δ ≥ baseline **AND** soundness proof. Closes the gap between "self-extension" and "verifiable self-improvement" — the actual AGI-relevant property.

### 3.4 Keeping the no-overclaim culture & benchmark discipline
- **Keep `canClaimAGI: false`** until a verifier-synthesized rule, **soundness-proven in Lean**, improves held-out accuracy on a **third-party-authored, entity-disjoint** pack across **≥3 seeds** with CI excluding zero, judged by **≥2 independent families** with **κ≥0.40**. The logic/dependent-type work *raises* the bar, never lowers it.
- **Kill the decaying-claim framing.** Stop headlining the anti-fabrication Δ on weak models. Re-label honestly: *"abstention layer effective on weak/uncensored models; advantage decays toward zero on frontier models (self-evidenced)."* Replace the flagship with the **verifier-synthesis compounding curve** — the one asset that does not decay.
- **The failure ledger stays the source of truth.** Every new mechanism gets a ledger entry with boundary conditions *before* it's built (extend `agi-proof/PRE-REGISTRATION.md` to the logic/Lean work).
- **Third-party independence is the real gate, not more seeds.** The ledger repeatedly shows "self-authored pack" as the residual caveat. **One** real third-party pack is worth more than 10 more self-runs.

---

## 4. Phased roadmap

### Short-term (0–3 months) — highest-leverage, compounds the non-decaying asset
1. **Merge the three open PRs with the conflict map resolved** (#129 code-RLVR, #131 Spark, #132 IP). The handover doc already did the conflict analysis — execute it. Land in-flight lanes before opening new ones.
2. **Embed a Datalog engine (`soufflé` or `problog`) under the provenance gate.** Port the closed-world abstention rules (the `legal_citation_exists`-style verifiers) into it. **First experiment:** does the logic-engine abstention reproduce the validated 12.5pt Δ *deterministically* (no LLM judge) on the existing 290-case set? If yes → a **stronger** version of the only validated claim: judge-free, reproducible.
3. **Lean-check the existing `verifier_synthesis` output.** Wire `lean_backend.py` into the flywheel promote step as a *soundness* gate. **First experiment:** on the math-RLVR sympy verifiers, can Lean prove input→output soundness? Converts "adapter +0.10 held-out" into "adapter +0.10 *via sound verifiers*."
4. **Write the 3 dependently-typed moral invariants** (F\* or Agda), extract to Python guards, assert in `conscience_runtime.py`. **First experiment:** do existing `test_conscience_*.py` still pass *and* can you construct a test the type system rejects (a `block`-when-hard-rule-fires case)?

### Medium-term (3–9 months) — architecture shifts
1. **Promote the logic engine to the OKF contradiction/retraction layer.** Belief-graph consistency as a Datalog query. This is where neuro-symbolic integration becomes real: graph = neural side (embeddings, retrieval); Datalog = symbolic side (consistency, retraction, derivation).
2. **Partial evaluation in the flywheel.** Implement verifier specialization (Futamura-style) → narrow, Lean-checkable rules. Measure: does specialization reduce verifier false-alarms on held-out?
3. **Modal/epistemic operators** over the OKF graph; `metacognition.py` *derives* uncertainty from structure.
4. **One real third-party pack.** Commission/solicit an independent hidden pack + independent semantic review. Highest-value validation action; orthogonal to architecture work — run in parallel.

### Long-term (9–24+ months) — autonomous reasoning under governance
1. **Verifiable self-modification as the core loop:** promote = held-out Δ **AND** Lean soundness proof **AND** conscience-gate (logic-engine) approval. At this point the system improves itself on a non-decaying axis while provenance & conscience are *theorems*, not classifiers. That is the only configuration that honestly earns "self-extending + verified conscience."
2. **Code-RLVR as the second compounding domain** (PR #129's foundation) — exec-pass tests are base-model-independent verifiers, so on-strategy.
3. **Hold `canClaimAGI: false`** until the above produces cross-domain transfer on a third-party pack. **Do not move the goalposts to accommodate progress.**

---

## 5. What to explicitly *not* do
- **Do not adopt Julia, Elixir, Haskell-as-app-language, Zig, or a separate Prolog *service*.** Each is individually defensible; collectively they are the plan's most likely failure mode. The polyglot vision is correct in *principle*, fatal in *dosage*.
- **Do not frame verifier-synthesis or logic-engine work as "AGI progress"** in commits or docs. It is *infrastructure for non-decaying validation*. Overclaiming temptation is highest exactly here.
- **Do not let the multi-agent PR sprawl continue unmanaged.** Three open PRs with a handover-required conflict map is already coordination debt.

---

## 6. Single strongest direction & first action

**Direction:** Make Sophia's compounding asset the thing that does not decay as language models improve — **machine-checkable verification and provable, logic-engine-backed abstention** — by adding *one* embedded symbolic substrate (Datalog/Prolog) under the provenance gate and conscience, and *one* formal-verification backend (Lean 4, already half-wired) in the flywheel's promote step. Keep Python as the spine. Type only the fail-closed invariants. Everything else stays.

This is the only plan that (a) is feasible for one person, (b) genuinely differentiates Sophia from "yet another LLM orchestration framework," and (c) directly answers the self-admitted fatal flaw — because **a soundness proof and a derivable `abstain` do not vanish when GPT-6 stops hallucinating.**

**First action (this week):** Resolve & merge PRs #129/#131/#132 per the handover's conflict map to clear coordination debt, then run **one experiment** — port the closed-world abstention rules behind the existing validated 290-case provenance set into an embedded Datalog engine (`problog` or `soufflé`, in-process, no new Docker service) and test whether logic-derived abstention reproduces or beats the validated −12.5pt hallucination Δ **without any LLM judge in the loop**. If it does, you have just converted Sophia's one decaying validated claim into a non-decaying, deterministic, reproducible one — and that result, not a new feature, should set the direction for the next phase.

---

## Appendix A — Evidence base consulted
- `VERSION` (0.9.1), `git log` (1232 commits), `git branch -a`.
- `RESULTS.md` (published-results page, no-overclaim gate wording).
- `agi-proof/definition.md` (operational AGI definition), `agi-proof/failure-ledger.md` (full ledger, incl. `calibration-advantage-is-model-dependent-2026-06-25`).
- `agi-proof/PRE-REGISTRATION.md`, `agi-proof/REPLICATION.md`.
- `agent/conscience.py`, `agent/ssil_moral_parliament.py`, `agent/lean_backend.py`, `agent/proof_search.py`.
- `selfextend/*.py` (~1010 LOC), `okf/*.py`.
- `eval/results/{memory,coding}_eval.json` (both explicitly harness-evidence-only).
- `HANDOVER-FROM-GLM5.2.md` (open-PR conflict map).
- `CONTRACT.md` / `AGENTS.md` (git-crypt encrypted — not readable; judgment does not depend on them).

## Appendix B — Honesty flag on this very document
This plan is advisor analysis, not a claim. The two "first experiments" (Datalog reproduction of the 12.5pt Δ; Lean soundness on sympy verifiers) are **predictions**, not results. They must be run before any of this plan is cited as evidence. If the Datalog reproduction *fails*, the implication is serious: Sophia's validated advantage may be partly an LLM-judge artifact, not a real abstention property — which would itself be the most important finding of the phase and must be recorded in the failure ledger as such.
