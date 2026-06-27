# Open-Problems Self-Growth Framework — Rigorous Critique

> **⚠️ Restoration + supersession banner (restored to `main` 2026-06-27).** This
> roadmap was authored on branch `feat/governed-sparse-quant` (commit `d78a65e2`),
> which never merged — its scoped-down sibling merged instead, dropping this file.
> It is restored here **verbatim** (content below unchanged from `d78a65e2`) so the
> cross-references in `docs/06-Roadmap/Lean-L0-Trace-Deadlock.md` ("critique §4",
> "critique §5.1") resolve on `main`. **One timeline is superseded:** §5 step 1's
> "get to L0 *this month* — one real Lean green check" is obsolete. The lean-dojo
> `trace()` deadlock that blocks L0 has since been reproduced on macOS-arm64 and (per
> PR #189) on Linux/CI — see `docs/06-Roadmap/Lean-L0-Trace-Deadlock.md`. L0 is a
> *blocked* research/engineering problem, not one install away. The doc's own
> load-bearing claim — "without this, everything above is speculative" — stands
> unchanged; only the ETA is stale. (For reference: steps 3 and 4 of §5 have since
> shipped as PR #147 and PR #157.)

**Status:** evaluation, not a claim. Documents the gaps in a proposed "use the
unsolved Millennium / open-physics problems as primary self-growth benchmarks"
framework, grounded in the *actual* deployed v0.9.1 state.
**Author:** GLM-5.2 (ZCode), 2026-06-26.
**Scope:** self-growth / RSI measurement plan for Sophia-AGI. Read alongside
`docs/06-Roadmap/Two-Paths-To-Novelty.md` (Path B = Lean proof search) and
`agi-proof/agi-verification/agi-verification-report.json`.

---

## 0. Corrections to the framing context (verified against the repo)

Three characterizations in the original proposal needed tightening before scoring:

1. **The 0% fabrication / −12.5pt hallucination result is real but coverage-limited.**
   `agi-proof/benchmark-results/published-results.json`: `delta: 0.125`,
   CI `[0.0556, 0.1944]`, `falsePositiveCost: 0.0` — on **n=24 false cases** with
   **coverage 0.3462**. The gate fires on ~35% of items and abstains on the rest.
   Defensible, preregistered, multi-judge — but it is an *abstention result under
   bounded coverage*, not a blanket "0% fabrication across all inputs." Any
   framework built atop it must not over-anchor to the bare number.
2. **"Proof ladder at Level 2" is exactly right and honestly bounded.**
   `agi-verification-report.json`: `highestMachineVerifiedLevel: level2`,
   `targetPassed: false` (level 3), `canClaimAGI: false`. Recommended public
   wording is "AGI-candidate verifier-gated epistemic agent framework." This is
   the strongest piece of epistemic discipline in the project.
3. **The Lean / PhysLean integration is wired but never run.** `agent/lean_backend.py`,
   `agent/proof_search.py`, `LeanProofSession` exist and are fail-closed / opt-in /
   `candidateOnly`. But `lean_available()` returns `False` in the default/CI env,
   the LeanDojo calls are defensive wrappers the code itself flags as
   version-variable, and **no gated run has ever produced a Lean-verified,
   non-retrieved proof.** Infrastructure on paper, not a demonstrated capability.
   That distinction is load-bearing for everything below.

---

## 1. Strengths (specific)

- **Formal-proof is the *only* correct domain given the roofline.** The
  `reasoning/deliberation_roofline.py` result (every output ∈ train ∪ retrieved,
  filtered by a verifier) logically entails that self-certifying formal proofs are
  the single novelty pathway that does not break fail-closed discipline. Tiering
  from Lean 4 + LeanDojo is the right *direction* — the open analogue of
  AlphaProof, mapping 1:1 onto the generate→retrieve→verify idiom.
- **Mapping to existing seams is accurate.** `agent/verifier_synthesis.py` is
  genuinely well-built: disjoint fit/val/test splits, independent oracle labels,
  held-out test precision/recall, an explicit WITHOUT-meta-verification ablation,
  abstention on out-of-library tasks. The correct substrate to extend.
- **Negative results as valid progress is methodologically right.**
  Independence/unsolvability results and "shift-degenerate → substrate is bounded"
  are legitimate outputs. `agent/verified_world_model.py`'s `shiftDegenerate`
  verdict already encodes this discipline.
- **Proof-ladder alignment is correct.** Pushing Level 2 → Level 3 via a
  machine-checkable artifact (a real Lean-verified proof) is the literal
  definition of moving up the project's own ladder. Neither hype nor overreach.

## 2. Weaknesses & gaps (direct)

- **Premature benchmarking.** Using the 6 Millennium Problems and major open
  physics problems as *primary self-growth benchmarks* is mis-scaled by ~4–5
  orders of difficulty relative to where the formal-math stack actually is
  (never run). FrontierMath Tier 1 (~olympiad-hard) is still beyond current SOTA
  agents; the Millennium problems are not a benchmark, they are a *research
  frontier that may be inaccessible for decades*. As a **headline target** they
  produce theater, not signal.
- **The self-growth metrics are ill-defined for this substrate.** "Performance
  delta after self-extension iterations" is circular *unless the held-out set is
  fixed, disjoint across iterations, and out-of-distribution from the flywheel's
  training traces.* `selfextend/evolve.py` enforces this for scalar artifacts —
  but `propose_verifier_candidates` is a **deterministic single-candidate
  decision-stump synthesizer** (`n: int = 1`). That is a held-out-gated rule
  fitter, not RSI. Iteration count over it is not a "self-growth metric."
- **The RSI alignment-of-improvers problem is unaddressed.** The hard RSI
  result (DGM, HyperAgents): *task skill and self-modification skill are
  different axes*; an agent can improve at theorem-proving without improving at
  improving-itself. The framework conflates "got better at Lean" with "got better
  at self-extension." These must be measured separately or "self-growing" is
  empty. The conscience/kernel/OKF stack has no hook measuring the *meta*-process.
- **PhysLean / physics formalization is a stronger claim than admitted.**
  PhysLean (Lean 4 physics) is far less mature than mathlib. Formalizing even
  the Yang–Mills mass gap or a vacuum-energy sub-calculation in Lean is itself an
  open research problem, not a benchmark one applies. You cannot "integrate
  PhysLean" as a Tier-1 step; you'd be *contributing to* PhysLean. The proposal
  treats physics formalization as Lean theorem-proving with different lemmas. It
  isn't — it requires the *formalization itself* (contested definitional choices)
  before any proof search begins.
- **Verification cost of large proofs is missing.** A Millennium-class attempt
  produces proof objects and tactic searches that are enormously expensive to
  elaborate; a solo dev has no compute envelope for RL-over-Lean-search at
  AlphaProof scale. `search_proof`'s `max_nodes=50`, `max_depth=12` defaults are
  toy-scale. No feasibility analysis of *who verifies the search itself is
  tractable*.
- **Embodiment / simulation grounding for physics is absent.** "Self-growing
  ability" on physics implicitly assumes the system can *do physics* —
  numerically, symbolically, with world-model reasoning. Sophia has none of that
  substrate. The world-model path (DreamerV3-style RSSM) is tested on *harness
  decision traces*, not physical dynamics. No bridge from "learned dynamics over
  my own logs" to "learned dynamics of a Navier–Stokes flow."
- **Conscience/kernel don't help here, and pretending they do is a trap.** The
  moral parliament / metacognition / 7-path conscience kernel is an *epistemic
  and behavioral* safety layer. It contributes nothing to *whether a proof is
  correct* (Lean does that) or *whether a physics model generalizes* (held-out
  dynamics do that). The framework leans on it as if it were a capability
  amplifier. Its value is exactly that it gates against overclaiming — so its
  *right* use here is to refuse the Millennium-problem framing.

## 3. Critical corrections / clarifications needed

1. **Demote open problems from "primary benchmarks" to "north-star reference
   points."** A benchmark must be (a) attemptable, (b) graded partial credit,
   (c) enough trials for statistical signal. The Millennium set fails (a) and
   (c). Long-horizon direction only, never measured KPIs.
2. **Define self-growth as two *orthogonal* metrics**, not one:
   **(G1) domain-task delta** (held-out Lean proofs solved, fixed eval, OOD from
   the flywheel's training traces) and **(G2) improver-quality delta** (does
   iteration N+1's *promoted candidate* generalize better than iteration N's,
   measured on a frozen meta-held-out set). G2 is the only thing that earns the
   word "self-growing"; current `evolve.py` cannot yet produce it.
3. **Make the novelty probe honest about its ceiling.** `lean_backend.novelty_check`
   is char-trigram Jaccard, threshold 0.92. It detects *near-verbatim* retrieval,
   not *semantic* retrieval (a re-proof via a different tactic path the model saw
   in training will score "novel"). For a true novelty signal: (a) the proof's
   *tactic graph* must not match any library proof's dependency DAG, and (b) the
   proof must be on a theorem outside the training set's namespace. Trigram
   Jaccard is necessary, not sufficient; the framework treats it as sufficient.
4. **Separate "formalization" from "proof search" as two Tier levels.** You
   cannot search a proof for a theorem with no Lean statement. For physics
   especially, *writing the `theorem ... : ...` line* is the hard, contested,
   research-grade step. Add a "formalization tier" before the "search tier";
   its success metric is *human acceptance of the formalization into
   PhysLean/mathlib*, not a machine number.

## 4. Recommended improvements / alternative structures

- **Replace Foundation→Tier1–4 with a viability ladder keyed to *what has run*:**
  - **L0** — LeanDojo elaborates a bundled trivial proof in the local env (the
    project is *below* this today: `lean_available()` is False by default, so the
    first gate is one real Lean green check).
  - **L1** — reproduce a mathlib proof the system retrieves.
  - **L2** — produce a Lean-verified proof of a *new-to-the-system* olympiad-style
    lemma with a non-trivial tactic path.
  - **L3** — the same on a theorem absent from the model's training data, with a
    passing (tactic-DAG) novelty probe.
  - **L4** — contribute a formalization to PhysLean / mathlib.

  Each level has a clean pass/fail and a held-out split — the discipline the
  proof ladder already wants.
- **Proxy benchmarks that validate the infrastructure before any open problem:**
  - **miniF2F** (Lean 4) — the standard eval for this exact stack; what every
    Lean-LLM paper reports. Run it. If you can't move miniF2F, you cannot move
    FrontierMath.
  - **ProofNet** (Lean 4 undergrad-math formalization) — tests the
    *formalization* axis, not just search.
  - **A self-authored held-out theorem set**, graded by Lean, namespace-disjoint
    from training — the only place "self-grown novelty" is honestly measurable at
    this scale.
- **Harden the novelty probe to a tactic-DAG hash** (normalize commutative/
  associative rewrites, hash the dependency graph of lemmas used). Ship this
  before claiming any "novel verified proof."
- **Split G1/G2 instrumentation**, make G2 the headline *wisdom* metric:
  *fraction of self-extension attempts that abstain rather than overclaim*.
  Conscience kernel + OKF retraction semantics are directly measurable here —
  abstention-precision under adversarial pressure is a number no other AGI
  project reports, and the one place Sophia is genuinely distinctive. Lead with
  that, not Millennium-problem theatrics.
- **Drop physics formalization from the near-term plan.** Keep as a 5-year north
  star. Near-term physics "self-growth" is better measured as *numerical*
  generalization (held-out PDE solutions via the verified-world-model scaffold)
  than as Lean proofs of statements no one has formalized.

## 5. Concrete next steps (prioritized, solo-dev realistic)

1. **Get to L0 this month:** install elan + Lean 4 + lean-dojo behind
   `requirements-theorem.txt`, make `lean_available()` return True in *one* local
   env, run `verify_proof` on a bundled `trivial_true`. Commit the green check as
   the first non-`candidateOnly` artifact. Without this, everything above is
   speculative.
2. **Wire `default_proposer` (LLM tactic proposer) to a real model and run
   miniF2F-test pass@1.** Report the number. First honest external yardstick;
   exactly what reviewers ask for. ~1–2 weeks infra, then search compute.
3. **Upgrade `novelty_check` to a tactic-DAG hash** and define the
   namespace-disjoint held-out set. Defensible novelty measurement.
4. **Instrument G1 and G2 separately** in `selfextend/evolve.py`: extend
   `propose_verifier_candidates` to emit ≥3 candidates (real selection) and add a
   *frozen meta-held-out* split the improver never trains on, scored across
   iterations. Only G2 deltas justify "self-growing."
5. **Run DreamerV3-style Path A on real harness traces** (already planned) — the
   cheaper, independent signal of whether the substrate can generalize at all. If
   `shiftDegenerate` fires, the ceiling is real: a publishable negative result
   that prevents over-investing in Path B.
6. **Defer Millennium/physics** to a "north star" appendix. Zero compute on them
   until miniF2F is competitive and L2 novelty is demonstrated.

## 6. Overall verdict

**Score: 5 / 10** — *directionally right, mis-scaled and over-claiming as written.*

**Justification.** Architectural instincts are correct and better than typical
AGI-roadmap output: it identifies the one legitimate novelty pathway the roofline
permits (formal proof), builds on real disciplined substrate
(`verifier_synthesis` meta-verification, `evolve` canary gate, proof ladder at an
honest Level 2 with `canClaimAGI: false`), and treats negative results as valid.
That earns the 5.

It loses 5 for three concrete failures. **(a) Mis-scaling:** the Millennium /
open-physics set as *primary benchmarks* is off by 4–5 orders of difficulty from a
formal-math stack that has **never run Lean once** — exactly the
appearance-of-progress the thesis says it is against. **(b) Metric circularity:**
self-growth metrics are not yet measurable here — `evolve.py` is a
single-candidate scalar stump-fitter and G1/G2 (task-skill vs. improver-skill) are
not separated, so "self-growing" would be asserted, not measured. **(c) Physics
naivety:** PhysLean is treated as an integration target when it is an open
research contribution; the formalization step (the hard part) is invisible in the
tiering.

**The fix is not to abandon the direction.** Swap the headline benchmark from
Millennium problems to miniF2F / ProofNet + a namespace-disjoint self-authored
set, separate G1 from G2, lead with the one genuinely distinctive metric
(abstention-precision under adversarial pressure — *measured wisdom*), and treat
the open problems as a north-star appendix until L0–L2 are actually demonstrated.
Do that and this becomes an 8/10 roadmap. As written it risks being the theater
the conscience kernel exists to prevent.

---

## ASI realism check (maximum truth-seeking)

From a maximally truth-seeking view, **how much of this advances credible claims
toward ASI?** Almost none, today — and that is the correct answer, not a failure.
The deployed system's defensible claims are narrow and epistemic: fail-closed
provenance gating, measured abstention over fabrication, a verifier-synthesis
flywheel with honest held-out validation, and an AGI-verification gate that
refuses to claim AGI. None of those are ASI progress; they are *epistemic
infrastructure that, if the formal-proof and self-extension axes ever fire, would
make any resulting capability claim trustworthy.* That is the contribution.

**Real, defensible evidence of self-growing capability** in this setup would be:
a *fixed, OOD, namespace-disjoint* held-out proof set on which iteration-N+1 of a
G2-measured improver solves strictly more than iteration-N, with the promoted
artifact's provenance and held-out delta both machine-logged — and the
abstention-precision on adversarial traps not degrading across iterations. That
is a number with a defensible denominator. Anything short of it — "we searched
harder," "we proved more theorems from the training distribution," "the LLM
proposed better tactics" — is **misleading theater**: it measures task skill, not
self-growth, and conflating the two is precisely the trap RSI literature warns
against.
