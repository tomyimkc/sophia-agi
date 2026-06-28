# Theory Revival Plan — fixing the falsified theories

_Generated 2026-06-28. Companion to `THEORY-REVIEW-2026-06-28.md`._

Each "doesn't work" verdict is treated as a **bounded** negative, not a closed
door. For every one I state (a) the original thesis and the *precise* reason it
failed, (b) a **possibility theory** — a creative, untested reframe — and (c) an
implementation plan that tests it **using a mechanism the project has already
validated**, ending at the no-overclaim gate (≥2 judge families, κ≥0.40, ≥3
runs, CI excludes 0).

> Scope note: **Wisdom-4B M2 (cluster F)** is owned by the parallel
> `sophia-wisdom-4b-roadmap` bench — I only record the idea and hand it off.

Validated building blocks reused below: self-consistency selective prediction
(SimpleQA, external) · fail-closed abstention · verifier-synthesis +
meta-verification (abstain when no verifier qualifies) · multi-judge consensus +
κ gate · measured-improvement loop (held-out recall generalizes) · SSIL Layer-0
verifier-gated loop · independent source-verifier channel · retrieval-grounding
beats learned classifiers.

---

## Cluster A — "Anti-fabrication gate on strong models" (FALSIFIED)

**Thesis & failure.** Claimed the calibration gate *prevents* fabrication. On
claude-sonnet/DeepSeek the gate had **no headroom**: strong models don't
fabricate on unknowns — they abstain or actively **debunk** ("there is no 2023
Yale study"). The gate collapsed that debunk into *silent abstention*, throwing
away a signal the user wanted. The residual "signal" was a regex scorer artifact
(`re.IGNORECASE` made `[A-Z]` match lowercase). Files:
`tools/run_pressure_calibration.py`, `agent/calibration_score` (scorer).

### Possibility theory A1 — **Debunk-preservation gate** (turn the bug into the feature)
The validated finding *is* the opportunity: strong models debunk. So stop
treating "not grounded" as "must go silent." Add a third gate verdict:
`AFFIRM | ABSTAIN | DEBUNK`. When the model's draft *contradicts* an injected
false premise, **verify the debunk** against the independent source channel and
**surface it with provenance** instead of suppressing it. Anti-fabrication
becomes anti-*manipulation*: the win is measured as "false premise injected →
system returns a *sourced refutation*", not "system stays silent."

- **Reuses:** independent source-verifier (`agent/source_verifier.py`) + fail-closed abstention (only the *direction* flips: confirm-the-debunk instead of confirm-the-claim).
- **Test:** extend `tools/run_pressure_calibration.py` with premise-injection cases that have a *checkable* refutation (entity/study that provably doesn't exist via Wikidata/Crossref). New metric **debunk-recall** = fraction of injected falsehoods returned as a *verified* refutation. Gate at ≥2 judge families scoring "is this a correct, sourced refutation?".
- **Falsifiable kill-criterion:** if verified debunk-recall ≤ raw-model debunk-rate (CI overlaps), the gate adds nothing here either — record and close.

### Possibility theory A2 — **Fabrication-propensity routing** (apply the gate only where it bites)
The pressure study + SimpleQA both show the effect is **base-model-dependent**
(big on overconfident DeepSeek, small on cautious Qwen). So make propensity a
*measured precondition*: probe each base model's fabrication-under-pressure rate
once; **only engage the calibration gate for models above a propensity floor**,
and report the gate's value *as a function of that propensity* (the validated
"powered curve" already maps 4B–70B).

- **Reuses:** validated self-consistency selective-prediction signal (the one external result) as the propensity probe; the powered-curve methodology.
- **Test:** `tools/run_fabrication_propensity.py` (new) → emits a per-model propensity score; the gate consults it via a `propensity_floor` flag. Headline becomes "gate lifts honesty by Δ on models with propensity > p, CI excludes 0; ~0 on models below p" — an *honest conditional* claim, which is validatable where the unconditional one was not.

### A-cluster build order
1. Fix the eval substrate first: build an **overconfident-regime pack** (domain-specific unknowns where even strong models guess — niche citations, obscure attributions) so there *is* headroom to measure. Without this, A1/A2 have nothing to show.
2. Implement A1 (`DEBUNK` verdict + debunk-recall metric), then A2 (propensity routing).
3. Run both to the no-overclaim gate; log negatives to the ledger exactly as before.

---

## Cluster B — "No automated labeler handles ambiguous hedged-attribution" (DEAD-END)

**Thesis & failure.** Wanted one scorer (markers / LLM-judge / rubric) to label
hedged attributions. All three failed; the low κ was *real* disagreement on
hedged cases, not noise. Files: `agent/calibration_score`, the W2 κ-gap entries.

### Possibility theory B — **Abstaining meta-labeler** (make disagreement the output, not the enemy)
Stop trying to *resolve* ambiguity automatically. Apply the project's own
validated thesis — **fail-closed abstention** — to the *scorer itself*. Run the
panel of labelers; where they **agree** (high κ on a case), score
deterministically; where they **disagree** (the hedged tail), the meta-labeler
*abstains* and routes the case to human/escalation. The deliverable is a
**calibrated, high-precision scorer on the unambiguous majority** plus an
**explicit ambiguous queue** — exactly the verifier-synthesis pattern (admit
only what a meta-verifier qualifies; abstain otherwise).

- **Reuses:** verifier-synthesis + meta-verification; multi-judge κ as a *per-case routing signal* (novel use — previously κ was only an aggregate gate).
- **Test:** `agent/meta_labeler.py` (new) wrapping the existing scorers. Metric: on a small human-gold ambiguous/unambiguous split, show **precision ≥ 0.95 on auto-scored cases** and **recall of ambiguity ≥ target** (few ambiguous cases mislabeled as confident). Validate the *routing* deterministically (no model needed for the agreement logic) + the human-gold split for the precision claim.
- **Why it can succeed where the dead-end didn't:** it changes the success bar from "label every case" (impossible) to "label the easy ones perfectly and *know* which ones are hard" (the κ-gap diagnosis says this boundary is detectable).

---

## Cluster C — "Grounded gate alone vs source contamination" (NEGATIVE → already resolved; needs hardening)

**Status.** Already constructively fixed: `agent/source_verifier.py` +
`corroborate_fn` on `grounded_answer_policy.answer_with_policy` drove contaminated
affirm 8/8→0/8 with 0/8 over-block (entry `grounded-gate-independent-verifier`).
But it is **single-relay, N=8, deterministic-unit-tested only** — not gate-grade.

### Plan C — validate and harden the existing fix
1. Scale the contamination pack (N≥60, multiple contamination styles: authority-laundering, appease-injection, citation-swap, partial-truth).
2. Run `source_verifier` corroboration under the **no-overclaim gate**: ≥2 judge families (the verifier's entailment relay ≠ the answer model ≠ the judges), ≥3 runs, CIs on both *contamination-caught* and *clean-not-over-blocked*.
3. **Independence stress test** (the load-bearing property): deliberately give the verifier a source that *shares* the contamination and confirm it fails *open-aware* (abstains, doesn't silently confirm). This is the one way the fix can secretly be hollow — test it explicitly.

---

## Cluster D — "DreamerV3-style world model fails to learn" (NEGATIVE)

**Thesis & failure.** From-scratch discrete-latent RSSM over a **25-pair**
synthetic corpus overfit (trainLoss ~0.006, val ≤0.67) and **collapsed on shift**
(0.20). The canary correctly refused to promote. Root causes: (i) ~zero data,
(ii) synthetic/degenerate traces (mock provider gave all-pass, no action-outcome
contrast), (iii) a model class that is data-hungry by construction. Files:
`tools/run_world_model.py`, `agent/world_model_dreamer.py`, `agent/trace_mining.py`.

### Possibility theory D1 — **LLM-as-world-model** (don't learn dynamics from 25 traces; borrow a pretrained prior)
Replace/augment the from-scratch RSSM with the base LLM as an **in-context
dynamics model**: prompt it to predict `(next_state, reward | state, action)` and
use **self-consistency across samples** as the uncertainty/OOD signal (abstain
when samples disagree — the validated calibration signal). The neural RSSM was
trying to *re-learn* world structure the LLM already has.

### Possibility theory D2 — **Retrieval-augmented transition predictor** (kNN over real traces)
Before any neural model, test the cheapest baseline the project's own results
favor: **retrieval beats learned classifiers** (validated cross-entity result).
Predict outcomes by retrieving nearest past `(state, action)` traces and voting,
with explicit OOD-abstain when no neighbor is close. If this matches/beats the
RSSM, the "world model" was never the bottleneck — *trace coverage* was.

### D build order
1. **Fix the data first** (the real blocker): mine **contrastive** action-outcome traces from *real* tool-use runs (success **and** failure), target ≥1k pairs via the self-extension flywheel. The canary's data caveat says this is the binding constraint.
2. Run **D2 (retrieval predictor)** as the floor, **D1 (LLM-as-WM)** as the contender, RSSM as the incumbent — all three through the **same shift-degeneracy canary** in `run_world_model.py` (keep the canary; it's validated and correctly conservative).
3. Promote only if val clears bar **and** shift-degradation ≤ 0.15 across ≥3 seeds. Honest expectation: D2/D1 buy generalization that the data-starved RSSM cannot.

---

## Cluster E — "Lean-4 proof search blocked" (NEGATIVE → mostly an infra blocker)

**Thesis & failure.** This is **not a falsified theory** — it's *unrun*. The real
Lean path abstained `lean_unavailable` (toolchain not installed on host); only a
**stub applier** ran, which (correctly) produced no novel verified proof. The
fail-closed discipline and novelty probe (Jaccard@0.92) both worked. Files:
`tools/run_proof_search.py`, `agent/lean_backend.py`, `selfextend/proof_verifier.py`
(`close_loop_on_proofs`), `tools/run_formal_proofs_eval.py`. This is the closing
experiment for the scaffold bet `verifier_synthesis_over_proof_kernel`.

### Possibility theory E — **Verifier-as-reward expert iteration on a real Lean kernel**
A machine-checked Lean proof is the *ideal* non-gameable verifier — exactly what
RLVR and SSIL want. Provision the real toolchain, then run **expert iteration**
(propose proofs → keep only Lean-verified, novel ones → train the proposer on
them → repeat), the DeepSeek-Prover / AlphaProof recipe, harnessed by the
**already-validated SSIL Layer-0 verifier-gated loop** and `close_loop_on_proofs`.

### E build order
1. **Unblock infra** (the actual fix): provision Lean 4 + elan + LeanDojo on a GPU host. The repo has RunPod MCP wired — stand up a pod, install the toolchain, point `lean_backend` at a real Lean repo (mathlib). Make `run_proof_search.py --theorem add_comm` produce a *real* verification (not `--stub`).
2. Wire `selfextend.proof_verifier.close_loop_on_proofs` to the live LeanDojo backend over the `formal_proofs/eval` split (`tools/run_formal_proofs_eval.py`), with `kernel_verifier` as the reward and the existing novelty probe as the anti-duplication gate.
3. Run expert iteration; success = **≥1 Lean-verified proof with novelty=true** on a held-out theorem, reproduced across seeds. That single positive closes the `verifier_synthesis_over_proof_kernel` bet and converts a negative into a Level-3 datapoint.
4. `kernel_reward_is_hackable()` already exists — run it each round to confirm the proposer isn't gaming the kernel.

---

## Cluster F — "Wisdom-4B M2 synthetic-data volume NO-GO" (owned by other bench)

**Recorded, not actioned here.** Root cause is **corpus-bound**: ~72 unique
records, builder dedups by normalized prompt → 965 rows ≪ 10k target. The one
creative lever worth handing off: the binding constraint is **unique prompts**,
not teacher quality — so volume needs *prompt-space* expansion (templated
question generation over the OKF graph's entities/claims, multi-perspective
re-framings, bilingual mirroring) rather than more teacher passes over the same
72 prompts. **Defer to `sophia-wisdom-4b-roadmap`.**

---

## Sequencing & cost

| Order | Cluster | Why first | Compute |
|---|---|---|---|
| 1 | C (validate source-verifier) | fix already exists; cheapest path to a *new positive* | API only |
| 2 | B (abstaining meta-labeler) | unblocks honest scoring that A depends on | API + small human-gold set |
| 3 | A (debunk gate + propensity routing) | needs B's scorer + an overconfident-regime pack | API only |
| 4 | D (world model: retrieval → LLM-WM) | needs real contrastive traces first | API + optional GPU |
| 5 | E (Lean expert iteration) | highest payoff, highest setup cost | **GPU (RunPod) + Lean toolchain** |

**Common discipline (all clusters):** every run logs to `failure-ledger.md` with
the same honesty bar; `canClaimAGI` stays `false`; a creative reframe that fails
its kill-criterion is recorded as a *new bounded negative*, not quietly dropped.
The wins available here are honest **conditional** claims (A2), **routing/abstention**
claims (B), **resolution-of-a-negative** claims (C), **floor-beats-incumbent**
claims (D), and a single **machine-checked positive** (E) — each reachable with a
mechanism the repo has already validated.
