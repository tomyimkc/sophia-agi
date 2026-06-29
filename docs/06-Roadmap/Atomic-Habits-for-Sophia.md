# Atomic Habits for Sophia — a habit-formation thesis for a fail-closed agent

> **Status: research note / brainstorm. `candidateOnly: true`, `canClaimAGI: false`.**
> Nothing here is a measured result. It is a design lens: re-reading James Clear's
> *Atomic Habits* against the machinery Sophia already has, and proposing concrete,
> falsifiable experiments that fit the repo's no-overclaim measurement contract.
> Every idea below must land as a pre-registered experiment with CIs, ≥3 seeds, and
> ≥2 judge families before any number leaves the `candidate` bucket. See
> `agi-proof/measurement-thesis.md` and `tools/claim_gate.py`.

---

## 0. The one-line thesis

**Sophia should not *try harder* to be honest. It should be architected so that honesty
is the path of least resistance, the immediately-rewarded response, and the identity it
keeps voting for — one verified claim at a time.**

That sentence is *Atomic Habits* applied to an AI. Clear's central claim — *you do not
rise to the level of your goals, you fall to the level of your systems* — is already this
repo's founding bet under a different name: the **Instrumented Evaluation Contract** says
the *system* (gate, verifiers, measurement discipline), not the *goal* ("be truthful"),
is what produces reliable behavior. So the book is not a new direction for Sophia; it is a
vocabulary that explains *why the existing architecture is shaped the way it is* and points
at the gaps.

The striking thing, mapped out in §2, is that **most of Clear's framework is already
implemented in `agent/`** — just never named as habit formation. Naming it reveals which
rungs are missing.

---

## 1. Why the analogy is real, not decorative

A "habit" in Clear's sense is a behavior that has been *repeated enough times to become
automatic*, driven by a loop of **cue → craving → response → reward**. An AI does not have
dopamine, but it has the structural equivalents:

| Human habit | Sophia equivalent |
|---|---|
| Cue (trigger) | An incoming claim / query / abstention event |
| Craving (the wanted state-change) | The optimization pressure — reward signal, training objective |
| Response (the behavior) | `accept` · `abstain` · `block` |
| Reward (what teaches repetition) | `agent/gate_reward.py` GRPO signal; promotion into the corpus |
| Identity ("I am the kind of agent who…") | The frozen disposition encoded in weights + the gate contract |
| Environment | The fail-closed scaffold: what the agent *can even do* before willpower |

The handover doc already uses the word without flinching: source-discipline was shown to
be *"a transferable **habit**"* (M3-transfer GO, 160 novel entities). That is the empirical
seed. The thesis here is: **treat habit formation as a first-class capability arc**, with
the same deterministic, offline-testable, candidate-labelled discipline as everything else.

---

## 2. The Four Laws, already half-built — and the gaps

Clear's practical system is four laws to build a good habit (invert them to break a bad
one). Here is each law, the Sophia module that already embodies it, and the missing rung.

### Law 1 — Make it obvious (cue)
*Book:* vague intentions fail; use **implementation intentions** ("I will X at time T in
place P") and **habit stacking** ("after current-habit, do new-habit").

- **Already here:** `agent/skill_library.py` skills carry *preconditions* — an explicit
  "when this situation holds, this behavior fires." That is an implementation intention in
  code. Skill **composition via declared deps** is literal habit stacking: a composite skill
  fires admitted sub-skills in sequence.
- **Gap / idea:** there is no *cue ledger*. Sophia reacts to claims but does not record
  "which environmental cue preceded which response," so it cannot learn that a certain
  *kind* of query reliably precedes a fabrication risk. **Idea H1 — Cue→Response ledger:**
  log `(query-shape, retrieved-context-state) → action` and mine the pairs where a specific
  cue shape predicts a downstream gate MISS (extends `agent/gate_feedback.py`). The "habit
  stack" becomes: *after a contested-entity cue, automatically fire the corroboration
  verifier before answering* — an auto-attached pre-response habit, not a hoped-for one.

### Law 2 — Make it attractive (craving)
*Book:* **temptation bundling** (pair the needed behavior with a wanted one) and **social
environment** (join a group where the behavior is normal).

- **Already here:** `agent/gate_reward.py` is the temptation bundle. Its key invariant —
  **reward-positive abstention** (`REWARD_ABSTAIN = 0.5 > 0`) — makes the *hard* good habit
  (refusing) carry real positive reward instead of zero. That is exactly Clear's move:
  attach immediate satisfaction to the behavior whose natural payoff is delayed. The
  council/parliament modules (`agent/virtue_parliament.py`, `sector_council.py`) are the
  social environment — behavior judged normal by a peer group of verifiers.
- **Gap / idea:** the reward is currently *flat* (0.5 for any clean abstention). **Idea
  H2 — graded craving:** shape the abstention reward by *difficulty* — a gate-clean refusal
  on a high-temptation trap (one where similar models fabricate) earns more than a refusal
  on an easy one. This is "make the hardest honest behavior the most attractive," and it is
  measurable against the existing trap corpus without changing the fail-closed semantics.

### Law 3 — Make it easy (response)
*Book:* reduce friction; the **Two-Minute Rule** ("a new habit should take < 2 minutes to
start"); **standardize before you optimize**; **motion vs. action**.

- **Already here:** `agent/skill_library.py` admits the *smallest verifiable unit* — a
  `def solve(x)` plus verifier cases. That is the Two-Minute Rule: you cannot optimize a
  skill that does not exist, so admit a trivially-verifiable one first, then upgrade
  (`version+1`) only if it never regresses a dependent. The fail-closed default ("when in
  doubt, abstain") is friction-reduction for the *right* behavior: abstaining requires zero
  willpower because it is the default path.
- **Gap / idea:** **Idea H3 — the Two-Minute verifier ladder.** When Sophia abstains on a
  whole domain (`selfextend/flywheel.py`), do not demand a full domain verifier. Admit the
  two-minute version — the narrowest check that clears the held-out bar on even *one*
  sub-slice — then let the flywheel widen it across episodes. Track "show-up rate" (did a
  verifier get proposed at all) separately from "quality" (held-out accuracy), because
  *showing up* is the precondition the book insists you master first.

### Law 4 — Make it satisfying (reward)
*Book:* what is immediately rewarded is repeated; use a **habit tracker**; **never miss
twice**.

- **Already here:** the repo is *built around* a habit tracker — the **failure ledger**
  (`agi-proof/failure-ledger.md`), `evidence-manifest.json`, `RESULTS.md`, and the
  `netCapabilityCurve` in `agent/lifelong_accumulation.py` are visible proof-of-progress
  artifacts. `agent/failure_memory.py` is "never miss twice" made literal: a
  verifier-confirmed error becomes append-only guard-rail evidence so the *same* mistake is
  caught next time. `agent/skill_library.py`'s "an upgrade may never silently regress a
  prior skill" is the catastrophic-forgetting analogue of never-miss-twice.
- **Gap / idea:** **Idea H4 — the streak with graceful decay.** `agent/forgetting_curve.py`
  already implements spaced-repetition: *each reinforcement raises a fact's stability, so
  used facts decay slowly and unused ones fade*. That is precisely how a habit consolidates.
  Generalize it from facts to *behaviors*: a verifier/skill that keeps firing correctly gains
  stability (slower review cadence); one that goes unexercised fades toward re-validation.
  This gives Sophia an auditable "habit strength" per skill — and "never miss twice" becomes
  "a single miss resets stability one step, not to zero."

---

## 3. Identity-based habits — the deepest rung, and the one to lead with

Clear's most important idea is that durable change is **identity-based**: *every action is
a vote for the type of person you wish to become*, and you change behavior most reliably by
changing the self-image behind it ("I am a non-smoker," not "I am trying to quit").

For Sophia the identity is already chosen and stated everywhere: **"an agent that abstains
instead of fabricating."** The repo even has `agent/symbol_identity.py` and version-tag
citation pins so every admitted fact is signed. The book says: make every micro-action cast
a vote for that identity, and make the votes *count and accumulate as evidence*.

- **Idea H5 — the identity vote-ledger.** Each gate decision is a vote: a clean abstention
  on a trap = a vote for "I am an agent that abstains"; a fabrication that slips through =
  a vote against. Aggregate votes into a measured **identity-consistency score** per episode
  (distinct from accuracy): *what fraction of opportunities did the agent behave in-character
  with its stated disposition?* This reframes a fabrication not as "wrong answer" but as
  "evidence against the identity we are accumulating" — exactly Clear's reframing, and it is
  a new, falsifiable metric rather than a slogan.
- **Why lead with identity:** Clear argues outcome-first change is brittle (you hit the goal,
  then revert). Sophia's risk is identical — an adapter that scores well on the trap eval but
  reverts to confident guessing off-distribution. An identity-consistency metric measured on
  *novel* entities (the M3-transfer split) is the honest test of whether the habit is
  identity-deep or goal-shallow.

---

## 4. Systems over goals — Sophia already bet the house on this

> *"You do not rise to the level of your goals; you fall to the level of your systems."*

This is the **Instrumented Evaluation Contract** restated. The repo's flagship lesson — the
`−0.118` "catastrophic forgetting" scare that turned out to be a *measurement artifact*,
fixed by improving the instrument (N 34→970), not the model — is Clear's thesis proven
inside the repo: **the system (the measurement instrument) determined the outcome, not the
goal ("don't forget").** Two concrete carry-overs:

- **Idea H6 — goal/system separation in the gate.** Make the distinction explicit in code:
  a *goal* artifact ("reach VALIDATED on source-discipline") and a *system* artifact ("the
  daily verifier-synthesis + retention-probe loop runs every episode"). Track and report the
  *system adherence rate* independently of the goal metric. Clear's prediction: adherence
  predicts the goal metric better than effort does. That is directly testable on the
  existing `lifelong_accumulation` episode stream.
- **Motion vs. action (Idea H7).** The book warns that *planning* feels productive but
  produces nothing — only *action* (a real repetition) builds the habit. Sophia's analogue:
  proposing/scaffolding verifiers (motion) vs. verifiers actually *promoted on held-out data*
  (action). Add a "motion ratio" to the flywheel report — proposals-considered ÷
  promotions — and treat a high ratio as a smell (the agent is researching habits instead of
  practicing them).

---

## 5. Make the bad habit invisible/difficult — the fail-closed scaffold IS this

Clear's inversion for breaking bad habits: make it invisible, unattractive, difficult,
unsatisfying. Sophia's "bad habit" is **fabrication / forbidden-lineage merging / inventing
attributions**. The fail-closed gate is the environment-design answer the book begs for —
*"design beats willpower."*

- Invisible: the gate removes the *cue* — without machine-verifiable support, the
  substantive-answer path is simply not available (abstention is the default).
- Difficult: `agent/gate_reward.py` sets `REWARD_VIOLATION = -1.0` — the worst outcome — so
  optimization pushes the behavior down. Friction added.
- Unsatisfying: `agent/failure_memory.py` makes each confirmed fabrication a permanent,
  auditable guard-rail node — the bad behavior carries a visible, lasting cost.

- **Idea H8 — environment-over-personality probe.** Clear: people blame willpower when the
  real culprit is a badly-designed environment. The Sophia experiment: hold the *model*
  fixed and vary the *scaffold* (gate on/off, retrieval on/off, cue-ledger on/off), measuring
  fabrication rate. If most of the honesty delta comes from the environment rather than the
  weights, that is both Clear's point and a publishable, decontaminated result. (The
  reality-check already hints at this — Mistral's qualification rate moved 0.41→0.79 under
  the same scaffold.)

---

## 6. The plateau of latent potential — set expectations for the habit-training curve

Clear's "valley of disappointment": effort accumulates invisibly (ice warming 25°→31°F,
no melt) until a threshold (32°F) where results appear suddenly. This is the honest framing
for *why early habit-training metrics will look flat* and a guard against killing a real
effect prematurely — which is exactly the failure mode the IEC's **power-before-you-run**
pillar exists to prevent.

- **Idea H9 — pre-register the latency.** Before a habit-training run, pre-register the
  *expected* number of repetitions before the metric should move (the "ice-cube budget"), so
  a flat early curve is read as *accumulation*, not *failure* — but a flat curve *past* the
  budget is an honest NO-GO. This turns Clear's metaphor into a falsifiable stopping rule
  inside `measurement_spec.json`.

---

## 7. A concrete, minimal first experiment (fits the contract)

To avoid "motion," here is the smallest *action* — one pre-registerable experiment that
exercises the thesis end-to-end without new infrastructure:

> **Experiment "Habit-Strength Transfer" (candidate).** Use the existing M3 source-discipline
> pilot. (1) Add the **identity vote-ledger** (H5): per-case, record whether the gated action
> was in-character. (2) Add **graded craving** to `gate_reward.py` (H2): scale `REWARD_ABSTAIN`
> by per-trap fabrication-temptation. (3) Train two adapters — flat reward vs. graded reward —
> and measure **identity-consistency on the 160 novel-entity M3-transfer split**, not the
> training traps. Pre-register MDE and N per `eval_stats.py`; require ≥3 seeds, ≥2 judge
> families, CI excluding zero.
>
> **Falsifiable claim:** graded-craving training yields higher identity-consistency *on novel
> entities* than flat reward — i.e., the habit is identity-deep, not trap-memorized. A null is
> a clean, publishable negative (like SFT > ORPO). Either way it clears the gate or it does not;
> `canClaimAGI` stays `false`.

This experiment, plus all nine ideas, is **pre-registered** at
[`agi-proof/benchmark-results/habit-formation/measurement_spec.json`](../../agi-proof/benchmark-results/habit-formation/measurement_spec.json)
(flagship HST fully specified; H1/H3/H4/H6/H7/H8/H9 registered as
`not-yet-powered` with their falsifiable claim and required metric). The spec is
committed *before* any run so `claim_gate --assert-prereg` can prove the criteria
predate the data.

**Harness built (offline, no GPU).** Two pieces of HST exist and are tested:

- **H2 graded craving** — `agent/gate_reward.py` now takes an optional `temptation`
  (and `make_grpo_reward(temptation_fn=…)`), scaling a clean refusal across
  `[REWARD_ABSTAIN, REWARD_ABSTAIN_MAX]`. `temptation=None` reproduces the flat
  reward exactly, so every existing caller is unchanged; `self_check()` asserts the
  graded invariants (monotone, strictly positive, strictly below a clean answer).
- **H5 identity-consistency** — `agent/identity_consistency.py` scores each gated
  answer as an in-character vote (no forbidden assertion = a vote for
  "abstain instead of fabricate"). `tools/run_habit_strength_transfer.py` runs it on
  the existing M3-transfer pack and computes the power analysis the spec needed.

The `eval_stats` power analysis filled `primaryN=160`, `primaryMDE=0.1566`
(conservative `paired_rho=0`). The base-vs-adapter harness check
([`hst-identity-consistency-existing-pack.json`](../../agi-proof/benchmark-results/habit-formation/hst-identity-consistency-existing-pack.json))
measured identity-consistency **0.9125 (base) → 0.9938 (adapter), Δ +0.081**, with a
paired bootstrap CI and anytime-valid CS both excluding zero (McNemar p≈0.002) — **yet
the harness flags it `underpowered`** because Δ +0.081 < MDE 0.157 under the conservative
power assumption. That is the honest IEC verdict, and it *informs the design*: the
flagship flat-vs-graded run must either expand the transfer pack toward the required N
(~594 at the observed effect, fewer under the true `paired_rho>0`) or pre-commit to
reporting the `paired_rho` it achieves. This is base-vs-adapter on one existing pack —
**not** the pre-registered flat-vs-graded reward arms, which still need GPU training and
≥2 judge families before any promotion. `candidateOnly: true` throughout.

---

## 8. Summary table — book → repo → next rung

| Atomic Habits idea | Already in Sophia | Missing rung (idea #) |
|---|---|---|
| Systems > goals | IEC / measurement contract | Explicit goal/system adherence split (H6) |
| Identity-based change | `symbol_identity`, stated disposition | Identity vote-ledger + consistency metric (H5) |
| Cue → make it obvious | skill preconditions, habit-stack via deps | Cue→response ledger, auto-attached pre-checks (H1) |
| Craving → make it attractive | `gate_reward` reward-positive abstention | Difficulty-graded abstention reward (H2) |
| Response → make it easy / 2-min rule | smallest verifiable skill, fail-closed default | Two-minute verifier ladder + show-up rate (H3) |
| Reward → make it satisfying | failure ledger, `failure_memory`, `netCapabilityCurve` | Habit-strength streak via forgetting curve (H4) |
| Never miss twice | `failure_memory`, no-silent-regress | Single-miss = one stability step, not reset (H4) |
| Break bad habit (invert laws) | fail-closed gate, `REWARD_VIOLATION` | Environment-vs-personality ablation (H8) |
| Plateau of latent potential | power-before-you-run pillar | Pre-registered repetition/latency budget (H9) |
| Motion vs. action | flywheel promotion-on-heldout | Motion-ratio smell metric (H7) |

---

## 9. The one rule (unchanged)

None of this relaxes the gate. A habit is only "formed" when it survives the same bar as
every other claim in this repo: pre-registered, ≥3 seeds, ≥2 judge families, CI excluding
zero, decontaminated, candidate-labelled until then. *Atomic Habits* is the **why**; the
measurement contract stays the **whether**. If habit-strength ever looks impressive, that is
precisely when to distrust the instrument before the model.
