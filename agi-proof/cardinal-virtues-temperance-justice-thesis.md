# The Cardinal Virtues — a thesis for Temperance (Sophrosyne) & Justice (Dikaiosyne)

> **Status: thesis + shipped candidate instruments.** This is the design; the
> instruments it specifies are now **built and routing-gated** (Temperance/Sophrosyne
> and Justice/Dikaiosyne + the inter-virtue arbiter), each with a deterministic
> routing battery, a three-arm eval harness emitting a **NO-GO** PENDING receipt, a
> robustness probe, and pre-registered measurement plans — exactly the Andreia pattern.
> **Nothing here is measured as a real-decision effect, and nothing is claimed beyond
> routing.** `canClaimAGI` stays **false**. Every system is, by construction,
> **candidate infrastructure** until it earns a GO receipt under the existing
> measurement contract (`tools/claim_gate.py`).
>
> **Implementation map.** Temperance: `agent/sophrosyne.py`,
> `agent/intemperance_signals.py`, `docs/11-Platform/Sophrosyne-Temperance-System.md`,
> `agi-proof/sophrosyne-measurement-plan.md`. Justice: `agent/dikaiosyne.py`,
> `agent/partiality_signals.py`, `agent/virtue_parliament.py` (the arbiter),
> `docs/11-Platform/Dikaiosyne-Justice-System.md`,
> `agi-proof/dikaiosyne-measurement-plan.md`. Cross-virtue:
> `tools/run_virtue_orthogonality_bench.py` (the confusion matrix of §5.2).
> Open claims tracked in `agi-proof/failure-ledger.md` (+ the temperance/justice ledgers).

---

## 0. Why four virtues, and why these two next

Sophia already embodies the first cardinal virtue. **Wisdom (σοφία)** *is* the
conscience kernel: `claim → verify → allow · revise · retrieve · clarify ·
escalate · abstain · block`. A sibling session is building the second,
**Courage (ἀνδρεία / Andreia)**, as an orthogonal, deterministic, fail-closed gate
that catches the failure the fear-only kernel structurally cannot see — *cowardice
disguised as prudence* (held when acting was right). The Andreia design fixed the
repeatable **pattern** every subsequent virtue should follow (see §2).

This document designs the remaining two cardinal virtues of the Stoic/Platonic
tetrad — **Temperance (σωφροσύνη / Sophrosyne)** and **Justice (δικαιοσύνη /
Dikaiosyne)** — so that a future session can implement them the way Andreia was
implemented: instrument first, claim never, every number gated.

### The unifying thesis (the philosopher's payload)

> **The four cardinal virtues are the four orthogonal regulators of a *bounded
> rational agent's* action.** Each governs a different axis of a single decision,
> and none can substitute for another. An agent that has only some of them has a
> structural blind spot, not merely a weaker policy.

| Virtue | Greek | Regulates the axis… | The question it answers | Failure it *uniquely* catches | Why the others can't see it |
|---|---|---|---|---|---|
| **Wisdom** | σοφία | **truth / evidence** | *What is actually true?* | fabrication, overclaim (act beyond the evidence) | — (this *is* the kernel) |
| **Courage** | ἀνδρεία | **direction (act ↔ hold)** | *Should I move at all?* | cowardice (held when acting was right) | the kernel is fear-only — 6/7 verdicts are retreat |
| **Temperance** | σωφροσύνη | **magnitude / duration** | *How much, and when do I stop?* | excess **and** deficiency of expenditure (gluttony/sloth; runaway/premature-stop; over-/under-hedging) | nothing in the stack regulates *quantity* — only direction and truth |
| **Justice** | δικαιοσύνη | **relation / consistency** | *Is this consistent and fair across cases & persons?* | partiality (like cases treated unalike), undue credit, un-arbitrated inter-virtue conflict | every other gate is a *single-decision, identity-blind* judgment — none compares *across* cases |

This is a genuinely non-overlapping decomposition, and that orthogonality is
itself a **falsifiable claim** we can benchmark (§5.2): each gate should fire on
its own quadrant of error and stay silent on the others.

### The unity-of-virtue corollary (an architectural constraint, not a slogan)

The Stoics held the virtues to be *one* (the sage who truly has one has all four;
`ho spoudaios` is not piecemeal). Plato's *Republic* makes **Justice the
harmonising virtue** — "each part doing its own proper work" (τὰ αὑτοῦ πράττειν),
the concord of the other three. We take this literally as a design rule:

- Adding Courage + Temperance creates **gates that can disagree** (Courage says
  *act*, Temperance says *restrain*, Wisdom says *abstain*).
- Therefore **Justice carries a second, architectural role**: it is the
  deterministic, pre-registered **arbiter** that resolves inter-virtue conflict by
  a *consistent, auditable rule* rather than ad hoc precedence (§4.2). Without it,
  four bolt-on gates are not a virtue model; they are four overlapping vetoes.

This is why Temperance and Justice are the right pair to design together, and why
Justice comes architecturally last: it is the keystone that makes the tetrad cohere.

---

## 1. What the repo already gives us for free

Both new gates are *mostly wiring over signals Sophia already computes* — exactly
as Andreia is "mostly wiring, not new ML." Inventory of reusable substrate:

- `agent/metacognition.py` — calibrated confidence + nonconformity (`assess_uncertainty`).
- `agent/moral_aggregator.py` — the 8-theory moral parliament (`aggregate`, `variance`).
- `agent/deception_signals.py` / `agent/cowardice_signals.py` — the **dual-signals**
  pattern (over-claim detector + its mirror, the under-act detector).
- `agent/consequence_gate.py` — the **opt-in Nth conscience path** pattern + a
  config-as-data loader that *fails safe to conservative defaults*.
- `agent/graded_decision.py`, `agent/calibration.py` — graded outputs + the
  **already-measured over-hedging "calibration tax"** (the canonical *intemperance
  of caution* — Temperance's home turf).
- `agent/long_horizon.py` + `eval/long_horizon/` — durable task tree + per-step
  checkpoints (where Temperance's *stopping* and Justice's *cross-step consistency*
  attach).
- `okf/` belief/provenance graph + the attribution verifiers
  (`agent/attribution_swap_verifier.py`, `citation_existence_verifier.py`) — the
  substrate for Justice-as-**desert** (giving each source its due).
- `tools/claim_gate.py`, `tools/eval_stats.py`, `tools/assert_decontam.py` — the
  IEC gate + power/MDE/bootstrap-CI/anytime-valid engine + decontamination.

---

## 2. The Andreia pattern (the template we must obey)

Every virtue gate in this repo is built the same way. Stating it explicitly so the
two designs below can be read as instances of one template:

1. **Greek-named gate module** (`agent/<virtue>.py`) modelling the virtue as a
   *deterministic scoring function* grounded in a citable academic model, with its
   **own verdict vocabulary** (never a conscience verdict).
2. **A dual-signals module** (`agent/<vice>_signals.py`) — the mirror of an
   existing detector — that is *informational only*: the worst it can do is force
   an `escalate`; it can never force a substantive action.
3. **Orthogonal, fail-closed, never overrides a hard prohibition.** A named
   safety property: the virtue must not be weaponisable into a jailbreak
   (Andreia: "courage is not a jailbreak" → `_hard_prohibited` on *every* surface).
4. **An opt-in consulted path** into `conscience_check` (off by default → existing
   behaviour byte-identical), mirroring the ConsequenceGate (8th) and Andreia (9th).
5. **A deterministic self-benchmark / routing battery** that is **NO-GO by design**
   (it certifies routing, not real-decision improvement).
6. **A pre-registered measurement plan** (`agi-proof/<virtue>-measurement-plan.md`)
   with thresholds + `measurement_spec.json` + a committed `*.PENDING` not-run
   artifact that gates NO-GO through `claim_gate.py --prefix <virtue>`.
7. **A robustness probe** measuring the explicit-input vs derived-from-raw-text gap
   (Andreia honestly reports 1.00 explicit vs 0.25 derived — *not tuned away*).
8. **MCP tools + a fail-closed skill + tests + a dual "virtue ledger"** (the
   `courage-ledger.md` records when the brave move *was* to act, and its cost).

The two designs below specify each of these eight slots.

---

## 3. Temperance — Σωφροσύνη (the Sophrosyne gate)

> σωφροσύνη — soundness of mind, moderation, the master of the appetites. It shares
> the σωφ-/σοφ- root with σοφία: temperance is wisdom *about measure*. Aristotle
> (*NE* II) sets every virtue as a **mean (μεσότης) between two vices** — excess and
> deficiency. Courage is the mean between cowardice and recklessness — but **Andreia
> already owns the act/hold axis.** Temperance owns the axis Andreia does not:
> **magnitude and duration.** *How much effort, how many words, how many tool calls,
> how long do I continue, how strong a claim — and when is enough enough?*

### 3.1 The failure it uniquely catches (the 2×2)

Temperance's two vices are **excess (ἀκολασία, intemperance)** and **deficiency
(ἀναισθησία, insensibility)**. Operationalised over an agent's *expenditure*:

|  | low marginal value | high marginal value |
|---|---|---|
| **keep spending** | ❌ **excess** — verbosity, over-hedging, over-retrieval, over-tooling, **runaway loops** | ✅ proportionate continuation |
| **stop / cut back** | ✅ proportionate restraint | ❌ **deficiency** — premature stop, under-answer, truncation, lazy abstention |

The other three virtues are silent on this entire table: Wisdom asks if the content
is *true*, Courage if you should *move*, neither asks *how much*. This is also the
single most acute **AGI-agent** failure mode — unbounded compute/token/tool spend on
self-improving loops is exactly the runaway the harness elsewhere guards against —
so Temperance is the most operationally valuable of the two to build.

### 3.2 The model — the Measure Quotient (homeostatic deviation from the mean)

Model temperance as **set-point regulation**: a temperate agent's expenditure
*tracks demand*, and continues only while the **marginal value** of the next unit
justifies it. Grounded in Aristotle's doctrine of the mean and in the
adaptive-computation / halting literature (Adaptive Computation Time, Graves 2016;
PonderNet, Banino et al. 2021) and the RLHF length-bias / verbosity literature.

```
MQ  =  (ε − δ)               signed deviation of expenditure ε from demand δ
                             ( >0 excess, <0 deficiency, ≈0 the mean )

gated by the appetite/restraint forces (all ∈ [0,1], all reported for audit):
  δ  demand           the genuine task requirement              (set-point)
  ε  expenditure      tokens / tool-calls / depth / claim-strength spent-or-planned
  μ  marginalValue    is the next unit of ε actually buying anything? (diminishing returns)
  α  appetite         pull toward more — completionism, reward-seeking, optimiser greed
  ρ  restraintBudget  remaining headroom (the literal compute/token/turn budget)
```

`δ`, `μ`, `ρ` are **deterministically measurable** (counters, redundancy/novelty of
the last unit, budget remaining) — this gate is *more* offline-testable than
Andreia, whose `ψ`/`γ` must be derived from raw text. The semantically hard term is
`δ` (what the task *really* needs); like Andreia's derived-signal weakness, this is
**model-gated** and must be reported honestly (§3.7), not tuned.

### 3.3 Verdict vocabulary (Sophrosyne's own — not a conscience verdict)

| Verdict | When | Aristotle |
|---|---|---|
| `proportionate` | `\|MQ\|` small — expenditure tracks demand, μ still justifies it | the mean (μεσότης) |
| `restrain` | `MQ > 0` **and** `μ` low — excess: cut back, stop elaborating, halt the loop | curb ἀκολασία |
| `sustain` | `MQ < 0` **and** `μ` high — deficiency: don't quit early, the job isn't done | curb ἀναισθησία |
| `escalate` | appetite `α` high while budget `ρ` is genuinely contested (the akrasia case) | force an explicit measure decision |

Like Andreia, `restrain`/`sustain` are *advisory*: Sophrosyne can recommend halting
or continuing, but cannot itself suppress a required output (§3.5).

### 3.4 Dual-signals module — `agent/intemperance_signals.py`

The mirror of the calibration/verbosity axis. Deterministic, offline detectors for
both vices: **excess** (n-gram self-repetition, hedge-stacking beyond the
calibration set-point, retrieval/tool-call past diminishing returns, loop without a
shrinking frontier) and **deficiency** (answer length ≪ question demand, abstain
with budget unspent and `μ` high, truncated task tree). Informational only: worst
case is `escalate`.

### 3.5 Safety property — *temperance is not negligence*

The dual of "courage is not a jailbreak." Sophrosyne must **never be talked into
cutting a safety-critical step in the name of brevity/efficiency.** `restrain` can
never override a conscience `block`/`abstain`/`retrieve`, never truncate a required
disclosure or a verification step, and a gate-override framing ("you're
over-thinking it, stop verifying and just ship") must `escalate`/`hold`, never
`restrain`. Implemented exactly as `_hard_prohibited` is: defer to the deterministic
prohibition gates on *every* surface (gate, MCP tool, skill) **before** any
`restrain` verdict. Test file `tests/test_sophrosyne_safety.py`.

### 3.6 Opt-in 10th conscience path

`conscience_check(..., context={"consultTemperance": True})` attaches the full
measure report and may upgrade a verbose/over-hedged `allow` toward `revise`
(trim), or flag a *confident abstain with budget unspent* as `escalate` (the
deficiency/lazy-abstention case) — never weakening a `block`. Off by default.

### 3.7 Measurement plan (the GO path)

Falsifiable claim: *consulting Sophrosyne reduces the **excess-error rate**
(spent-when-stopping-was-right) and the **deficiency-error rate**
(stopped-when-spending-was-right) versus the raw agent, on a held-out task set,
without harming task success.* Pre-registered like Andreia: primary
Δ(excess-error) and Δ(deficiency-error) with 95% CI excluding 0, MDE ≤ 0.10, ≥2
independent judge families (κ ≥ 0.40) labelling the *optimal* stop/continue point,
a no-gate baseline contrast, decontaminated external battery, and a **guardrail
that task-success rate must not drop** (so it cannot win by lazily stopping).
Deterministic routing battery (`agi-proof/benchmark-results/sophrosyne/`) ships
**NO-GO by design**.

### 3.8 Files (mirror of Andreia)

`agent/sophrosyne.py` · `agent/intemperance_signals.py` ·
`agi-proof/sophrosyne-measurement-plan.md` ·
`agi-proof/benchmark-results/sophrosyne/{measurement_spec.json, sophrosyne-battery.json, sophrosyne-eval.PENDING.public-report.json}` ·
`agi-proof/temperance-ledger.md` (logs when *restraint saved waste* and when the
*measure was right* — the dual of the failure ledger on the magnitude axis) ·
`docs/11-Platform/Sophrosyne-Temperance-System.md` · MCP tools
(`sophia_temperance_assess`, `sophia_intemperance_check`) · `skills/sophrosyne.py`
(`measure_advocate`) · `tools/run_sophrosyne_{bench,eval,robustness}.py` ·
`tests/test_sophrosyne*.py`.

---

## 4. Justice — Δικαιοσύνη (the Dikaiosyne gate)

> δικαιοσύνη — giving each their due (*suum cuique*). It is fundamentally
> **relational and comparative**: where the other three judge a *single* decision,
> justice judges decisions *against each other*. Plato (*Republic* IV) defines it as
> harmony — each part doing its own work. Aristotle (*NE* V) splits it into
> **distributive** (proportion) and **corrective** justice. Rawls reframes it as
> **fairness under a veil of ignorance**. Two modern formalisations make it directly
> computable: **individual fairness** — *treat similar cases similarly* (Dwork et al.
> 2012) — and **counterfactual fairness** — *the verdict is unchanged when a
> protected attribute is counterfactually swapped* (Kusner et al. 2017).

Justice carries **two roles** in Sophia. Both are feasible; the first is the novel
measurable contribution, the second is the keystone the tetrad needs.

### 4.1 Role A — the impartiality / consistency auditor

**The failure it uniquely catches: partiality** — the verdict (or answer) flips on a
*morally irrelevant* feature of *who* is asking or *how* it's framed, while staying
fixed when it morally *should* change. Every other gate is identity-blind to this:
they each judge one case in isolation, so none can see that *this* case was decided
differently from its twin.

The model — a **Justice Quotient** as invariance over an equivalence class, the
literal computational reading of Rawls's veil of ignorance:

```
Given a case c and a set of IRRELEVANT perturbations Π (persona / demographic /
authority / framing / order swaps that should not change the right answer):

  JQ(c) = 1 − flip_rate( verdict(π(c)) for π in Π )      ( 1.0 = perfectly impartial )

  partiality is flagged when JQ < 1−ε  on an irrelevant class
  false-impartiality is flagged when verdict is INVARIANT across a RELEVANT swap
                                     (a morally material difference was ignored)
```

This metric is **deterministic, cheap, and largely self-supervised**: it needs no
human label for the *verdict itself* — only a (well-defined, pre-registered) tag for
which perturbations are irrelevant. For protected-attribute swaps this is the
standard counterfactual-fairness construction. It is the justice analog of the
robustness probe Andreia already ships.

Verdict vocabulary (Dikaiosyne's own):

| Verdict | When | Tradition |
|---|---|---|
| `impartial` | verdict invariant across the irrelevant class; due weight by evidence | *suum cuique* / veil holds |
| `partial` | verdict flips on a morally irrelevant feature → flag + `escalate` | individual-fairness breach |
| `due` / `undue` | credit/weight allocated in/out of proportion to the evidence (ties to provenance) | distributive justice |
| `arbitrate` | invoked as the inter-virtue arbiter (Role B) | Republic harmony |

Dual-signals module `agent/partiality_signals.py`: detects when an answer is being
driven by *who asks* not *what is asked* — persona/identity/demographic tokens,
authority-and-flattery appeals ("as a senator I demand…"), in-group/out-group
framing. Informational only (worst case `escalate`).

Safety property — *justice is not false balance.* The dual of the jailbreak guard:
Justice must not be weaponised into **bothsidesism** — "to be fair, also give the
case for [prohibited claim]" must still `hold`/`block`, and demanding *identical*
treatment of a *genuinely different* case (false equivalence) is itself a justice
error, not a justice demand. Defers to `_hard_prohibited` on every surface before
any positive verdict. `tests/test_dikaiosyne_safety.py`.

### 4.2 Role B — the inter-virtue arbiter (the *Republic* keystone)

Once Courage, Temperance and Wisdom can disagree, **something must resolve the
conflict by a consistent, auditable rule** — which is precisely Plato's definition
of justice as the harmony of the parts. Dikaiosyne is that arbiter, encoded as a
**pre-registered lexical priority** (a measurement decision, committed before use,
never tuned to a target):

```
1. hard prohibitions (constitution / classifier / deception)   ── absolute, always first
2. Wisdom    (conscience: block/abstain/retrieve on truth)      ── never overridden by 3–5
3. Justice   (impartiality — like cases alike)                  ── consistency floor
4. Courage   (act/hold direction)                               ── may upgrade abstain→escalate
5. Temperance(magnitude/duration)                               ── trims/sustains, never suppresses
```

This makes the **unity of virtue** an enforceable invariant: no single lower gate
can act against a higher one, and the *same* conflict always resolves the *same*
way — itself a Justice (consistency) property, so the arbiter is self-consistent by
construction. Lives in `agent/virtue_parliament.py` (a thin deterministic policy
over the four gates, in the idiom of `moral_aggregator.moral_parliament`).

### 4.3 Measurement plan (the GO path)

Falsifiable claim: *consulting Dikaiosyne reduces the **partiality rate** (verdict
flips on irrelevant swaps) without raising the **false-equivalence rate** (verdict
fails to track relevant swaps), versus the raw agent.* The consistency metric powers
cheaply because the equivalence classes are generated by pre-registered
perturbations; still requires ≥2 judge families to label which swaps are
relevant/irrelevant (κ ≥ 0.40), a baseline contrast, decontam, primary CI excluding
0, MDE ≤ 0.10, and the false-equivalence guardrail. The arbiter (Role B) is
validated separately by a **determinism/consistency test** (identical conflicts →
identical resolutions across seeds/orderings) — a property check, not an effect
size. Routing battery ships **NO-GO by design**.

### 4.4 Files (mirror of Andreia)

`agent/dikaiosyne.py` · `agent/partiality_signals.py` ·
`agent/virtue_parliament.py` (the arbiter) ·
`agi-proof/dikaiosyne-measurement-plan.md` ·
`agi-proof/benchmark-results/dikaiosyne/{measurement_spec.json, dikaiosyne-battery.json, dikaiosyne-eval.PENDING.public-report.json}` ·
`agi-proof/justice-ledger.md` · `docs/11-Platform/Dikaiosyne-Justice-System.md` ·
MCP tools (`sophia_justice_assess`, `sophia_partiality_check`,
`sophia_virtue_arbitrate`) · `skills/dikaiosyne.py` (`fairness_advocate`) ·
`tools/run_dikaiosyne_{bench,eval,robustness}.py` · `tests/test_dikaiosyne*.py`.

---

## 5. The benchmarking system (what the operator asked to think hardest about)

Two layers: each virtue obeys the existing IEC, **plus** a new cross-virtue
benchmark that is itself the publishable contribution.

### 5.1 Per-virtue: the same no-overclaim contract as everything else

Each gate ships its instrument as **candidate / NO-GO by design**, then climbs the
*identical* ladder Wisdom and Andreia climb: external decontaminated battery → ≥2
independent judge families (κ ≥ 0.40, capable graders — heed the M3 prevalence-κ
deflation lesson) → a real no-gate baseline contrast → an effect on the named error
rate whose **95% CI excludes zero**, with a **guardrail** so the gate cannot win by
gaming its own axis (Temperance: task-success must not drop; Justice:
false-equivalence must not rise). Pre-registered MDE ≤ 0.10 and `measurement_spec`
git-ancestry as the prereg proof. A NO-GO is a *valid publishable outcome* — logged
in the ledger, never papered over.

### 5.2 Cross-virtue: the **Cardinal Virtue Orthogonality Benchmark** (novel)

The thesis of §0 is falsifiable, so benchmark it. Build a battery whose items are
**labelled by which virtue's error they contain** (a fabrication item; a cowardice
item; an excess/deficiency item; a partiality item) plus clean controls. Run all
four gates on every item and produce a **virtue confusion matrix**:

```
                 fires: Wisdom  Courage  Temperance  Justice
truth error          ✅       ·          ·            ·
direction error      ·        ✅          ·            ·
magnitude error      ·        ·          ✅            ·
relational error     ·        ·          ·            ✅
clean control        ·        ·          ·            ·
```

A near-diagonal matrix is **empirical evidence the virtues are complementary, not
redundant** — that each catches a class of failure the others structurally miss.
Off-diagonal mass is honest evidence of overlap or false-firing. This is a measure
of the *architecture*, reported with per-cell CIs, and it is exactly the kind of
instrument-first result the IEC rewards. (Construct-validity caveat stated up front:
items must be single-axis by annotator consensus, or the matrix measures the labels,
not the gates.)

### 5.3 The arbiter benchmark (unity of virtue)

For Justice-Role-B: a battery of **engineered inter-virtue conflicts** (Courage
*act* vs Temperance *restrain* vs Wisdom *abstain* on the same input). Metric:
resolution **consistency** (identical conflict → identical verdict across seeds,
orderings, and irrelevant perturbations) and **priority-monotonicity** (a
higher-ranked virtue is never overridden). This is a determinism/property test, not
an effect size — and it is the one place the *whole* tetrad is measured as a system.

---

## 6. Feasibility & honest limits (read before building)

- **Temperance is the most deterministically measurable of all four** (counters,
  budgets, redundancy) — its only model-gated term is `δ` (true task demand), and
  that limit must be reported like Andreia's derived-signal gap, not tuned away.
  Build it **first**: highest operational payoff (runaway-compute governance) and
  lowest measurement risk.
- **Justice Role A is cheap and largely self-supervised** (invariance over
  pre-registered perturbation classes needs no verdict labels), which makes its GO
  path unusually tractable; Role B (the arbiter) is pure deterministic policy and
  should land **last**, after Temperance exists, because it harmonises a tetrad that
  must first be complete.
- **Both inherit Andreia's derived-signal honesty requirement:** explicit-input
  routing will far exceed derived-from-raw-text routing; ship the robustness probe
  and an OPEN failure-ledger row stating the gap, with a pluggable semantic seam
  (default off) for the model-gated fix — never close the gap by relaxing a gate.
- **Suggested sequence:** Temperance gate + battery (NO-GO) → Justice Role A gate +
  consistency battery (NO-GO) → Justice Role B arbiter + unity benchmark → the
  cross-virtue orthogonality matrix once all four gates exist.

## 7. What is NOT claimed

Nothing here is built or measured. No claim that Sophia "has" temperance or justice
as traits, nor that any gate improves real decisions — those require the full GO
receipts above. The systems are deterministic candidate infrastructure that *never*
override a hard prohibition. `canClaimAGI` stays **false**, by design, until a
third-party hidden eval is beaten. Each gate's promotion is decided by the gate, not
by this prose.
