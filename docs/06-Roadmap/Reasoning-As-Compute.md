# Reasoning as Compute — AGI features inspired by the HPC operator discipline

> **What this is.** A design note, not a build order. It takes the *engineering
> philosophy* of a high-performance operator/communication/compiler team (the DeepSeek
> 高性能算子/通信/编译器 role this branch's roadmap targets) and maps it onto AGI-candidate
> features that fit Sophia's existing trust layer. Companion to
> [`HPC-Operator-Compiler-Roadmap.md`](HPC-Operator-Compiler-Roadmap.md), which is the
> *systems* track; this is the *reasoning* track the same mindset suggests.
>
> **Scope, stated plainly.** These are **research features, not AGI claims** (see
> [`VISION.md`](../../VISION.md)). Each must still clear the no-overclaim measurement gate
> before any number headlines. Nothing here asserts sentience or general intelligence.

## The core transfer

Sophia already obsesses over one axis: **verifiability** — it abstains instead of
fabricating, traces every claim, and refuses to promote a hypothesis it cannot check. The
HPC discipline contributes a *second* axis Sophia under-invests in: **efficiency measured
against a physical limit** — not "better than a baseline," but "how far from the
theoretical ceiling," with near-obsessive care for every cycle and every watt.

Pair them and the thesis is one line:

> **Treat *thinking* the way an operator team treats *compute*: a bounded physical
> resource, spent against a measurable ideal.**

Sophia knows *when to abstain*. This adds *how much to spend, and how far the result is
from optimal.* Call it **bounded-optimal verifiable reasoning.**

---

## The five features

Each: the operator principle → the reasoning feature → where it lands in today's code →
the honest caveat.

### 1. A reasoning roofline — measure against the ideal, not a baseline
- **Principle.** Never compare to a baseline or another implementation; only measure
  distance to the theoretical ceiling.
- **Feature.** Stop scoring Sophia as "better than the raw model." Define an **oracle
  ceiling** (perfect retrieval + perfect verification on the task) and report *% of that
  ceiling*, with a **ridge point** where the binding constraint flips from *retrieval* to
  *reasoning* — the analogue of a memory-bound/compute-bound transition.
- **Lands in.** `agent/calibration.py` (ECE, risk-coverage) already has the measurement
  machinery; this supplies the denominator. Same instinct as the `RESULTS.md` gate.
- **Caveat.** The oracle ceiling is itself an estimate; report it with the same CIs and
  ≥3-run discipline, and label it candidate until a third-party oracle exists.

### 2. Test-time compute as "cycles and watts" — metacognitive budgeting
- **Principle.** Every cycle and every watt is accounted for.
- **Feature.** A self-model that decides *how much deliberation a query deserves* —
  best-of-N width, council size, retrieval depth — by the **marginal verified-confidence
  gain per token**, and stops at the ridge point where more thinking buys nothing. That
  same flattening is a principled trigger for **abstention**, not just early-exit.
- **Lands in.** `agent/best_of.py`, `agent/council_deliberate.py`,
  `agent/graded_decision.py` (the answer/hedge/abstain router).
- **Caveat.** "Marginal gain" needs a live confidence signal that is actually predictive;
  Sophia's own measurements show provenance-confidence is a *weak* correctness predictor,
  so the budgeter must be evaluated, not assumed.

### 3. A reasoning *compiler* — lower intent → IR → verified actions
- **Principle.** DSL → IR → optimization passes → CodeGen, with simple, stable interfaces
  over high-performance internals.
- **Feature.** Treat a goal as something you **lower** into a typed *plan-IR*, run
  **passes** over, then "CodeGen" to tool calls. Sophia's belief/justification graph is
  already a graph IR — so the passes write themselves:
  - *constant-folding* = claim dedup / canonicalization,
  - *dead-code elimination* = prune branches with no grounded support,
  - *type-checking* = provenance + consistency verification before "emit,"
  - *CSE* = reuse a verified sub-conclusion across branches.
- **Lands in.** `okf/graph.py` (min-over-chain propagation, contradiction ledger) +
  `okf/counterfactual.py` (counterfactual removal/retraction = a "what-if this pass were
  dropped" analysis) + the verifier gate as the final lowering check.
- **Caveat.** The highest-leverage item *because* it reuses existing structure — but the
  pass framework must keep the interface simple (the operator team's actual hard part is
  maintainability, not cleverness).

### 4. A cost-modeled memory hierarchy
- **Principle.** register → shared → HBM → interconnect; *data movement*, not arithmetic,
  is usually the bottleneck.
- **Feature.** Tier agent memory — working context / cached KG / vector store / cold
  archive — with an **explicit access-cost model**, and a locality-aware retrieval planner
  that minimizes expensive long-context and remote fetches (the reasoning analogue of
  avoiding HBM and cross-node traffic).
- **Lands in.** `agent/memory.py`, `agent/rag_pipeline.py`, and the OKF wiki tiers — which
  are already tiers, just not cost-modeled.
- **Caveat.** Don't optimize movement at the expense of provenance; a cache hit must carry
  the same lineage as a cold read, or the gate weakens.

### 5. Communication-efficient agent collectives
- **Principle.** Collective comms (NCCL/HCCL/DeepEP) — overlap compute and communication,
  be topology-aware, and decide *what to send when*.
- **Feature.** Treat councils as a **collective-communication problem**: a
  provenance-carrying "belief all-reduce" that minimizes redundant inter-agent messages,
  routes by role/topology, overlaps independent deliberation with exchange, and treats the
  confidentiality firewall as the "what may cross which link" policy.
- **Lands in.** `agent/sector_council.py`, `agent/council_deliberate.py`, with
  `agent/security/labels.py` + `agent/dataflow/firewall.py` as the routing/lawful-link
  layer.
- **Caveat.** Bandwidth-saving must not silently drop a dissenting agent's evidence;
  "reduce" has to preserve minority provenance, not just majority vote.

---

## Priority

Prototype **#3 (reasoning compiler)** and **#1 (reasoning roofline)** first: both reuse
machinery that already exists (`okf/graph.py`, `agent/calibration.py`) rather than opening
new scope, so they are the cheapest path to a measurable result that clears the gate.
#2, #4, #5 follow once there is a roofline to measure improvements against.

## From thesis to adoptable steps

The thesis is only useful if it becomes a procedure. The **deliberation-roofline protocol**
(feature #2, generalizes to any verifier-gated best-of-N / council loop):

1. **Instrument.** Log, per query, the deliberation budget spent (samples / council width /
   retrieval depth) and whether the emitted answer was *verifier-accepted* and *actually
   correct* (on a labeled slice). One row per (query, budget).
2. **Build the curve.** Plot effective quality `Q(N)` = P(emit an actually-correct answer)
   against budget `N`. Expect it concave and saturating.
3. **Estimate the ceiling.** Fit / read off the asymptote. With a verifier of recall `r`
   and false-positive rate `f` over difficulty mix `{p_i}`, the ceiling is
   `mean_i (p_i·r)/(p_i·r + (1-p_i)·f)` — a property of the **verifier**, not the budget.
4. **Set the operating budget at the ridge.** Pick `N* = ` smallest budget reaching ~95%
   of the ceiling. Spend there; everything past `N*` is wasted compute.
5. **If the ceiling is below target, fix the verifier — not the budget.** This is the
   load-bearing consequence: when `Q(∞) < target`, no amount of additional thinking helps;
   the lever is verifier precision (lower `f`) / recall (higher `r`), or abstaining more.

This plugs into `agent/graded_decision.py` (the answer/hedge/abstain router gains a
*budget* dimension) and `agent/best_of.py` / `agent/council_deliberate.py` (which gain an
early-exit at `N*`).

## Tested: the deliberation roofline holds (and the ceiling is the verifier)

The protocol's claims are implemented and **run** as a falsifiable experiment in
[`reasoning/deliberation_roofline.py`](../../reasoning/deliberation_roofline.py) — a
verifier-gated best-of-N over a 90-item easy/medium/hard task, Monte-Carlo simulated **and
checked against a closed-form derivation** (offline, seeded, no GPU/keys). Saved output:
[`reasoning/results/deliberation_roofline.txt`](../../reasoning/results/deliberation_roofline.txt).

Measured verdict (trials=800, seed=1234):

| Verifier (recall, fpr) | Ceiling `Q(∞)` | Ridge `N*` (95%) | `Q` at N=64 | Compute past ridge |
|---|---|---|---|---|
| oracle (1.0, 0.0)  | **1.000** | 16 | 1.000 | 4× → +0.025 |
| good   (0.95, 0.05)| **0.905** | 16 | 0.905 | 4× → +0.010 |
| leaky  (0.85, 0.15)| **0.777** | 8  | 0.777 | 8× → +0.017 |

- **H1 — concave / diminishing returns:** confirmed in all three.
- **H2 — finite ridge point:** confirmed — e.g. the leaky verifier is within 5% of its
  ceiling by `N*=8`; the 8× compute from 8→64 buys **+0.017** quality.
- **H3 — ceiling set by the verifier, not compute:** confirmed — oracle plateaus at 1.000,
  leaky at **0.777**, and *no budget crosses it*. You cannot deliberate past a leaky verifier.
- **Soundness:** Monte-Carlo matches the closed form within **0.0019** — the model is
  validated, not assumed.

One honest, non-obvious side finding the run surfaced: task-level *accuracy-on-answered*
**drifts down** as budget grows (e.g. good-verifier 0.954→0.905), because more deliberation
broadens coverage to *harder* items that have lower per-item verified-accuracy. Higher
coverage is not free accuracy — another reason `N*` matters.

Reproduce:

```bash
python reasoning/deliberation_roofline.py --run        # the experiment + verdict
python reasoning/deliberation_roofline.py --self-test   # assert the invariants
```

**Implication for Sophia.** This is the verifiability axis and the efficiency axis meeting:
the system's quality ceiling is a function of its *verifier*, and beyond a measurable ridge
point, the right move is not more compute but a better check — or principled abstention.
That is exactly the discipline `VISION.md` already commits to, now with a budget attached.

## Non-goals

- No claim that any of this constitutes AGI, sentience, or general intelligence.
- No efficiency win that costs traceability — every feature here is subordinate to the
  verifiability axis, not a license to weaken it.
- No headline number that hasn't cleared the no-overclaim measurement gate.
