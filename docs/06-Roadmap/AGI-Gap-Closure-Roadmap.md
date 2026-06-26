# AGI Gap-Closure Roadmap — turning honest gaps into closed-loop strengths

**Status:** planning document. Nothing here is an AGI claim. Every workstream ships under the
no-overclaim gate (`tools/lint_claims.py`) and leaves `canClaimAGI` **False** until its own
acceptance gate is met *and* recorded in `agi-proof/failure-ledger.md`.

This roadmap responds to an external repo-grounded gap audit. It restates the findings against
the **verified** state of this tree (some audit references were corrected — see
[Corrections to the audit](#corrections-to-the-audit)), then converts each weakness into a
concrete fix with an acceptance gate tied to a real file and command.

---

## 1. Diagnosis (verified)

The repo is not short of architectural surface area. It has gates, councils, intake contracts,
a single per-case pipeline with eight independently-toggleable ablation flags, long-context and
long-horizon harnesses, a proof package, and ~215 test files. The honest deficit is **closed-loop
proof that the stack buys general competence at acceptable cost**:

- **Integration:** promising modules (`agent/graded_decision.py`, `agent/layered_memory.py`,
  the AGI-pillar toys, `agent/claim_router.py`) are not driven by `run_case()`. They exist as
  offline harnesses or SSIL infrastructure, so their value is never *measured on the live path*.
- **Measurement:** the Level-3 harnesses exist and are unit-tested, but the evidence runs
  (fresh hidden pack, full ablation matrix, manual semantic review, external benchmark, timed
  long-horizon, clean-clone replication) are mostly open in `agi-proof/TODO.md`. The strongest
  real evidence today is the **calibration / abstain** finding (multi-judge corroborated,
  κ≥0.40, ≥3 runs, CI excludes 0) — and even that carries a self-authored-pack independence caveat.
- **Training loop:** decontamination and dataset discipline are solid; a *promoted* local model
  with a populated checkpoint registry and a closed eval ladder is not.

**The compass.** Every other bet — long-context packing, tool router, RLVR promotion, council
wiring — needs the same question answered before it earns headline status: does it **help, hurt,
or only change abstention**? Adding another architecture layer before the delta loop is closed
produces more scaffold without guidance on what actually helps. So the sequencing below is
**evidence first, integration second.**

---

## 2. Corrections to the audit

Grounding the audit against the live tree (`tools/run_hidden_eval_sophia.py`,
`agent/`, `agi-proof/`) surfaced three discrepancies worth fixing so the roadmap stays honest:

| Audit statement | Verified reality |
|---|---|
| `agi-proof/architecture-bets.json` tracks bet status (`selective-tool-router`, `hybrid-memory`). | **No such file exists.** Bet/status narrative lives in `VISION.md`, `agi-proof/baseline-ablation/README.md`, and the failure ledger. Treat "create a single bets registry" as itself a small roadmap item (W0). |
| The best unwired integration target is `graded_decision.decide()` into `run_case()`. | Correct that it is **unwired** — but the *lowest-risk* live seam is already present: `agent/gate.py:check_response(..., route_claims=...)` calls `agent/claim_router.py`, and `run_case()` calls it with `route_claims` effectively off (`domain=None`, line ~970/1027). Flipping that behind a new ablation flag is a smaller, safer first integration than grafting `graded_decision`. |
| Backends are "expired token / unset key." | Confirmed in this environment: `DEEPSEEK_API_KEY` unset and no Grok/XAI key. Every evidence run below is **blocked on a live backend** and must run where a key exists (local with `.env`, or CI/RunPod with secrets). The harnesses themselves are ready. |

The eight live ablation flags actually accepted by `run_case()` are:
`raw_system`, `use_kb`, `use_evidence`, `use_council`, `use_gate`, `use_memory`, `use_tools`,
`allow_repair` (`tools/run_hidden_eval_sophia.py`, `Ablation` dataclass). The wired verifier path
is `gate.check_response` → `_legal_gate`/`_numeric_gate` → `agent/verifiers.py`
(`legal_citation_exists`, `legal_holding_faithful`, `math_sound`). These are the real levers; the
roadmap is written against them.

---

## 3. Workstreams

Each workstream lists **Why**, **Build**, **Acceptance** (the gate that flips it from open to
closed in the failure ledger), **Blocker** (what it needs that this environment lacks), and an
**Honest bound** (what it does *not* prove). Ordered by leverage.

### W0 — Single bets registry *(cheap, do first; unblocks tracking)*

**Why:** the audit assumed a `architecture-bets.json` that does not exist. Without one, "which
module is wired, scaffold, or retired, and what would close it" is scattered across prose.

**Build:** create `agi-proof/architecture-bets.json` — one record per bet
(`graded_decision`, `claim_router`, `layered_memory`, `planner_mcts`,
`predictive_world_model`, `selective_tool_router`, `hybrid_memory`) with fields
`{status: scaffold|wired|measured|retired, live_caller: <file:fn|null>, ablation_flag,
closing_experiment, ledger_id}`. Add a unit test asserting every `agent/*pillar*` /
unwired module named here resolves to a real file, and a `tools/lint_claims.py` hook that fails
if a bet is marked `wired` but no `live_caller` is set.

**Acceptance:** registry committed; `pytest tests/test_architecture_bets.py` green; each bet's
`status` matches a verifiable grep of the live path.

**Blocker:** none.

**Honest bound:** a registry tracks intent; it proves nothing about capability.

---

### W1 — Fresh hidden-pack run through the full pipeline *(highest leverage — the compass)*

**Why:** the preregistered Level-3 bar (`agi-proof/preregistered-thresholds.md`) centers on
fresh hidden tasks, not a new layer. Current packs are **spent** and the only valid full-pipeline
run produced 0/8 nonempty answers (`failure-ledger.md:hidden-fresh-pack-sophia-grok-2026-06-19`).

**Build:**
1. Obtain/author one **unspent, reviewer-controlled** pack (schema:
   `agi-proof/hidden-reviewer-packs/schema.json`). Independence matters more than size —
   a third-party-authored pack closes the `calibration-self-authored-pack-2026-06-22` caveat.
2. Run `python tools/run_hidden_eval_sophia.py <pack.json> --backend <live>` with logs intact.

**Acceptance:** ≥8/8 nonempty answers, **0 backend failures**, artifact under
`agi-proof/benchmark-results/`; ledger entry `hidden-full-sophia-valid-run-not-yet-run` updated
with the real auto-score and strict-pass count reported **separately**.

**Blocker:** live backend key. **Fail conditions:** 0 answers, a spent pack reused, or any
backend failure (reuse the existing failure IDs).

**Honest bound:** a valid full-pipeline run is *evidence the pipeline executes*, not yet evidence
it beats raw — that is W2.

---

### W2 — Full ablation matrix on the abstain/calibration pack *(closes the delta loop)*

**Why:** this is the provenance-delta question the whole repo is built to answer. The protocol,
multi-judge history, and deterministic scorer already exist
(`agi-proof/baseline-ablation/`, `tools/run_ablation_sophia.py`, `tools/run_calibration_judge.py`).

**Build:** run the seven-mode matrix (`--modes all`: sophia-full, raw, raw+tools, no-gate,
no-kb, no-council, no-memory) ≥3 seeds on the abstain pack. Report, per arm: fabrication rate,
calibration Δ, **false-positive / over-abstain cost**, and 95% CIs.

**Acceptance:** fabrication Δ CI excludes 0 **and** FP/over-abstain cost is documented; ledger
updated. **Fail:** CI includes 0, or full loses to raw on task success without an FP analysis.

**Blocker:** live backend key (≥3 seeds × 7 modes × N cases).

**Honest bound:** a calibration win is *epistemic conservatism measured well*. It is not a
general-competence claim until task-success (not just abstention) also improves — report both,
even when they diverge, as the ledger already does.

---

### W3 — Two-pass manual semantic review *(unblocks "strict pass")*

**Why:** strict-pass is 0/8 across spent packs **because manual review is pending**, not
necessarily because answers are wrong (`failure-ledger.md`, multiple hidden entries). Regex/auto
score alone cannot promote a claim past "candidate."

**Build:** complete `agi-proof/hidden-reviewer-packs/MANUAL-SEMANTIC-REVIEW.md` two-pass review
on the W1 pack; record reviewer-signed aggregate with strict-pass reported **separately** from
the regex score.

**Acceptance:** reviewer-signed aggregate committed; strict-pass rate published distinct from auto
score. **Fail:** promoting any result from regex/auto score alone.

**Blocker:** a human reviewer (can be the author for a first pass, but a third party is required
to clear the independence caveat).

**Honest bound:** internal manual review raises confidence; it does not substitute for the
clean-clone third-party replication in `agi-proof/third-party-replication/`.

---

### W4 — Wire one unwired bet behind an ablation flag *(integration, measurement-gated)*

**Why:** the measurement-first philosophy says new wiring earns its place only as an *ablatable
lever*. The smallest such lever already exists: `route_claims` in `agent/gate.py:check_response`.

**Build (pick the lower-risk seam first):**
- **Option A (recommended, smallest):** add a `use_claim_router` ablation flag to
  `run_case()`; when on, call `check_response(..., route_claims=True)` so `agent/claim_router.py`
  is exercised on the live path. No new module, just an existing dormant seam.
- **Option B:** graft `agent/graded_decision.decide()` as a post-gate confidence grade behind a
  `use_graded_decision` flag, replacing the binary gate with the existing confidence curve.

**Acceptance:** ablation artifact shows a measurable routing/abstention change vs the flag-off
arm; new unit test green; **no regression** on the W2 calibration pack. **Fail:** any silent
behavior change with no ablation artifact, or a regression on calibration.

**Blocker:** the *unit test* needs no backend; the *ablation delta* needs a live backend (depends
on W2 infrastructure).

**Honest bound:** wiring a module makes it *measurable*, not *proven*. Keep the bet at
`status: wired` (not `measured`) in W0's registry until the delta CI excludes 0.

---

### W5 — One external-benchmark pilot *(outward measurement)*

**Why:** every internal pack carries a self-authoring caveat. One external benchmark with a
public provenance trail breaks that. Plan and checklist already exist
(`agi-proof/external-benchmarks/README.md`, `PROVENANCE-DELTA-CHECKLIST.md`).

**Build:** run one pilot — GSM8K-style numeric exact-match or an ARC-lite slice — raw vs
sophia-full, with the gate-coverage cost stated. Record commit hash, backend version, raw aggregate.

**Acceptance:** filled benchmark-run artifact with `status != not_run`; ledger
`external-benchmarks-not-run` updated. **Fail:** claiming success without logs/checksums.

**Blocker:** live backend key; dataset license discipline (use *style* samples until the official
set is licensed, per the preregistered-thresholds note).

**Honest bound:** one pilot is a *foothold*, not the full external suite (ARC-AGI, GAIA,
SWE-bench) the Level-4 bar requires.

---

### W6 — Register one checkpoint candidate *(close the training loop's edge)*

**Why:** the MLOps registry is a schema with no artifacts. Even a **rejected** candidate with
real eval refs is more honest than an empty registry plus a verbal "we trained something."

**Build:** after any gated training run (the RLVR-math rung is already cleared with a real
3-seed N=60 artifact — `agi-proof/self-extension/math-rlvr-3seed-n60/`), add one entry to
`agi-proof/mlops/checkpoint-registry.json` with `config_hash`, eval-artifact refs, promotion
verdict (promote **or** reject), and `canClaimAGI: false`.

**Acceptance:** registry has ≥1 entry with config hash + eval refs + explicit verdict. **Fail:**
empty registry alongside any training narrative.

**Blocker:** GPU (gated to CI/RunPod); the *registration step itself* can reuse the
already-completed RLVR-math artifact with no new GPU run.

**Honest bound:** a registered checkpoint proves *provenance discipline*, not model quality —
quality is whatever the linked eval refs actually show.

---

## 4. Two-week sequencing

| Week | Focus | Workstreams | Gate to advance |
|---|---|---|---|
| 0 (now) | Tracking + no-backend integration prep | **W0** | bets registry committed, test green |
| 1 | Evidence (needs backend) | **W1 → W2 → W3** | valid hidden run, ablation Δ CI excludes 0, signed manual review |
| 2 | Integration + outward measurement | **W4 → W5 → W6** | ablatable wiring with no regression, one external pilot, one registry entry |

**Dependency note.** W1–W3 and W5 are all blocked on a live backend; W6's GPU step is gated to
CI/RunPod. The only fully-unblocked items in this environment are **W0** and the **unit-test half
of W4**. Do those here; run the backend-gated evidence where a key exists.

---

## 5. What "fixed strength" means here

A weakness becomes a strength in this repo only when it is **closed in the failure ledger** with:
acceptable false-positive / over-abstain cost, ≥3 seeds where numbers are cited, two-pass manual
review for anything promoted past "candidate", and an explicit residual-independence note. The
bar is deliberately the repo's own (`agi-proof/preregistered-thresholds.md`) — this roadmap does
not move it. It only sequences the work so the architecture is judged by the measurement loop,
never as a substitute for it.

> Bottom line: the missing piece is not an AGI architecture diagram. It is closed-loop proof that
> the architecture buys general competence at acceptable cost — on fresh hidden tasks, with
> ablations, false-positive accounting, and independence where the gates require it. Build
> architecture only to test hypotheses that loop raises.
