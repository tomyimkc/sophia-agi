# Handover: adversarial review of five AGI-proof evidence tools for `tomyimkc/sophia-agi`

You are reviewing a PR package produced by another AI research assistant ("Sophia Research
Advisor") on 2026-07-01. Your job is to **adversarially verify** it against the live repo and
report back a structured verdict. Do not rewrite the tools yet — first find what is wrong,
unbound, or overclaimed. You have (or can fetch) the `tomyimkc/sophia-agi` tree at `main`.

---

## 1. Context you need

The repo is an AGI-candidate architecture whose honest headline is a **selective-prediction /
calibration** result (self-consistency selective prediction lifts selective accuracy; local-model
attribution hallucination cut ~36%→24% at 0% FP). Its own diagnosis (`AGI-Gap-Closure-Roadmap.md`)
is: **rich in architecture, poor in independently-measured closed-loop evidence.** As of `main`
the failure ledger has ~98 Open entries and `agi-proof/TODO.md` has 23 open items. The compass is
*evidence first, integration second*, under a strict discipline:

- **fail-closed** everywhere (no backend / no dep / missing data → abstain or an "environment
  artifact, not a score" report; never fabricate a number, never crash);
- **no overclaim** (every artifact carries `candidateOnly:true`, `level3Evidence:false`,
  `canClaimAGI:false`; public numbers need ≥2 judge families or a CI excluding zero);
- **heavy deps opt-in** (torch/mlx/adapters lazy-imported, abstain when absent; core stays stdlib);
- **decontamination floor** on any eval data (shingle/Jaccard guard).

The package implements the five highest-leverage OPEN TODO items as runnable, unit-tested drop-in
files. **Critical framing the author asserts — verify it holds:** these are *instruments, not
results*. They were unit-tested offline on synthetic fixtures; **none was run with a live backend.**
No metric/CI/score is claimed; all five ledger rows stay **Open** with an acceptance gate.

## 2. What to review (files in the package)

| WS | File | Claims to close |
|---|---|---|
| A | `tools/make_independent_hidden_pack.py` (+ `examples/reviewer_pack_input.json`) | Third-party reproduction — TODO 15–20, 22 |
| B | `tools/run_t1_gated_self_training.py` | Verifier-gated self-training w/ shift-split transfer arm (Thesis T1); `rlvr-live-run-not-yet-gated` |
| C | `tools/run_arc_agi_sophia.py` | External benchmark ARC-AGI — TODO 11 |
| D | `tools/ablation_no_executor.patch.md` + `tools/run_long_horizon_timed.py` | No-executor ablation cell — TODO 2; long-horizon timed runs — TODO 7–10 |
| — | `tests/` (17 offline tests), `ledger/failure-ledger-additions.md`, `README.md` | — |

## 3. The interface bindings to VERIFY (this is the highest-value check)

The author fetched these from GitHub raw and bound the tools to them **by name**. Confirm each
still exists with that signature on current `main`; flag any drift. If a name moved or a signature
changed, the tool will import-fail or mis-bind — that is the most likely real defect.

- `agent/gate_reward.py` → `reward(completion, *, question=None, temptation=None) -> float`
  and `is_abstention(text) -> bool`. **Check:** WS-B and WS-C treat `reward(...) >= 0.0` as the
  "grounded/accepted" floor. Is `0.0` actually the right accept boundary given `REWARD_MIN/MAX`
  and `REWARD_ABSTAIN`? The tools set `ACCEPT_FLOOR = 0.0` — is that too permissive or too strict?
- `agent/continual_plasticity.py` → `evaluate_update(candidate, *, target_suite, min_target_delta=0.03,
  max_protected_regression=0.01, require_artifacts=2, ...)`, and dataclasses `EvalMetric(suite,
  before, after, protected=False)`, `UpdateCandidate(id, kind, metrics, verifier_artifacts=(),
  contaminated=False, notes="")`, `PromotionDecision(candidate_id, verdict, reasons, metrics, ...)`.
  **Check:** WS-B builds an `UpdateCandidate` and calls `evaluate_update(..., target_suite=
  "heldout_shifted")`. Does it mark protected suites correctly? `require_artifacts=2` — does WS-B
  supply ≥2 `verifier_artifacts`? (It passes the two held-out paths — is that what the gate expects,
  or does it want verifier RUN artifacts?)
- `tools/run_hidden_eval_sophia.py` → the frozen `Ablation` dataclass (fields: `label, raw_system,
  use_kb, use_evidence, use_council, use_gate, use_memory, use_tools, allow_repair, use_intake,
  use_claim_router`) and the `ABLATION_MODES` dict. **Check:** WS-D's patch adds `use_executor` and a
  `sophia-no-executor` mode. Does the executor actually have a single dispatch site in `run_case` to
  gate, or is execution spread across stages (in which case the patch is incomplete)? Find the real
  executor call site and confirm the diff targets it.
- `agent/long_horizon.py` → `build_ledger(goal, subtasks, *, ledger_id, ledgers_dir=...) ->
  TaskLedger`, `run_long_horizon(ledger, *, client=None, recovery=None, approve_tools=False,
  max_nodes=256) -> LongHorizonResult(ledger_id, ok, completed, failed, blocked, total_cost_usd,
  ledger_path)`. **Check:** WS-D's timed wrapper enforces the `--minutes` budget only as a *post-hoc*
  timed-out marker, not a cooperative in-harness deadline. Is there a real deadline hook in
  `run_long_horizon` (or a per-node callback) it should use instead so a 1-day budget actually stops?
- `agi-proof/hidden-reviewer-packs/schema.json` → WS-A validates against it (jsonschema if present,
  else a stricter structural fallback). **Check:** does the fallback accept anything the real runner
  would reject, or vice-versa? Does `tools/run_hidden_eval_sophia.py` read any pack field the WS-A
  generator does not emit?
- `agent/model.py` → the tools assume a `Model(spec)` adapter with `.complete(prompt)`. **Check
  hard:** confirm the class name, constructor spec format, and completion method. If the real adapter
  differs (e.g. `ModelClient`, `generate()`, a factory function), every tool's `load_generator` /
  `_make_client` is mis-bound and silently falls to fail-closed — the tools would "run" but never use
  a backend. This is the single most important thing to check.

## 4. Deeper questions to pressure-test (name the strongest objection to each)

1. **WS-B transfer arm.** The 3-way outcome classifier (heldout_lift / verifier_overfit /
   reward_hacking) depends on a genuine distribution SHIFT between `heldout_scored` and
   `heldout_shifted`. What operationalizes "shift" here, and could a lazy split (same items,
   reworded) make the overfit canary un-fireable? The author left `gen_after=None` (no trainer
   wired) — is the promotion gate meaningfully exercised in a dry run, or is it decorative until a
   real SFT/DPO step exists?
2. **WS-B reward validity.** `gate_reward.reward` intentionally does NOT pass `question` to the
   trap-grader. Using it as a *training* reward — does that create a reward-hacking surface the gate
   was never designed to resist (e.g. reward for confident abstention on answerable items)? Is the
   `graded_abstain_reward` temptation scaling relevant here?
3. **WS-C ARC.** Grid parsing stops at the first non-grid line. Is that robust to models that
   preface the grid with prose, or emit it fenced? Is exact-grid-match the right ARC scorer for
   ARC-AGI-3 (which is interactive), or only ARC-AGI-1? Should low accuracy + high abstention be
   framed as a *result* at all, or is it uninformative without the raw-vs-gated ablation?
4. **WS-A independence.** The tool STAMPS `reviewer.status:"third-party"` but cannot verify
   authorship. What out-of-band signature/attestation flow should accompany it so "independent" is
   more than a self-applied label? Does the decontam corpus (`wiki/`) cover the actual training
   distribution, or only a slice — i.e. can a pack pass decontam and still be contaminated against
   data not in `wiki/`?
5. **WS-D ablation.** Is Sophia-full-vs-no-executor the *only* missing ablation cell, or are there
   others (e.g. no-intake × no-gate interactions)? Does bypassing the executor change the fabrication
   scorer's inputs in a way that makes the delta not apples-to-apples?

## 5. Report back to the originating assistant in THIS structure

Return a single markdown report with these sections:

1. **Binding audit** — a table: `symbol | expected signature | actual on main | status (OK / DRIFTED / MISSING)`. This decides whether the tools import at all.
2. **Blocking defects** — anything that makes a tool import-fail, silently fail-closed forever, or violate the fail-closed/no-overclaim discipline. Ordered by severity.
3. **Answers to the five pressure-test questions** — for each, state the strongest objection and whether it's fatal, fixable, or acceptable-as-scoped.
4. **Acceptance-gate realism** — for each of the five ledger rows, is the stated gate the right bar, and what is the minimum live run (backend, dataset, hours) to close it?
5. **Verdict + minimal change set** — merge as-is / merge-after-fixes / redesign, plus the smallest diff that makes each tool live-runnable, and which ONE workstream to run first for maximum evidence-per-effort.

Cite exact file paths and line numbers from the current tree for every defect. Do not invent repo
symbols — if you can't find one, say so and mark it MISSING rather than guessing. Where you'd change
a tool, give the concrete diff, not prose.
