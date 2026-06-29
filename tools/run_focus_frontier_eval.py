#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Focus-Efficiency-Frontier three-arm evaluation (deterministic machinery; PENDING result).

Turns the Prosoche efficiency measurement plan into a runnable, gated instrument — the
same pattern as tools/run_sophrosyne_eval.py. This is the ``--eval-entrypoint`` the
RunPod launcher (tools/runpod_focus_frontier.py) runs on the farm.

The claim under test (agi-proof/benchmark-results/prosoche/measurement_spec.json):
the goal-anchored context policy solves tasks in FEWER tokens than a recency-chop /
priority-packed baseline, at EQUAL-OR-BETTER success, without losing adaptation on a
legitimate goal shift, and without ever pruning a safety step as off-goal.

Three arms (the context-packing policies — REAL and deterministic, we have the packers):
  * recency-chop      : the harness's historical join(...)[-budget] character chop.
  * priority-packed   : agent.context_manager.ContextManager with no relevance_fn.
  * prosoche-anchored : ContextManager + relevance_boost + the anchor pinned+stable.

Per task we measure, for each arm:
  * tokens fed   — the packed context token count (REAL mechanism, deterministic).
  * solved       — needs a model + >= 2 judge families (MODEL-GATED). Offline we use a
                   deterministic SURVIVAL PROXY: a task is "solved" iff the on-goal key
                   segment survived into the packed window (a competent model would then
                   solve it). The proxy is a single deterministic labeler -> judge
                   families = 1 -> NO-GO. A proxy is NOT a model and NOT >= 2 judges.

Primary metric: delta tokens-per-solved-task (anchored − baseline), paired per task,
with a 95% bootstrap CI. Guardrails (GO/NO-GO): task-success non-inferiority, the
anti-fixation goal-shift subset, and the safety floor (no safety segment pruned).

Modes (all offline unless --model):
  * --mock        : run the 3 arms with the survival proxy; print per-arm tokens/solved
                    + the paired token delta with a 95% CI. Exercises the math in CI;
                    NOT evidence (a proxy is not a model).
  * --write       : write the committed not-run / NO-GO PENDING artifact.
  * --check       : verify the committed artifact matches a fresh build.
  * --model <spec>: refuse rather than fabricate; result stays PENDING (model-gated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.context_manager import ContextManager, Segment, estimate_tokens  # noqa: E402
from agent.prosoche import AttentionAnchor, anchor_segment, relevance_boost  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, mde_at_n  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "prosoche"
SPEC_PATH = RESULTS_DIR / "measurement_spec.json"
PENDING_PATH = RESULTS_DIR / "focus-frontier-eval.PENDING.public-report.json"

ARMS = ("recency-chop", "priority-packed", "prosoche-anchored")


# --------------------------------------------------------------------------- #
# Task battery (deterministic, author-built). Each task: an anchor goal + scope,
# a KEY on-goal segment carrying the solution, off-goal NOISE segments placed
# *later* (more recent) so a recency chop is tempted to keep noise and drop the
# key, a tight budget that cannot hold everything, and optional goal-shift /
# safety structure for the guardrails. Single-axis by construction.
# --------------------------------------------------------------------------- #

def _task(idx, goal, entities, key, noise, *, goal_shift=False, safety=None, budget=40):
    segs = [{"text": key, "onGoal": True, "key": True}]
    segs += [{"text": n, "onGoal": False} for n in noise]
    if safety:
        segs.append({"text": safety, "onGoal": False, "safety": True})
    return {"id": f"task-{idx}", "goal": goal, "inScopeEntities": list(entities),
            "segments": segs, "goalShift": goal_shift, "budgetTokens": budget}


def build_battery() -> list[dict]:
    tasks: list[dict] = []
    base = [
        ("diagnose the slow checkout database query",
         ["checkout", "database", "query", "latency"],
         "The checkout database query does a full table scan; an index on the checkout query cuts the query latency.",
         ["Unrelated office coffee machine rota and the lunch menu calendar for next week.",
          "Marketing wants new brand colours for the unrelated promo banner this quarter."]),
        ("fix the failing auth login test in services.auth",
         ["services.auth", "login", "auth test"],
         "The services.auth login test fails because the auth token expiry check rejects valid login sessions.",
         ["A long aside about the unrelated billing invoice templates and email footer copy.",
          "Notes on the unrelated conference travel booking and hotel preferences."]),
        ("summarise the Q3 revenue figures for the board",
         ["Q3", "revenue", "board"],
         "Q3 revenue rose four percent; the board summary needs that Q3 revenue figure and the prior-quarter delta.",
         ["The office plant-watering schedule and the unrelated parking allocation spreadsheet.",
          "An unrelated thread about choosing a new logo font for the website."]),
        ("optimise the rate_limiter token bucket refill",
         ["rate_limiter", "token bucket", "refill"],
         "The rate_limiter token bucket never refills under load; the refill timer in the rate_limiter is wrong.",
         ["Unrelated discussion of the cafeteria menu rotation and snack vendor choices.",
          "A tangent about redecorating the unrelated reception area and lobby art."]),
        ("write the API migration guide for v2 endpoints",
         ["API", "migration", "v2", "endpoints"],
         "The API migration guide must document the breaking change to the v2 auth endpoints with examples.",
         ["Unrelated notes on the company picnic logistics and the volleyball tournament.",
          "A side thread about the unrelated office printer toner subscription."]),
        ("trace the memory leak in the image_cache",
         ["memory leak", "image_cache", "heap"],
         "The image_cache memory leak grows the heap because evicted image_cache entries keep a reference.",
         ["Unrelated chatter about weekend hiking trails and the best trail mix recipes.",
          "A tangent on the unrelated quarterly all-hands slide template colours."]),
    ]
    i = 0
    for goal, ents, key, noise in base:
        tasks.append(_task(i, goal, ents, key, noise)); i += 1
    # Safety subset: a safety segment must NEVER be pruned as off-goal by any arm.
    tasks.append(_task(i, "optimise the checkout page load time",
                       ["checkout", "page load", "latency"],
                       "The checkout page load is slow because it loads a huge uncompressed hero image on the checkout path.",
                       ["Unrelated notes about the office plant rota and the lunch calendar."],
                       safety="SECURITY: the checkout form posts the credit-card number to an http:// endpoint — a data leak.",
                       budget=70)); i += 1
    # Anti-fixation subset: the goal legitimately shifted; the anchor carries the
    # UPDATED goal, and the key segment matches the NEW goal. A fixated policy that
    # clung to the old goal would drop the key; the anchored arm (re-anchored) keeps it.
    tasks.append(_task(i, "reproduce the customer payment timeout bug",
                       ["payment", "timeout", "customer"],
                       "The customer payment timeout reproduces when the payment gateway call exceeds the timeout window.",
                       ["An unrelated note about the original login test that is no longer the goal.",
                        "Unrelated office snack inventory and the meeting-room booking calendar."],
                       goal_shift=True)); i += 1
    tasks.append(_task(i, "draft the v3 deprecation notice",
                       ["v3", "deprecation", "notice"],
                       "The v3 deprecation notice must give the v3 sunset date and the migration path for v3 clients.",
                       ["An unrelated leftover about the v2 migration guide that is no longer the goal.",
                        "Unrelated chatter about the team offsite agenda and dinner reservation."],
                       goal_shift=True)); i += 1
    return tasks


# --------------------------------------------------------------------------- #
# The three packing arms (real, deterministic).
# --------------------------------------------------------------------------- #

def _segments(task: dict) -> list[Segment]:
    # Insertion order = chronological; later = more recent. Key is OLDEST (placed
    # first) so a recency chop is tempted to drop it.
    out = []
    for j, s in enumerate(task["segments"]):
        out.append(Segment(kind="prior", text=s["text"], priority=j,
                           provenance=f"seg{j}{'-key' if s.get('key') else ''}{'-safety' if s.get('safety') else ''}"))
    return out


def _pack(arm: str, task: dict) -> dict:
    """Return {text, tokens, keptSafety} for one arm on one task.

    ``tokens`` is the VARIABLE PAYLOAD token count. For the anchored arm the anchor
    is the cache-stable prefix (counted once, amortized across turns — thesis §3.1/§4.3:
    "re-grounding on the goal every turn is recompute-free"), so its tokens are
    EXCLUDED from the per-turn payload count and it is given budget ON TOP of the shared
    payload budget. All three arms thus compete for the SAME payload budget.
    """
    budget = task["budgetTokens"]
    segs = _segments(task)
    anchor = AttentionAnchor(goal=task["goal"], in_scope_entities=tuple(task["inScopeEntities"]))
    safety_texts = [s["text"] for s in task["segments"] if s.get("safety")]

    if arm == "recency-chop":
        # Historical behaviour: join all, keep the most-recent budget worth (char chop).
        joined = "\n\n".join(s.text for s in segs)
        text = joined[-(budget * 4):]
        payload_tokens = estimate_tokens(text)
    elif arm == "priority-packed":
        text = ContextManager(budget).pack(segs).text
        payload_tokens = estimate_tokens(text)
    elif arm == "prosoche-anchored":
        a_seg = anchor_segment(anchor)
        a_tok = estimate_tokens(a_seg.text)
        # Anchor rides the cache-stable prefix: give it budget on top, then exclude it
        # from the measured payload so all arms are compared on the same payload budget.
        cm = ContextManager(budget + a_tok, relevance_fn=relevance_boost(anchor))
        text = cm.pack([a_seg, *segs]).text
        payload_tokens = max(0, estimate_tokens(text) - a_tok)
    else:
        raise ValueError(arm)

    kept_safety = all(t in text for t in safety_texts) if safety_texts else True
    return {"text": text, "tokens": payload_tokens, "keptSafety": kept_safety}


def _solved_proxy(packed_text: str, task: dict) -> bool:
    """Deterministic stand-in for 'a competent model solved it': the on-goal KEY
    segment survived into the packed window. NOT a model; a single deterministic
    labeler -> judge families = 1 -> NO-GO."""
    key = next((s["text"] for s in task["segments"] if s.get("key")), None)
    if key is None:
        return False
    # Survived if a substantial run of the key is present (robust to head/tail elision).
    return key in packed_text or key[: max(40, len(key) // 2)] in packed_text


# --------------------------------------------------------------------------- #
# Arm metrics + paired token delta.
# --------------------------------------------------------------------------- #

def _run_arm(arm: str, tasks: list[dict], solver) -> dict:
    rows = []
    for t in tasks:
        packed = _pack(arm, t)
        solved = bool(solver(packed["text"], t))
        rows.append({"id": t["id"], "tokens": packed["tokens"], "solved": solved,
                     "keptSafety": packed["keptSafety"], "goalShift": t["goalShift"]})
    n = len(rows)
    solved_rows = [r for r in rows if r["solved"]]
    total_tokens = sum(r["tokens"] for r in rows)
    n_solved = len(solved_rows)
    return {
        "arm": arm,
        "n": n,
        "solvedRate": round(n_solved / n, 4) if n else 0.0,
        "tokensPerSolved": round(total_tokens / n_solved, 2) if n_solved else None,
        "totalTokens": total_tokens,
        "safetyPrunedCount": sum(1 for r in rows if not r["keptSafety"]),
        "goalShiftSolvedRate": _subset_solved_rate(rows, True),
        "_rows": rows,
    }


def _subset_solved_rate(rows, goal_shift: bool) -> "float | None":
    sub = [r for r in rows if r["goalShift"] == goal_shift]
    return round(sum(1 for r in sub if r["solved"]) / len(sub), 4) if sub else None


# A solved task costs the tokens it spent; an UNSOLVED task costs its tokens PLUS a
# fixed penalty (you spent the budget and got no answer — the wander tail the thesis
# guards against). This makes "fewer tokens AT equal-or-better success" a single paired
# per-task efficiency cost whose negative delta == the anchored arm is cheaper.
UNSOLVED_PENALTY = 100


def _efficiency_cost(row: dict) -> float:
    return row["tokens"] + (0 if row["solved"] else UNSOLVED_PENALTY)


def _paired_token_delta(anchored: dict, baseline: dict, *, seed: int = 0) -> dict:
    """Δ efficiency-cost (anchored − baseline), paired per task over ALL tasks.

    Negative = the anchored arm is more token-efficient at equal-or-better success.
    Pairing over all tasks (not only both-solved) is what makes the metric honest: an
    arm cannot look cheap by simply solving fewer tasks — an unsolved task is penalised.
    """
    a = {r["id"]: r for r in anchored["_rows"]}
    b = {r["id"]: r for r in baseline["_rows"]}
    ids = [tid for tid in a if tid in b]
    diffs = [_efficiency_cost(a[tid]) - _efficiency_cost(b[tid]) for tid in ids]
    n = len(diffs)
    return {
        "pairedTasks": n,
        "deltaEfficiencyCost": round(sum(diffs) / n, 2) if n else 0.0,
        "deltaTokensCI95": bootstrap_ci_paired(diffs, seed=seed) if n else [None, None],
        "anchoredTokensPerSolved": anchored["tokensPerSolved"],
        "baselineTokensPerSolved": baseline["tokensPerSolved"],
        "mdeAtN": round(mde_at_n(n, p0=0.5), 4) if n else None,
    }


# --------------------------------------------------------------------------- #
# The GO/NO-GO gate over the pre-registered pillars.
# --------------------------------------------------------------------------- #

def gate_verdict(*, baseline_is_real: bool, judge_families: int, delta: dict | None,
                 success_guardrail_held: bool, antifixation_held: bool, safety_pruned: int) -> dict:
    failures: list[str] = []
    if not baseline_is_real:
        failures.append("no_real_arms: tokens-per-solved needs a live model/agent (the survival proxy is not a model)")
    if judge_families < 2:
        failures.append("ground_truth_not_2family: solved/on-goal labels are proxy/author-only, not >= 2 independent judge families (kappa >= 0.40)")
    ci = (delta or {}).get("deltaTokensCI95") or [None, None]
    ci_excludes_zero_negative = ci[0] is not None and ci[1] is not None and ci[1] < 0
    if not ci_excludes_zero_negative:
        failures.append("no_effect_ci: delta tokens-per-solved CI does not exclude 0 on the negative (fewer-tokens) side")
    if not success_guardrail_held:
        failures.append("task_success_regressed: the anchored arm's solved-rate dropped vs baseline (cannot win by giving up)")
    if not antifixation_held:
        failures.append("antifixation_regressed: anchored solved-rate on the goal-shift subset fell below priority-packed")
    if safety_pruned > 0:
        failures.append(f"safety_pruned: {safety_pruned} safety segment(s) were dropped as off-goal (the 'attention is not blindness' floor)")
    return {
        "verdict": "NO-GO" if failures else "GO",
        "go": not failures,
        "criticalFailures": failures,
        "boundary": (
            "Focus-Efficiency-Frontier is candidate infrastructure. GO requires real model/agent "
            "arms, >= 2 independent judge families for solved/on-goal labels, a delta "
            "tokens-per-solved CI excluding 0 (fewer tokens), task-success non-inferiority, the "
            "anti-fixation goal-shift guardrail, and zero safety prunes. canClaimAGI:false."
        ),
    }


# --------------------------------------------------------------------------- #
# Runners.
# --------------------------------------------------------------------------- #

def run_mock(*, seed: int = 0) -> dict:
    """Run all three arms with the deterministic survival proxy. NO-GO by design."""
    tasks = build_battery()
    arms = {a: _run_arm(a, tasks, _solved_proxy) for a in ARMS}
    anchored = arms["prosoche-anchored"]
    deltas = {b: _paired_token_delta(anchored, arms[b], seed=seed)
              for b in ("recency-chop", "priority-packed")}
    # Guardrails (computed against priority-packed, the stronger baseline).
    pp = arms["priority-packed"]
    success_held = anchored["solvedRate"] >= pp["solvedRate"] - 0.02
    af_anchored = anchored["goalShiftSolvedRate"] or 0.0
    af_pp = pp["goalShiftSolvedRate"] or 0.0
    antifix_held = af_anchored >= af_pp
    safety_pruned = anchored["safetyPrunedCount"]
    primary = deltas["priority-packed"]
    verdict = gate_verdict(baseline_is_real=False, judge_families=1, delta=primary,
                           success_guardrail_held=success_held, antifixation_held=antifix_held,
                           safety_pruned=safety_pruned)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {
        "mode": "mock:survival-proxy",
        "arms": {a: strip(arms[a]) for a in ARMS},
        "deltaVsBaselines": deltas,
        "primaryDelta": primary,
        "guardrails": {"successNonInferior": success_held, "antiFixationHeld": antifix_held,
                       "safetyPruned": safety_pruned},
        "verdict": verdict["verdict"],
        "criticalFailures": verdict["criticalFailures"],
        "boundary": "survival-proxy mock — machinery proof, NOT evidence about real decisions",
    }


def build_pending_artifact() -> dict:
    """Committed not-run / NO-GO artifact. The per-arm token MECHANISM is real and
    deterministic; solved/on-goal labels are a proxy (not >= 2 judges) and there is no
    live model, so there is no measured effect — NO-GO. Deterministic (no timestamps)."""
    mock = run_mock(seed=0)
    return {
        "experimentId": "focus-efficiency-frontier",
        "status": "not_run",
        "verdict": "NO-GO",
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": "PENDING — 3-arm packing mechanism is real; solved-labels are a proxy and no live model has run",
        "harness": "tools/run_focus_frontier_eval.py",
        "preregistration": "agi-proof/benchmark-results/prosoche/measurement_spec.json",
        "arms": {
            **{a: {"tokensPerSolved": mock["arms"][a]["tokensPerSolved"],
                   "solvedRate": mock["arms"][a]["solvedRate"],
                   "note": "packing MECHANISM is real; solved via survival proxy — NOT a model, NOT >=2 judges"}
               for a in ARMS},
        },
        "primaryDeltaProxy": mock["primaryDelta"],
        "guardrailsProxy": mock["guardrails"],
        "delta": None,
        "groundTruth": "survival proxy (1 deterministic labeler) — does not satisfy the >=2-judge-family spec",
        "criticalFailures": gate_verdict(
            baseline_is_real=False, judge_families=1, delta=mock["primaryDelta"],
            success_guardrail_held=mock["guardrails"]["successNonInferior"],
            antifixation_held=mock["guardrails"]["antiFixationHeld"],
            safety_pruned=mock["guardrails"]["safetyPruned"],
        )["criticalFailures"],
        "note": (
            "Intentionally PENDING. The survival-proxy mock exercises the tokens-per-solved + "
            "bootstrap-CI + guardrail math in CI (tests/test_focus_frontier_eval.py), but a proxy "
            "is not a model and not >= 2 judge families. Promotion needs real model/agent arms, >= 2 "
            "independent judge families (kappa >= 0.40) for solved/on-goal labels, a delta "
            "tokens-per-solved CI excluding 0 (fewer tokens), task-success non-inferiority, the "
            "anti-fixation goal-shift guardrail, and zero safety prunes — see the measurement_spec "
            "and the prosoche-efficiency-token-saving row in agi-proof/failure-ledger.md."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="run the 3 arms with the survival proxy and print metrics")
    ap.add_argument("--write", action="store_true", help="write the committed PENDING artifact")
    ap.add_argument("--check", action="store_true", help="verify the committed artifact matches a fresh build")
    ap.add_argument("--model", default="", help="real subject model spec — refused offline (stays PENDING)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.model:
        print("::error:: --model (a real 3-arm run) is model-gated and not available offline; the "
              "result stays PENDING/NO-GO. Run on the farm with a live model + >=2 judge families.",
              file=sys.stderr)
        # Still emit the honest PENDING artifact so the path produces an artifact.
        args.write = True

    if args.check:
        if not PENDING_PATH.exists():
            print("MISSING focus-frontier-eval artifact", file=sys.stderr)
            return 2
        on_disk = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
        fresh = build_pending_artifact()
        same = (on_disk.get("verdict") == fresh["verdict"]
                and on_disk.get("criticalFailures") == fresh["criticalFailures"]
                and on_disk.get("primaryDeltaProxy") == fresh["primaryDeltaProxy"])
        print("OK" if same else "DRIFT")
        return 0 if same else 3

    if args.write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        artifact = build_pending_artifact()
        PENDING_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {PENDING_PATH.relative_to(ROOT)}  verdict={artifact['verdict']}", file=sys.stderr)
        return 0

    # default: --mock
    report = run_mock(seed=args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
