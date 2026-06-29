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


# --------------------------------------------------------------------------- #
# Real-model arm (model-gated). A subject model answers each task from each arm's
# packed context; >= 2 INDEPENDENT judge families label solved/unsolved against the
# known key fact (judge != subject). Tokens = the deterministic packed-payload size
# (the efficiency axis); SOLVED = whether a real model produced the key fact.
# --------------------------------------------------------------------------- #

# Pre-registered power floor: below this many items the result is an UNDERPOWERED
# PILOT (MDE >> 0.10), so it can never be a GO no matter how clean the direction.
MIN_ITEMS_FOR_POWER = 100
MIN_SEEDS = 3
MDE_TARGET = 0.10  # pre-registered: the powered regime requires MDE@N <= 0.10 (N ~ 400)


def _task_question(task: dict) -> str:
    return (f"Using ONLY the context above, explain: {task['goal']}. "
            "If the context does not contain the answer, reply exactly: INSUFFICIENT CONTEXT.")


def _retry(fn, *args, attempts: int = 5, base: float = 1.5, **kw):
    """Call ``fn`` with exponential backoff on transient API errors (429 / 5xx).
    Re-raises only after the last attempt; callers degrade a final failure to a
    conservative outcome rather than crashing the whole run."""
    import time
    import urllib.error

    for i in range(attempts):
        try:
            return fn(*args, **kw)
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            if exc.code in (429, 500, 502, 503, 529) and i < attempts - 1:
                time.sleep(base * (2 ** i))
                continue
            raise
        except Exception:  # noqa: BLE001 — network blips: retry, then give up
            if i < attempts - 1:
                time.sleep(base * (2 ** i))
                continue
            raise


def _judge_label(judge_complete, question: str, key: str, answer: str) -> str:
    sysmsg = ("You are a strict grader for a context-retrieval test. Given a QUESTION, the "
              "REQUIRED FACT a correct answer must convey, and a candidate ANSWER, reply with "
              "exactly one word: SOLVED if the ANSWER conveys the required fact, or UNSOLVED "
              "otherwise (including if it says the context is insufficient).")
    user = f"QUESTION: {question}\nREQUIRED FACT: {key}\nANSWER: {answer}\n\nVerdict (SOLVED or UNSOLVED):"
    out = (judge_complete(sysmsg, user) or "").strip().upper()
    return "solved" if out.startswith("SOLVED") or ("SOLVED" in out and "UNSOLVED" not in out) else "unsolved"


def load_powered_battery(split: str = "public") -> "list[dict] | None":
    """Load the committed powered battery (tools/build_focus_battery.py), filtered to a
    split. Returns None if it has not been built yet (callers fall back to the inline
    fixture)."""
    p = RESULTS_DIR / "focus-frontier-battery.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    tasks = [t for t in data.get("tasks", []) if t.get("split") == split] if split else data.get("tasks", [])
    return tasks or None


def run_real(subject, subject_id: str, judges: list, *, seeds: int = 1, cache=None,
             battery: "list[dict] | None" = None, max_workers: int = 1) -> dict:
    """subject: complete(system,user)->str. judges: list of (family_id, complete). Returns a report.

    A task is SOLVED only on judge CONSENSUS (all judges say solved) — conservative /
    fail-closed. Pairwise inter-judge kappa is reported across all families. NO
    promotion happens here; the gate decides, and an underpowered battery (MDE > 0.10)
    stays NO-GO by the pre-registered power floor. ``max_workers`` > 1 fans the model
    calls out concurrently so a powered (N >= 400) run is feasible.
    """
    from tools.eval_stats import cohen_kappa, gwet_ac1

    tasks = battery if battery is not None else (load_powered_battery("public") or build_battery())
    families = [fid for fid, _ in judges]
    judge_families = len(set(families))
    n_distinct_tasks = len(tasks)

    subj_sys = ("Answer the user's question using ONLY the provided context. Be concise. "
                "If the context lacks the answer, reply exactly: INSUFFICIENT CONTEXT.")

    def _work(unit):
        seed, t, arm = unit
        q = _task_question(t)
        key = next((s["text"] for s in t["segments"] if s.get("key")), "")
        packed = _pack(arm, t)
        errored = False
        try:
            answer = _retry(subject, subj_sys, f"CONTEXT:\n{packed['text']}\n\n{q}")
        except Exception:  # noqa: BLE001 — a persistent subject failure -> conservative "no answer"
            answer, errored = "INSUFFICIENT CONTEXT", True
        votes = []
        for _fid, jfn in judges:
            try:
                votes.append(_retry(_judge_label, jfn, q, key, answer))
            except Exception:  # noqa: BLE001 — a persistent judge failure -> conservative "unsolved"
                votes.append("unsolved")
                errored = True
        return (seed, t["id"], arm, {"id": f"{t['id']}#s{seed}", "tokens": packed["tokens"],
                                     "solved": all(v == "solved" for v in votes),
                                     "keptSafety": packed["keptSafety"], "goalShift": t["goalShift"],
                                     "votes": votes, "errored": errored})

    units = [(seed, t, arm) for seed in range(seeds) for t in tasks for arm in ARMS]
    results: dict = {}
    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for seed, tid, arm, row in ex.map(_work, units):
                results[(seed, tid, arm)] = row
    else:
        for u in units:
            seed, tid, arm, row = _work(u)
            results[(seed, tid, arm)] = row

    # Aggregate in deterministic order (independent of thread completion order).
    arm_rows: dict[str, list] = {a: [] for a in ARMS}
    judge_label_lists: list[list[str]] = [[] for _ in judges]
    for seed in range(seeds):
        for t in tasks:
            for arm in ARMS:
                row = results[(seed, t["id"], arm)]
                arm_rows[arm].append(row)
                if arm == "prosoche-anchored":
                    for ji, v in enumerate(row["votes"]):
                        judge_label_lists[ji].append(v)

    def _arm_summary(arm):
        rows = arm_rows[arm]
        n = len(rows); ns = sum(1 for r in rows if r["solved"]); tot = sum(r["tokens"] for r in rows)
        return {"arm": arm, "n": n, "solvedRate": round(ns / n, 4) if n else 0.0,
                "tokensPerSolved": round(tot / ns, 2) if ns else None, "totalTokens": tot,
                "safetyPrunedCount": sum(1 for r in rows if not r["keptSafety"]),
                "goalShiftSolvedRate": _subset_solved_rate(rows, True), "_rows": rows}

    arms = {a: _arm_summary(a) for a in ARMS}
    anchored, pp = arms["prosoche-anchored"], arms["priority-packed"]
    primary = _paired_token_delta(anchored, pp, seed=0)
    # Pairwise inter-judge agreement across ALL family pairs; the gate uses the MIN
    # pairwise kappa (most conservative) so a single disagreeing family cannot be
    # hidden behind an agreeing pair.
    import itertools

    pair_kappas: dict[str, "float | None"] = {}
    for (i, fi), (j, fj) in itertools.combinations(enumerate(families), 2):
        pair_kappas[f"{fi}|{fj}"] = cohen_kappa(judge_label_lists[i], judge_label_lists[j])
    present = [k for k in pair_kappas.values() if k is not None]
    kappa = min(present) if present else None
    ac1 = gwet_ac1(judge_label_lists[0], judge_label_lists[1]) if len(judge_label_lists) >= 2 else None
    n_agreement = len(judge_label_lists[0]) if judge_label_lists else 0
    success_held = anchored["solvedRate"] >= pp["solvedRate"] - 0.02
    antifix_held = (anchored["goalShiftSolvedRate"] or 0.0) >= (pp["goalShiftSolvedRate"] or 0.0)
    safety_pruned = anchored["safetyPrunedCount"]
    mde = round(mde_at_n(n_distinct_tasks, p0=0.5), 4)
    using_powered = bool(tasks) and all("split" in t for t in tasks)
    split_used = tasks[0].get("split") if using_powered else "inline"

    verdict = gate_verdict(baseline_is_real=True, judge_families=judge_families, delta=primary,
                           success_guardrail_held=success_held, antifixation_held=antifix_held,
                           safety_pruned=safety_pruned)
    failures = list(verdict["criticalFailures"])
    # Pre-registered power + replication + decontam floors — surfaced honestly, never tuned.
    if mde > MDE_TARGET:
        failures.append(f"underpowered: MDE@N={mde} > {MDE_TARGET} (N={n_distinct_tasks} distinct tasks)")
    if seeds < MIN_SEEDS:
        failures.append(f"insufficient_seeds: {seeds} < {MIN_SEEDS} seeds")
    kappa_ok = kappa is not None and kappa >= 0.40
    if not kappa_ok:
        failures.append(f"judge_agreement: min pairwise Cohen kappa={kappa} does not clear >= 0.40 (or undefined on a degenerate label set)")
    if not using_powered:
        failures.append("no_decontam: the inline fixture is not the decontaminated, powered battery (build_focus_battery.py)")
    elif split_used != "private":
        failures.append(f"sealed_split_not_scored: scored on the '{split_used}' split; the held-out PRIVATE split has not been scored for the final claim")
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {
        "mode": "real",
        "status": "powered" if (using_powered and mde <= MDE_TARGET) else "pilot",
        "subject": subject_id,
        "judges": families,
        "judgeFamilies": judge_families,
        "seeds": seeds,
        "nDistinctTasks": n_distinct_tasks,
        "split": split_used,
        "mdeAtN": mde,
        "apiErrors": sum(1 for a in ARMS for r in arm_rows[a] if r.get("errored")),
        "arms": {a: strip(arms[a]) for a in ARMS},
        "primaryDelta": primary,
        "interJudge": {"minPairwiseCohenKappa": kappa, "pairwiseCohenKappa": pair_kappas,
                       "gwetAC1": ac1, "n": n_agreement,
                       "note": "pairwise across all judge families on the anchored arm; gate uses the min"},
        "guardrails": {"successNonInferior": success_held, "antiFixationHeld": antifix_held,
                       "safetyPruned": safety_pruned},
        "verdict": "NO-GO" if failures else "GO",
        "go": not failures,
        "canClaimAGI": False,
        "criticalFailures": failures,
        "boundary": (
            "REAL pilot: a live subject model + >= 2 independent judge families, but on a small, "
            "non-decontaminated author battery -> UNDERPOWERED. Candidate signal only; the gate "
            "stays NO-GO until a decontaminated, powered (N >= 100), >= 3-seed run clears every "
            "guardrail with a delta CI excluding 0. canClaimAGI:false."
        ),
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


def _env_judges():
    """Build the judge panel from env. Model ids are read from the environment so no
    specific proprietary model string is ever hardcoded in (or committed via) source;
    the committed artifact records the FAMILY label (e.g. 'llmhub:anthropic'), not a
    version. Two distinct families are required for the >= 2-family contract.

    Env: FOCUS_JUDGE_A_MODEL / FOCUS_JUDGE_A_FAMILY (frontier judge),
         FOCUS_JUDGE_B_MODEL (default 'qwen-flash') / FOCUS_JUDGE_B_FAMILY (default 'qwen').
    """
    import os

    from agent.llmhub_llm import make_complete as hub_complete

    ja_model = os.environ.get("FOCUS_JUDGE_A_MODEL")
    ja_family = os.environ.get("FOCUS_JUDGE_A_FAMILY", "anthropic")
    jb_model = os.environ.get("FOCUS_JUDGE_B_MODEL", "qwen-flash")
    jb_family = os.environ.get("FOCUS_JUDGE_B_FAMILY", "qwen")
    if not ja_model:
        return None
    judges = [(f"llmhub:{ja_family}", hub_complete(model=ja_model, max_tokens=8))]
    if jb_model and jb_model.lower() != "none":  # 'none' skips B (e.g. to use A + an OpenRouter family)
        judges.append((f"llmhub:{jb_family}", hub_complete(model=jb_model, max_tokens=8)))
    # Optional third, INDEPENDENT family via OpenRouter (e.g. a non-US Mistral model).
    # Model id read from env so no specific id is hardcoded/committed; the artifact
    # records only the family label.
    jc_model = os.environ.get("FOCUS_JUDGE_C_MODEL")
    if jc_model and os.environ.get("OPENROUTER_API_KEY"):
        from agent.openrouter_client import make_complete as or_complete

        jc_family = os.environ.get("FOCUS_JUDGE_C_FAMILY", "mistral")
        judges.append((f"openrouter:{jc_family}", or_complete(model=jc_model, max_tokens=8)))
    return judges


def _run_real_from_env(args) -> int:
    """Build the real panel from env keys/models and run the pilot. Keys AND specific
    model ids are read from the environment ONLY; they are never written to disk,
    printed, or hardcoded in source."""
    import os

    from agent.deepseek_llm import make_complete as ds_complete

    if not (os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("LLMHUB_API_KEY")):
        print("::error:: need DEEPSEEK_API_KEY (subject) and LLMHUB_API_KEY (judges) in the environment.",
              file=sys.stderr)
        return 2
    judges = _env_judges()
    if judges is None:
        print("::error:: set FOCUS_JUDGE_A_MODEL (the frontier judge model id) in the environment.",
              file=sys.stderr)
        return 2
    subject_model = os.environ.get("FOCUS_SUBJECT_MODEL", "deepseek-chat")
    subject_temp = float(os.environ.get("FOCUS_SUBJECT_TEMP", "0.0"))  # > 0 for genuine seed variance
    subject_id = f"deepseek:{subject_model}"
    subject = ds_complete(model=subject_model, temperature=subject_temp, max_tokens=160)
    battery = load_powered_battery(args.split) if args.split else None
    if args.limit and battery:
        battery = battery[: args.limit]
    report = run_real(subject, subject_id, judges, seeds=args.seeds,
                      battery=battery, max_workers=args.max_workers)
    report["batterySource"] = (f"powered:{args.split}" if battery is not None else "inline-fixture")
    out = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}  verdict={report['verdict']}", file=sys.stderr)
    else:
        print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="run the 3 arms with the survival proxy and print metrics")
    ap.add_argument("--write", action="store_true", help="write the committed PENDING artifact")
    ap.add_argument("--check", action="store_true", help="verify the committed artifact matches a fresh build")
    ap.add_argument("--model", default="", help="real subject model spec — refused offline (stays PENDING)")
    ap.add_argument("--real", action="store_true",
                    help="run the REAL panel (subject=DeepSeek, judges=llmhub claude+qwen) from env keys")
    ap.add_argument("--seeds", type=int, default=1, help="(--real) number of passes")
    ap.add_argument("--split", default="", help="(--real) powered-battery split: public|private (empty = inline fixture)")
    ap.add_argument("--limit", type=int, default=0, help="(--real) cap the number of tasks (cheap sampling)")
    ap.add_argument("--max-workers", type=int, default=1, help="(--real) concurrent model calls")
    ap.add_argument("--out", default="", help="(--real) write the report JSON here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.real:
        return _run_real_from_env(args)

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
