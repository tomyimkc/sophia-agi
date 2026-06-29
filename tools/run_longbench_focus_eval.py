#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Naturalistic Focus-Efficiency-Frontier eval on real multi-doc QA (LongBench slice).

The honest contrast to the synthetic battery. Here the gold-bearing passage sits at a
NATURAL, varying position among distractors, so recency baselines sometimes win — a pass
would actually generalise. ``solved`` is graded against the dataset's GOLD answer two ways
(deterministic EM/F1 AND a >=2-family judge panel, cross-checked), never a synthetic proxy.

Arms (real packers, reused from run_focus_frontier_eval._pack):
  recency-chop / priority-packed / prosoche-anchored (relevance to the QUESTION), PLUS a
  CLOSED-BOOK control (no context) that measures how much the subject answers from parametric
  memory — if closed-book is high the dataset is memorised and the comparison is confounded.

The gate (reused) keeps the Robbins anytime-valid CS requirement and adds a parametric-leakage
guardrail. NO promotion; the gate decides. canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.context_manager import Segment, estimate_tokens  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, cohen_kappa, confidence_sequence_mean, mde_at_n  # noqa: E402
from tools.run_focus_frontier_eval import (  # noqa: E402 — reuse the packers + gate + retry
    UNSOLVED_PENALTY,
    _pack,
    _retry,
    gate_verdict,
)

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "prosoche"
SLICE_PATH = RESULTS_DIR / "focus-longbench-slice.json"
ARMS = ("recency-chop", "priority-packed", "prosoche-anchored")
MDE_TARGET = 0.10
MIN_SEEDS = 3
F1_SOLVED = 0.5            # deterministic cross-check threshold
LEAKAGE_MARGIN = 0.15     # closed-book within this of anchored -> parametric-leakage confound


# --------------------------------------------------------------------------- #
# Deterministic EM / F1 vs gold (SQuAD/HotpotQA normalisation).
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em(pred: str, golds: list[str]) -> int:
    p = _norm(pred)
    return int(any(_norm(g) == p for g in golds) or any(_norm(g) and _norm(g) in p for g in golds))


def _f1(pred: str, golds: list[str]) -> float:
    pt = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not pt or not gt:
            continue
        common = {}
        for w in pt:
            common[w] = min(pt.count(w), gt.count(w))
        ov = sum(common.values())
        if ov == 0:
            continue
        prec, rec = ov / len(pt), ov / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return round(best, 4)


def _task_fft(t: dict) -> dict:
    """Adapt a LongBench task to the run_focus_frontier_eval task schema for _pack()."""
    return {"goal": t["question"], "inScopeEntities": [], "budgetTokens": t["budgetTokens"],
            "segments": [{"text": p} for p in t["passages"]]}


def _judge_match(judge_complete, question: str, golds: list[str], answer: str) -> str:
    sysmsg = ("You grade a QA answer. Given a QUESTION, the GOLD answer(s), and a candidate "
              "ANSWER, reply with exactly one word: CORRECT if the ANSWER matches any GOLD answer "
              "in meaning, else WRONG (including 'I don't know' / insufficient context).")
    user = f"QUESTION: {question}\nGOLD: {' | '.join(golds)}\nANSWER: {answer}\n\nVerdict (CORRECT or WRONG):"
    out = (judge_complete(sysmsg, user) or "").strip().upper()
    return "solved" if out.startswith("CORRECT") or ("CORRECT" in out and "WRONG" not in out) else "unsolved"


def run(subject, subject_id: str, judges: list, tasks: list[dict], *, seeds: int = 1,
        max_workers: int = 1, budget: int | None = None) -> dict:
    if budget:
        tasks = [{**t, "budgetTokens": budget} for t in tasks]
    families = [fid for fid, _ in judges]
    subj_sys = ("Answer the question using ONLY the provided context, as briefly as possible (a few "
                "words). If the context lacks the answer, reply exactly: INSUFFICIENT CONTEXT.")
    cb_sys = ("Answer the question as briefly as possible (a few words) from your own knowledge. "
              "If you do not know, reply exactly: I DON'T KNOW.")
    all_arms = (*ARMS, "closed-book")

    def _ask_subject(arm, t):
        if arm == "closed-book":
            return cb_sys, t["question"], 0
        packed = _pack(arm, _task_fft(t))
        return subj_sys, f"CONTEXT:\n{packed['text']}\n\nQUESTION: {t['question']}", packed["tokens"]

    def _work(unit):
        seed, t, arm = unit
        sysmsg, user, tokens = _ask_subject(arm, t)
        try:
            answer = _retry(subject, sysmsg, user)
        except Exception:  # noqa: BLE001
            answer = "INSUFFICIENT CONTEXT"
        golds = t["goldAnswers"]
        votes = []
        for _fid, jfn in judges:
            try:
                votes.append(_retry(_judge_match, jfn, t["question"], golds, answer))
            except Exception:  # noqa: BLE001
                votes.append("unsolved")
        return (seed, t["id"], arm, {"id": f"{t['id']}#s{seed}", "tokens": tokens,
                                     "judgeSolved": all(v == "solved" for v in votes),
                                     "f1": _f1(answer, golds), "em": _em(answer, golds),
                                     "votes": votes})

    units = [(seed, t, arm) for seed in range(seeds) for t in tasks for arm in all_arms]
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

    arm_rows: dict[str, list] = {a: [] for a in all_arms}
    jl: list[list[str]] = [[] for _ in judges]
    for seed in range(seeds):
        for t in tasks:
            for arm in all_arms:
                r = results[(seed, t["id"], arm)]
                arm_rows[arm].append(r)
                if arm == "prosoche-anchored":
                    for ji, v in enumerate(r["votes"]):
                        jl[ji].append(v)

    def _summ(arm):
        rows = arm_rows[arm]
        n = len(rows)
        ns = sum(1 for r in rows if r["judgeSolved"])
        return {"arm": arm, "n": n, "solvedRate": round(ns / n, 4) if n else 0.0,
                "f1Mean": round(sum(r["f1"] for r in rows) / n, 4) if n else 0.0,
                "emRate": round(sum(r["em"] for r in rows) / n, 4) if n else 0.0,
                "f1SolvedRate": round(sum(1 for r in rows if r["f1"] >= F1_SOLVED) / n, 4) if n else 0.0,
                "tokensPerSolved": round(sum(r["tokens"] for r in rows) / ns, 2) if ns else None,
                "_rows": rows}

    arms = {a: _summ(a) for a in all_arms}
    anchored, pp, cb = arms["prosoche-anchored"], arms["priority-packed"], arms["closed-book"]

    # Paired efficiency-cost delta (anchored vs priority-packed), tokens + penalty if unsolved.
    a = {r["id"]: r for r in anchored["_rows"]}
    b = {r["id"]: r for r in pp["_rows"]}
    ids = [i for i in a if i in b]
    diffs = [(a[i]["tokens"] + (0 if a[i]["judgeSolved"] else UNSOLVED_PENALTY))
             - (b[i]["tokens"] + (0 if b[i]["judgeSolved"] else UNSOLVED_PENALTY)) for i in ids]
    n = len(diffs)
    delta = {"pairedTasks": n, "deltaEfficiencyCost": round(sum(diffs) / n, 2) if n else 0.0,
             "deltaTokensCI95": bootstrap_ci_paired(diffs, seed=0) if n else [None, None],
             "deltaTokensCS95": confidence_sequence_mean(diffs) if n else [None, None]}

    kappa = cohen_kappa(jl[0], jl[1]) if len(jl) >= 2 else None
    # Judge<->deterministic agreement (the 'both, cross-checked' validity check).
    cross = [(1 if r["judgeSolved"] else 0) == (1 if r["f1"] >= F1_SOLVED else 0)
             for r in anchored["_rows"]]
    judge_f1_agreement = round(sum(cross) / len(cross), 4) if cross else None

    n_tasks = len(tasks)
    mde = round(mde_at_n(n_tasks, p0=0.5), 4)
    success_held = anchored["solvedRate"] >= pp["solvedRate"] - 0.02
    leakage = cb["solvedRate"] >= anchored["solvedRate"] - LEAKAGE_MARGIN

    verdict = gate_verdict(baseline_is_real=True, judge_families=len(set(families)), delta=delta,
                           success_guardrail_held=success_held, antifixation_held=True, safety_pruned=0)
    failures = list(verdict["criticalFailures"])
    if mde > MDE_TARGET:
        failures.append(f"underpowered: MDE@N={mde} > {MDE_TARGET} (N={n_tasks})")
    if seeds < MIN_SEEDS:
        failures.append(f"insufficient_seeds: {seeds} < {MIN_SEEDS}")
    if not (kappa is not None and kappa >= 0.40):
        failures.append(f"judge_agreement: kappa={kappa} < 0.40 (or degenerate)")
    if leakage:
        failures.append(f"parametric_leakage: closed-book solved {cb['solvedRate']} is within "
                        f"{LEAKAGE_MARGIN} of anchored {anchored['solvedRate']} — the result is confounded by memorisation")
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {
        "mode": "real-naturalistic", "dataset": "LongBench v1 hotpotqa/2wikimqa (decontaminated slice)",
        "subject": subject_id, "judges": families, "judgeFamilies": len(set(families)),
        "seeds": seeds, "nTasks": n_tasks, "mdeAtN": mde,
        "arms": {a: strip(arms[a]) for a in all_arms},
        "closedBookControl": {"solvedRate": cb["solvedRate"], "f1Mean": cb["f1Mean"],
                              "note": "high closed-book => parametric leakage (HotpotQA may be memorised)"},
        "primaryDelta": delta,
        "interJudge": {"cohenKappa": kappa, "n": len(jl[0]) if jl else 0},
        "crossCheck": {"judgeVsF1Agreement": judge_f1_agreement, "f1SolvedThreshold": F1_SOLVED},
        "guardrails": {"successNonInferior": success_held, "parametricLeakage": leakage},
        "verdict": "NO-GO" if failures else "GO", "go": not failures, "canClaimAGI": False,
        "criticalFailures": failures,
        "boundary": ("Naturalistic real-QA eval: gold passage at a varying position, graded vs the "
                     "dataset's gold answer (EM/F1 + >=2 judges). A GO here would NOT be construct-bounded "
                     "the way the synthetic battery is — but is still corpus-bound and NOT AGI. canClaimAGI:false."),
    }


def _run_from_env(args) -> int:
    import os

    from agent.deepseek_llm import make_complete as ds_complete
    from tools.run_focus_frontier_eval import _env_judges

    if not (os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("LLMHUB_API_KEY")):
        print("::error:: need DEEPSEEK_API_KEY + LLMHUB_API_KEY in env.", file=sys.stderr)
        return 2
    judges = _env_judges()
    if judges is None:
        print("::error:: set FOCUS_JUDGE_A_MODEL.", file=sys.stderr)
        return 2
    sl = json.loads(SLICE_PATH.read_text(encoding="utf-8"))
    tasks = [t for t in sl["tasks"] if (not args.split or t["split"] == args.split)]
    if args.limit:
        tasks = tasks[: args.limit]
    subject_model = os.environ.get("FOCUS_SUBJECT_MODEL", "deepseek-chat")
    subject = ds_complete(model=subject_model, temperature=float(os.environ.get("FOCUS_SUBJECT_TEMP", "0.0")), max_tokens=64)
    report = run(subject, f"deepseek:{subject_model}", judges, tasks, seeds=args.seeds,
                 max_workers=args.max_workers, budget=args.budget or None)
    report["packBudgetTokens"] = args.budget or tasks[0]["budgetTokens"] if tasks else None
    out = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
        print(f"wrote {args.out}  verdict={report['verdict']}", file=sys.stderr)
    else:
        print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--split", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--budget", type=int, default=0, help="override the per-task pack budget (tokens)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.real:
        return _run_from_env(args)
    print("Use --real (with env keys) to run the naturalistic eval.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
