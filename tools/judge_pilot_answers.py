#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent multi-judge semantic re-score of the M3-pilot base-vs-adapter answers.

The pilot's headline metrics are DETERMINISTIC MARKER-BASED; a large qualification/route
delta could be the adapter learning source-discipline FORMAT rather than substance. This
script tests that with ≥2 INDEPENDENT LLM judge families (distinct from the subject
gemma-3-4b AND from agent/gate.py): for each held-out case it shows the base and adapter
answers in RANDOMIZED A/B order and asks which better practices source discipline given the
gold reference. Reports adapter-vs-base win rates + inter-judge Cohen's κ.

Judges share NO code with the gate; this is the RESULTS.md anti-circularity bar.

    OPENROUTER_API_KEY=... python3 tools/judge_pilot_answers.py \
      --answers agi-proof/benchmark-results/wisdom-market/M3-pilot-answers-seed1.json \
      --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct \
      --out agi-proof/benchmark-results/wisdom-market/M3-pilot-judge.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.model import default_client  # noqa: E402

# Source/provenance families the pilot's claims rest on (skip retention/tool axes).
SOURCE_FAMILIES = {
    "source_discipline", "false_attribution", "provenance_qa", "authorship_boundary",
    "tradition_boundary", "contested_religion", "fake_citation",
}

JUDGE_SYSTEM = (
    "You are a strict scholar of textual provenance and source discipline. You judge which of two "
    "answers better practices SOURCE DISCIPLINE: accurate attribution, appropriate qualification of "
    "uncertain/legendary attributions, keeping distinct traditions separate, refusing to fabricate "
    "citations, and saying what cannot be verified — judged for SUBSTANCE, not style or verbosity. "
    "Reply with a single token: A, B, or TIE."
)

# Pre-registered variant (--forced-choice): identical rubric, but TIE is disallowed so
# BOTH judges face the same forced binary decision. This removes the tie-rate asymmetry
# (one judge ties often, the other never) that depresses chance-corrected κ even when
# the judges agree directionally. The shared no-TIE policy is the single methodological
# change; results are reported once, whatever they are (no goalpost-moving).
JUDGE_SYSTEM_FORCED = JUDGE_SYSTEM.rsplit(".", 2)[0] + (
    ". You MUST choose the better answer — TIE is NOT allowed. Reply with a single token: A or B."
)


def _judge_prompt(case: dict, ans_a: str, ans_b: str, *, forced: bool = False) -> str:
    forb = "; ".join(case.get("forbidden_assertions") or []) or "(none listed)"
    feats = "; ".join(case.get("acceptable_answer_features") or []) or "(none listed)"
    tail = ("Answer exactly one token: A or B (you MUST pick one; TIE is not allowed)."
            if forced else "Answer exactly one token: A, B, or TIE.")
    return (
        f"QUESTION:\n{case['prompt']}\n\n"
        f"REFERENCE — claims that would be WRONG to assert: {forb}\n"
        f"REFERENCE — features a good answer shows: {feats}\n\n"
        f"ANSWER A:\n{ans_a[:1400]}\n\n"
        f"ANSWER B:\n{ans_b[:1400]}\n\n"
        "Which answer better practices source discipline (substance, per the reference)? "
        + tail
    )


def _parse(verdict: str, *, forced: bool = False) -> "str | None":
    t = (verdict or "").strip().upper()
    if forced:
        m = re.search(r"\b(A|B)\b", t)
        return m.group(1) if m else None          # no clear pick -> dropped, never a silent TIE
    m = re.search(r"\b(A|B|TIE)\b", t)
    return m.group(1) if m else "TIE"


def _ab_order(case_id: str) -> bool:
    """Deterministic per-case A/B assignment (reproducible, no global RNG): True => A=adapter."""
    return (int(__import__("hashlib").sha1((case_id or "").encode()).hexdigest(), 16) % 2) == 0


def judge_one(client, case: dict, *, forced: bool = False) -> "str | None":
    a_is_adapter = _ab_order(case.get("id", ""))
    ans_a = case["adapter_answer"] if a_is_adapter else case["base_answer"]
    ans_b = case["base_answer"] if a_is_adapter else case["adapter_answer"]
    try:
        res = client.generate(JUDGE_SYSTEM_FORCED if forced else JUDGE_SYSTEM,
                              _judge_prompt(case, ans_a, ans_b, forced=forced))
        v = _parse(getattr(res, "text", ""), forced=forced)
    except Exception:
        return None
    if v is None:
        return None
    if v == "TIE":
        return "tie"
    picked_a = (v == "A")
    return "adapter" if (picked_a == a_is_adapter) else "base"


def _kappa(labels_x: list, labels_y: list) -> "float | None":
    pairs = [(x, y) for x, y in zip(labels_x, labels_y) if x and y]
    if len(pairs) < 2:
        return None
    cats = ["adapter", "base", "tie"]
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    px = {c: sum(1 for x, _ in pairs if x == c) / n for c in cats}
    py = {c: sum(1 for _, y in pairs if y == c) / n for c in cats}
    pe = sum(px[c] * py[c] for c in cats)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else None


# --------------------------------------------------------------------------- #
# Honest-statistics panel. The capability claim is a WIN-RATE (vs 0.5); κ measures
# only judge RELIABILITY and is provably deflated on same-direction skewed pairwise
# data (Feinstein & Cicchetti 1990; Warrens 2010). So we report a panel, never a
# single number: per-judge win-rate (Wilson CI + exact binomial vs 0.5 + bootstrap),
# the Byrt (1993) 2x2 decomposition (observed agreement, κ, PABAK, bias/prevalence
# indices), and a majority vote when ≥3 judges. NB: do NOT compare PABAK/AC1 against
# a κ-derived 0.40 bar — the Landis & Koch cutoffs are not transferable (Zec 2023).
# --------------------------------------------------------------------------- #

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> "list[float] | None":
    """Wilson score 95% CI for a binomial proportion (stable at small n / extreme p)."""
    if n == 0:
        return None
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def _binom_two_sided_p(wins: int, n: int, p0: float = 0.5) -> "float | None":
    """Exact two-sided binomial (sign) test that the win-rate differs from p0."""
    if n == 0:
        return None
    probs = [math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k) for k in range(n + 1)]
    obs = probs[wins]
    return round(min(1.0, sum(pr for pr in probs if pr <= obs + 1e-12)), 5)


def _bootstrap_winrate_ci(binary: "list[str]", *, B: int = 5000, seed: int = 0) -> "list[float] | None":
    """Percentile bootstrap 95% CI for adapter win-rate over binary {adapter, base}."""
    wins = [1 if v == "adapter" else 0 for v in binary]
    n = len(wins)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = sorted(sum(wins[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B) - 1], 4)]


def _winrate_stats(labels: "list[str]") -> dict:
    """Adapter win-rate over base on binary verdicts (ties/None dropped) + honest CIs."""
    binary = [v for v in labels if v in ("adapter", "base")]
    n, wins = len(binary), sum(1 for v in labels if v == "adapter")
    return {
        "nBinary": n, "adapterWins": wins, "baseWins": n - wins,
        "adapterWinrate": round(wins / n, 4) if n else None,
        "wilson95": _wilson_ci(wins, n),
        "bootstrap95": _bootstrap_winrate_ci(binary),
        "binomialTwoSidedP_vs_0_5": _binom_two_sided_p(wins, n),
        "significantVs0_5_at_0_05": (lambda p: p is not None and p < 0.05)(_binom_two_sided_p(wins, n)),
    }


def _pairwise_panel(labels_x: "list[str]", labels_y: "list[str]") -> "dict | None":
    """Byrt (1993) 2x2 decomposition on binary {adapter, base} for two judges."""
    pairs = [(x, y) for x, y in zip(labels_x, labels_y)
             if x in ("adapter", "base") and y in ("adapter", "base")]
    n = len(pairs)
    if n < 2:
        return None
    a = sum(1 for x, y in pairs if x == "adapter" and y == "adapter")  # both adapter
    d = sum(1 for x, y in pairs if x == "base" and y == "base")        # both base
    b = sum(1 for x, y in pairs if x == "adapter" and y == "base")
    c = sum(1 for x, y in pairs if x == "base" and y == "adapter")
    p0 = (a + d) / n
    px1, py1 = (a + b) / n, (a + c) / n
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    kappa = (p0 - pe) / (1 - pe) if pe != 1 else None
    return {
        "nBinaryPairs": n,
        "observedAgreement": round(p0, 4),
        "cohenKappaBinary": round(kappa, 4) if kappa is not None else None,
        "PABAK": round(2 * p0 - 1, 4),            # Byrt 1993 prevalence-adjusted bias-adjusted κ
        "biasIndex": round((b - c) / n, 4),       # between-judge marginal asymmetry (signed)
        "prevalenceIndex": round((a - d) / n, 4),  # both-adapter vs both-base imbalance (drives κ deflation)
        "table": {"bothAdapter": a, "bothBase": d, "xAdapter_yBase": b, "xBase_yAdapter": c},
    }


def _majority_labels(per_judge: dict, specs: list, n_rows: int) -> "list[str]":
    """Per-case majority verdict across ≥3 judges (ties in the vote -> 'tie')."""
    out = []
    for i in range(n_rows):
        votes = [per_judge[s][i] for s in specs if per_judge[s][i] in ("adapter", "base")]
        a, b = votes.count("adapter"), votes.count("base")
        out.append(None if not votes else "adapter" if a > b else "base" if b > a else "tie")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--judges", default="openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct")
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M3-pilot-judge.json")
    ap.add_argument("--raw-out", type=Path, default=None,
                    help="Also write per-item per-judge raw verdicts (adapter/base/tie/null) here. "
                         "This is the labelling-step artifact tools/assemble_uplift_judgments.py "
                         "consumes to build the A3 judgments.json (the summary --out drops them).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed number to stamp into --raw-out (this file judges ONE seed's answers).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--forced-choice", action="store_true",
                    help="Pre-registered variant: disallow TIE so both judges face the same forced "
                         "binary choice (removes tie-rate asymmetry that depresses Cohen's κ).")
    args = ap.parse_args()

    rows = json.loads(args.answers.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("task_family") in SOURCE_FAMILIES
            and r.get("base_answer") and r.get("adapter_answer")]
    if args.limit:
        rows = rows[:args.limit]
    judge_specs = [j.strip() for j in args.judges.split(",") if j.strip()]
    print(f"judging {len(rows)} source-family cases with {len(judge_specs)} families ...")

    per_judge = {}
    for spec in judge_specs:
        client = default_client(spec)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            verdicts = list(ex.map(lambda r: judge_one(client, r, forced=args.forced_choice), rows))
        per_judge[spec] = verdicts
        ok = [v for v in verdicts if v]
        tally = {k: ok.count(k) for k in ("adapter", "base", "tie")}
        n = len(ok)
        print(f"  [{spec}] n={n} adapter={tally['adapter']} base={tally['base']} tie={tally['tie']} "
              f"adapter_winrate={round(tally['adapter']/n,3) if n else None}")

    specs = list(per_judge)
    kappa = _kappa(per_judge[specs[0]], per_judge[specs[1]]) if len(specs) >= 2 else None

    def summary(verdicts):
        ok = [v for v in verdicts if v]
        n = len(ok) or 1
        return {"n": len([v for v in verdicts if v]),
                "adapter": ok.count("adapter"), "base": ok.count("base"), "tie": ok.count("tie"),
                "adapter_winrate": round(ok.count("adapter") / n, 4),
                "base_winrate": round(ok.count("base") / n, 4)}

    # consensus: cases where BOTH non-tie judges agree
    both = [(per_judge[specs[0]][i], per_judge[specs[1]][i]) for i in range(len(rows))] if len(specs) >= 2 else []
    consensus_adapter = sum(1 for x, y in both if x == "adapter" and y == "adapter")
    consensus_base = sum(1 for x, y in both if x == "base" and y == "base")

    # --- honest-statistics panel (win-rate is the capability claim; κ is reliability only) ---
    winrate = {s: _winrate_stats(per_judge[s]) for s in specs}
    pairwise = ({f"{specs[i]} vs {specs[j]}": _pairwise_panel(per_judge[specs[i]], per_judge[specs[j]])
                 for i in range(len(specs)) for j in range(i + 1, len(specs))}
                if len(specs) >= 2 else {})
    majority = None
    if len(specs) >= 3:
        maj = _majority_labels(per_judge, specs, len(rows))
        majority = {"perCase_winrate": _winrate_stats(maj),
                    "ties_in_vote": sum(1 for v in maj if v == "tie")}

    report = {
        "pilot_judge": "sophia-wisdom-4b-m3",
        "answers": str(args.answers.relative_to(ROOT) if args.answers.is_relative_to(ROOT) else args.answers),
        "judges": judge_specs, "nCasesJudged": len(rows),
        "protocol": "forced-choice (TIE disallowed)" if args.forced_choice else "tie-allowed",
        "perJudge": {s: summary(per_judge[s]) for s in specs},
        "interJudgeKappa": kappa,
        "consensus": {"adapter_better": consensus_adapter, "base_better": consensus_base},
        # capability claim = win-rate vs 0.5 (Wilson + exact-binomial + bootstrap):
        "winRate": winrate,
        # reliability (Byrt 1993 panel) — read PABAK/indices ALONGSIDE κ, never instead-of:
        "interJudgeReliability": pairwise,
        "majorityVote": majority,
        "statsNote": ("Win-rate vs 0.5 (Wilson/binomial/bootstrap) is the capability evidence; "
                      "κ/PABAK/bias+prevalence indices report JUDGE RELIABILITY (Feinstein & "
                      "Cicchetti 1990; Byrt 1993; Warrens 2010). Do NOT compare PABAK/AC1 to a "
                      "κ-derived 0.40 bar — Landis & Koch cutoffs are not transferable (Zec 2023)."),
        "boundary": ("Independent LLM judges (≠ subject, ≠ gate). Validates whether the adapter's "
                     "marker-based source-discipline gains hold up SEMANTICALLY. Single seed's answers; "
                     "not a market or AGI claim."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Persist per-item per-judge raw verdicts (the labelling artifact A3 needs). The summary
    # report above keeps only aggregates; assemble_uplift_judgments.py maps these pairwise
    # verdicts -> the content-uplift schema run_lora_uplift_validation.py consumes.
    if args.raw_out:
        raw = {
            "answers": report["answers"],
            "seed": args.seed,
            "judges": judge_specs,
            "protocol": "forced-choice" if args.forced_choice else "tie-allowed",
            "subjectHint": "allenai/OLMoE-1B-7B-0924-Instruct",
            "items": [
                {"id": rows[i].get("id", f"item_{i}"),
                 "task_family": rows[i].get("task_family"),
                 "verdicts": {s: per_judge[s][i] for s in specs}}
                for i in range(len(rows))
            ],
        }
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_out.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote raw verdicts -> {args.raw_out}")
    print(f"\nκ(3-cat)={kappa}  consensus adapter-better={consensus_adapter} base-better={consensus_base}")
    for s in specs:
        w = winrate[s]
        print(f"  win-rate [{s}]: {w['adapterWinrate']} (n={w['nBinary']}) "
              f"Wilson95={w['wilson95']} binomP_vs0.5={w['binomialTwoSidedP_vs_0_5']} "
              f"sig={w['significantVs0_5_at_0_05']}")
    for pair, pan in pairwise.items():
        if pan:
            print(f"  reliability [{pair}]: obsAgree={pan['observedAgreement']} "
                  f"κ={pan['cohenKappaBinary']} PABAK={pan['PABAK']} "
                  f"bias={pan['biasIndex']} prevalence={pan['prevalenceIndex']}")
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
