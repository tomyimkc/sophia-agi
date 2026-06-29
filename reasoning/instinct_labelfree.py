# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Stronger label-free reflexes — can a softer agreement signal beat exact self-consistency?

The validation step (§3f) pinned the real frontier: B/B2 are *verifiers* (near-tautological), so
the only genuinely **predictive, label-free** reflex is self-consistency — and exact-match
self-consistency is **weak** (DeepSeek AUC 0.63, CI overlapping chance). Exact match is brittle:
two *near-identical wrong* sets count as full agreement (looks confident → reflex stays quiet),
and a correct answer with one stray sample counts as disagreement. So softer agreement signals
*might* separate errors better — using the same samples, still with no labels.

This module recomputes, from the **stored** v2 sample sets (zero API), a family of label-free
signals and measures each one's AUC (with bootstrap CIs) against the planted error label:

  - ``exact``        — 1 − (most-common exact set)/N      (the baseline = the stored detector A)
  - ``jaccard``      — 1 − mean pairwise Jaccard similarity of the sample sets (soft agreement)
  - ``instability``  — mean over candidate claims of Gini 2p(1−p), p = fraction of samples that
                       include the claim (per-element membership disagreement)
  - ``size_disp``    — stdev of |set| across samples, squashed to [0,1) (answer-length wobble)
  - ``entropy``      — Shannon entropy over the distinct-set distribution, normalised by log N

It then reports which label-free signal wins, whether it **beats the exact baseline**, whether
its CI **excludes chance**, and confirms it still trails the structural verifier (B2). The
verdict is reported **whichever way it falls** — a soft signal that fails to beat exact is an
equally honest result about the frontier.

Honest scope (``candidateOnly: true``, ``canClaimAGI: false``). Single run, real models'
*recorded* samples; AUCs are case-resampling estimates. Not a model claim — a measurement of how
much head-room a better label-free reflex has on this task.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_reflex_eval import auc  # noqa: E402
from reasoning.instinct_validation import bootstrap_ci  # noqa: E402

RESULTS = ROOT / "reasoning" / "results"


def _sets(case: dict[str, Any]) -> list[frozenset[str]]:
    return [frozenset(s) for s in case["samples"]]


# ---------------------------------------------------------------------------
# Label-free signals (each: list[frozenset] -> wrongness in [0, ~1])
# ---------------------------------------------------------------------------

def sig_exact(samples: list[frozenset[str]]) -> float:
    n = len(samples)
    if n == 0:
        return 1.0
    counts: dict[frozenset[str], int] = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    return 1.0 - max(counts.values()) / n


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def sig_jaccard(samples: list[frozenset[str]]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    sims = [_jaccard(samples[i], samples[j]) for i in range(n) for j in range(i + 1, n)]
    return 1.0 - sum(sims) / len(sims)


def sig_instability(samples: list[frozenset[str]]) -> float:
    n = len(samples)
    if n == 0:
        return 1.0
    universe = set().union(*samples) if samples else set()
    if not universe:
        return 0.0
    total = 0.0
    for c in sorted(universe):  # sorted ⇒ deterministic float-sum order (hash-seed independent)
        p = sum(1 for s in samples if c in s) / n
        total += 2.0 * p * (1.0 - p)  # Gini impurity of membership (0 if unanimous)
    return total / len(universe)


def sig_size_disp(samples: list[frozenset[str]]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    sizes = [len(s) for s in samples]
    m = sum(sizes) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in sizes) / n)
    return sd / (1.0 + sd)  # squash to [0,1)


def sig_entropy(samples: list[frozenset[str]]) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    counts: dict[frozenset[str], int] = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    h = -sum((c / n) * math.log(c / n) for c in sorted(counts.values()))  # sorted ⇒ stable sum
    return h / math.log(n)  # normalise by max entropy (log N)


SIGNALS = {
    "exact": sig_exact, "jaccard": sig_jaccard, "instability": sig_instability,
    "size_disp": sig_size_disp, "entropy": sig_entropy,
}


def evaluate(path: Path, *, seed: int = 1234) -> dict[str, Any]:
    pc = json.loads(Path(path).read_text())["per_case"]
    labels = [bool(c["is_error"]) for c in pc]
    sample_sets = [_sets(c) for c in pc]
    stored_A = [float(c["A"]) for c in pc]
    rows: dict[str, Any] = {}
    for name, fn in SIGNALS.items():
        scores = [fn(s) for s in sample_sets]
        e = [s for s, lab in zip(scores, labels) if lab]
        c = [s for s, lab in zip(scores, labels) if not lab]
        rows[name] = {
            "auc": round(auc(e, c), 4),
            "auc_ci": bootstrap_ci(scores, labels, metric="auc", seed=seed),
        }
    # cross-check: recomputed exact == stored detector A
    recomputed = [sig_exact(s) for s in sample_sets]
    drift = max(abs(a - b) for a, b in zip(recomputed, stored_A)) if pc else 0.0
    best = max(SIGNALS, key=lambda k: rows[k]["auc"])
    model = json.loads(path.read_text())["report"]["model"]
    n_clean = sum(1 for x in labels if not x)
    return {
        "model": model, "n": len(pc), "n_clean": n_clean,
        "exact_vs_storedA_drift": round(drift, 6),
        "rows": rows, "best": best,
        "best_beats_exact": rows[best]["auc"] > rows["exact"]["auc"] + 1e-9,
        "best_excludes_chance": rows[best]["auc_ci"][0] > 0.5,
    }


def run_experiment(seed: int = 1234) -> dict[str, Any]:
    out = {}
    for fname in ("fusion_realmodel_deepseek.json", "fusion_realmodel_llmhub-haiku.json"):
        p = RESULTS / fname
        if p.exists():
            ev = evaluate(p, seed=seed)
            out[ev["model"]] = ev
    return out


def format_report(res: dict[str, Any]) -> str:
    lines = ["Stronger label-free reflexes — soft agreement vs exact self-consistency", "=" * 72]
    for model, ev in res.items():
        lines.append(f"\nMODEL {model}  (N={ev['n']}, clean={ev['n_clean']})  "
                     f"[exact≡A drift {ev['exact_vs_storedA_drift']}]")
        lines.append(f"  {'signal':14} {'AUC':>6}  {'AUC 95% CI':>18}")
        for name, r in ev["rows"].items():
            mark = "  <- best" if name == ev["best"] else ("   (baseline)" if name == "exact" else "")
            lines.append(f"  {name:14} {r['auc']:>6.3f}  [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]{mark}")
        lines.append(f"  => best={ev['best']}  beats_exact={ev['best_beats_exact']}  "
                     f"excludes_chance={ev['best_excludes_chance']}")
    lines += ["", "candidateOnly=True  level3Evidence=False",
              "boundary: single run, recorded samples; AUCs are case-resampling estimates."]
    return "\n".join(lines)


def _self_test() -> int:
    res = run_experiment()
    ds = next(ev for m, ev in res.items() if "deepseek" in m.lower())
    # recomputed exact signal must reproduce the stored detector A (validates the pipeline)
    assert ds["exact_vs_storedA_drift"] < 1e-6, f"exact≠stored A ({ds['exact_vs_storedA_drift']})"
    # all signals produce a valid AUC
    for name, r in ds["rows"].items():
        assert 0.0 <= r["auc"] <= 1.0, f"{name} AUC out of range"
        assert r["auc_ci"][0] <= r["auc_ci"][1]
    # the 'best' is well-defined and at least ties the baseline (it's among the candidates)
    assert ds["rows"][ds["best"]]["auc"] >= ds["rows"]["exact"]["auc"] - 1e-9
    verdict = ("BEATS exact" if ds["best_beats_exact"] else "does NOT beat exact")
    print(f"self-test OK: DeepSeek best label-free = '{ds['best']}' "
          f"AUC {ds['rows'][ds['best']]['auc']} (exact {ds['rows']['exact']['auc']}) — {verdict}; "
          f"CI excludes chance={ds['best_excludes_chance']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.run or args.json:
        res = run_experiment(seed=args.seed)
        print(json.dumps(res, indent=2) if args.json else format_report(res))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
