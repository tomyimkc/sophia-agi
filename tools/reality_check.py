#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Market reality-check: the trained adapter vs FRONTIER models on the SAME held-out cases.

M1 measured the gate's uplift on same-size *base* models. It never asked the question that actually
gates a product decision: does the narrow source-discipline advantage survive against the models
people actually use (GPT/Claude/Gemini), which may already qualify contested claims and refuse false
attributions natively? This tool answers it under the measurement contract — absolute rates, the
powered N, and an MDE-based ahead/parity/behind verdict per metric. No new eval: it reuses the SAME
benchmark + scorers as the adapter eval.

Producing the frontier baseline (needs a rotated OpenRouter key; ~$1 of spend):
    OPENROUTER_API_KEY=... python tools/run_same_size_market_baselines.py \
        --models "openrouter:openai/gpt-4o,openrouter:anthropic/claude-3.5-sonnet,openrouter:google/gemini-2.0-flash-001" \
        --conditions raw --runs 3 --out agi-proof/benchmark-results/wisdom-market/frontier_baselines.json
Then:
    python tools/reality_check.py --frontier agi-proof/benchmark-results/wisdom-market/frontier_baselines.json

Exit 0 always (this is a decision aid, not a gate). Prints the honest verdict + writes a report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_stats import mde_at_n  # noqa: E402

WM = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market"

# metric -> higher_is_better. The narrow axis the adapter was trained to move.
DIRECTION = {
    "qualification_rate_on_contested": True,
    "moral_route_accuracy": True,
    "provenance_accuracy": True,
    "citation_fidelity": True,
    "false_attribution_rate": False,
    "contested_fabrication_rate": False,
    "tradition_merge_rate": False,
    "over_abstention_rate": False,
}


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _adapter_prompt_abs(ev: dict) -> dict:
    """Reconstruct adapter(prompt) ABSOLUTE rates = base(prompt) + signed improvement delta."""
    base_prompt = ((ev.get("base") or {}).get("prompt") or {}).get("metrics") or {}
    deltas = ev.get("adapterPromptVsBasePrompt") or {}
    out = {}
    for m, hib in DIRECTION.items():
        b = (base_prompt.get(m) or {}).get("mean")
        d = (deltas.get(m) or {}).get("delta")
        if b is None:
            continue
        if d is None:
            out[m] = b
        else:
            # delta is stored as an IMPROVEMENT (positive = better); convert to absolute.
            out[m] = b + d if hib else b - d
        out[m] = max(0.0, min(1.0, out[m]))
    return out


def _frontier_abs(front: dict, condition: str = "raw") -> dict:
    """{model_spec: {metric: absolute mean}} for each frontier model at the given condition."""
    res = {}
    for mod in (front.get("models") or []):
        spec = mod.get("spec", "?")
        cm = ((mod.get("conditions") or {}).get(condition) or {}).get("metrics") or {}
        res[spec] = {m: (cm.get(m) or {}).get("mean") for m in DIRECTION if m in cm}
    return res


def _compare(adapter: dict, frontier: dict, mde: float, headline_axis: list) -> list:
    rows = []
    for spec, fr in frontier.items():
        for m, hib in DIRECTION.items():
            a, f = adapter.get(m), fr.get(m)
            if a is None or f is None:
                continue
            gap = (a - f) if hib else (f - a)        # >0 => adapter ahead (sign-normalized)
            verdict = "AHEAD" if gap > mde else ("BEHIND" if gap < -mde else "PARITY")
            rows.append({"model": spec, "metric": m, "adapter": round(a, 3), "frontier": round(f, 3),
                         "gap_adapter_minus_frontier": round(gap, 3), "mde": mde, "verdict": verdict,
                         "headline": m in headline_axis})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-eval", type=Path, default=WM / "M3-pilot-eval.json")
    ap.add_argument("--frontier", type=Path, default=WM / "frontier_baselines.json",
                    help="frontier RAW baseline (default/product framing)")
    ap.add_argument("--frontier-prompt", type=Path, default=WM / "frontier_prompt_baselines.json",
                    help="frontier WITH the same scaffold (isolates the LoRA's contribution)")
    ap.add_argument("--out", type=Path, default=WM / "reality-check.json")
    args = ap.parse_args()

    ev = _load(args.adapter_eval)
    if not ev:
        print(f"adapter eval not found: {args.adapter_eval}", file=sys.stderr)
        return 0
    n = (ev.get("nCases") or 354) * (ev.get("runs") or 3)
    mde = round(mde_at_n(max(1, n)), 3)
    adapter = _adapter_prompt_abs(ev)

    front = _load(args.frontier)
    if not front:
        print("FRONTIER BASELINE NOT PRESENT — reality-check is PENDING.\n")
        print("This is the single most decision-relevant number you do not yet have. To produce it")
        print("(needs a rotated OpenRouter key; ~$1):\n")
        print('  OPENROUTER_API_KEY=... python tools/run_same_size_market_baselines.py \\')
        print('    --models "openrouter:openai/gpt-4o,openrouter:anthropic/claude-3.5-sonnet,'
              'openrouter:google/gemini-2.0-flash-001" \\')
        print(f'    --conditions raw --runs 3 --out {args.frontier.relative_to(ROOT)}')
        print("\nThen re-run: python tools/reality_check.py")
        print(f"\nAdapter(prompt) absolute rates (the bar frontier must clear), N={n}, MDE={mde}:")
        for m, v in adapter.items():
            print(f"  {m:34s} {v:.3f}  ({'higher' if DIRECTION[m] else 'lower'} better)")
        return 0

    headline_axis = ["qualification_rate_on_contested", "false_attribution_rate", "tradition_merge_rate"]
    frontier_raw = _frontier_abs(front, "raw")
    rows = _compare(adapter, frontier_raw, mde, headline_axis)

    # Optional fairness framing: frontier WITH the same scaffold (prompt condition). Isolates
    # whether the LoRA adds anything BEYOND a portable prompt any model could use.
    front_p = _load(args.frontier_prompt)
    frontier_prompt = _frontier_abs(front_p, "prompt") if front_p else {}
    rows_prompt = _compare(adapter, frontier_prompt, mde, headline_axis) if frontier_prompt else []

    # honest headline: on the axis the adapter was trained for, is it ahead of frontier or not?
    head = [r for r in rows if r["headline"]]
    ahead = [r for r in head if r["verdict"] == "AHEAD"]
    head_p = [r for r in rows_prompt if r["headline"]]
    ahead_p = [r for r in head_p if r["verdict"] == "AHEAD"]

    if head and not ahead:
        summary = ("FRONTIER PARITY-OR-BETTER (raw): a 4B adapter's source-discipline edge does NOT "
                   "beat frontier models. Do not ship as a market product.")
    elif ahead and head_p and not ahead_p:
        summary = (f"Adapter beats frontier RAW on {len(ahead)}/{len(head)} headline metric(s), but the "
                   f"edge VANISHES once frontier gets the SAME scaffold ({len(ahead_p)}/{len(head_p)} "
                   "ahead vs frontier(prompt)). The win is a PORTABLE PROMPT, not the LoRA — ship the "
                   "scaffold/gate, not the weights.")
    elif ahead and ahead_p:
        summary = (f"Adapter AHEAD of frontier on {len(ahead)}/{len(head)} headline metric(s) RAW and "
                   f"{len(ahead_p)}/{len(head_p)} even WITH the scaffold given to frontier — the LoRA "
                   "adds real consistency beyond a prompt. A genuine narrow edge; still not a general-LLM "
                   "claim, and pending a semantic-judge cross-check (markers reward the trained format).")
    elif ahead and not head_p:
        summary = (f"Adapter beats frontier RAW on {len(ahead)}/{len(head)} headline metric(s). FAIRNESS "
                   "PENDING: run frontier WITH the scaffold (frontier_prompt_baselines.json) before any "
                   "claim — the raw gap may just be a portable prompt. Markers also need a judge cross-check.")
    else:
        summary = "Insufficient overlapping metrics to judge."

    report = {"benchmark": "wisdom-market heldout_v1 (same cases as adapter eval)", "N": n, "mde": mde,
              "adapterPromptAbsolute": adapter, "frontierRawAbsolute": frontier_raw,
              "frontierPromptAbsolute": frontier_prompt or None,
              "comparison_vs_raw": rows, "comparison_vs_prompt": rows_prompt or None,
              "headlineSummary": summary,
              "boundary": ("adapter(prompt) vs frontier(raw) = default/product framing; vs frontier(prompt) "
                           "= isolates LoRA beyond scaffold; absolute MARKER rates (judge cross-check pending); "
                           "candidate_only; canClaimAGI:false")}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"=== MARKET REALITY-CHECK (N={n}, MDE={mde}) ===")
    print("\n-- adapter(prompt) vs frontier(RAW) — the default/product framing --")
    for r in head:
        print(f"  [{r['verdict']:6s}] {r['metric']:34s} adapter {r['adapter']:.3f} vs "
              f"{r['model'].split('/')[-1]:24s} {r['frontier']:.3f}  (gap {r['gap_adapter_minus_frontier']:+.3f})")
    if head_p:
        print("\n-- adapter(prompt) vs frontier(PROMPT, same scaffold) — isolates the LoRA --")
        for r in head_p:
            print(f"  [{r['verdict']:6s}] {r['metric']:34s} adapter {r['adapter']:.3f} vs "
                  f"{r['model'].split('/')[-1]:24s} {r['frontier']:.3f}  (gap {r['gap_adapter_minus_frontier']:+.3f})")
    else:
        print("\n-- frontier(PROMPT) not present yet — fairness framing PENDING --")
    print(f"\n{summary}\nwrote -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
