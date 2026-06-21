#!/usr/bin/env python3
"""Run the measured, contamination-free self-improvement loop.

Falsifiable claim: held-out recall (catching lineage-merges in *unseen phrasings*)
rises cycle over cycle as the system learns rules from its TRAIN failures, while
held-out false-positive cost stays ~0. Train and held-out phrasings are disjoint,
and rules are learned only from train — so the metric cannot be gamed.

    python tools/run_improvement_loop.py
    python tools/run_improvement_loop.py --batch 4 --cycles 10 --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import improvement  # noqa: E402
from provenance_bench.dataset import DATA_DIR  # noqa: E402


def _load() -> tuple[list[dict], list[dict]]:
    mis = json.loads((DATA_DIR / "misattributions.json").read_text(encoding="utf-8"))["misattributions"]
    pairs = [{"claimed": m["claimed_author"], "work": m["work"]} for m in mis]
    true = json.loads((DATA_DIR / "wikidata_snapshot.json").read_text(encoding="utf-8"))["attributions"]
    # only clean single-author true controls (skip "anonymous"/parenthetical golds)
    controls = [
        {"gold": t["gold_author"], "work": t["work"]}
        for t in true if "(" not in t["gold_author"] and "anonymous" not in t["gold_author"].lower()
        and " and " not in t["gold_author"]
    ]
    return pairs, controls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--model", default=None,
                    help="source TRAIN text from a model (exploratory; default = deterministic templates)")
    ap.add_argument("--json", default=None, help="write the curve to this path")
    args = ap.parse_args(argv)

    answer_fn = None
    if args.model:
        from agent.model import default_client

        client = default_client(args.model)
        _sys = "Answer the attribution question in one sentence."
        answer_fn = lambda claimed, work: getattr(  # noqa: E731
            client.generate(_sys, f"Who wrote {work}?"), "text", "") or ""

    pairs, controls = _load()
    # sealed-hash of the held-out probe set (proves the test set didn't change)
    sealed = hashlib.sha256(
        json.dumps([(p["claimed"], p["work"]) for p in pairs], sort_keys=True).encode()
    ).hexdigest()[:16]

    result = improvement.run_loop(pairs, controls, batch=args.batch, cycles=args.cycles, answer_fn=answer_fn)
    result["trainSource"] = args.model or "deterministic-template"
    result["pairs"] = len(pairs)
    result["trueControls"] = len(controls)
    result["sealedSetHash"] = sealed

    print(f"pairs={len(pairs)} controls={len(controls)} sealed={sealed}")
    print("cycle  rules  heldoutRecall  heldoutFP")
    for c in result["curve"]:
        print(f"  {c['cycle']:>2}   {c['rulesLearned']:>4}      {c['heldoutRecall']:>6.1%}      {c['heldoutFalsePositive']:>6.1%}")
    print(
        f"\nfinalRecall={result['finalRecall']:.1%} "
        f"monotoneNonDecreasing={result['monotoneNonDecreasing']} "
        f"maxFalsePositive={result['maxFalsePositive']:.1%}"
    )
    # the falsifiable assertions
    rose = result["curve"][0]["heldoutRecall"] < result["finalRecall"]
    ok = result["monotoneNonDecreasing"] and rose and result["maxFalsePositive"] == 0.0
    print("LOOP VERIFIED ✓" if ok else "LOOP CLAIM NOT MET ✗")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
