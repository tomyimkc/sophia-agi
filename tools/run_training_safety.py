#!/usr/bin/env python3
"""LoRA leakage guard + contamination control (#7) — falsifiable, on the real corpus.

  - Leakage: the real training corpus has 0 unsafe (confidential/PII/secret)
    examples, and a planted confidential example IS dropped by the guard.
  - Contamination: a planted near-duplicate is flagged; disjoint text is clean.

Exits non-zero if any invariant fails. Deterministic, offline.

    python tools/run_training_safety.py [--json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.training_safety import filter_examples, is_safe_to_train, make_canary  # noqa: E402
from eval.contamination import overlap_report  # noqa: E402


def _text(ex: dict) -> str:
    return "\n".join(str(m.get("content", "")) for m in ex.get("messages", []) or [])


def run() -> dict:
    examples = [json.loads(Path(p).read_text(encoding="utf-8"))
                for p in sorted(glob.glob(str(ROOT / "training" / "examples" / "*.json")))]
    filt = filter_examples(examples)

    # planted confidential example (canary + classification) must be dropped
    canary = make_canary("leakage-demo")
    confidential = {
        "messages": [{"role": "user", "content": "internal"},
                     {"role": "assistant", "content": f"api_key={canary}"}],
        "metadata": {"classification": "confidential"},
    }
    confidential_dropped = not is_safe_to_train(confidential)

    # REAL contamination: split the safe corpus into train vs held-out (the same
    # benchmark/trap holdout the LoRA prep uses) and measure near-dup overlap.
    from tools.prepare_lora_dataset import is_holdout, load_benchmark_ids

    bench_ids, bench_q = load_benchmark_ids()
    train_ex = [e for e in filt["safe"] if not is_holdout(e, bench_ids, bench_q)]
    held_ex = [e for e in filt["safe"] if is_holdout(e, bench_ids, bench_q)]
    train_texts = [_text(e) for e in train_ex if _text(e).strip()]
    held_texts = [_text(e) for e in held_ex if _text(e).strip()]
    real = overlap_report(train_texts or ["x"], held_texts, n=8, threshold=0.6) if held_texts else None

    # Metric self-test (synthetic controls verify the detector itself works).
    base = train_texts[0] if train_texts else "the dao de jing is a classic daoist text by laozi"
    pos = overlap_report([base], [base], n=8, threshold=0.6)                          # verbatim -> flagged
    neg = overlap_report([base], ["zebra xylophone quasar nimbus unrelated sentence"], n=8, threshold=0.6)

    invariants = {
        "real_corpus_has_no_unsafe_example": filt["nDropped"] == 0,
        "confidential_example_is_dropped": confidential_dropped,
        "real_train_heldout_not_contaminated": (real is None) or real["contaminationRate"] == 0.0,
        "detector_flags_verbatim": pos["contaminationRate"] == 1.0,
        "detector_clean_on_disjoint": neg["contaminationRate"] == 0.0,
    }
    return {
        "corpus": {"n": filt["nIn"], "safe": filt["nSafe"], "dropped": filt["nDropped"],
                   "reasons": filt["reasonsHistogram"]},
        "contamination": {
            "realSplit": {"train": len(train_texts), "heldout": len(held_texts),
                          "rate": (real["contaminationRate"] if real else None),
                          "maxContainment": (real["maxContainment"] if real else None)},
            "selfTest": {"verbatimRate": pos["contaminationRate"], "disjointRate": neg["contaminationRate"]},
        },
        "invariants": invariants,
        "ok": all(invariants.values()),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = run()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    c = r["corpus"]
    rs = r["contamination"]["realSplit"]
    st = r["contamination"]["selfTest"]
    print("LoRA leakage guard + contamination control (#7)")
    print("=" * 50)
    print(f"\nreal corpus: {c['n']} examples -> {c['safe']} safe, {c['dropped']} dropped {c['reasons']}")
    print(f"contamination (REAL split): train={rs['train']} heldout={rs['heldout']} "
          f"rate={rs['rate']} maxContainment={rs['maxContainment']}")
    print(f"detector self-test: verbatim={st['verbatimRate']} disjoint={st['disjointRate']}")
    print("\nFalsifiable invariants:")
    for k, v in r["invariants"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\n" + ("ALL INVARIANTS HOLD" if r["ok"] else "INVARIANT FAILURE"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
