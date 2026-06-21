#!/usr/bin/env python3
"""Fetch a real external benchmark and convert it to the eval's JSONL contract.

`tools/run_external_eval.py` scores a model against EXTERNAL gold answers (never
the gate), reading ``{question, answer}`` JSONL. This downloads a public dataset
and writes that exact shape, so the external-eval harness produces a *citable*
number instead of running only the committed style-sample.

    python tools/fetch_eval_dataset.py --dataset gsm8k --split test --limit 200
    python tools/run_external_eval.py --dataset eval/external/gsm8k-test.jsonl --model <spec>

Honest scope: this only downloads and reshapes public data — it makes no model
call and no capability claim. A quotable number still requires you to run the eval
with a named model and report N + dataset version. Network access is required and
this tool is intentionally NOT run in CI.

Sources (public):
  - GSM8K: openai/grade-school-math (MIT) — native ``{question, answer}`` with a
    ``#### N`` final-answer marker, which the eval's extractor already understands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Raw public mirrors keyed by (dataset, split).
SOURCES = {
    ("gsm8k", "test"): "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    ("gsm8k", "train"): "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
}

DEFAULT_OUT = {"gsm8k": "eval/external/gsm8k-{split}.jsonl"}


def convert_gsm8k_lines(lines) -> list:
    """Parse GSM8K JSONL lines into the eval contract ``{question, answer}``.

    GSM8K is already ``{question, answer}`` with the gold ending in ``#### N``;
    we keep both fields verbatim (the eval's extractor reads the ``#### N``
    marker). Malformed / empty lines are skipped. Pure function — no network — so
    it is unit-tested offline.
    """
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        q, a = obj.get("question"), obj.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            out.append({"question": q, "answer": a})
    return out


CONVERTERS = {"gsm8k": convert_gsm8k_lines}


def _fetch(url: str) -> list:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:   # noqa: S310 — pinned public mirror
        return resp.read().decode("utf-8").splitlines()


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gsm8k", choices=sorted({d for d, _ in SOURCES}))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="cap items (0 = all)")
    ap.add_argument("--out", default=None, help="output JSONL path")
    args = ap.parse_args(argv)

    key = (args.dataset, args.split)
    if key not in SOURCES:
        print(f"no source for {key}; available: {sorted(SOURCES)}", file=sys.stderr)
        return 2

    out_path = Path(args.out or DEFAULT_OUT[args.dataset].format(split=args.split))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"fetching {args.dataset}/{args.split} …")
    try:
        lines = _fetch(SOURCES[key])
    except Exception as exc:                                  # noqa: BLE001 — surface any network error
        print(f"download failed: {exc}\n(network required; this tool is not run in CI)", file=sys.stderr)
        return 1

    items = CONVERTERS[args.dataset](lines)
    if args.limit:
        items = items[: args.limit]
    out_path.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8")

    print(f"wrote {len(items)} items -> {out_path}")
    print("\nNext (produces a citable external number — needs a model):")
    print(f"  python tools/run_external_eval.py --dataset {out_path} --model <spec> --limit {args.limit or len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
