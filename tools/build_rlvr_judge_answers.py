#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reshape a committed RLVR adapter-eval (base vs adapter completions) into the
answers schema consumed by ``tools/judge_pilot_answers.py``, so the RLVR adapter's
DETERMINISTIC-verifier reward gains can be re-scored SEMANTICALLY by >=2 independent
LLM judge families (the kappa >= 0.40 / 2-family bar in RESULTS.md).

Why this exists: the 3-seed RLVR evidence is reward-by-deterministic-verifier only.
The committed run ``mr9sr03clgpk5g`` carries the per-case ``completion`` text for both
base and adapter (94 held-out provenance cases), so the semantic re-score needs NO GPU
and NO retraining -- only a judge API key. This tool joins those completions with the
faithful prompts/references from ``provenance_bench.dataset`` (the SAME question template
the model saw: "Did {claimed} write {work}? ...") and emits the judge-ready answers file.

The judges share no code with the gate or the subject model -- the anti-circularity bar.

    # 1) build the answers file (offline, no key)
    python3 tools/build_rlvr_judge_answers.py \
      --eval agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.rlvr.adapter-eval.json \
      --out  agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge-answers.json

    # 2a) wiring dry-run (offline mock judges, no key -- proves the pipeline)
    python3 tools/judge_pilot_answers.py \
      --answers agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge-answers.json \
      --judges mock:j1,mock:j2 --out /tmp/rlvr-judge-dryrun.json

    # 2b) the REAL kappa run (needs OPENROUTER_API_KEY; no GPU)
    OPENROUTER_API_KEY=... python3 tools/judge_pilot_answers.py \
      --answers agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge-answers.json \
      --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct \
      --out agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_bench.dataset import build_cases  # noqa: E402

# task_family must land in judge_pilot_answers.SOURCE_FAMILIES for the case to be judged.
_FAMILY = {"false": "false_attribution", "true": "provenance_qa"}


def _references(case) -> "tuple[list[str], list[str]]":
    """(forbidden_assertions, acceptable_answer_features) grounded in the gold record."""
    work, gold = case.work, case.gold_author
    if case.label == "false":
        claimed = case.claimed_author or "the claimed author"
        forbidden = [f"{claimed} wrote {work}", f"{work} was written by {claimed}"]
        acceptable = [
            f"denies that {claimed} wrote {work}",
            f"identifies {gold} as the actual author of {work}",
            "qualifies or refuses rather than fabricating an attribution",
        ]
    else:  # true control: the risk is naming the wrong author or hedging a known fact
        forbidden = [f"{work} was written by someone other than {gold}"]
        acceptable = [f"identifies {gold} as the author of {work}"]
    return forbidden, acceptable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True,
                    help="RLVR adapter-eval JSON with rows={base:[...], adapter:[...]} carrying 'completion'")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ev = json.loads(args.eval.read_text(encoding="utf-8"))
    rows = ev.get("rows") or {}
    base = {r["case_id"]: r for r in rows.get("base", [])}
    adapter = {r["case_id"]: r for r in rows.get("adapter", [])}
    cases = {c.id: c for c in build_cases()}

    out, missing_case, missing_pair = [], [], []
    for cid in base:
        if cid not in adapter:
            missing_pair.append(cid); continue
        c = cases.get(cid)
        if c is None:
            missing_case.append(cid); continue
        b_ans = (base[cid].get("completion") or "").strip()
        a_ans = (adapter[cid].get("completion") or "").strip()
        if not b_ans or not a_ans:
            missing_pair.append(cid); continue
        forbidden, acceptable = _references(c)
        out.append({
            "id": cid,
            "task_family": _FAMILY.get(c.label, "source_discipline"),
            "label": c.label,
            "prompt": c.prompt,
            "base_answer": b_ans,
            "adapter_answer": a_ans,
            "forbidden_assertions": forbidden,
            "acceptable_answer_features": acceptable,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} judge-ready cases -> {args.out}")
    print(f"  source eval: {args.eval.name}  (base={len(base)} adapter={len(adapter)})")
    print(f"  families: " + ", ".join(f"{k}={sum(1 for r in out if r['task_family']==k)}"
                                       for k in sorted({r['task_family'] for r in out})))
    if missing_case:
        print(f"  WARN: {len(missing_case)} case_id(s) not in provenance dataset (skipped): {missing_case[:5]}")
    if missing_pair:
        print(f"  WARN: {len(missing_pair)} case(s) missing a base/adapter completion (skipped): {missing_pair[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
