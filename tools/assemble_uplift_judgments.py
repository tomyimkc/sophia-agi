#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Assemble the A3 judgments.json from per-seed pairwise judge sidecars (the A2→A3 bridge).

`judge_pilot_answers.py` (A2) emits **pairwise forced-choice** verdicts — for each item, each
judge family says which answer is better: ``adapter`` / ``base`` / ``tie`` / null. But
`run_lora_uplift_validation.py` (A3) consumes **independent per-family content booleans**
(``baseContent`` / ``adapterContent``). Nothing produced that file, so A3 crashed on a missing
path. This tool is the missing labelling-step assembler.

The faithful pairwise → content encoding (documented, not hidden):

  verdict  | adapterContent[fam] | baseContent[fam]   meaning
  ---------+---------------------+-----------------   ------------------------------------------
  adapter  | True                | False              judge preferred the adapter answer
  base     | False               | True               judge preferred the base answer
  tie      | True                | True               judge rated them at-least-equal
  null     | (omitted)           | (omitted)          judge failed/abstained -> row dropped for
                                                       that family (A3 drops rows not labelled by
                                                       ALL families, so a half-labelled item is
                                                       excluded consistently with mean κ)

Under this encoding the per-seed delta A3 computes (adapter pass-rate − base pass-rate) equals
the **net head-to-head preference margin** (adapter_wins − base_wins)/n, with tie counted as
at-least-as-good on BOTH sides. So the A3 "CI excludes zero" check tests whether the adapter is
**significantly preferred over base** — a relative-quality claim, NOT an absolute-correctness one.
This is stated plainly so no one reads the resulting `validated` as more than it is. `canClaimAGI`
stays False regardless.

Usage:
  python tools/assemble_uplift_judgments.py \
      --subject allenai/OLMoE-1B-7B-0924-Instruct \
      --raw out/seed1-raw.json out/seed2-raw.json out/seed3-raw.json \
      --out out/judgments.json
  python tools/assemble_uplift_judgments.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Key content booleans by the SAME family key the gate uses (run_lora_uplift_validation derives
# fams = [_family_key(j) for j in judges] and looks up content[fam]). Keying by the raw spec
# string instead silently drops EVERY row -> meanDelta 0.0, κ undefined, spurious NO-GO. Import
# the gate's own function so the two can never diverge (the whole point of _family_key's docstring).
from tools.run_lora_uplift_validation import _family_key  # noqa: E402


def _content_pair(verdict: "str | None") -> "tuple[bool, bool] | None":
    """(adapterContent, baseContent) for one pairwise verdict, or None to drop the family."""
    if verdict == "adapter":
        return True, False
    if verdict == "base":
        return False, True
    if verdict == "tie":
        return True, True
    return None  # null / unknown -> family did not label this item


def assemble(raws: list[dict], subject: str) -> dict:
    """Build the run_lora_uplift_validation judgments schema from per-seed raw sidecars.

    Judge family set is taken from the first raw file and asserted identical across the rest
    (the gate keys κ on a fixed family order; a drifting judge set would corrupt it)."""
    if not raws:
        raise ValueError("no raw sidecars provided")
    judges = list(raws[0].get("judges", []))
    if not judges:
        raise ValueError("first raw sidecar has no judges[]")
    for r in raws:
        if list(r.get("judges", [])) != judges:
            raise ValueError("judge set differs across seeds; refusing to assemble a mixed κ")

    seeds = []
    for idx, r in enumerate(raws):
        seed_no = r.get("seed")
        if seed_no is None:
            seed_no = idx
        items = []
        for it in r.get("items", []):
            base_c: dict[str, bool] = {}
            adapter_c: dict[str, bool] = {}
            for spec, verdict in (it.get("verdicts") or {}).items():
                pair = _content_pair(verdict)
                if pair is None:
                    continue  # drop this family for this item (kept consistent with A3)
                fam = _family_key(spec)  # MUST match the gate's per-judge key, not the raw spec
                if fam in adapter_c:
                    raise ValueError(f"two judge specs map to the same family {fam!r}; the gate "
                                     f"would collapse them — use distinct judge families")
                adapter_c[fam], base_c[fam] = pair
            # Only emit items at least one family labelled; A3 drops not-all-families rows itself.
            if base_c or adapter_c:
                items.append({"id": it.get("id"),
                              "baseContent": base_c, "adapterContent": adapter_c})
        seeds.append({"seed": seed_no, "items": items})

    return {
        "subjectModel": subject,
        "judges": judges,
        "seeds": seeds,
        "_provenance": {
            "assembledBy": "tools/assemble_uplift_judgments.py",
            "encoding": "pairwise forced-choice -> content booleans (adapter/base/tie); "
                        "delta == net head-to-head preference margin, not absolute correctness",
            "sources": [r.get("answers") for r in raws],
        },
    }


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    # Two families, 3 seeds, adapter clearly preferred -> assembled schema is well-formed and
    # A3 can run on it. We assert the encoding, not a VALIDATED verdict.
    raws = []
    for s in range(3):
        items = []
        for i in range(8):
            v = "adapter" if i < 6 else ("tie" if i == 6 else "base")
            items.append({"id": f"item_{i}",
                          "verdicts": {"ollama:qwen": v,
                                       "vllm:mlx-community/Llama-3.3-70B": v}})
        # one item where the 70B family abstained -> must drop that family only
        items.append({"id": "item_abstain",
                      "verdicts": {"ollama:qwen": "adapter",
                                   "vllm:mlx-community/Llama-3.3-70B": None}})
        raws.append({"seed": s, "judges": ["ollama:qwen", "vllm:mlx-community/Llama-3.3-70B"],
                     "answers": f"seed{s}.json", "items": items})
    j = assemble(raws, "allenai/OLMoE-1B-7B-0924-Instruct")
    assert j["subjectModel"] == "allenai/OLMoE-1B-7B-0924-Instruct"
    assert len(j["seeds"]) == 3
    # CONTENT MUST be keyed by the gate's _family_key, NOT the raw spec — else A3 drops every
    # row (meanDelta 0.0, spurious NO-GO). This is the assertion that catches the 2026-06-30 bug
    # and it needs no numpy, so it runs in CI too.
    fams = {_family_key(spec) for spec in ["ollama:qwen", "vllm:mlx-community/Llama-3.3-70B"]}
    assert fams == {"ollama", "mlx-community"}, fams
    s0 = j["seeds"][0]["items"]
    it0 = next(x for x in s0 if x["id"] == "item_0")
    assert set(it0["adapterContent"]) == fams, it0["adapterContent"]
    assert it0["adapterContent"] == {"ollama": True, "mlx-community": True}
    assert it0["baseContent"] == {"ollama": False, "mlx-community": False}
    # tie item -> both true
    it6 = next(x for x in s0 if x["id"] == "item_6")
    assert it6["adapterContent"]["ollama"] is True and it6["baseContent"]["ollama"] is True
    # abstain item -> only the qwen family present
    ita = next(x for x in s0 if x["id"] == "item_abstain")
    assert set(ita["adapterContent"]) == {"ollama"}
    assert "mlx-community" not in ita["baseContent"]
    # null verdict maps to None pair
    assert _content_pair(None) is None and _content_pair("tie") == (True, True)
    # End-to-end: feed it to the real aggregator and confirm it runs + computes a delta.
    # The aggregator's bootstrap CI needs numpy (present on the Spark, maybe not in CI); the
    # encoding assertions above stand on their own, so degrade gracefully if numpy is absent.
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        print("ok assemble_uplift_judgments selftest (encoding verified; "
              "skipped end-to-end aggregate — numpy not installed)")
        return 0
    from tools.run_lora_uplift_validation import aggregate
    rep = aggregate(j)
    assert rep["subjectFamily"] == "allenai"
    assert rep["judgeFamilies"] == 2
    assert rep["meanDelta"] > 0  # adapter preferred in the synthetic data
    assert rep["canClaimAGI"] is False
    print("ok assemble_uplift_judgments selftest "
          f"(meanDelta={rep['meanDelta']} κ={rep['meanPairwiseKappa']} "
          f"families={rep['judgeFamilies']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, nargs="+",
                    help="per-seed raw verdict sidecars from judge_pilot_answers.py --raw-out")
    ap.add_argument("--subject", default="allenai/OLMoE-1B-7B-0924-Instruct",
                    help="subject model id (its family must differ from every judge family)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true", help="offline self-test of the encoding")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.raw:
        ap.error("provide --raw FILE... or --selftest")
        return 2

    raws = [json.loads(p.read_text(encoding="utf-8")) for p in args.raw]
    judgments = assemble(raws, args.subject)
    text = json.dumps(judgments, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {sum(len(s['items']) for s in judgments['seeds'])} items across "
              f"{len(judgments['seeds'])} seeds -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
