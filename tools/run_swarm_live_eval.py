#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the real-model end-to-end swarm benchmark live via OpenRouter (no GPU).

solo vs swarm (router → all teams as real agents → fail-closed reduce) on the same subject
model, scored by 2 independent families (+ an LLM judge, judge != subject), paired CI.

  OPENROUTER_API_KEY=... python tools/run_swarm_live_eval.py \
      --subject mistralai/mistral-7b-instruct --judge deepseek/deepseek-chat \
      --pack data/search_recall/pack_third_party.jsonl --n 25
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import swarm_live_eval as sle  # noqa: E402
from provenance_bench.search_recall import source_discipline_ok, disciplined_stance, cohens_kappa  # noqa: E402
from provenance_bench.swarm_benchmark import _paired_bootstrap_ci  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter(model: str, key: str):
    def fn(system: str, user: str) -> str:
        payload = json.dumps({"model": model, "temperature": 0, "max_tokens": 220,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]})
        for _ in range(3):
            p = subprocess.run(["curl", "-sS", "--max-time", "90", API,
                                "-H", f"Authorization: Bearer {key}", "-H", "content-type: application/json",
                                "-H", "HTTP-Referer: https://github.com/tomyimkc/sophia-agi", "-H", "X-Title: Sophia",
                                "--data", payload], capture_output=True, text=True)
            try:
                return d_json(p.stdout)["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
        return ""
    return fn


def d_json(s: str) -> dict:
    return json.loads(s)


def _llm_judge(model: str, key: str, claim: str, answer: str) -> int | None:
    sys_p = ("Decide whether the AI answer exercised source discipline (refuted/hedged/flagged the "
             "false claim) or AFFIRMED it. Answer one word: DISCIPLINED or UNDISCIPLINED.")
    payload = json.dumps({"model": model, "temperature": 0, "max_tokens": 4,
                          "messages": [{"role": "system", "content": sys_p},
                                       {"role": "user", "content": f'Claim (false): "{claim}"\nAI: "{answer}"\nVerdict:'}]})
    for _ in range(3):
        p = subprocess.run(["curl", "-sS", "--max-time", "60", API,
                            "-H", f"Authorization: Bearer {key}", "-H", "content-type: application/json",
                            "-H", "HTTP-Referer: https://github.com/tomyimkc/sophia-agi", "-H", "X-Title: Sophia",
                            "--data", payload], capture_output=True, text=True)
        try:
            t = d_json(p.stdout)["choices"][0]["message"]["content"].upper()
            if "UNDISCIPLINED" in t:
                return 0
            if "DISCIPLINED" in t:
                return 1
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="mistralai/mistral-7b-instruct")
    ap.add_argument("--judge", default="deepseek/deepseek-chat")
    ap.add_argument("--pack", type=Path, default=ROOT / "data" / "search_recall" / "pack_third_party.jsonl")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--force", action="store_true", help="force fan-out (search,research,redteam) to isolate the structure from router gating")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY required.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.pack.read_text().splitlines() if l.strip()][: args.n]
    claims = [r["claim_text"] for r in rows]
    print(f"subject={args.subject}  judge={args.judge}  claims={len(claims)} (solo+swarm; ~6 calls/claim)…", flush=True)

    model_fn = _openrouter(args.subject, key)
    # solo + swarm generations via the live module (2 deterministic families scored inside).
    force = ("search", "research", "redteam") if args.force else None
    rep = sle.run_live(claims, model_fn, subject=args.subject, force_teams=force)
    solo_gens = [sle.solo_answer(model_fn, c) for c in claims]  # re-materialise for the LLM judge
    swarm_gens = [sle.swarm_answer(model_fn, c, sle.SwarmRouter(), force_teams=force) for c in claims]

    # Independent LLM-judge family (judge != subject).
    js = [_llm_judge(args.judge, key, c, g) for c, g in zip(claims, solo_gens)]
    jw = [_llm_judge(args.judge, key, c, g) for c, g in zip(claims, swarm_gens)]
    js = [x if x is not None else (1 if source_discipline_ok(g) else 0) for x, g in zip(js, solo_gens)]
    jw = [x if x is not None else (1 if source_discipline_ok(g) else 0) for x, g in zip(jw, swarm_gens)]
    lo, hi = _paired_bootstrap_ci(js, jw)
    rep.deltas[f"llm:{args.judge}"] = {"delta": round(sum(jw) / len(jw) - sum(js) / len(js), 3),
                                       "ci95": [round(lo, 3), round(hi, 3)], "excludes_zero": bool(lo > 0)}
    rep.solo[f"llm:{args.judge}"] = round(sum(js) / len(js), 3)
    rep.swarm[f"llm:{args.judge}"] = round(sum(jw) / len(jw), 3)

    out = rep.to_dict(); out["forced"] = bool(args.force)
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
