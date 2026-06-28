#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a decomposition/coverage pack — broad questions + gold aspects (DeepSeek-authored).

Each row is a broad analytical question plus 4-5 distinct key aspects a complete answer must
cover. The subject model never sees the aspects; they are the held-out coverage targets for
``provenance_bench/swarm_coverage_eval.py``. Keys from env only.

  OPENROUTER_API_KEY=... python tools/build_coverage_pack.py --n 15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "coverage" / "pack_v1.jsonl"
MANIFEST = ROOT / "data" / "coverage" / "manifest.json"
API = "https://openrouter.ai/api/v1/chat/completions"

TOPICS = [
    "the decline of the Roman Empire", "the causes of the 2008 financial crisis",
    "the impact of social media on democracy", "antibiotic resistance",
    "the transition to renewable energy", "the ethics of gene editing",
    "urban housing affordability", "the effects of remote work",
    "ocean plastic pollution", "the rise of large language models",
    "vaccine hesitancy", "deforestation in the Amazon",
    "the gig economy", "electric vehicle adoption", "misinformation online",
    "water scarcity", "the opioid epidemic", "space commercialization",
]


def _gen(topic: str, key: str, model: str) -> dict | None:
    sys_p = ("Generate ONE broad analytical question about the topic and list 4-5 DISTINCT key "
             "aspects a complete answer must cover (short noun phrases, e.g. 'economic causes', "
             "'regulatory response', 'social impact'). Output STRICT JSON: "
             '{"question": "...", "aspects": ["...", "...", "..."]}')
    payload = json.dumps({"model": model, "temperature": 0.4, "max_tokens": 220,
                          "messages": [{"role": "system", "content": sys_p},
                                       {"role": "user", "content": f"Topic: {topic}"}]})
    p = subprocess.run(["curl", "-sS", "--max-time", "60", API,
                        "-H", f"Authorization: Bearer {key}", "-H", "content-type: application/json",
                        "-H", "HTTP-Referer: https://github.com/tomyimkc/sophia-agi", "-H", "X-Title: Sophia",
                        "--data", payload], capture_output=True, text=True)
    try:
        txt = json.loads(p.stdout)["choices"][0]["message"]["content"]
        s, e = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[s:e + 1])
        q = obj.get("question", "").strip()
        aspects = [a.strip() for a in (obj.get("aspects") or []) if a.strip()]
        if q and 3 <= len(aspects) <= 6:
            return {"question": q, "aspects": aspects}
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY required.", file=sys.stderr)
        return 2

    rows = []
    for topic in TOPICS:
        r = _gen(topic, key, args.model)
        if r:
            rows.append(r)
        if len(rows) >= args.n:
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps({
        "schema": "sophia.coverage_pack.v1", "n": len(rows),
        "totalAspects": sum(len(r["aspects"]) for r in rows), "sha256": sha,
        "author_model": args.model, "purpose": "decomposition/coverage benchmark — the fair positive test for the swarm",
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} coverage questions ({sum(len(r['aspects']) for r in rows)} aspects) → {OUT} (sha {sha[:16]})")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
