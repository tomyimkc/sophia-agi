#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a FORMAT-ROBUST source-discipline SFT corpus (distilled, decontaminated).

The cross-model negative (Mistral) was diagnosed as council-FORMAT overfitting: the council
traces teach a verbose multi-seat "Council panel:" layout, which Qwen partly resists but
Mistral adopts wholesale (30/30) — losing the direct refutation that *is* source discipline.

This builds the fix: plain ``question → direct disciplined answer`` pairs, NO multi-seat
scaffolding. Training claims are drawn from sources DISJOINT from the third-party eval pack
(first-party attribution traps + a FRESH batch of fact-checked claims on different queries,
deduped against the eval), and the gold disciplined answers are DISTILLED from an
independent model (OpenRouter DeepSeek) with an explicit "be direct, no council format"
instruction. The result is a format-robust corpus to retest cross-model transfer.

  OPENROUTER_API_KEY=... GOOGLE_FACTCHECK_API_KEY=... python tools/build_discipline_sft.py

Writes ``data/discipline_sft/format_robust.jsonl`` (+ manifest). Keys from env only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.search_recall import PACK_V2_PATH  # noqa: E402

OUT = ROOT / "data" / "discipline_sft" / "format_robust.jsonl"
MANIFEST = ROOT / "data" / "discipline_sft" / "manifest.json"
EVAL_PACK = ROOT / "data" / "search_recall" / "pack_third_party.jsonl"
OR_API = "https://openrouter.ai/api/v1/chat/completions"
FC_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

SYS = "You are a source-disciplined search agent. Cite sources; abstain if you cannot ground a claim."
GEN_SYS = (
    "You write CONCISE, source-disciplined answers. Given a claim that is false, disputed, "
    "or a misattribution, write a direct 2-4 sentence answer that refutes or flags it, names "
    "the kind of evidence/source, and hedges honestly. CRITICAL: plain prose only — NO "
    "'Council panel', NO multi-seat or persona layout, NO markdown headers, NO bullet lists."
)
# Fresh FC queries, deliberately different from the eval pack's queries.
FRESH_FC = [
    "shark attack statistics", "microwave radiation danger", "cracking knuckles arthritis",
    "swallowed gum digest", "lightning same place twice", "tongue taste map",
    "dogs see black white", "hair grows back thicker shaving", "blood is blue in veins",
    "full moon behavior", "coffee stunts growth", "reading dim light eyes",
    "wait 24 hours missing person", "alcohol kills brain cells", "spinach iron popeye",
    "carrots night vision", "stomach ulcers stress", "antibiotics virus cold",
]


def _curl_json(url: str, headers: list[str], data: str | None = None) -> dict:
    cmd = ["curl", "-sS", "--max-time", "60", url]
    for h in headers:
        cmd += ["-H", h]
    if data is not None:
        cmd += ["--data", data]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}


def _deepseek(claim: str, key: str, model: str) -> str | None:
    payload = json.dumps({"model": model, "temperature": 0.3, "max_tokens": 160,
                          "messages": [{"role": "system", "content": GEN_SYS},
                                       {"role": "user", "content": f'Claim: "{claim}"\n\nDisciplined answer:'}]})
    d = _curl_json(OR_API, ["Authorization: Bearer " + key, "content-type: application/json",
                            "HTTP-Referer: https://github.com/tomyimkc/sophia-agi", "X-Title: Sophia"], payload)
    try:
        return d["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _fresh_fc_claims(key: str, exclude: set[str], limit: int) -> list[str]:
    out: list[str] = []
    _FALSE = re.compile(r"\b(false|misleading|unproven|incorrect|myth|no evidence|debunk|baseless|inaccurate)\b", re.I)
    for q in FRESH_FC:
        url = f"{FC_API}?{urllib.parse.urlencode({'query': q, 'languageCode': 'en', 'pageSize': 5, 'key': key})}"
        for c in _curl_json(url, []).get("claims", []):
            t = (c.get("text") or "").strip()
            norm = re.sub(r"\s+", " ", t.lower())
            if not t or len(t) > 220 or norm in exclude:
                continue
            if any(_FALSE.search((rev.get("textualRating") or "")) for rev in (c.get("claimReview") or [])):
                exclude.add(norm)
                out.append(t)
        if len(out) >= limit:
            break
    return out[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--fresh-fc", type=int, default=25)
    args = ap.parse_args()
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    fc_key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
    if not or_key:
        print("OPENROUTER_API_KEY required.", file=sys.stderr)
        return 2

    # Decontaminate: collect the eval pack's claim texts to EXCLUDE from training.
    eval_norm = set()
    if EVAL_PACK.exists():
        for l in EVAL_PACK.read_text().splitlines():
            if l.strip():
                eval_norm.add(re.sub(r"\s+", " ", json.loads(l)["claim_text"].lower()))

    # Training claims: first-party attribution traps (pack_v2) + fresh decontaminated FC claims.
    claims: list[str] = []
    for l in Path(PACK_V2_PATH).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            q = r["query"]
            norm = re.sub(r"\s+", " ", q.lower())
            if norm not in eval_norm:
                claims.append(q)
    if fc_key:
        claims += [f'Is the following claim accurate? "{c}"' for c in _fresh_fc_claims(fc_key, eval_norm, args.fresh_fc)]

    rows = []
    for i, claim in enumerate(claims):
        ans = _deepseek(claim, or_key, args.model)
        if not ans or "council panel" in ans.lower() or ans.count("**") >= 4:
            continue  # reject degenerate / formatted distillations
        rows.append({"messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": claim},
                                  {"role": "assistant", "content": ans}]})
        if (i + 1) % 10 == 0:
            print(f"  distilled {len(rows)}/{i+1}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps({
        "schema": "sophia.discipline_sft.v1", "corpus": "format_robust",
        "n": len(rows), "sha256": sha, "distill_model": args.model,
        "format": "plain prose, no council/multi-seat scaffolding (fixes the cross-model format-overfit)",
        "decontaminated_against": "data/search_recall/pack_third_party.jsonl (eval)",
        "claim_sources": ["pack_v2 attribution traps (first-party)", "fresh Google FactCheck claims (decontaminated)"],
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} format-robust SFT pairs → {OUT} (sha {sha[:16]})")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
