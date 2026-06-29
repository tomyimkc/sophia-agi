#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Re-grade a saved generation set with the SELF-HOSTABLE open judge (Leiden value 5).

This is the harness for open-judge acceptance criterion #2: corroborate a result with a judge
family that runs on a NON-PROPRIETARY path (open weights + self-hosted inference), rather than
a proprietary API. It mirrors tools/llm_judge_score.py's verdict contract but routes through
``agent.open_judge`` (a local OpenAI-compatible endpoint configured via OPEN_JUDGE_* env).

It does NOT provision GPUs and it does NOT call any proprietary service. It is fail-closed:
without a configured endpoint it refuses to run (exit 2) — it never silently falls back to a
proprietary judge. On a per-item *unparsed* response it falls back to the LOCAL deterministic
lexical heuristic (also non-proprietary) and counts how often that happened.

    OPEN_JUDGE_BASE_URL=http://localhost:8000/v1 OPEN_JUDGE_MODEL=qwen2.5-32b-instruct \
        python tools/run_open_judge_regrade.py \
        training/swarm_router/theta_search_2family_result.json \
        --pack data/search_recall/pack_third_party.jsonl \
        --out agi-proof/benchmark-results/wisdom-market/open-judge-regrade.json

The emitted receipt records the judge id, its non-proprietary classification, the delta + CI,
and Cohen's κ vs the local heuristic families — enough for claim_gate-style scrutiny.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import judge_registry, open_judge  # noqa: E402
from provenance_bench.search_recall import (  # noqa: E402
    cohens_kappa, source_discipline_ok, disciplined_stance,
)
from provenance_bench.swarm_benchmark import _paired_bootstrap_ci  # noqa: E402


def _claims_from_pack(pack: Path) -> "list[str]":
    out = []
    for line in pack.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out.append(rec.get("claim_text") or rec.get("claim") or "")
    return out


def regrade(result: dict, claims: "list[str]", *, env: "Optional[dict]" = None,
            transport=None) -> dict:
    """Grade base + seed arms with the open judge; return a receipt dict.

    Raises RuntimeError if the open judge endpoint is not configured (fail-closed).
    """
    if not open_judge.available(env):
        raise RuntimeError("open judge endpoint not configured (OPEN_JUDGE_BASE_URL / "
                           "OPEN_JUDGE_MODEL) — refusing to grade (fail-closed)")
    raw = result.get("raw_generations") or {}
    if "base" not in raw:
        raise RuntimeError("result has no raw_generations.base to grade")
    seed_keys = sorted(k for k in raw if k.startswith("seed"))
    n = len(raw["base"])
    if len(claims) < n:
        raise RuntimeError(f"pack has {len(claims)} claims < {n} generations")

    fallbacks = {"count": 0}

    def judge_arm(gens: "list[str]") -> "list[int]":
        labels = []
        for claim, ans in zip(claims, gens):
            v = open_judge.score(claim, ans, env=env, transport=transport)
            if v is None:  # unparsed -> LOCAL heuristic (still non-proprietary), counted
                fallbacks["count"] += 1
                v = 1 if source_discipline_ok(ans) else 0
            labels.append(v)
        return labels

    base_lab = judge_arm(raw["base"])
    seed_lab = {sk: judge_arm(raw[sk]) for sk in seed_keys}

    per_seed, deltas, pb, pa = {}, [], [], []
    br = sum(base_lab) / n
    for sk in seed_keys:
        ar = sum(seed_lab[sk]) / n
        per_seed[sk] = {"before": round(br, 3), "after": round(ar, 3), "delta": round(ar - br, 3)}
        deltas.append(ar - br)
        pb += base_lab
        pa += seed_lab[sk]
    lo, hi = _paired_bootstrap_ci(pb, pa, iters=4000, seed=0)

    all_gens = raw["base"] + [g for sk in seed_keys for g in raw[sk]]
    open_all = base_lab + [v for sk in seed_keys for v in seed_lab[sk]]
    lex_all = [1 if source_discipline_ok(g) else 0 for g in all_gens]
    stance_all = [1 if disciplined_stance(g) else 0 for g in all_gens]

    # Build the judge id from the (non-secret) model name read directly, NOT from
    # open_judge.endpoint_config(): that returns a dict containing the API key alongside the
    # model, and a clear-text-logging scanner taints the whole dict, so anything derived from
    # it (the model) would be treated as a secret once the receipt is printed/written. The API
    # key only ever flows into the HTTP request header inside open_judge.score(), never here.
    model = (env if env is not None else os.environ).get("OPEN_JUDGE_MODEL", "")
    jid = f"local:{model or 'unconfigured'}"
    independence = judge_registry.classify_judge(jid)
    return {
        "_comment": ("Open-judge re-grade receipt — non-proprietary corroboration "
                     "(Leiden value 5). Generated by tools/run_open_judge_regrade.py."),
        "canClaimAGI": False,
        "judge_id": jid,
        "judge_independence": independence,
        "non_proprietary_path": independence["non_proprietary_path"],
        "n_traps": n,
        "seeds": seed_keys,
        "base_model": result.get("model"),
        "open_judge": {
            "base_rate": round(br, 3),
            "per_seed": per_seed,
            "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0),
        },
        "kappa_open_vs_lexical": cohens_kappa(open_all, lex_all),
        "kappa_open_vs_stance": cohens_kappa(open_all, stance_all),
        "heuristic_fallbacks": fallbacks["count"],
        "boundary": ("Independent self-hosted open-weights judge (!= subject, != gate). "
                     "Single saved generation set; corroboration, not a market or AGI claim."),
    }


def main(argv: "Optional[list[str]]" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("--pack", type=Path,
                    default=ROOT / "data" / "search_recall" / "pack_third_party.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not open_judge.available():
        print("OPEN_JUDGE_BASE_URL / OPEN_JUDGE_MODEL not set (fail-closed). This harness only "
              "runs against a self-hosted open endpoint; it never uses a proprietary judge.",
              file=sys.stderr)
        return 2
    result = json.loads(args.result.read_text(encoding="utf-8"))
    claims = _claims_from_pack(args.pack)
    try:
        # env defaults to None -> open_judge reads os.environ internally only where it builds
        # the HTTP request (the API key flows to the request header, never into the returned
        # receipt). We deliberately do NOT pass dict(os.environ) through here: that would route
        # the whole environment (incl. the API key) into a value that is later printed, which a
        # clear-text-logging scanner flags even though no secret actually reaches the receipt.
        receipt = regrade(result, claims)
    except RuntimeError as exc:
        print(f"re-grade refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
