#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Learn skill triggers from routing telemetry — make triggering self-improving.

``agent/skills.py::select(..., log_path=...)`` appends one row per routing decision:
``{goal, skill_id, score, via}``. Pair those goals with an OUTCOME signal (did the
selected skill's verifier accept? did the task succeed?) and you can LEARN triggers:
goal tokens that co-occur with ACCEPTED outcomes for a skill are promoted into that
skill's ``triggers``; tokens that co-occur with misfires are demoted. Triggering then
rides the same verifier-gated flywheel as the skills themselves.

This is a PROPOSER, not an auto-committer: it prints suggested trigger additions and,
with ``--apply``, adds only tokens whose acceptance rate clears ``--min-precision``
over at least ``--min-support`` observations. It never removes a hand-authored trigger.

Telemetry row (JSONL), produced by the router + your outcome logger::

    {"goal": "fix the failing auth test", "skill_id": "coding-debugging",
     "score": 20.8, "via": "keyword", "accepted": true}

``accepted`` is optional; rows without it count as support but not as positive signal.

Run:  python tools/learn_triggers.py --log agi-proof/skill-trigger-log.jsonl [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "skills" / "registry"
DEFAULT_LOG = ROOT / "agi-proof" / "skill-trigger-log.jsonl"

_STOP = set("""
the a an and or of to in on for with without this that these those is are be use using used
your you it its as at by from into out up down when whenever before after run i my me we our
fix make do please can could would should help need want get set the".split() also new
""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 2]


def _load_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("skill_id") and d.get("goal"):
            rows.append(d)
    return rows


def propose(rows: list[dict], *, min_support: int, min_precision: float) -> dict[str, list[dict]]:
    """Return {skill_id: [{token, support, accepts, precision}]} for tokens worth adding."""
    # (skill, token) -> [support, accepts]
    stat: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        sid = r["skill_id"]
        acc = bool(r.get("accepted"))
        for tok in set(_tokens(r["goal"])):
            stat[(sid, tok)][0] += 1
            if acc:
                stat[(sid, tok)][1] += 1

    out: dict[str, list[dict]] = defaultdict(list)
    for (sid, tok), (support, accepts) in stat.items():
        if support < min_support:
            continue
        precision = accepts / support if support else 0.0
        if precision >= min_precision:
            out[sid].append({"token": tok, "support": support, "accepts": accepts,
                             "precision": round(precision, 3)})
    for sid in out:
        out[sid].sort(key=lambda x: (-x["precision"], -x["support"], x["token"]))
    return out


def _existing_triggers(skill_id: str) -> tuple[Path | None, dict | None]:
    for p in REGISTRY.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("name") == skill_id and "triggers" in data:
            return p, data
    return None, None


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="routing telemetry JSONL")
    ap.add_argument("--min-support", type=int, default=3, help="min observations for a token")
    ap.add_argument("--min-precision", type=float, default=0.8, help="min acceptance rate to promote a token")
    ap.add_argument("--apply", action="store_true", help="add promoted tokens to the skills' triggers")
    args = ap.parse_args(argv)

    rows = _load_log(args.log)
    if not rows:
        print(f"learn-triggers: no telemetry in {args.log} "
              f"(enable with agent.skills.select(..., log_path=...)). Nothing to learn yet.")
        return 0

    proposals = propose(rows, min_support=args.min_support, min_precision=args.min_precision)
    if not proposals:
        print(f"learn-triggers: {len(rows)} rows, no token cleared "
              f"support>={args.min_support} & precision>={args.min_precision}.")
        return 0

    added_total = 0
    for sid, toks in sorted(proposals.items()):
        path, data = _existing_triggers(sid)
        have = set(t.lower() for t in (data.get("triggers", []) if data else []))
        fresh = [t for t in toks if t["token"] not in have]
        if not fresh:
            continue
        print(f"\n{sid}  ({path.name if path else 'no registry spec — forge/project first'}):")
        for t in fresh:
            print(f"  + {t['token']:<18} precision={t['precision']} support={t['support']}")
        if args.apply and path and data is not None:
            data["triggers"] = sorted(set(data.get("triggers", [])) | {t["token"] for t in fresh})
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            added_total += len(fresh)

    if args.apply:
        print(f"\nlearn-triggers: added {added_total} trigger token(s). "
              f"Re-run tools/build_skill_index.py to refresh the unified index.")
    else:
        print("\n(proposal only — re-run with --apply to add these triggers.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
