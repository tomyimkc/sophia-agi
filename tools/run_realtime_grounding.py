#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive the verifier-gated real-time grounding loop over a batch of live claims.

Two modes:

  --offline (default; CI-safe, no network)
    Uses the committed FixtureFactBackend so the full loop runs deterministically:
    retrieve -> fact-check -> conformal -> decontam/valid-time -> belief store.

  --online (opt-in; keyless live sources)
    Uses agent.live_sources.LiveFactBackend (Wikidata/Crossref/OpenAlex/macro),
    respecting the dataflow firewall. No API keys required. Never on by default so
    hidden/private prompts are not sent to third parties by accident.

    python3 tools/run_realtime_grounding.py --offline
    python3 tools/run_realtime_grounding.py --offline --as-of 2026-07-01 --eval-cutoff 2026-01-01
    python3 tools/run_realtime_grounding.py --online --claims data/realtime/demo_claims.jsonl

Honest scope: this only admits fact-checked live data to an external belief store;
it changes no weights and makes no capability claim (candidateOnly).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_grounding as rg  # noqa: E402
from agent.conformal_gate import fit_conformal_policy, load_jsonl  # noqa: E402
from agent.live_sources import FixtureFactBackend, LiveFactBackend  # noqa: E402

DEFAULT_CLAIMS = ROOT / "data" / "realtime" / "demo_claims.jsonl"
DEFAULT_FIXTURE = ROOT / "data" / "realtime" / "fixtures_v1.json"
DEFAULT_CALIB = ROOT / "data" / "realtime" / "conformal_calib.jsonl"
DEFAULT_STORE = ROOT / "agent" / "memory" / "realtime" / "belief_store.jsonl"
DEFAULT_OUT = ROOT / "agent" / "memory" / "realtime" / "grounding-report.json"


def _load_claims(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use committed FixtureFactBackend (default)")
    mode.add_argument("--online", action="store_true", help="use keyless LiveFactBackend (opt-in network)")
    ap.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB, help="conformal calibration rows (nonconformity/correct/risk)")
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage budget")
    ap.add_argument("--as-of", default=date.today().isoformat(), help="query as-of date for valid-time")
    ap.add_argument("--eval-cutoff", default="2026-01-01", help="frozen eval cutoff for temporal decontam")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--no-store", action="store_true", help="do not persist ingested rows")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mark-stale", action="store_true", help="after grounding, flag lapsed beliefs for re-verification")
    args = ap.parse_args()

    online = bool(args.online)
    if online:
        backend = LiveFactBackend()
    else:
        backend = FixtureFactBackend.from_file(args.fixture)

    policy = None
    if args.calib.exists():
        policy = fit_conformal_policy(load_jsonl(args.calib), alpha=args.alpha)

    claims = _load_claims(args.claims)
    store_path = None if args.no_store else args.store
    report = rg.run_grounding(
        claims,
        backend=backend,
        policy=policy,
        as_of=args.as_of,
        eval_cutoff=args.eval_cutoff,
        root=ROOT,
        store_path=store_path,
    )
    report["mode"] = "online" if online else "offline"
    if policy is not None:
        report["conformalPolicy"] = policy.to_dict()

    if args.mark_stale and store_path:
        report["reverify"] = rg.mark_stale(store_path, args.as_of)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = report["counts"]
    print(f"REAL-TIME GROUNDING [{report['mode']}] as-of={report['asOf']} cutoff={report['evalCutoff']} "
          f"evalSurface={report['evalSurfaceSize']}")
    print(f"  claims={report['nClaims']} counts={counts} writtenToStore={report['nWrittenToStore']}")
    for r in report["rows"]:
        print(f"  [{r['ingestState']:>11}] {r['claim'][:60]!r}  ({r['reason']})")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
