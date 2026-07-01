#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 benchmark for claim C1 (verifier-as-truth-filter) of the real-time grounding loop.

Offline + deterministic by default: labeled `eval/fact_check/heldout_v1.jsonl` scored with
the committed `FixtureFactBackend`. Reports admission precision/recall/fabricationRate per
arm, the verifier-vs-raw-RAG contrast, a control-sanity guard, and an HONEST power verdict
(underpowered at the committed N — a floor, not a GO). Pre-registration lives in
`data/realtime/benchmark/measurement_spec.json`.

    python3 tools/run_realtime_benchmark.py --offline
    python3 tools/run_realtime_benchmark.py --offline --threshold 0.10 --alpha 0.1
    python3 tools/run_realtime_benchmark.py --online   # Phase-1 seam (keyless LiveFactBackend)

Honest scope: C1 floor only. C2 (live passAt1) and C3 (drift/forgetting) need a model and
are Phase-1/3. canClaimAGI stays false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_benchmark as rb  # noqa: E402
from agent.conformal_gate import fit_conformal_policy, load_jsonl  # noqa: E402
from agent.live_sources import FixtureFactBackend, LiveFactBackend  # noqa: E402

DEFAULT_PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
DEFAULT_FIXTURE = ROOT / "eval" / "fact_check" / "fixtures_v1.json"
DEFAULT_CALIB = ROOT / "data" / "realtime" / "conformal_calib.jsonl"
DEFAULT_OUT = ROOT / "agent" / "memory" / "realtime" / "benchmark-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="use committed FixtureFactBackend (default)")
    mode.add_argument("--online", action="store_true", help="Phase-1 seam: keyless LiveFactBackend")
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage budget")
    ap.add_argument("--threshold", type=float, default=0.10, help="practical MDE for the power verdict")
    ap.add_argument("--as-of", default="2026-07-01")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    backend = LiveFactBackend() if args.online else FixtureFactBackend.from_file(args.fixture)
    policy = fit_conformal_policy(load_jsonl(args.calib), alpha=args.alpha) if args.calib.exists() else None
    rows = rb.load_pack(args.pack)

    report = rb.run_c1_benchmark(
        rows,
        backend=backend,
        policy=policy,
        eval_prompts=None,   # benchmarking the verifier ON the held-out; no self-decontam
        eval_cutoff=None,    # held-out has no source timestamps; temporal gate is a Phase-1 arm
        as_of=args.as_of,
        practical_threshold=args.threshold,
        live_backend=bool(args.online),
    )
    report["mode"] = "online" if args.online else "offline"
    rb.write_report(report, args.out)

    c = report["controlSanity"]
    print(f"REAL-TIME BENCHMARK C1 [{report['mode']}]  N={report['n']}  labels={report['labelCounts']}")
    print(f"  control-sanity: knownTrueAdmit={c['knownTrueAdmitRate']} knownFalseReject={c['knownFalseRejectRate']} ok={c['ok']}")
    print(f"  {'arm':<13} {'prec':>6} {'recall':>7} {'f1':>6} {'fabric':>7} {'admAcc':>7}")
    for arm in rb.ARMS:
        m = report["arms"][arm]
        print(f"  {arm:<13} {m['precision']:>6} {m['recall']:>7} {m['f1']:>6} {m['fabricationRate']:>7} {m['admissionAccuracy']:>7}")
    vvr = report["verifierVsRawRag"]
    print(f"  full vs raw_rag: ΔadmAcc={vvr['deltaAdmissionAccuracy']}  McNemar p={vvr['mcnemar']['p']}  "
          f"power={vvr['powerVerdict']['verdict']} (MDE={vvr['powerVerdict']['mde']})")
    print(f"  VERDICT: {report['verdict']['label']} — {report['verdict']['reason']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
