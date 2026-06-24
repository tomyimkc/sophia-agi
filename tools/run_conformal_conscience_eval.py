#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agent.conformal_gate import write_conformal_report
DEFAULT_IN=ROOT/"eval"/"conscience"/"calibration_v1.jsonl"
DEFAULT_OUT=ROOT/"agi-proof"/"conscience"/"conformal-conscience.public-report.json"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN)); ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--alpha", type=float, default=0.1); args=ap.parse_args()
    r=write_conformal_report(args.inp,args.out,alpha=args.alpha); print(json.dumps({"ok":r["ok"],"out":args.out,"risks":list(r["byRisk"])},indent=2)); return 0 if r["ok"] else 1
if __name__=="__main__": raise SystemExit(main())
