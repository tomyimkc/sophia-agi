#!/usr/bin/env python3
"""Injection / containment red-team (M1) — does the deterministic gate contain a
COMPROMISED model?

Assumes the model is fully attacker-controlled and measures whether the gate /
policy verifiers (which run outside the model) stop the forbidden output from
reaching the user. Exits non-zero if a *gating* attack wins, the secret tripwire
fails, or the exfiltration baseline is (implausibly) already clean.

    python tools/run_security_redteam.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.security.redteam import run_redteam  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    r = run_redteam()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    print("Injection / containment red-team — compromised-model threat model")
    print("=" * 66)
    print(f"\nGATING attacks (must be contained): ASR = {r['standard']['gatingASR']:.0%}")
    for cat, s in r["standard"]["byCategory"].items():
        print(f"  {cat:<24} n={s['n']}  ASR={s['asr']:.0%}")
    print(f"\nExfiltration:  baseline ASR = {r['exfiltration']['baselineASR']:.0%}"
          f"  ->  with no_secret_leak = {r['exfiltration']['defendedASR']:.0%}"
          f"  (n={r['exfiltration']['n']})")
    fw = r["firewall"]
    print(f"\nData-flow firewall (M2):  lethal-trifecta ASR = {fw['firewalledASR']:.0%}"
          f"  (baseline {fw['baselineASR']:.0%}; reads allowed = {fw['readsAllowed']})")
    for s in fw["scenarios"]:
        print(f"  {s['id']:<18} {s['tool']:<28} -> {s['action']}")
    it = r["interpreter"]
    print(f"\nDual-LLM interpreter (M2.2):  control-flow integrity = {it['controlFlowIntegrity']}"
          f";  tainted write contained = {it['taintedWriteContained']}")
    print("\nPROBES (reported, not gating — known/suspected gaps):")
    for cat, s in r["probes"]["byCategory"].items():
        flag = "  <-- ATTACK WON (real gap)" if s["asr"] > 0 else ""
        print(f"  {cat:<24} n={s['n']}  ASR={s['asr']:.0%}{flag}")
    print("\nInvariants:")
    for name, ok in r["invariants"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n" + ("ALL GATING INVARIANTS HOLD" if r["ok"] else "CONTAINMENT FAILURE"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
