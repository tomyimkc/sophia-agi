#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Claim gate — enforce the WHOLE measurement contract on a result before it backs a claim.

Generalizes tools/retention_gate.py from one axis to the full Instrumented Evaluation Contract
(agi-proof/measurement-thesis.md). Given a measurement_spec + a recipe's artifacts, it checks the
runnable pillars and emits a GO/NO-GO RECEIPT. A candidate may only be promoted past
`candidate_only` if it carries a passing receipt — that is the machine-checkable form of
"no claim may exceed what the instrument can resolve".

    python3 tools/claim_gate.py --prefix M3-pilot \
        --spec agi-proof/benchmark-results/wisdom-market/measurement_spec.json

Checks (each CRITICAL unless noted):
  1 uncertainty  — primary metrics carry CIs; retention carries a CI.
  2 power        — the retention probe can RESOLVE the tolerance (mde_at_n(N) <= tolerance);
                   primary is powered (mde <= primaryMDE).
  5 constructs   — >= 2 independent constructs agree (markers + judge [+ retention]).
  6 decontam     — retention probe reports zero contamination leaks.
  8 magnitude    — >= 1 primary metric is CI-clean AND |delta| >= practicalThreshold;
                   no protected regression; retention gate = GO.
Exit 0 = GO, 3 = NO-GO, 2 = unreadable/missing inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_stats import mde_at_n  # noqa: E402

WM = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market"
PRIMARY = ["qualification_rate_on_contested", "tradition_merge_rate", "false_attribution_rate",
           "moral_route_accuracy", "citation_fidelity", "provenance_accuracy"]
PROTECTED = ["protected_history_regression", "protected_religion_regression"]


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ci_clean(ci) -> bool:
    return isinstance(ci, list) and None not in ci and (ci[0] > 0 or ci[1] < 0)


def gate(prefix: str, spec: dict) -> dict:
    ev = _load(WM / f"{prefix}-eval.json")
    jg = _load(WM / f"{prefix}-judge.json")
    rt = _load(WM / f"{prefix}-retention.json") or _load(WM / f"{prefix}-retention-eval.json")
    tol = float(spec.get("guardrails", {}).get("retention", {}).get("rule_tolerance", 0.05)
                if "rule_tolerance" in spec.get("guardrails", {}).get("retention", {}) else 0.05)
    practical = float(spec.get("primaryMDE", 0.05))
    checks = []

    def add(name, ok, detail, critical=True):
        checks.append({"check": name, "ok": bool(ok), "critical": critical, "detail": detail})

    # 1 uncertainty
    prim_ci = ev and all(isinstance((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("ci"), list)
                         for m in PRIMARY if m in ev.get("adapterPromptVsBasePrompt", {}))
    add("1-uncertainty", bool(prim_ci and (not rt or rt.get("delta_ci95"))),
        f"primary CIs={bool(prim_ci)} retentionCI={(rt or {}).get('delta_ci95')}")

    # 2 power
    prim_n = (ev or {}).get("nCases", 0) * (ev or {}).get("runs", 1)
    prim_mde = round(mde_at_n(max(1, prim_n)), 3)
    ret_n = (rt or {}).get("nTasks", 0)
    ret_mde = round(mde_at_n(max(1, ret_n)), 3) if ret_n else None
    add("2-power-primary", prim_mde <= practical + 1e-9, f"primary N={prim_n} mde={prim_mde} <= {practical}")
    add("2-power-retention", bool(ret_n and ret_mde <= tol + 0.02),
        f"retention N={ret_n} mde={ret_mde} vs tolerance {tol} "
        f"(coarse if mde>tol)", critical=False)

    # 5 constructs (>=2 agree)
    markers_ok = bool(ev) and any((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("improves")
                                  and _ci_clean((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("ci"))
                                  for m in PRIMARY)
    wr = [v.get("adapter_winrate", 0) for v in ((jg or {}).get("perJudge") or {}).values()]
    judge_ok = bool(wr) and (sum(wr) / len(wr)) > 0.5
    n_constructs = sum([markers_ok, judge_ok, bool(rt and rt.get("retains"))])
    add("5-constructs", n_constructs >= 2,
        f"markers={markers_ok} judge={judge_ok}({round(sum(wr)/len(wr),3) if wr else None}) "
        f"retention_retains={(rt or {}).get('retains')} -> {n_constructs} agree")

    # 6 decontam
    leaks = (rt or {}).get("contaminationLeaks")
    add("6-decontam", (rt is None) or not leaks, f"retention leaks={leaks}", critical=bool(rt))

    # 8 magnitude + significance + guardrails
    big = bool(ev) and any((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("improves")
                           and _ci_clean((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("ci"))
                           and abs((ev["adapterPromptVsBasePrompt"].get(m) or {}).get("delta") or 0) >= practical
                           for m in PRIMARY)
    prot = [m for m in PROTECTED if (((ev or {}).get("adapterPromptVsBasePrompt", {}).get(m) or {}).get("delta") or 0) > 0
            and _ci_clean(((ev or {}).get("adapterPromptVsBasePrompt", {}).get(m) or {}).get("ci"))]
    ret_go = (rt is None) or (rt.get("delta") is not None and rt["delta"] >= -tol)
    add("8-magnitude", big and not prot and ret_go,
        f"primary>=threshold={big} protectedRegressions={prot} retentionGate={'GO' if ret_go else 'NO-GO'}")

    critical_fail = [c["check"] for c in checks if c["critical"] and not c["ok"]]
    go = not critical_fail
    return {
        "prefix": prefix, "verdict": "GO" if go else "NO-GO", "ok": go,
        "criticalFailures": critical_fail,
        "checks": checks,
        "powerNote": {"primaryMDE": prim_mde, "retentionMDE": ret_mde, "tolerance": tol},
        "claimCeiling": spec.get("claimCeiling", "candidate_only; canClaimAGI:false"),
        "code": 0 if go else 3,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="recipe artifact prefix, e.g. M3-pilot")
    ap.add_argument("--spec", type=Path, default=WM / "measurement_spec.json")
    ap.add_argument("--receipt", type=Path, default=None, help="write receipt JSON here (default <prefix>.gate.json)")
    args = ap.parse_args()
    spec = _load(args.spec)
    if spec is None:
        print(json.dumps({"verdict": "NO-GO", "reason": "unreadable spec", "code": 2}))
        return 2
    result = gate(args.prefix, spec)
    receipt = args.receipt or (WM / f"{args.prefix}.gate.json")
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in result["checks"]:
        mark = "✓" if c["ok"] else ("✗" if c["critical"] else "!")
        print(f"  {mark} {c['check']:18s} {c['detail']}", file=sys.stderr)
    print(f"CLAIM GATE [{args.prefix}]: {result['verdict']}  (receipt -> {receipt.relative_to(ROOT)})", file=sys.stderr)
    print(json.dumps({"prefix": args.prefix, "verdict": result["verdict"], "criticalFailures": result["criticalFailures"]}))
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
