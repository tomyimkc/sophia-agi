#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ingest a real RLVR adapter-eval report and run it through the SSIL Layer-1 gate.

Turns a trained weight delta's held-out eval into an SSIL verdict — the single
command that makes Layer 1 real instead of mock. It reads the JSON produced by

    python3 tools/eval_rlvr_adapter.py --mode real --adapter <ckpt> --out <report>

and maps it onto `agent/ssil_layer1.adapter_candidate`, then runs the IDENTICAL
orchestrator (G2/G4/G5/G6) that gates skills:

  before/after capability  <- base.meanReward / adapterScore.meanReward   (G4 headline)
  protected integrity      <- 1 - trueFalsePositiveRate  (lower FP = higher integrity;
                              a rise in false-positives shows up as a protected regression)
  contamination            <- entityIntersection non-empty -> G4 reject

Fail-closed: if the report lacks a before/after capability pair, this errors with a
clear message instead of inventing numbers. Output carries candidateOnly /
canClaimAGI=false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_layer1 import adapter_candidate, run_layer1  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-layer1-real.public-report.json"

# The durable, committed evidence archive. Re-gating a file from here would decouple the
# gate from the actual training (a fresh run could "promote" by re-reading stale numbers).
# Defense-in-depth: refuse it structurally, so even a broad glob upstream cannot reintroduce
# the stale-promote bug. Only freshly copied-back pod reports (runpod-rlvr/) may be gated.
ARCHIVE_DIR_NAME = "rlvr-replication"


def _dig(report: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first present value among dotted key paths; None if none found."""
    for path in paths:
        cur: Any = report
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def map_report(report: dict[str, Any], *, adapter_id: str | None = None) -> dict[str, Any]:
    task = report.get("task") or "provenance"
    # Capability metric: passAt1 for math/code (objective, ungameable per-item reward),
    # meanReward for provenance (the FP-aware aggregate reward).
    if task in ("math", "code"):
        before = _dig(report, ("base", "passAt1"), ("baseScore", "passAt1"))
        after = _dig(report, ("adapterScore", "passAt1"), ("adapter", "passAt1"))
        capability_metric = "passAt1"
        contam_key = "familyIntersection"
    else:
        before = _dig(report, ("base", "meanReward"), ("baseScore", "meanReward"))
        after = _dig(report, ("adapterScore", "meanReward"), ("adapter", "meanReward"))
        capability_metric = "meanReward"
        contam_key = "entityIntersection"
    if before is None or after is None:
        raise SystemExit(
            f"ERROR: report has no before/after capability pair for task={task} "
            f"(expected base.{capability_metric} + adapterScore.{capability_metric}). "
            "Produce it with: python3 tools/eval_rlvr_adapter.py --mode real --adapter <ckpt>"
        )
    entity_intersection = (
        _dig(report, (contam_key,), ("split", contam_key), ("split", "entityIntersection")) or []
    )
    contaminated = bool(entity_intersection)
    aid = adapter_id or _dig(report, ("adapter",)) or report.get("model") or "rlvr-adapter"
    aid = str(aid).rsplit("/", 1)[-1]
    # Protected-integrity axis, task-aware. Provenance has a false-positive rate (lower better
    # -> invert to integrity); a rise in FP registers as a protected regression. Fail-closed: a
    # MISSING FP rate is unverified integrity — refuse rather than assume perfect. Code/math
    # reward is per-item objective (interpreter / sympy = ground truth), so there is NO separate
    # false-positive axis to protect — the protected floor is a no-op (1.0); the gate's teeth
    # for those tasks are the capability delta + contamination + the solver-checked invariant
    # suite. (A per-item-objective reward cannot be "gamed into passing" without actually
    # passing — that is exactly why these tasks are RLVR-friendly.)
    if task == "provenance":
        base_fp = _dig(report, ("base", "trueFalsePositiveRate"), ("baseScore", "trueFalsePositiveRate"))
        adapter_fp = _dig(report, ("adapterScore", "trueFalsePositiveRate"), ("adapter", "trueFalsePositiveRate"))
        if base_fp is None or adapter_fp is None:
            raise SystemExit(
                "ERROR: provenance report is missing trueFalsePositiveRate on base and/or adapter. "
                "The Layer-1 gate is fail-closed: unverified integrity cannot promote. "
                "Re-run tools/eval_rlvr_adapter.py so the eval reports false-positive rates."
            )
        prot_before = round(1.0 - float(base_fp), 4)
        prot_after = round(1.0 - float(adapter_fp), 4)
        protected_axis = "1_minus_trueFalsePositiveRate"
    else:
        prot_before = prot_after = 1.0
        protected_axis = "none_objective_reward_no_fp_axis"
    mapped = {
        "id": aid, "task": task, "before": float(before), "after": float(after),
        "capabilityMetric": capability_metric,
        "protected_before": prot_before, "protected_after": prot_after,
        "protectedAxis": protected_axis,
        "contaminated": contaminated, "entityIntersection": entity_intersection,
    }
    # Capability-panel deltas (attribution / hallucination / calibration), if the
    # report carries them (eval_rlvr_adapter.py --capability-panel). Fail-OPEN on
    # absence: old reports and non-panel runs still ingest exactly as before; the
    # legacy G4/G5 headline (meanReward + FP integrity) is unaffected by these.
    panel = report.get("capabilityPanel")
    if isinstance(panel, dict) and panel.get("delta"):
        pdelta = panel["delta"]
        mapped["capabilityPanelDelta"] = {
            "verdictAccuracy": pdelta.get("verdictAccuracy"),
            "hallucinationRate": pdelta.get("hallucinationRate"),
            "integrityRecall": pdelta.get("integrityRecall"),
            "calibrationScore": pdelta.get("calibrationScore"),
            "fabricationRate": pdelta.get("fabricationRate"),
        }
    return mapped


def ingest(report_path: str | Path, *, adapter_id: str | None = None) -> dict[str, Any]:
    p = Path(report_path)
    if ARCHIVE_DIR_NAME in p.parts:
        raise SystemExit(
            f"ERROR: refusing to gate a file under {ARCHIVE_DIR_NAME}/ — that is the durable, "
            "committed evidence archive, not a fresh pod result. Re-gating it would decouple the "
            "gate from the actual training (stale-promote). Point this at the fresh "
            "runpod-rlvr/ report produced by tools/eval_rlvr_adapter.py."
        )
    report = json.loads(p.read_text(encoding="utf-8"))
    m = map_report(report, adapter_id=adapter_id)
    candidate = adapter_candidate(
        m["id"], before=m["before"], after=m["after"],
        protected_before=m["protected_before"], protected_after=m["protected_after"],
        contaminated=m["contaminated"],
    )
    record = run_layer1(candidate)
    record["sourceReport"] = str(report_path)
    record["mappedMetrics"] = m
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest an RLVR adapter-eval report into the SSIL Layer-1 gate")
    ap.add_argument("report", help="path to *.rlvr.adapter-eval*.json (from tools/eval_rlvr_adapter.py)")
    ap.add_argument("--adapter-id", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--hardened", action="store_true",
                    help="also run the feedable SSIL hardening gates (GUARD/GOOD, +G8 if a validated "
                         "probe exists) and combine worst-wins with Layer-1; lists the pending gates")
    ap.add_argument("--registry", default=None,
                    help="registry.jsonl path; canonical-promote entries seed GUARD rollback targets")
    args = ap.parse_args(argv)

    record = ingest(args.report, adapter_id=args.adapter_id)

    # Optional hardening pass (default OFF so the live ingest path is unchanged). Additive:
    # attaches a combined fail-closed verdict over the gates a provenance eval can actually
    # feed, and lists the rest as pending. Only --hardened changes the exit code.
    hardened = None
    if args.hardened:
        from agent.ssil_ingest_hardened import harden_from_report

        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        hardened = harden_from_report(report, record, adapter_id=args.adapter_id, registry_path=args.registry)
        record["hardened"] = hardened

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    m = record["mappedMetrics"]
    if args.print:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"adapter={m['id']} capability {m['before']}->{m['after']} "
          f"integrity {m['protected_before']}->{m['protected_after']} contaminated={m['contaminated']}")
    print(f"SSIL VERDICT={record['verdict']} blocking={record['blockingGates']} -> {args.out}")
    if hardened is not None:
        print(f"HARDENED combinedVerdict={hardened['combinedVerdict']} "
              f"enforced={hardened['enforcedGates']} pending={sorted(hardened['pendingGates'])}")
        # Under --hardened the combined fail-closed verdict is authoritative for the exit code.
        return 0 if hardened["combinedVerdict"] == "promote" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
