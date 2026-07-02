#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A1 — Sophia Trajectory Packs: harness records -> (s,a,o,v) training trajectories.

Agents-A1 (arXiv 2606.30616 §2.2) shows the capability gain of horizon scaling
lives in trajectory SFT data whose every step carries a verifier outcome and
whose non-agent content is loss-masked. Sophia's harnesses ALREADY record all
of that — this tool serializes it as training data:

  * run_case-shaped records (tools/run_hidden_eval_sophia.py): prompt, answer,
    gate verdict, toolLog (action->observation), sources;
  * long-horizon event logs (tools/run_long_horizon.py): goal / tool_call /
    verification / self_correction events with per-step `passed`.

Acceptance gates = the paper's five criteria, applied fail-closed per record:
  verifiable        -> must carry >=1 verifier outcome (gate verdict or
                       verification event); else DROPPED
  valid             -> final verdict accepted / objective passed; FAILED
                       trajectories are not discarded — they are routed to the
                       DPO-negatives output (failures are preference evidence,
                       the paper keeps failures in its KAG for credit
                       assignment)
  process-informative-> >=2 action steps (a one-shot lookup teaches nothing
                       about the horizon); else DROPPED
  evidence-covering -> >=1 source/evidence reference (waivable per-domain via
                       --no-require-evidence); else DROPPED
  no-shortcut       -> NOT automatable here; every row is stamped
                       shortcutScreened:false until the A5 forge (or manual
                       review) screens it. Never silently claimed.

Output rows are chat-format (mlx --mask-prompt compatible): tool observations
ride as role:"tool" messages (masked by role), per-step verifier outcomes in
metadata.steps, so the pack composes with build_local_sophia_dataset
decontamination when registered as a source. candidateOnly:true throughout —
building a pack claims nothing; only a gated training run can.

Usage:
  PYTHONPATH=. python3 tools/build_trajectory_pack.py --cases cases.jsonl \
      --out training/trajectories/pack.jsonl
  PYTHONPATH=. python3 tools/build_trajectory_pack.py --long-horizon-events \
      agi-proof/long-horizon-runs/run-X.jsonl --out training/trajectories/lh.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sophia.trajectory_pack.v1"
ACCEPTED_VERDICTS = {"accepted", "pass", "passed", True}


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": str(content)}


def case_to_trajectory(rec: "dict[str, Any]", *, require_evidence: bool = True) -> "tuple[str, dict | None]":
    """Convert one run_case-shaped record. Returns (disposition, row|None).

    Dispositions: 'sft' | 'dpo_negative' | 'dropped_<criterion>'.
    """
    prompt = str(rec.get("prompt") or rec.get("question") or "").strip()
    answer = str(rec.get("answer") or "").strip()
    gate = rec.get("gate") or {}
    verdict = gate.get("verdict")
    tool_log = rec.get("toolLog") or []
    sources = rec.get("sources") or []
    if not prompt or not answer:
        return "dropped_no_text", None
    if verdict is None:
        return "dropped_not_verifiable", None
    steps = []
    messages = [_msg("user", prompt)]
    for t in tool_log:
        action = f"{t.get('tool', 'tool')}({json.dumps(t.get('args', {}), ensure_ascii=False)})"
        messages.append(_msg("assistant", f"ACTION: {action}"))
        messages.append(_msg("tool", str(t.get("output", ""))[:2000]))  # observation: masked by role
        steps.append({"action": action, "verdict": t.get("verdict", None)})
    if len(steps) < 2:
        return "dropped_not_process_informative", None
    if require_evidence and not sources:
        return "dropped_not_evidence_covering", None
    messages.append(_msg("assistant", answer))
    row = {
        "messages": messages,
        "metadata": {
            "schema": SCHEMA, "caseId": rec.get("id"),
            "steps": steps, "gateVerdict": verdict,
            "sources": sources[:20], "shortcutScreened": False,
            "labelStatus": "trajectory", "candidateOnly": True,
        },
    }
    return ("sft" if verdict in ACCEPTED_VERDICTS else "dpo_negative"), row


def long_horizon_to_trajectory(events: "list[dict[str, Any]]",
                               *, require_evidence: bool = False) -> "tuple[str, dict | None]":
    """Convert one long-horizon run's event stream (tools/run_long_horizon.py)."""
    goal = next((e for e in events if e.get("kind") == "goal"), None)
    if goal is None:
        return "dropped_no_text", None
    verifications = [e for e in events if e.get("kind") == "verification"]
    if not verifications:
        return "dropped_not_verifiable", None
    actions = [e for e in events if e.get("kind") in ("tool_call", "self_correction")]
    if len(actions) < 2:
        return "dropped_not_process_informative", None
    artifacts = [e for e in events if e.get("kind") == "artifact"]
    if require_evidence and not artifacts:
        return "dropped_not_evidence_covering", None
    messages = [_msg("user", str(goal.get("detail") or goal.get("goal") or "long-horizon goal"))]
    steps = []
    for e in actions:
        action = str(e.get("detail") or e.get("argv") or e.get("kind"))
        messages.append(_msg("assistant", f"ACTION: {action}"))
        obs = str(e.get("stdoutTail") or e.get("output") or "")[:2000]
        if obs:
            messages.append(_msg("tool", obs))
        steps.append({"action": action[:200],
                      "verdict": e.get("passed", None)})
    all_passed = all(bool(v.get("passed")) for v in verifications)
    messages.append(_msg("assistant",
                         "OBJECTIVE " + ("COMPLETE" if all_passed else "NOT MET")))
    row = {
        "messages": messages,
        "metadata": {"schema": SCHEMA, "steps": steps,
                     "verifications": [{"passed": bool(v.get("passed"))} for v in verifications],
                     "shortcutScreened": False, "labelStatus": "trajectory",
                     "candidateOnly": True},
    }
    return ("sft" if all_passed else "dpo_negative"), row


def build_pack(records: "list[dict[str, Any]]", *, mode: str = "cases",
               require_evidence: bool = True) -> "dict[str, Any]":
    sft, negatives, dispositions = [], [], {}
    if mode == "cases":
        items = [case_to_trajectory(r, require_evidence=require_evidence) for r in records]
    else:  # one long-horizon event stream per call
        items = [long_horizon_to_trajectory(records, require_evidence=require_evidence)]
    for disp, row in items:
        dispositions[disp] = dispositions.get(disp, 0) + 1
        if row is None:
            continue
        (sft if disp == "sft" else negatives).append(row)
    if not sft and not negatives:
        return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                "reason": "no record survived the acceptance gates (fail-closed)",
                "dispositions": dispositions}
    return {"schema": SCHEMA, "ok": True, "candidateOnly": True,
            "level3Evidence": False, "canClaimAGI": False,
            "sft": sft, "dpoNegatives": negatives, "dispositions": dispositions,
            "acceptanceGates": ["verifiable", "valid->sft/else dpo", "process-informative(>=2 steps)",
                                f"evidence-covering(required={require_evidence})",
                                "no-shortcut: NOT screened (shortcutScreened:false)"]}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A1 trajectory pack builder")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", type=Path, help="run_case-shaped records JSONL")
    src.add_argument("--long-horizon-events", type=Path, help="one run's event-log JSONL")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-require-evidence", action="store_true")
    args = ap.parse_args(argv)

    if args.cases:
        result = build_pack(load_jsonl(args.cases), mode="cases",
                            require_evidence=not args.no_require_evidence)
    else:
        result = build_pack(load_jsonl(args.long_horizon_events), mode="long-horizon",
                            require_evidence=not args.no_require_evidence)
    if not result["ok"]:
        print(json.dumps(result, indent=2))
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in result["sft"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    neg_path = args.out.with_name(args.out.stem + "_dpo_negatives.jsonl")
    with neg_path.open("w", encoding="utf-8") as fh:
        for row in result["dpoNegatives"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {k: v for k, v in result.items() if k not in ("sft", "dpoNegatives")}
    report["counts"] = {"sft": len(result["sft"]), "dpoNegatives": len(result["dpoNegatives"])}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
