#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Issue → skill: forge/update a verifier-gated skill from logged failures.

This is the bridge the failure ledger was missing. It reads structured failure rows
(``agi-proof/failures.jsonl`` or stdin), groups them by task, and runs each group
through the EXISTING verifier gate (``tools.sophia_skill_forge.forge_skill`` ->
``gateway.skill_flywheel.synthesize_gate``). Nothing unproven ships: a skill is
written only if its synthesized verifier clears held-out validation; a rejection is a
valid, logged outcome — never tune to force a pass.

Failure row schema (one JSON object per line)::

    {"task": "danger", "text": "delete the production database", "label": true,
     "issue_id": "2026-06-29-dropped-prod", "note": "agent almost ran this"}

``label`` is the oracle: true = a skill SHOULD flag/accept this text, false = it should
not. Give each task both positive and negative rows (the gate needs >=4 examples with
both classes) so the verifier can actually separate them.

Default is DRY-RUN: it synthesizes in a throwaway dir and reports the verdict without
touching the committed registry. ``--apply`` forges into ``skills/generated/``, updates
``forge_index.json``, projects the promoted skill into a routable registry spec
(``tools/project_forge_to_registry.py``), and prints a ledger-ready line.

Run:
    python tools/skill_from_failure.py                 # dry-run on agi-proof/failures.jsonl
    python tools/skill_from_failure.py --apply         # actually forge + project
    cat rows.jsonl | python tools/skill_from_failure.py --stdin --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sophia_skill_forge import forge_skill  # noqa: E402

DEFAULT_FAILURES = ROOT / "agi-proof" / "failures.jsonl"


def _load_rows(path: Path | None, use_stdin: bool) -> list[dict]:
    if use_stdin:
        text = sys.stdin.read()
    elif path and path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        return []
    rows: list[dict] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "text" in obj and "label" in obj:
            rows.append(obj)
    return rows


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        task = str(r.get("task") or r.get("domain") or "issue")
        groups.setdefault(task, []).append(r)
    return groups


def _spec(task: str, rows: list[dict]) -> dict:
    descs = [r.get("note") or r.get("description") for r in rows if r.get("note") or r.get("description")]
    return {
        "task_id": task,
        "description": (descs[0] if descs else f"flag {task} cases seen in failures"),
        "examples": [{"text": str(r["text"]), "label": bool(r["label"])} for r in rows],
    }


def _classes_ok(rows: list[dict]) -> bool:
    labels = {bool(r["label"]) for r in rows}
    return len(rows) >= 4 and labels == {True, False}


def run(rows: list[dict], *, apply: bool, proposer_model: str = "mock") -> list[dict]:
    results: list[dict] = []
    for task, group in sorted(_group(rows).items()):
        spec = _spec(task, group)
        if not _classes_ok(group):
            results.append({"task": task, "status": "skipped",
                            "reason": f"need >=4 rows with both labels, got {len(group)} "
                                      f"labels={sorted({bool(r['label']) for r in group})}"})
            continue
        if apply:
            out = forge_skill(spec, proposer_model=proposer_model, update_registry=True)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                out = forge_skill(spec, out_root=Path(tmp), proposer_model=proposer_model,
                                  update_registry=False)
        promoted = bool(out.get("created"))
        results.append({
            "task": task, "status": "forged" if promoted else "rejected",
            "skill_id": out.get("skill_id"),
            "promoted": promoted,
            "n_examples": len(group),
            "validation": (out.get("rules") or [{}])[0].get("validation") if promoted else None,
            "report": out.get("promotion", {}),
        })
    return results


def _ledger_line(r: dict) -> str:
    if r["status"] == "forged":
        v = r.get("validation") or {}
        return (f"- skill-from-failure {r['skill_id']}: FORGED (verifier-gated), "
                f"n={r['n_examples']} val_acc={v.get('accuracy')} — routable via projected registry spec.")
    if r["status"] == "rejected":
        return (f"- skill-from-failure {r['skill_id']}: NO SKILL SHIPPED — verifier failed "
                f"held-out validation on {r['n_examples']} examples (valid outcome, not tuned).")
    return f"- skill-from-failure {r['task']}: SKIPPED — {r.get('reason')}"


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("failures", nargs="?", type=Path, default=DEFAULT_FAILURES,
                    help="JSONL of failure rows (default: agi-proof/failures.jsonl)")
    ap.add_argument("--stdin", action="store_true", help="read rows from stdin instead of a file")
    ap.add_argument("--apply", action="store_true",
                    help="actually forge into skills/generated + forge_index + project to registry")
    ap.add_argument("--proposer-model", default="mock", help="mock|off|deepseek|openrouter:...")
    args = ap.parse_args(argv)

    rows = _load_rows(None if args.stdin else args.failures, args.stdin)
    if not rows:
        src = "stdin" if args.stdin else str(args.failures)
        print(f"skill-from-failure: no usable failure rows in {src}. "
              f"Append rows like {{'task':..,'text':..,'label':true}} and retry.")
        return 0

    results = run(rows, apply=args.apply, proposer_model=args.proposer_model)

    forged = [r for r in results if r["status"] == "forged"]
    if args.apply and forged:
        # make every newly promoted skill routable by the agent harness
        from tools.project_forge_to_registry import build as project_build
        project_build(check=False)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== skill-from-failure ({mode}) ===")
    for r in results:
        print(_ledger_line(r))
    print(json.dumps({"mode": mode, "forged": len(forged),
                      "rejected": sum(r["status"] == "rejected" for r in results),
                      "skipped": sum(r["status"] == "skipped" for r in results)}, ensure_ascii=False))
    if args.apply and forged:
        print("\nLedger-ready lines appended above — record them in agi-proof/failure-ledger.md.")
    elif not args.apply:
        print("\n(dry-run: nothing written. Re-run with --apply to forge + project the promoted skills.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
