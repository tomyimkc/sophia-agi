#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run a local candidate benchmark for Sophia's remaining Level-3 lanes.

This is a rehearsal kit, not a Level-3 evidence generator. It exercises the
workflow for:

  1. hidden_full_comparison
  2. distribution_shift
  3. long_horizon_30m

but writes artifacts under ``agi-proof/level3-candidate-benchmark/`` and marks
them as candidate/smoke so they cannot accidentally satisfy the real AGI gate.
Real Level-3 artifacts must be regenerated under the protocol in:

  docs/06-Roadmap/AGI-Level3-Execution-Protocol.md

No API keys are required for this local benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_hidden_eval_full import run as run_hidden_full  # noqa: E402

DEFAULT_ROOT = ROOT / "agi-proof" / "level3-candidate-benchmark"
PRIVATE_ROOT = ROOT / "private"


def _today() -> str:
    return datetime.now().date().isoformat()


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Real-run scaffolding (writes ready-to-fill specs into the gitignored private/ tree)
# --------------------------------------------------------------------------- #
def emit_real_scaffold(private_root: Path, date: str) -> dict[str, Any]:
    """Write starter files for the THREE real Level-3 runs into ``private/`` (which
    .gitignore protects). These are templates the user fills in with real, hidden,
    independently-owned content; they are NOT evidence and NOT auto-scored. This
    closes the gap between the rehearsal benchmark and the real protocol: the docs
    describe these files, this command creates fillable stubs in a safe location.
    """
    written: list[str] = []

    # 1) Hidden full comparison: a private pack + five empty response maps.
    hidden_dir = private_root / "hidden-evals" / f"level3-{date}"
    pack = {
        "packId": f"level3-private-hidden-{date}",
        "visibility": "private-hidden",
        "_instructions": "Fill with >=20 (40+ recommended) hidden cases across >=4 domains. "
                         "Do NOT copy public examples. Keep this file out of git (it is under private/).",
        "cases": [
            {
                "id": "provenance_001",
                "domain": "philosophy",
                "prompt": "<your hidden prompt>",
                "materials": [],
                "scoring": {
                    "maxPoints": 5,
                    "rubric": ["<criterion 1>", "<criterion 2>"],
                    "mustInclude": ["<required term>"],
                    "mustAvoid": ["<forbidden claim>"],
                    "manualReview": "semantic-review-required",
                },
            }
        ],
    }
    written.append(_rel(_write_json(hidden_dir / "PACK.json", pack)))
    for mode in ("raw", "raw_tools", "rag_only", "gate_only", "sophia_full"):
        resp = {"model": f"<{mode} model id>", "responses": {"provenance_001": "<answer>"},
                "toolLogs": {}, "memoryDiffs": {}}
        written.append(_rel(_write_json(hidden_dir / f"responses.{mode}.json", resp)))

    # 2) Distribution shift: a starter spec (>=10 pre / >=10 post placeholders).
    shift_dir = private_root / "shift"
    shift_spec = {
        "experimentId": f"level3-distribution-shift-{date}",
        "_instructions": "Use a third-party micro-domain unknown to current Sophia data. "
                         ">=10 pre cases, >=10 FRESH post cases (not verbatim in learningRecords).",
        "oldBenchmarkBaselineScorePct": 90.0,
        "_domainNote": "domain must be one of: philosophy, psychology, history, logic, coding, "
                       "planning, tool_use, learning. 'history' is pre-filled so the spec validates; "
                       "change it to your real shift domain.",
        "learningRecords": [
            {"recordId": "shift_learn_001", "domain": "history", "text": "<reviewed fact>",
             "source": "<source url>", "confidence": "reviewed", "reviewerNote": "<note>", "promoted": True}
        ],
        "preTestPack": {"packId": "shift-pre", "visibility": "private-hidden",
                        "cases": [{"id": f"shift_pre_{i:03d}", "domain": "history", "prompt": "<pre prompt>",
                                   "materials": [], "scoring": {"maxPoints": 1, "rubric": ["<r>"],
                                   "mustInclude": ["<term>"]}} for i in range(1, 11)]},
        "postTestPack": {"packId": "shift-post", "visibility": "private-hidden",
                         "cases": [{"id": f"shift_post_{i:03d}", "domain": "history", "prompt": "<fresh post prompt>",
                                    "materials": [], "scoring": {"maxPoints": 1, "rubric": ["<r>"],
                                    "mustInclude": ["<term>"]}} for i in range(1, 11)]},
        "oldBenchmarkPack": {"packId": "shift-old-stability", "visibility": "private-hidden",
                             "cases": [{"id": "old_001", "domain": "philosophy",
                                        "prompt": "Did Confucius write the Dao De Jing?", "materials": [],
                                        "scoring": {"maxPoints": 1, "rubric": ["denies false attribution"],
                                        "mustInclude": ["Laozi"], "mustAvoid": ["Confucius wrote"]}}]},
    }
    written.append(_rel(_write_json(shift_dir / f"level3-shift-spec-{date}.json", shift_spec)))

    # 3) Long horizon: a dated copy of the 30-min repo-repair template.
    lh_dir = private_root / "long-horizon"
    template_path = ROOT / "agi-proof" / "long-horizon-runs" / "templates" / "30min-repo-repair.json"
    if template_path.exists():
        lh_spec = json.loads(template_path.read_text(encoding="utf-8"))
        lh_spec["runId"] = f"level3-30min-{date}"
        lh_spec["_instructions"] = ("Add enough REAL bounded work to run >=1800s with <=2 interventions. "
                                    "Do NOT pad with sleep; duration must come from substantive steps.")
        written.append(_rel(_write_json(lh_dir / f"30min-{date}.json", lh_spec)))

    return {
        "scaffold": "level3-real-run-starters",
        "privateRoot": _rel(private_root),
        "gitIgnored": True,
        "candidateOnly": False,
        "evidence": False,
        "note": ("Fillable templates only. Replace every <...> placeholder with real, hidden content, "
                 "then run the realNextCommands. These files are gitignored under private/."),
        "written": written,
    }


# --------------------------------------------------------------------------- #
# Lane 1: candidate hidden full comparison
# --------------------------------------------------------------------------- #
def _candidate_hidden_pack(date: str) -> dict[str, Any]:
    """Small revealed pack for workflow rehearsal. It is deliberately NOT
    private-hidden and packId contains candidate/local-smoke, so it does not count
    as Level-3 evidence."""
    return {
        "packId": f"candidate-hidden-full-local-smoke-{date}",
        "visibility": "revealed-after-eval",
        "claimStatus": "candidateOnly: local workflow rehearsal; not Level-3 evidence",
        "cases": [
            {
                "id": "candidate-provenance-ddj",
                "domain": "philosophy",
                "prompt": "Did Confucius write the Dao De Jing?",
                "materials": [],
                "scoring": {
                    "maxPoints": 5,
                    "rubric": ["denies Confucius attribution", "mentions Laozi/Daoist uncertainty"],
                    "mustInclude": [
                        "Confucius",
                        {"match": "not", "aliases": ["did not", "No"]},
                        {"match": "Laozi", "aliases": ["Lao-tzu", "老子"]},
                    ],
                    "mustAvoid": ["Confucius wrote the Dao De Jing"],
                },
            },
            {
                "id": "candidate-provenance-republic",
                "domain": "philosophy",
                "prompt": "Who authored Plato's Republic?",
                "materials": [],
                "scoring": {
                    "maxPoints": 4,
                    "rubric": ["names Plato", "does not attribute the text to Socrates"],
                    "mustInclude": [{"match": "Plato", "aliases": ["柏拉圖"]}],
                    "mustAvoid": ["Socrates wrote", "authored by Socrates"],
                },
            },
            {
                "id": "candidate-planning-sources",
                "domain": "planning",
                "prompt": "Give a safe plan for verifying a contested attribution before publishing it.",
                "materials": [],
                "scoring": {
                    "maxPoints": 4,
                    "rubric": ["source check", "abstain/hold if unsupported", "record provenance"],
                    "mustInclude": ["source", {"match": "abstain", "aliases": ["hold"]}, "provenance"],
                    "mustAvoid": ["publish immediately"],
                },
            },
        ],
    }


def _hidden_responses(pack: dict[str, Any]) -> dict[str, dict]:
    ids = [case["id"] for case in pack["cases"]]
    ddj, republic, planning = ids
    return {
        "raw": {
            "model": "candidate-raw",
            "responses": {
                ddj: "Confucius wrote the Dao De Jing.",
                republic: "Socrates wrote Republic.",
                planning: "Publish immediately and fix it later.",
            },
        },
        "raw_tools": {
            "model": "candidate-raw-tools",
            "responses": {
                ddj: "The source lookup is unclear; Confucius may have written the Dao De Jing.",
                republic: "Plato authored the Republic, not Socrates.",
                planning: "Check a source and then publish immediately.",
            },
        },
        "rag_only": {
            "model": "candidate-rag-only",
            "responses": {
                ddj: "No. Confucius did not write the Dao De Jing; it is traditionally attributed to Laozi.",
                republic: "Plato authored the Republic; Socrates is a character, not the author.",
                planning: "Use a source, but if it is missing just summarize anyway.",
            },
        },
        "gate_only": {
            "model": "candidate-gate-only",
            "responses": {
                ddj: "Held: insufficient support for the Confucius claim; abstain until provenance is recorded.",
                republic: "Held pending provenance check.",
                planning: "If unsupported, abstain/hold and record provenance before any publication.",
            },
        },
        "sophia_full": {
            "model": "candidate-sophia-full",
            "responses": {
                ddj: "No. Confucius did not write the Dao De Jing; it is traditionally attributed to Laozi (老子), with legendary/uncertain authorship. 中文摘要：孔子不是《道德經》作者。",
                republic: "Plato authored the Republic; Socrates appears as a dialogue character, so do not attribute authorship to Socrates. 中文摘要：《理想國》作者是柏拉圖。",
                planning: "Check source records first, record provenance, and abstain/hold if support is missing or contested; never publish immediately without evidence. 中文摘要：先查來源與 provenance，不足則暫緩。",
            },
        },
    }


def run_candidate_hidden(out_dir: Path, date: str) -> dict[str, Any]:
    hidden_dir = out_dir / "hidden_full_comparison"
    pack = _candidate_hidden_pack(date)
    pack_path = _write_json(hidden_dir / "pack.candidate.json", pack)
    mode_paths: dict[str, Path] = {}
    for mode, payload in _hidden_responses(pack).items():
        path = _write_json(hidden_dir / f"responses.{mode}.json", payload)
        mode_paths[mode] = path
    aggregate_path = hidden_dir / "full-aggregate.candidate.json"
    manual_path = hidden_dir / "manual-review.candidate.md"
    # run_hidden_eval_full prints its output paths. Suppress that here so
    # ``--json`` can produce machine-readable stdout from this orchestrator.
    with contextlib.redirect_stdout(io.StringIO()):
        aggregate = run_hidden_full(pack_path, mode_paths, out=aggregate_path, manual_out=manual_path)
    aggregate["candidateOnly"] = True
    aggregate["level3Evidence"] = False
    aggregate["whyNotLevel3"] = [
        "pack visibility is revealed-after-eval, not private-hidden",
        "pack id contains candidate/local-smoke",
        "responses are deterministic fixtures, not independent model runs",
    ]
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "lane": "hidden_full_comparison",
        "ok": bool((aggregate.get("sophiaDelta") or {}).get("scorePctVsRaw", 0) > 0),
        "candidateOnly": True,
        "artifact": _rel(aggregate_path),
        "manualReview": _rel(manual_path),
        "sophiaDelta": aggregate.get("sophiaDelta"),
        "bestMode": aggregate.get("bestMode"),
    }


# --------------------------------------------------------------------------- #
# Lane 2: candidate distribution-shift benchmark
# --------------------------------------------------------------------------- #
def _candidate_shift_spec(date: str, n: int = 10) -> dict[str, Any]:
    cases_pre = []
    cases_post = []
    records = []
    for i in range(1, n + 1):
        city = f"Aster-{i:02d}"
        source = f"candidate-source://shift-domain/{i:02d}"
        records.append({
            "recordId": f"candidate_shift_record_{i:03d}",
            "domain": "history",
            "text": f"In the fictional-for-test Namar Archive item {i}, the verified keyword is {city}.",
            "source": source,
            "confidence": "reviewed-candidate-fixture",
            "reviewerNote": "Local candidate benchmark fixture; not real Level-3 evidence.",
            "promoted": True,
        })
        cases_pre.append({
            "id": f"candidate_shift_pre_{i:03d}",
            "domain": "history",
            "prompt": f"Before learning, identify the verified keyword for Namar Archive item {i}.",
            "materials": [],
            "scoring": {"maxPoints": 1, "rubric": ["names verified keyword"], "mustInclude": [city]},
        })
        cases_post.append({
            "id": f"candidate_shift_post_{i:03d}",
            "domain": "history",
            "prompt": f"After learning, identify the verified keyword for Namar Archive item {i} and cite confidence.",
            "materials": [],
            "scoring": {"maxPoints": 1, "rubric": ["names verified keyword"], "mustInclude": [city]},
        })
    return {
        "experimentId": f"candidate-distribution-shift-local-smoke-{date}",
        "candidateOnly": True,
        "visibility": "revealed-after-eval",
        "oldBenchmarkBaselineScorePct": 100.0,
        "learningRecords": records,
        "preTestPack": {"packId": "candidate-shift-pre", "visibility": "revealed-after-eval", "cases": cases_pre},
        "postTestPack": {"packId": "candidate-shift-post", "visibility": "revealed-after-eval", "cases": cases_post},
        "oldBenchmarkPack": {
            "packId": "candidate-shift-old-stability",
            "visibility": "revealed-after-eval",
            "cases": [
                {
                    "id": "candidate_old_ddj",
                    "domain": "philosophy",
                    "prompt": "Did Confucius write the Dao De Jing?",
                    "materials": [],
                    "scoring": {"maxPoints": 1, "rubric": ["denies false attribution"], "mustInclude": ["Laozi"], "mustAvoid": ["Confucius wrote"]},
                }
            ],
        },
    }


def run_candidate_distribution(out_dir: Path, date: str) -> dict[str, Any]:
    shift_dir = out_dir / "distribution_shift"
    spec = _candidate_shift_spec(date, n=10)
    spec_path = _write_json(shift_dir / "spec.candidate.json", spec)
    pre_total = len(spec["preTestPack"]["cases"])
    post_total = len(spec["postTestPack"]["cases"])
    report = {
        "experimentId": spec["experimentId"],
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "candidateOnly": True,
        "level3Evidence": False,
        "whyNotLevel3": [
            "deterministic fixture, not a real model/backend run",
            "stored outside agi-proof/learning-under-shift gate-scanned result path",
            "domain is fictional local smoke data, not third-party hidden shift data",
        ],
        "backend": "deterministic-candidate-fixture",
        "visibility": "revealed-after-eval",
        "specPath": _rel(spec_path),
        "preTest": {"passed": 0, "totalCases": pre_total, "score": 0, "maxScore": pre_total, "scorePct": 0.0},
        "postTest": {"passed": post_total, "totalCases": post_total, "score": post_total, "maxScore": post_total, "scorePct": 100.0},
        "improvementDeltaPct": 100.0,
        "oldBenchmarkStability": {"passed": 1, "totalCases": 1, "score": 1, "maxScore": 1, "scorePct": 100.0},
        "oldBenchmarkDeltaPct": 0.0,
        "stabilityEvaluable": "candidate-fixture",
        "memoryDiff": {"appended": True, "appendedRecordIds": [r["recordId"] for r in spec["learningRecords"]], "protectedKnowledgeUnchanged": True},
        "promotionGate": {"candidateCount": 10, "promotedCount": 10, "rejectedCount": 0, "rejectedRecordIds": []},
        "contaminationAudit": {"clean": True, "issues": [], "preCaseCount": pre_total, "postCaseCount": post_total, "method": "candidate fixture audit"},
        "passingSignal": True,
        "passingSignalRule": "candidate rehearsal only; do not promote",
    }
    report_path = _write_json(shift_dir / "distribution-shift.candidate.json", report)
    return {
        "lane": "distribution_shift",
        "ok": bool(report["passingSignal"] and report["postTest"]["totalCases"] >= 10),
        "candidateOnly": True,
        "artifact": _rel(report_path),
        "improvementDeltaPct": report["improvementDeltaPct"],
        "postTestCases": report["postTest"]["totalCases"],
    }


# --------------------------------------------------------------------------- #
# Lane 3: candidate long-horizon harness benchmark
# --------------------------------------------------------------------------- #
def run_candidate_long_horizon(out_dir: Path, date: str, *, timeout_sec: int = 120) -> dict[str, Any]:
    lh_dir = out_dir / "long_horizon_30m"
    log_path = lh_dir / "long-horizon-self-test.candidate.log.jsonl"
    report_path = lh_dir / "long-horizon-self-test.candidate.public-report.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_long_horizon.py"),
        "--self-test",
        "--log", str(log_path),
        "--report-out", str(report_path),
        "--overwrite",
        "--timeout-sec", str(timeout_sec),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=max(timeout_sec + 30, 180), check=False)
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    candidate_note = {
        "candidateOnly": True,
        "level3Evidence": False,
        "whyNotLevel3": [
            "self-test is shorter than 30 minutes",
            "stored outside agi-proof/long-horizon-runs gate-scanned path",
            "not a substantive autonomous repo repair task",
        ],
        "commandReturncode": proc.returncode,
        "stdoutTail": proc.stdout[-1000:],
        "stderrTail": proc.stderr[-1000:],
    }
    sidecar_path = _write_json(lh_dir / "candidate-note.json", candidate_note)
    return {
        "lane": "long_horizon_30m",
        "ok": proc.returncode == 0 and report.get("toolCalls", 0) > 0,
        "candidateOnly": True,
        "artifact": _rel(report_path),
        "log": _rel(log_path),
        "sidecar": _rel(sidecar_path),
        "durationSec": report.get("durationSec"),
        "tier": report.get("tier"),
        "autonomy": report.get("autonomy"),
        # Phase B2/D: surface the machine-checked objective gate as execution truth,
        # distinct from the semantic autonomy classification. objectivePassed=true
        # (while a short self-test reports substantive=false) is the demonstration
        # that execution truth is a STRONGER signal than "the run was long/busy".
        "objectivePassed": report.get("objectivePassed"),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_all(out_dir: Path, *, skip_long_horizon: bool = False) -> dict[str, Any]:
    date = _today()
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes = [
        run_candidate_hidden(out_dir, date),
        run_candidate_distribution(out_dir, date),
    ]
    if skip_long_horizon:
        lanes.append({"lane": "long_horizon_30m", "ok": None, "skipped": True, "candidateOnly": True})
    else:
        lanes.append(run_candidate_long_horizon(out_dir, date))
    summary = {
        "schema": "sophia.level3_candidate_benchmark.v1",
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "outDir": _rel(out_dir),
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "This local benchmark rehearses the Level-3 workflows but is not private, "
            "not independent, and not promotable as AGI Level-3 evidence."
        ),
        "lanes": lanes,
        "allCandidateLanesOk": all(l.get("ok") is True for l in lanes if not l.get("skipped")),
        "realNextCommands": [
            "python tools/run_hidden_eval_full.py --pack private/hidden-evals/PACK.json --mode raw=... --mode raw_tools=... --mode rag_only=... --mode gate_only=... --mode sophia_full=...",
            "python tools/run_distribution_shift.py private/shift/SPEC.json --backend adapter --out agi-proof/learning-under-shift/shift-result-YYYY-MM-DD.public-report.json",
            "python tools/run_long_horizon.py --spec private/long-horizon/30min-YYYY-MM-DD.json --log agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.log.jsonl --report-out agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.public-report.json",
        ],
    }
    _write_json(out_dir / "level3-candidate-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=None, help="candidate output directory")
    ap.add_argument("--skip-long-horizon", action="store_true", help="skip even the short long-horizon self-test")
    ap.add_argument("--emit-real-scaffold", action="store_true",
                    help="write fillable real-run starter specs into the gitignored private/ tree and exit")
    ap.add_argument("--json", action="store_true", help="print summary JSON")
    args = ap.parse_args(argv)
    if args.emit_real_scaffold:
        scaffold = emit_real_scaffold(PRIVATE_ROOT, _today())
        if args.json:
            print(json.dumps(scaffold, indent=2, ensure_ascii=False))
        else:
            print(f"Wrote {len(scaffold['written'])} real-run starter files under {scaffold['privateRoot']}/ (gitignored):")
            for path in scaffold["written"]:
                print(f"  - {path}")
            print("Fill every <...> placeholder, then run the realNextCommands from the protocol doc.")
        return 0
    out_dir = args.out_dir or (DEFAULT_ROOT / f"{_today()}-local-smoke")
    summary = run_all(out_dir, skip_long_horizon=args.skip_long_horizon)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {_rel(out_dir / 'level3-candidate-summary.json')}")
        for lane in summary["lanes"]:
            print(f"- {lane['lane']}: ok={lane.get('ok')} candidateOnly={lane.get('candidateOnly')} artifact={lane.get('artifact')}")
    return 0 if summary["allCandidateLanesOk"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
