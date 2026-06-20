#!/usr/bin/env python3
"""Third-party replication harness for the Sophia AGI-candidate proof.

Runs the clean-clone checklist from agi-proof/third-party-replication/README.md,
records the commit hash + environment + per-command returncodes, and emits a
reviewer-signature template with the machine-checkable items filled and the human
attestation left blank. It cannot self-certify: the reviewer identity, the
"created hidden tasks", and the signature are intentionally left for an
independent human to complete.

Builder commands mutate tracked artifacts (the manifest, web data), so they are
skipped unless --full is passed; in a real clean clone a reviewer runs --full.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def default_checks() -> list[dict[str, Any]]:
    return [
        {"name": "validate-attribution", "cmd": [PY, "tools/validate_attribution.py"], "category": "validation"},
        {"name": "build-agi-proof", "cmd": [PY, "tools/build_agi_proof_package.py"], "category": "builder"},
        {"name": "build-web-data", "cmd": [PY, "tools/build_web_data.py"], "category": "builder"},
        {"name": "pytest", "cmd": [PY, "-m", "pytest", "-q"], "category": "tests"},
    ]


def git_output(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.stdout.strip()


def pytest_available() -> bool:
    proc = subprocess.run([PY, "-c", "import pytest"], cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.returncode == 0


def run_check(check: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    base = {"name": check["name"], "category": check["category"], "cmd": " ".join(check["cmd"])}
    try:
        proc = subprocess.run(check["cmd"], cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return {**base, "returncode": 124, "passed": False, "timedOut": True, "stderrTail": f"timed out after {timeout_sec}s"}
    return {
        **base,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdoutTail": proc.stdout[-800:],
        "stderrTail": proc.stderr[-800:],
    }


def run_test_scripts(*, timeout_sec: int) -> dict[str, Any]:
    """Fallback when pytest is absent: run each tests/test_*.py as a script."""
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    per_test: list[dict[str, Any]] = []
    for path in test_files:
        try:
            proc = subprocess.run([PY, str(path)], cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec, check=False)
            per_test.append({"test": path.name, "passed": proc.returncode == 0, "stderrTail": proc.stderr[-300:]})
        except subprocess.TimeoutExpired:
            per_test.append({"test": path.name, "passed": False, "timedOut": True, "stderrTail": f"timed out after {timeout_sec}s"})
    passed = sum(1 for t in per_test if t["passed"])
    return {
        "name": "tests-direct",
        "category": "tests",
        "cmd": "python tests/test_*.py (pytest unavailable)",
        # Vacuous truth guard: no test files discovered is a FAILURE, not a pass.
        "passed": bool(per_test) and passed == len(per_test),
        "summary": f"{passed}/{len(per_test)} test scripts passed",
        "perTest": per_test,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Sophia clean-clone replication checklist")
    parser.add_argument("--full", action="store_true", help="Also run builder commands that mutate tracked artifacts")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    environment = {
        "python": sys.version.split()[0],
        "executable": PY,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    commit = git_output("rev-parse", "HEAD")
    dirty = bool(git_output("status", "--short"))

    results: list[dict[str, Any]] = []
    for check in default_checks():
        if check["category"] == "builder" and not args.full:
            results.append({**{k: check[k] for k in ("name", "category")}, "cmd": " ".join(check["cmd"]), "skipped": "builder skipped without --full (mutates tracked artifacts)"})
            continue
        if check["name"] == "pytest" and not pytest_available():
            print("[replication] pytest unavailable; running test scripts directly", flush=True)
            results.append(run_test_scripts(timeout_sec=args.timeout_sec))
            continue
        print(f"[replication] running {check['name']}", flush=True)
        results.append(run_check(check, timeout_sec=args.timeout_sec))

    ran = [r for r in results if "passed" in r]
    machine_pass = bool(ran) and all(r.get("passed") for r in ran)

    report = {
        "kind": "third-party-replication-check",
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "selfCertified": False,
        "note": (
            "Machine-checkable items are filled by this harness. Independence items "
            "(reviewer identity, hidden tasks created by the reviewer, signature) MUST "
            "be completed by an external human; this tool cannot self-certify."
        ),
        "commit": commit,
        "workingTreeDirty": dirty,
        "environment": environment,
        "ranBuilders": args.full,
        "machineChecklist": results,
        "machineChecksPassed": machine_pass,
        "reviewerChecklist": {
            "usedCleanClone": None,
            "recordedCommitAndEnvironment": True,
            "ranValidationAndTests": machine_pass,
            "createdHiddenTasksNotVisibleToSophia": None,
            "ranBaselinesAndAblationsOnSameHiddenPack": None,
            "reportedFailuresBesideSuccesses": None,
            "didNotDescribePendingExternalBenchmarksAsAchieved": None,
        },
        "reviewerSignature": {
            "reviewer": None,
            "date": None,
            "commit": commit,
            "environment": environment,
            "signature": None,
        },
    }

    out_path = args.out or (
        ROOT / "agi-proof" / "third-party-replication" / f"replication-check-{datetime.now().date().isoformat()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps({k: report[k] for k in ("commit", "machineChecksPassed", "ranBuilders")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
