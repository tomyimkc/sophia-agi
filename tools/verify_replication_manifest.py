# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline manifest checker for the verification-replication pack (NO network).

Reads agi-proof/verification-replication/EXPECTED-RESULTS.json and confirms, fail-closed,
that every module / test / bench tool / synthetic pack / committed live report it references
actually EXISTS on disk, and that every referenced live-report JSON has ``canClaimAGI=false``
(including EXPECTED-RESULTS.json itself). This is the machine-checkable floor a third party can
run after a clean clone, before spending any keys: if the manifest does not even point at real
files, the live runbook cannot be trusted.

Run:  python3 tools/verify_replication_manifest.py
Exit: 0 = PASS (every file present, every report canClaimAGI=false), 1 = FAIL.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "agi-proof" / "verification-replication" / "EXPECTED-RESULTS.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check() -> tuple[list[str], list[str]]:
    """Return (ok_messages, problems). ``problems`` empty == PASS."""
    ok: list[str] = []
    problems: list[str] = []

    if not MANIFEST.exists():
        return ok, [f"manifest missing: {MANIFEST.relative_to(ROOT)}"]
    try:
        data = _load(MANIFEST)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a hard fail
        return ok, [f"manifest unreadable: {exc}"]

    # The manifest itself must never claim AGI.
    if data.get("canClaimAGI") is not False:
        problems.append("EXPECTED-RESULTS.json: canClaimAGI must be false")
    else:
        ok.append("EXPECTED-RESULTS.json: canClaimAGI=false")

    # 1) Every listed file must exist on disk.
    listed_files: list[str] = []
    for key in ("modules", "tests", "bench_tools", "synthetic_packs"):
        listed_files.extend(data.get(key, []))
    report_paths: list[str] = []
    for name, exp in (data.get("experiments") or {}).items():
        rp = exp.get("report_path")
        if not rp:
            problems.append(f"experiment '{name}': missing report_path")
            continue
        report_paths.append(rp)
        listed_files.append(rp)

    for rel in listed_files:
        if (ROOT / rel).exists():
            ok.append(f"exists: {rel}")
        else:
            problems.append(f"MISSING file: {rel}")

    # 2) Every referenced live report JSON must have canClaimAGI=false.
    for rel in report_paths:
        p = ROOT / rel
        if not p.exists():
            continue  # already reported as missing above
        try:
            rep = _load(p)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rel}: unreadable JSON ({exc})")
            continue
        if rep.get("canClaimAGI") is not False:
            problems.append(f"{rel}: canClaimAGI must be false (got {rep.get('canClaimAGI')!r})")
        else:
            ok.append(f"{rel}: canClaimAGI=false")

    return ok, problems


def main() -> int:
    ok, problems = check()
    for line in ok:
        print(f"  ok   {line}")
    if problems:
        print("\nVERIFY REPLICATION MANIFEST: FAIL")
        for p in problems:
            print(f"  FAIL {p}")
        print(f"\n{len(problems)} problem(s). Fix the manifest or restore the missing/over-claiming file.")
        return 1
    print(f"\nVERIFY REPLICATION MANIFEST: PASS — {len(ok)} check(s), no problems.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
