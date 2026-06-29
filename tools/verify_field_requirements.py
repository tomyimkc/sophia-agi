#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verify the field-requirements capability proof.

Walks agi-proof/field-requirements/manifest.json and checks that every artifact it
cites actually exists in the repo: each module file exists AND compiles, each test
file exists, each evidence path exists (file or directory). This turns the
capability map from prose into something CI enforces — if a module is deleted or a
report goes missing, the proof FAILS rather than silently lying.

    python tools/verify_field_requirements.py           # table; exit 1 on any gap
    python tools/verify_field_requirements.py --json     # machine-readable
    python tools/verify_field_requirements.py --import    # also try importing modules (best-effort)

The check is structural by design (exists + compiles), so it is fast, dependency-
free, and deterministic. It does NOT re-run the cited tests or re-measure the
evidence — `pytest` and the benchmark runners do that. It proves the map points at
real, compiling code and present artifacts.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "agi-proof" / "field-requirements" / "manifest.json"


def _try_import(rel: str, path: Path) -> "tuple[bool, str]":
    """Best-effort import, non-fatal. Prefers a real dotted import for package files
    (so sibling imports and dataclasses resolve normally); falls back to file-location
    for standalone scripts (e.g. tools/*, which is not a package). Catches
    BaseException because some modules raise SystemExit at import when an optional
    dependency (e.g. the MCP server's FastMCP) is absent — an env gap, not a broken
    module."""
    pkg_root = rel.split("/", 1)[0]
    use_dotted = (ROOT / pkg_root / "__init__.py").exists()
    try:
        if use_dotted:
            importlib.import_module(rel[:-3].replace("/", "."))
        else:
            spec = importlib.util.spec_from_file_location(f"_frcheck_{path.stem}", path)
            if spec is None or spec.loader is None:
                return False, "no import spec"
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        return True, ""
    except BaseException as e:  # noqa: BLE001 - best-effort; SystemExit included
        return False, f"{type(e).__name__}: {e}"[:200]


def _check_module(rel: str, do_import: bool) -> dict:
    path = ROOT / rel
    out = {"path": rel, "exists": path.exists(), "compiles": False, "imported": None}
    if not out["exists"]:
        return out
    try:
        py_compile.compile(str(path), doraise=True)
        out["compiles"] = True
    except py_compile.PyCompileError as e:
        out["error"] = str(e).splitlines()[-1][:200]
        return out
    if do_import and rel.endswith(".py"):
        imported, err = _try_import(rel, path)
        out["imported"] = imported
        if not imported:
            out["importError"] = err
    return out


def _check_exists(rel: str) -> dict:
    return {"path": rel, "exists": (ROOT / rel).exists()}


def verify(manifest: dict, *, do_import: bool = False) -> dict:
    capabilities = []
    for cap in manifest.get("capabilities", []):
        modules = [_check_module(m, do_import) for m in cap.get("modules", [])]
        tests = [_check_exists(t) for t in cap.get("tests", [])]
        evidence = [_check_exists(e) for e in cap.get("evidence", [])]

        module_ok = all(m["exists"] and m["compiles"] for m in modules)
        tests_ok = all(t["exists"] for t in tests)
        evidence_ok = all(e["exists"] for e in evidence)
        ok = module_ok and tests_ok and evidence_ok

        capabilities.append({
            "id": cap.get("id"),
            "status": cap.get("status"),
            "marketCategory": cap.get("marketCategory"),
            "ok": ok,
            "modules": modules,
            "tests": tests,
            "evidence": evidence,
            "moduleOk": module_ok,
            "testsOk": tests_ok,
            "evidenceOk": evidence_ok,
        })

    passed = sum(1 for c in capabilities if c["ok"])
    return {
        "schema": "sophia.field_requirements.verification.v1",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "capabilitiesTotal": len(capabilities),
        "capabilitiesOk": passed,
        "allOk": passed == len(capabilities) and capabilities != [],
        "importChecked": do_import,
        "capabilities": capabilities,
    }


def _missing(check_list: list[dict], key: str = "exists") -> list[str]:
    return [c["path"] for c in check_list if not c.get(key)]


def _print_table(report: dict) -> None:
    print(
        f"Field-requirements capability proof — {report['capabilitiesOk']}/"
        f"{report['capabilitiesTotal']} capabilities verified "
        f"(import-checked: {report['importChecked']})\n"
    )
    for c in report["capabilities"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"  [{mark}] {c['id']:26} status={c['status']:14} — {c['marketCategory']}")
        if not c["ok"]:
            mm = [m["path"] for m in c["modules"] if not (m["exists"] and m["compiles"])]
            for p in mm:
                print(f"         module missing/uncompilable: {p}")
            for p in _missing(c["tests"]):
                print(f"         test missing: {p}")
            for p in _missing(c["evidence"]):
                print(f"         evidence missing: {p}")
    if report["importChecked"]:
        fails = [
            (c["id"], m["path"], m.get("importError"))
            for c in report["capabilities"] for m in c["modules"]
            if m.get("imported") is False
        ]
        if fails:
            print("\n  import warnings (best-effort; not fatal):")
            for cid, path, err in fails:
                print(f"    {cid}: {path} — {err}")
    print()
    print("ALL OK" if report["allOk"] else "INCOMPLETE — see FAILs above")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--import", dest="do_import", action="store_true",
                    help="also attempt to import each module (best-effort, non-fatal)")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = verify(manifest, do_import=args.do_import)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_table(report)
    return 0 if report["allOk"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
