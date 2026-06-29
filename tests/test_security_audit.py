#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the sophia-security-audit pre-flight (tools/security_audit.py).

Deterministic, offline. Uses only in-memory synthetic strings for the secret
scanner — never writes a real credential to any file.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import security_audit as sa  # noqa: E402


def test_run_audit_returns_three_checks() -> None:
    result = sa.run_audit()
    assert isinstance(result, dict)
    assert "checks" in result and "ok" in result
    assert set(result["checks"]) == {"no_overclaim", "redos_robustness", "secret_scan"}
    for check in result["checks"].values():
        assert check["status"] in {"passed", "failed", "skipped"}
    assert isinstance(result["ok"], bool)


def test_ok_is_true_unless_a_check_failed() -> None:
    result = sa.run_audit()
    failed = [c for c in result["checks"].values() if c["status"] == "failed"]
    assert result["ok"] == (not failed)


def test_scan_text_flags_synthetic_aws_key() -> None:
    # Synthetic, in-memory only — fake AWS-access-key-id shape.
    findings = sa.scan_text("aws_key = AKIA" + "0" * 16)
    assert findings, "synthetic AKIA-shaped string must be flagged"
    assert any(f["pattern"] == "aws-access-key-id" for f in findings)
    # Output is redacted (no full token echoed back).
    assert all(f["match"].endswith("...") for f in findings)


def test_scan_text_flags_synthetic_private_key_header() -> None:
    findings = sa.scan_text("-----BEGIN RSA PRIVATE KEY-----")
    assert any(f["pattern"] == "private-key-header" for f in findings)


def test_scan_text_passes_clean_text() -> None:
    assert sa.scan_text("the quick brown fox jumps over 1234 lazy dogs") == []
    assert sa.scan_text("normal prose with no secrets here") == []


def test_scan_text_ignores_placeholders() -> None:
    # Allowlisted placeholder values must not be reported.
    assert sa.scan_text("token = sk-...placeholder") == []
    assert sa.scan_text("HF_TOKEN=hf_your_token_here") == []


def test_offline_invariants_pass() -> None:
    ok, detail = sa.offline_invariants()
    assert ok, detail["checks"]


def test_check_cli_exits_per_ok() -> None:
    rc = sa.main(["--check"])
    result = sa.run_audit()
    assert rc == (0 if result["ok"] else 1)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} security-audit tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
