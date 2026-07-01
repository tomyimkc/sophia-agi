# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/gate_provenance.py — stamp/verify roundtrip and staleness detection."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import gate_provenance as gp  # noqa: E402


def _tmp(text: str, suffix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    p = d / f"file{suffix}"
    p.write_text(text, encoding="utf-8")
    return p


def _receipt(d: Path) -> Path:
    p = d / "receipt.json"
    p.write_text(json.dumps({"prefix": "M3-pilot", "verdict": "GO", "ok": True}) + "\n",
                 encoding="utf-8")
    return p


def test_fingerprint_is_deterministic_and_order_independent():
    a = _tmp("alpha", "_a.py")
    b = _tmp("beta", "_b.py")
    h1 = gp.gate_fingerprint([a, b])
    h2 = gp.gate_fingerprint([b, a])   # argument order must not matter
    assert h1 == h2, "fingerprint must be order-independent"
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)
    # A change to any byte moves the fingerprint.
    a.write_text("alpha!", encoding="utf-8")
    assert gp.gate_fingerprint([a, b]) != h1, "byte change must move fingerprint"


def test_fingerprint_empty_is_fail_closed():
    raised = False
    try:
        gp.gate_fingerprint([])
    except ValueError:
        raised = True
    assert raised, "empty certifier set must raise (fail-closed), not return a hash"


def test_stamp_then_verify_is_fresh():
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    c1 = _tmp("cert-one", "_c1.py")
    c2 = _tmp("cert-two", "_c2.json")
    receipt = _receipt(d)

    stamped = gp.stamp_receipt(receipt, [c1, c2], stamped_at=None)
    prov = stamped["gateProvenance"]
    assert prov["gateHash"] == gp.gate_fingerprint([c1, c2])
    assert prov["stampedAt"] is None, "no wallclock: stampedAt must be the injected value (None)"
    assert len(prov["stampedFiles"]) == 2

    # Re-read from disk and verify == FRESH.
    result = gp.verify_receipt(receipt)
    assert result["status"] == "FRESH", result
    assert result["fresh"] is True
    assert result["drifted"] == []


def test_mutate_a_stamped_file_makes_verify_stale():
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    c1 = _tmp("cert-one", "_c1.py")
    c2 = _tmp("cert-two", "_c2.json")
    receipt = _receipt(d)
    gp.stamp_receipt(receipt, [c1, c2], stamped_at="2026-07-01T00:00:00+00:00")

    # Mutate one certifying file after stamping — CI would re-open the claim.
    c1.write_text("cert-one-TAMPERED", encoding="utf-8")

    result = gp.verify_receipt(receipt)
    assert result["status"] == "STALE", result
    assert result["fresh"] is False
    drifted_names = result["drifted"]
    assert len(drifted_names) == 1
    assert drifted_names[0].endswith("_c1.py"), drifted_names
    # The other file must NOT be reported as drifted.
    assert not any(n.endswith("_c2.json") for n in drifted_names)


def test_injected_timestamp_is_recorded_verbatim():
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    c1 = _tmp("cert-one", "_c1.py")
    receipt = _receipt(d)
    ts = "2026-07-01T12:34:56+00:00"
    stamped = gp.stamp_receipt(receipt, [c1], stamped_at=ts)
    assert stamped["gateProvenance"]["stampedAt"] == ts


def test_unstamped_receipt_verifies_as_unstamped():
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    receipt = _receipt(d)   # never stamped
    result = gp.verify_receipt(receipt)
    assert result["status"] == "UNSTAMPED"
    assert result["fresh"] is False


def test_missing_certifier_after_stamp_is_stale():
    d = Path(tempfile.mkdtemp(prefix="gateprov_"))
    # Use a file inside ROOT so its relative path re-resolves under ROOT on verify.
    inside = ROOT / "tools" / "gate_provenance.py"
    receipt = _receipt(d)
    stamped = gp.stamp_receipt(receipt, [inside], stamped_at=None)
    # Simulate the certifier vanishing by rewriting the stamped manifest to a
    # non-existent (but ROOT-relative) file, then verifying.
    receipt_obj = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_obj["gateProvenance"]["stampedFiles"] = [
        {"file": "tools/__does_not_exist__.py", "sha256": "0" * 64}
    ]
    receipt.write_text(json.dumps(receipt_obj), encoding="utf-8")
    result = gp.verify_receipt(receipt)
    assert result["status"] == "STALE", result
    assert result["fresh"] is False
    assert result.get("reason") == "missing_certifier"


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run()
