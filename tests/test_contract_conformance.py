#!/usr/bin/env python3
"""Conformance: run the golden vectors (fixed claims -> expected verdicts) through
the live contract. This is the gate that proves the wire behaviour is stable; it
runs on every release. Deterministic, offline, no model in the loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract import CONTRACT_VERSION, SophiaContract  # noqa: E402
from sophia_contract.models import claim_id_for  # noqa: E402

VECTORS = ROOT / "schema" / "golden-vectors.json"
_FIXED_CLOCK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731  (determinism)


def _resolve_parents(request: dict) -> dict:
    """Translate __key__:<key> parent references to deterministic claim ids."""
    req = dict(request)
    parents = []
    for p in req.get("parents", []) or []:
        if isinstance(p, str) and p.startswith("__key__:"):
            parents.append(claim_id_for(p.split(":", 1)[1]))
        else:
            parents.append(p)
    if parents:
        req["parents"] = parents
    return req


def _run_setup(svc: SophiaContract, setup: list) -> None:
    for op in setup:
        kind = op["op"]
        if kind == "record":
            out = svc.record_claim(_resolve_parents(op["request"]))
            assert "error" not in out, f"setup record unexpectedly failed: {out}"
        elif kind == "supersede":
            svc.mark_superseded(claim_id_for(op["old_key"]), claim_id_for(op["new_key"]))
        elif kind == "human_verdict":
            svc.record_human_verdict(claim_id=claim_id_for(op["key"]), verdict=op["verdict"],
                                     note=op.get("note", ""))
        else:
            raise AssertionError(f"unknown setup op {kind!r}")


def _do_action(svc: SophiaContract, action: dict) -> dict:
    op = action["op"]
    if op == "verify":
        return svc.verify_claim({"claim_id": claim_id_for(action["key"])},
                                clearance=action.get("clearance", "UNCLASSIFIED"))
    if op == "verify_id":
        return svc.verify_claim({"claim_id": action["claim_id"]},
                                clearance=action.get("clearance", "UNCLASSIFIED"))
    if op == "record":
        return svc.record_claim(_resolve_parents(action["request"]))
    raise AssertionError(f"unknown action op {op!r}")


def _check(name: str, result: dict, expect: dict) -> None:
    kind = expect["kind"]
    if kind == "error":
        assert "error" in result, f"[{name}] expected error, got {result}"
        assert result["error"]["code"] == expect["code"], \
            f"[{name}] expected code {expect['code']}, got {result['error']['code']}"
        return
    assert "error" not in result, f"[{name}] unexpected error: {result}"
    if kind == "claim":
        assert "claim_id" in result, f"[{name}] expected a Claim, got {result}"
        if "same_id_as_key" in expect:
            assert result["claim_id"] == claim_id_for(expect["same_id_as_key"]), \
                f"[{name}] idempotent id mismatch"
        return
    if kind == "verdict":
        assert result["verdict"] == expect["verdict"], \
            f"[{name}] verdict {result['verdict']} != {expect['verdict']} ({result})"
        if "held_reason" in expect:
            assert result.get("held_reason") == expect["held_reason"], \
                f"[{name}] held_reason {result.get('held_reason')} != {expect['held_reason']}"
        if "confidence" in expect:
            assert abs(result["confidence"] - expect["confidence"]) < 1e-9, \
                f"[{name}] confidence {result['confidence']} != {expect['confidence']}"
        if "reasons_include" in expect:
            joined = " ".join(result["reasons"]).lower()
            assert expect["reasons_include"].lower() in joined, \
                f"[{name}] reasons missing {expect['reasons_include']!r}: {result['reasons']}"
        if "supersedes_key" in expect:
            assert result.get("supersedes") == claim_id_for(expect["supersedes_key"]), \
                f"[{name}] supersedes mismatch"
        # contract invariant: only 'accepted' is publishable
        if result["verdict"] != "accepted":
            assert result["verdict"] in ("rejected", "superseded", "held")
        return
    raise AssertionError(f"unknown expect kind {kind!r}")


def test_contract_version_matches_vectors() -> None:
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert data["contract_version"] == CONTRACT_VERSION


def test_golden_vectors() -> None:
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    for vec in data["vectors"]:
        svc = SophiaContract(clock=_FIXED_CLOCK, **(vec.get("service") or {}))
        _run_setup(svc, vec.get("setup", []))
        result = _do_action(svc, vec["action"])
        _check(vec["name"], result, vec["expect"])


def test_describe_handshake_shape() -> None:
    d = SophiaContract().describe()
    assert d["version"] == CONTRACT_VERSION
    assert set(("describe", "record_claim", "verify_claim")).issubset(set(d["capabilities"]))
    assert d["schema_url"].endswith("contract-1.1.0.json")
    assert isinstance(d["deprecations"], list)


def test_schema_enums_match_code() -> None:
    """The published schema must not drift from the implementation's enums."""
    from sophia_contract import BLP_LEVELS, ERROR_CODES, HELD_REASONS, VERDICTS

    schema = json.loads((ROOT / "schema" / "contract-1.1.0.json").read_text("utf-8"))
    defs = schema["$defs"]
    assert schema["x-contract-version"] == CONTRACT_VERSION
    assert defs["blp_level"]["enum"] == list(BLP_LEVELS)
    assert defs["verdict_value"]["enum"] == list(VERDICTS)
    assert set(defs["held_reason"]["enum"]) == set(HELD_REASONS)
    assert set(defs["error_code"]["enum"]) == set(ERROR_CODES)


def test_live_outputs_validate_against_schema_if_available() -> None:
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return  # dependency-free environments rely on the enum cross-check above
    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "schema" / "contract-1.1.0.json").read_text("utf-8"))

    def val(defname, instance):
        sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{defname}"}
        Draft202012Validator(sub).validate(instance)

    svc = SophiaContract(clock=_FIXED_CLOCK)
    val("describe_response", svc.describe())
    claim = svc.record_claim({"idempotency_key": "schema-1", "content": "x", "sources": ["s1"]})
    val("claim", claim)
    val("verdict", svc.verify_claim({"claim_id": claim["claim_id"]}))
    val("error", svc.verify_claim({"claim_id": "nope"}))


def main() -> int:
    test_contract_version_matches_vectors()
    test_golden_vectors()
    test_describe_handshake_shape()
    test_schema_enums_match_code()
    test_live_outputs_validate_against_schema_if_available()
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    print(f"test_contract_conformance: OK ({len(data['vectors'])} golden vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
