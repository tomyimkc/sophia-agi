#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for ed25519 witness signing of evidence artifacts (skipped without the
cryptography package)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # a distro-packaged cryptography can PANIC (not ImportError) on import — skip either way
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401
except BaseException as _exc:  # noqa: BLE001 — pyo3 PanicException is not an Exception
    pytest.skip(f"cryptography ed25519 unavailable: {_exc!r}", allow_module_level=True)

from tools.witness import keygen, sign, verify  # noqa: E402


@pytest.fixture()
def signed_artifact(tmp_path: Path) -> "tuple[Path, Path, str]":
    # key lives OUTSIDE the repo tree (tmp_path is), as the tool requires
    key = tmp_path / "keys" / "witness.key"
    pub = tmp_path / "keys" / "witness.pub.hex"
    keygen(key, pub)
    artifact = tmp_path / "published-results.json"
    artifact.write_text(json.dumps({"result": "candidate_only"}), encoding="utf-8")
    assert sign(key, [artifact]) == 0
    return artifact, key, pub.read_text(encoding="utf-8").strip()


def test_sign_verify_roundtrip(signed_artifact) -> None:
    artifact, _key, pub_hex = signed_artifact
    assert verify([artifact]) == 0
    assert verify([artifact], pub_hex=pub_hex) == 0


def test_tampered_artifact_fails(signed_artifact) -> None:
    artifact, _key, pub_hex = signed_artifact
    artifact.write_text(json.dumps({"result": "VALIDATED"}), encoding="utf-8")
    assert verify([artifact], pub_hex=pub_hex) == 1


def test_wrong_signer_fails(signed_artifact, tmp_path: Path) -> None:
    artifact, _key, _pub = signed_artifact
    other_key = tmp_path / "keys2" / "other.key"
    other_pub = tmp_path / "keys2" / "other.pub.hex"
    keygen(other_key, other_pub)
    assert verify([artifact], pub_hex=other_pub.read_text(encoding="utf-8").strip()) == 1


def test_forged_signature_fails(signed_artifact) -> None:
    artifact, _key, pub_hex = signed_artifact
    sidecar = artifact.with_name(artifact.name + ".witness.json")
    w = json.loads(sidecar.read_text(encoding="utf-8"))
    w["signature"] = "00" * 64
    sidecar.write_text(json.dumps(w), encoding="utf-8")
    assert verify([artifact], pub_hex=pub_hex) == 1


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "loose.json"
    artifact.write_text("{}", encoding="utf-8")
    assert verify([artifact]) == 1


def test_key_inside_repo_refused() -> None:
    with pytest.raises(SystemExit, match="inside the repo"):
        keygen(ROOT / "secret" / "nope.key", None)


def test_keygen_never_overwrites(tmp_path: Path) -> None:
    key = tmp_path / "k.key"
    keygen(key, None)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        keygen(key, None)
