# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-5 integrity layer: artifact checksums/SBOM and the
tamper-evident audit hash-chain."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import audit_chain
from tools import sign_artifacts


# ── artifact signing ──────────────────────────────────────────────────────────
def test_checksum_roundtrip_and_verify(tmp_path):
    art = tmp_path / "adapter.bin"
    art.write_bytes(b"weights-v1")
    sums = sign_artifacts.compute_checksums([str(art)])
    assert len(sums) == 1
    out = tmp_path / "SHA256SUMS"
    sign_artifacts.write_sums(sums, out)
    rep = sign_artifacts.verify_sums(out)
    assert rep["ok"] and rep["verified"] == 1


def test_verify_detects_tampering(tmp_path):
    art = tmp_path / "adapter.bin"
    art.write_bytes(b"weights-v1")
    out = tmp_path / "SHA256SUMS"
    sign_artifacts.write_sums(sign_artifacts.compute_checksums([str(art)]), out)
    art.write_bytes(b"weights-TAMPERED")          # mutate after signing
    rep = sign_artifacts.verify_sums(out)
    assert not rep["ok"]
    assert str(art) in rep["mismatched"]


def test_sbom_lists_components():
    sbom = sign_artifacts.build_sbom()
    assert sbom["bomFormat"] == "CycloneDX"
    assert isinstance(sbom["components"], list)


# ── audit hash-chain ──────────────────────────────────────────────────────────
def test_chain_append_and_verify(tmp_path):
    log = tmp_path / "audit.jsonl"
    audit_chain.append(log, {"event": "tool_call", "tool": "a"}, ts=1.0)
    audit_chain.append(log, {"event": "tool_call", "tool": "b"}, ts=2.0)
    rep = audit_chain.verify_chain(log)
    assert rep["ok"] and rep["length"] == 2 and rep["broken_at"] is None


def test_chain_detects_edit(tmp_path):
    log = tmp_path / "audit.jsonl"
    audit_chain.append(log, {"event": "x", "v": 1}, ts=1.0)
    audit_chain.append(log, {"event": "y", "v": 2}, ts=2.0)
    # Tamper with the first record's payload, keeping its stored hash.
    lines = log.read_text().splitlines()
    lines[0] = lines[0].replace('"v":1', '"v":999')
    log.write_text("\n".join(lines) + "\n")
    rep = audit_chain.verify_chain(log)
    assert not rep["ok"]
    assert rep["broken_at"] == 0


def test_chain_detects_deletion(tmp_path):
    log = tmp_path / "audit.jsonl"
    audit_chain.append(log, {"event": "a"}, ts=1.0)
    audit_chain.append(log, {"event": "b"}, ts=2.0)
    audit_chain.append(log, {"event": "c"}, ts=3.0)
    lines = log.read_text().splitlines()
    del lines[1]                                   # drop the middle record
    log.write_text("\n".join(lines) + "\n")
    rep = audit_chain.verify_chain(log)
    assert not rep["ok"]
