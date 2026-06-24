#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the 4 aihk-os integration adapters:
  1. contract over MCP (sophia_mcp.tools_impl contract fns)
  2. vault bridge (sophia_contract.vault.VaultGate)
  3. 9-role scope registry (sophia_contract.roles.ROLES_9)
  4. Langfuse export (sophia_contract.langfuse_export)
Deterministic, offline, no new dependencies.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from sophia_contract import SophiaContract  # noqa: E402
from sophia_contract.roles import ROLES_9, ROLE_NAMES  # noqa: E402
from sophia_contract.vault import VaultGate  # noqa: E402
from sophia_contract.langfuse_export import build_batch, export_spans  # noqa: E402

_CLK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


# ---------------------------------------------------------------- adapter 3: roles
def test_roles_registry_has_nine() -> None:
    assert len(ROLE_NAMES) == 9
    assert "role_06_content_marketing" in ROLE_NAMES and "role_09_agents" in ROLE_NAMES


def test_content_marketing_capped_at_unclassified() -> None:
    # the playbook's worked example: public content role cannot act above UNCLASSIFIED
    svc = SophiaContract(clock=_CLK, scopes=ROLES_9)
    out = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"],
                            "blp_level": "CONFIDENTIAL", "role": "role_06_content_marketing"})
    assert out["error"]["code"] == "UNAUTHENTICATED"


def test_agents_role_cleared_to_top_secret() -> None:
    svc = SophiaContract(clock=_CLK, scopes=ROLES_9)
    out = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"],
                            "blp_level": "TOP_SECRET", "role": "role_09_agents"})
    assert "error" not in out and out["blp_level"] == "TOP_SECRET"


# --------------------------------------------------------------- adapter 2: vault
def _note(tmp: Path, name: str, meta: dict, body: str = "Body text.") -> Path:
    p = tmp / name
    p.write_text(frontmatter.serialize(meta, body), encoding="utf-8")
    return p


def test_vault_gate_accepts_and_stamps() -> None:
    tmp = Path(tempfile.mkdtemp())
    gate = VaultGate(SophiaContract(clock=_CLK), vault_root=tmp)
    note = _note(tmp, "draft.md", {"sources": ["https://example.com/src"], "blp_level": "UNCLASSIFIED"})
    verdict = gate.gate_note(note)
    assert verdict["verdict"] == "accepted"
    meta, _ = frontmatter.parse(note.read_text())
    assert meta["gate_status"] == "accepted" and meta["provenance_id"].startswith("clm_")
    assert gate.is_publishable(note) is True


def test_vault_gate_holds_no_source_not_publishable() -> None:
    tmp = Path(tempfile.mkdtemp())
    gate = VaultGate(SophiaContract(clock=_CLK), vault_root=tmp)
    note = _note(tmp, "nosrc.md", {"blp_level": "UNCLASSIFIED"})
    verdict = gate.gate_note(note)
    assert verdict["verdict"] == "held" and verdict["held_reason"] == "no_source"
    assert gate.is_publishable(note) is False
    published = gate.publish_if_accepted(note, lambda p: "PUBLISHED")
    assert published["published"] is False and published["held_reason"] == "no_source"


def test_vault_gate_idempotent_provenance_id() -> None:
    tmp = Path(tempfile.mkdtemp())
    gate = VaultGate(SophiaContract(clock=_CLK), vault_root=tmp)
    note = _note(tmp, "idem.md", {"sources": ["s1"]})
    gate.gate_note(note)
    id1 = frontmatter.parse(note.read_text())[0]["provenance_id"]
    gate.gate_note(note)
    id2 = frontmatter.parse(note.read_text())[0]["provenance_id"]
    assert id1 == id2


def test_vault_gate_respects_role_scope() -> None:
    tmp = Path(tempfile.mkdtemp())
    gate = VaultGate(SophiaContract(clock=_CLK, scopes=ROLES_9), vault_root=tmp)
    note = _note(tmp, "client.md", {"sources": ["s1"], "blp_level": "CONFIDENTIAL",
                                    "role": "role_06_content_marketing"})
    out = gate.gate_note(note)
    assert "error" in out and out["error"]["code"] == "UNAUTHENTICATED"
    assert gate.is_publishable(note) is False  # never stamped accepted


# ------------------------------------------------------------ adapter 4: langfuse
def test_langfuse_batch_shape() -> None:
    spans = [{"id": "trace_x", "name": "verify_claim", "startTime": "t", "endTime": "t",
              "input": {"a": 1}, "output": {"verdict": "accepted"}, "level": "DEFAULT", "metadata": {}}]
    batch = build_batch(spans)
    assert batch["batch"][0]["type"] == "trace-create"
    assert batch["batch"][0]["body"]["name"] == "verify_claim"
    assert batch["batch"][0]["body"]["metadata"]["source"] == "sophia-contract"


def test_langfuse_export_offline_no_creds() -> None:
    spans = [{"id": "t1", "name": "n", "startTime": "t", "endTime": "t",
              "input": {}, "output": {}, "level": "DEFAULT", "metadata": {}}]
    r = export_spans(spans, host=None, public_key=None, secret_key=None)
    assert r["sent"] is False and r["count"] == 1 and "batch" in r


def test_langfuse_export_dry_run_with_creds() -> None:
    r = export_spans([], host="https://lf", public_key="pk", secret_key="sk", dry_run=True)
    assert r["sent"] is False and r["reason"] == "dry_run"


def test_contract_traces_feed_langfuse() -> None:
    # the contract's own spans must export cleanly (end-to-end shape check)
    svc = SophiaContract(clock=_CLK)
    c = svc.record_claim({"idempotency_key": "e", "content": "x", "sources": ["s"]})
    svc.verify_claim({"claim_id": c["claim_id"]})
    batch = build_batch(svc.tracer.events())
    assert len(batch["batch"]) >= 2
    assert all(e["type"] == "trace-create" for e in batch["batch"])


# ----------------------------------------------------------------- adapter 1: MCP
def test_mcp_contract_tools_roundtrip() -> None:
    from sophia_mcp import tools_impl

    prev = tools_impl._CONTRACT
    tools_impl._CONTRACT = SophiaContract(clock=_CLK, scopes=ROLES_9)
    try:
        assert tools_impl.contract_describe()["version"]  # handshake
        claim = tools_impl.record_claim("mcp-1", "A sourced claim.", sources=["s1"],
                                        role="role_05_copywriting")
        assert claim["claim_id"].startswith("clm_")
        verdict = tools_impl.verify_claim(claim["claim_id"], role="role_05_copywriting")
        assert verdict["verdict"] == "accepted"
        assert "explanation" in tools_impl.explain_verdict(claim["claim_id"])
        # role scope enforced through the MCP layer too
        denied = tools_impl.record_claim("mcp-2", "x", sources=["s"], blp_level="SECRET",
                                         role="role_06_content_marketing")
        assert denied["error"]["code"] == "UNAUTHENTICATED"
        # durable queue tools
        t = tools_impl.enqueue_task("job-1", "verify", payload={"claim_id": claim["claim_id"]})
        assert t["task_id"].startswith("task_")
        assert tools_impl.next_task()["task"]["task_id"] == t["task_id"]
        assert tools_impl.contract_health()["status"] == "ok"
    finally:
        tools_impl._CONTRACT = prev


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_contract_adapters: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
