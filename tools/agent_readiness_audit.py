#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Agent-readiness audit — a deterministic scorecard for the agent's own harness.

The session bootstrap prints orientation prose; the security audit checks release
gates. This tool checks the *agent plumbing itself*: is the MCP surface wired and
risk-covered, are the hooks real, are the skills readable, is the memory/claims
substrate healthy. Each line is PASS / WARN / FAIL with the fix inline — a
scorecard, not a vanity score.

Run:  python tools/agent_readiness_audit.py [--json]
Exit: 0 = no FAIL (WARNs allowed), 1 = at least one FAIL.

Offline, stdlib-only, no model calls; safe on a LOCKED git-crypt checkout (locked
skills are a WARN, not a FAIL, because web sessions legitimately run locked).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GITCRYPT_MAGIC = b"\x00GITCRYPT"

#: Tool-name fragments that imply a side effect; every matching MCP tool must have a
#: non-"low" entry in sophia_mcp.audit.TOOL_RISK (the fail-closed floor for writes).
_WRITE_HINTS = ("upsert", "export", "record_", "enqueue", "claim_resource",
                "memory_store", "trajectory_record", "retract", "revise")

#: Documented exemptions: these tools write ONLY to the contract substrate, whose own
#: governance (contract gates + kill switch, see sophia_contract/) covers them; they
#: predate the mcp risk table. Any NEW write-shaped tool must NOT be added here —
#: cover it with @audited(risk="medium"/"high") instead. Exemptions are still
#: reported in the audit detail so they stay visible.
_CONTRACT_GOVERNED = frozenset({
    "sophia_enqueue_task", "sophia_record_claim", "sophia_retract", "sophia_revise",
})


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def audit(root: Path = ROOT, *, env: "dict | None" = None) -> "list[dict]":
    import os

    env = os.environ if env is None else env
    checks: list[dict] = []

    # 1) MCP config + server entrypoints exist.
    mcp_json = root / ".mcp.json"
    try:
        servers = json.loads(mcp_json.read_text(encoding="utf-8")).get("mcpServers", {})
        if not servers:
            checks.append(_check("mcp.config", "FAIL", ".mcp.json has no mcpServers"))
        else:
            checks.append(_check("mcp.config", "PASS", f"servers: {', '.join(sorted(servers))}"))
        entry = root / "sophia_mcp" / "server.py"
        checks.append(_check("mcp.entrypoint", "PASS" if entry.is_file() else "FAIL",
                             str(entry.relative_to(root))))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(_check("mcp.config", "FAIL", f".mcp.json unreadable: {exc}"))

    # 2) Risk-table coverage: every write-shaped sophia_* tool must be risk >= medium.
    try:
        from sophia_mcp.audit import TOOL_RISK
        server_src = (root / "sophia_mcp" / "server.py").read_text(encoding="utf-8")
        impl_src = (root / "sophia_mcp" / "tools_impl.py").read_text(encoding="utf-8")
        tools = sorted(set(re.findall(r"def (sophia_[a-z0-9_]+)\(", server_src)))
        risky = [t for t in tools if any(h in t for h in _WRITE_HINTS)]
        # a tool is covered if named in TOOL_RISK directly or via an @audited(risk=...)
        # registration in tools_impl (the decorator populates TOOL_RISK at import).
        uncovered = [t for t in risky
                     if t not in _CONTRACT_GOVERNED
                     and TOOL_RISK.get(t, "low") == "low"
                     and not re.search(rf'@audited\("{t}",\s*risk="(medium|high)"', impl_src)]
        exempt = sorted(set(risky) & _CONTRACT_GOVERNED)
        if uncovered:
            checks.append(_check("mcp.risk_coverage", "FAIL",
                                 f"write-shaped tools at risk=low: {', '.join(uncovered)} — "
                                 f"add TOOL_RISK / @audited(risk=...) entries"))
        else:
            detail = f"{len(tools)} tools; {len(risky)} write-shaped, all covered"
            if exempt:
                detail += f" ({len(exempt)} contract-governed exemption(s): {', '.join(exempt)})"
            checks.append(_check("mcp.risk_coverage", "PASS", detail))
    except Exception as exc:  # noqa: BLE001 — an unimportable audit table is itself a finding
        checks.append(_check("mcp.risk_coverage", "FAIL", f"cannot evaluate: {exc!r}"))

    # 3) Gateway enforcement flags (informational; the safe default is off+identical).
    flags = {"SOPHIA_MCP_GATEWAY": "governed() pipeline",
             "SOPHIA_MCP_APPROVAL": "approval holds",
             "SOPHIA_MCP_OUTPUT_VERIFY": "output re-verification",
             "SOPHIA_MCP_APPROVE_WRITES": "medium/high tool writes"}
    on = [k for k in flags if env.get(k) == "1"]
    checks.append(_check("gateway.flags", "PASS",
                         ("enabled: " + ", ".join(on)) if on else
                         "all off (default surface; enable per-deployment)"))

    # 4) Hooks referenced by settings.json exist and are non-empty.
    settings = root / ".claude" / "settings.json"
    try:
        cfg = json.loads(settings.read_text(encoding="utf-8"))
        missing = []
        n_cmds = 0
        for event_blocks in (cfg.get("hooks") or {}).values():
            for block in event_blocks:
                for hook in block.get("hooks", []):
                    cmd = hook.get("command", "")
                    for token in cmd.split():
                        if token.endswith(".sh") or token.endswith(".py"):
                            n_cmds += 1
                            script = root / token
                            if not (script.is_file() and script.stat().st_size > 0):
                                missing.append(token)
        checks.append(_check("hooks.scripts", "FAIL" if missing else "PASS",
                             f"missing: {', '.join(missing)}" if missing
                             else f"{n_cmds} hook script(s) present"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(_check("hooks.scripts", "FAIL", f"settings.json unreadable: {exc}"))

    # 5) Skills readable (locked ciphertext = WARN — expected on a locked web clone).
    skills_dir = root / ".claude" / "skills"
    locked, empty, ok_n = [], [], 0
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        head = skill_md.read_bytes()[:9]
        if head == GITCRYPT_MAGIC:
            locked.append(skill_md.parent.name)
        elif skill_md.stat().st_size == 0:
            empty.append(skill_md.parent.name)
        else:
            ok_n += 1
    if empty:
        checks.append(_check("skills.readable", "FAIL", f"empty SKILL.md: {', '.join(empty)}"))
    elif locked:
        checks.append(_check("skills.readable", "WARN",
                             f"{ok_n} readable; git-crypt LOCKED: {', '.join(locked)} "
                             f"(set GITCRYPT_KEY_B64 to self-unlock)"))
    else:
        checks.append(_check("skills.readable", "PASS", f"{ok_n} skills readable"))

    # 6) Resource-claims store healthy; expired claims flagged.
    try:
        from agent.resource_claims import status as claims_status
        st = claims_status()
        if not st["ok"]:
            checks.append(_check("claims.store", "FAIL", st["reason"]))
        else:
            expired = [r for r, c in st["claims"].items() if c["expired"]]
            live = [r for r, c in st["claims"].items() if not c["expired"]]
            detail = f"live: {live or 'none'}; expired: {expired or 'none'}"
            checks.append(_check("claims.store", "WARN" if expired else "PASS", detail))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("claims.store", "FAIL", f"cannot evaluate: {exc!r}"))

    # 7) RAG index manifest present + under the ANN crossover tripwire.
    meta = root / "rag" / "index" / "embeddings.meta.json"
    if meta.exists():
        try:
            count = int(json.loads(meta.read_text(encoding="utf-8")).get("count", 0))
            status = "PASS" if count < 5000 else "WARN"
            checks.append(_check("rag.index", status,
                                 f"{count} chunks (ANN crossover tripwire at 5000)"))
        except (json.JSONDecodeError, ValueError) as exc:
            checks.append(_check("rag.index", "FAIL", f"manifest unreadable: {exc}"))
    else:
        checks.append(_check("rag.index", "WARN", "no committed index manifest"))

    # 8) Memory substrate writable (audit logs, traces, experience bank all land here).
    mem = root / "agent" / "memory"
    try:
        mem.mkdir(parents=True, exist_ok=True)
        probe = mem / ".readiness_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks.append(_check("memory.writable", "PASS", str(mem.relative_to(root))))
    except OSError as exc:
        checks.append(_check("memory.writable", "FAIL", f"{exc}"))

    return checks


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    checks = audit()
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    if args.json:
        print(json.dumps({"schema": "sophia.readiness_audit.v1", "checks": checks,
                          "fails": len(fails), "warns": len(warns)}, indent=2))
    else:
        for c in checks:
            print(f"[{c['status']:>4}] {c['name']} — {c['detail']}")
        print(f"\nreadiness: {len(checks) - len(fails) - len(warns)} pass, "
              f"{len(warns)} warn, {len(fails)} fail "
              f"(a clean audit is a filter, not a guarantee)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
