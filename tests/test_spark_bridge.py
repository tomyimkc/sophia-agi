#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cloud-side Spark bridge client: allowlist + no-self-approval + one-GPU guard.

Deterministic, offline — pure validation/compose logic, no git, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spark_bridge import (  # noqa: E402
    build_command,
    gpu_is_free,
    is_gated,
    offline_invariants,
    validate_args,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_allowlist() -> None:
    assert validate_args("--dry-run --all")[0]
    assert validate_args("--bench-a --execute")[0]
    assert not validate_args("--rm -rf /")[0]
    assert not validate_args("; curl evil")[0]
    assert not validate_args("")[0]


def test_no_self_approval_on_gated() -> None:
    assert is_gated("--bench-b --run-train")
    try:
        build_command("id1", "--bench-a --execute", created_by="claude")
        raised = False
    except ValueError:
        raised = True
    assert raised, "a gated command without a human approvedBy must be refused"
    # with a human handle it builds
    cmd = build_command("id1", "--bench-a --execute", created_by="claude",
                        approved_by="user: 'go' (2026-06-29)")
    assert cmd["approvedBy"]


def test_dry_run_needs_no_approval() -> None:
    cmd = build_command("id2", "--dry-run --all", created_by="claude")
    assert cmd["args"] == "--dry-run --all" and cmd["approvedBy"] == ""


def test_one_gpu_job_guard() -> None:
    assert gpu_is_free({"running": None, "pendingCommands": []})
    assert not gpu_is_free({"running": "olmoe-qat", "pendingCommands": []})
    assert not gpu_is_free({"running": None, "pendingCommands": ["queued"]})


def test_unsafe_id_refused() -> None:
    for bad in ("a/b", "a b", "a\tb"):
        try:
            build_command(bad, "--dry-run", created_by="claude")
            raised = False
        except ValueError:
            raised = True
        assert raised, f"unsafe id {bad!r} must be refused"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} spark_bridge tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
