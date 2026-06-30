# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline, deterministic tests for the Spark control panel's PURE security logic.

Only the action-registry + auth + argv-safety helpers are exercised here — NO real socket is bound,
no subprocess is spawned, no network. The HTTP server + subprocess streaming is the impure part,
tested live (not in CI). The #1 property under test: a web button that runs commands cannot be
turned into an RCE — only allowlisted fixed-argv ids are accepted, and off-localhost binds require a
constant-time-compared token.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import spark_control_panel as scp  # noqa: E402


# --- allowlist: unknown / free-form / injection ids refused --------------------------------------
def test_unknown_action_refused() -> None:
    for bad in ("nope", "definitely-not-real", "BRIDGE-STATUS"):
        try:
            scp.resolve_action(bad)
            raise AssertionError(f"expected refusal for {bad!r}")
        except KeyError:
            pass


def test_injection_attempt_refused() -> None:
    # The classic RCE shapes — only registry ids are ever accepted, never a command string.
    for evil in (
        "; rm -rf /",
        "bridge-status; ls",
        "bridge-status && curl evil",
        "$(reboot)",
        "`id`",
        "bash -c 'x'",
        "../../etc/passwd",
        "bridge-status\n trainwatch",
    ):
        try:
            scp.resolve_action(evil)
            raise AssertionError(f"expected refusal for {evil!r}")
        except KeyError:
            pass


def test_non_string_ids_refused() -> None:
    for bad in (None, 123, ["bridge-status"], {"action": "x"}):
        try:
            scp.resolve_action(bad)  # type: ignore[arg-type]
            raise AssertionError(f"expected refusal for {bad!r}")
        except KeyError:
            pass


def test_known_ids_resolve() -> None:
    for ok in ("bridge-status", "trainwatch", "gpu-free", "link-results",
               "board-refresh", "cert-t1", "bench-a"):
        entry = scp.resolve_action(ok)
        assert "label" in entry and "gpu" in entry


# --- is_gpu_action -------------------------------------------------------------------------------
def test_is_gpu_action_correct() -> None:
    assert scp.is_gpu_action("cert-t1") is True
    assert scp.is_gpu_action("bench-a") is True
    assert scp.is_gpu_action("bridge-status") is False
    assert scp.is_gpu_action("trainwatch") is False
    assert scp.is_gpu_action("gpu-free") is False
    assert scp.is_gpu_action("link-results") is False
    assert scp.is_gpu_action("board-refresh") is False


# --- argv safety: a list, no shell metacharacters ------------------------------------------------
def test_every_registry_argv_is_a_safe_list() -> None:
    for action_id, entry in scp.ACTION_REGISTRY.items():
        argv = entry["argv"]
        if argv is None:
            continue  # in-process action (gpu-free) — no argv by design
        assert isinstance(argv, list) and argv, f"{action_id}: argv must be a non-empty list"
        for tok in argv:
            assert isinstance(tok, str), f"{action_id}: argv token not a str: {tok!r}"
        assert scp.argv_is_safe(argv), f"{action_id}: argv has shell metacharacters"


def test_argv_is_safe_rejects_shell_metacharacters() -> None:
    assert scp.argv_is_safe([sys.executable, "tools/spark_bridge.py", "status"]) is True
    assert scp.argv_is_safe(None) is True
    assert scp.argv_is_safe(["bash", "-c", "rm -rf /; echo $HOME"]) is False
    assert scp.argv_is_safe(["python", "x | y"]) is False
    assert scp.argv_is_safe(["python", "x > /tmp/out"]) is False
    assert scp.argv_is_safe(["python", "$(reboot)"]) is False
    assert scp.argv_is_safe([]) is False
    assert scp.argv_is_safe("not a list") is False  # type: ignore[arg-type]


def test_no_action_uses_shell_true_metachars() -> None:
    # Defense-in-depth assertion: the registry never smuggles a shell pipeline into argv.
    for entry in scp.ACTION_REGISTRY.values():
        argv = entry["argv"]
        if argv is None:
            continue
        joined = " ".join(argv)
        for meta in (";", "|", "&", "$", "`", ">", "<"):
            assert meta not in joined


# --- auth: localhost read exemption + constant-time token off-localhost ---------------------------
def test_localhost_read_exemption() -> None:
    # A read on localhost needs no token.
    assert scp.auth_ok("127.0.0.1", None, None, is_read=True) is True
    assert scp.auth_ok("localhost", None, None, is_read=True) is True
    assert scp.auth_ok("::1", None, None, is_read=True) is True


def test_localhost_write_allowed_without_token() -> None:
    assert scp.auth_ok("127.0.0.1", None, None, is_read=False) is True


def test_remote_requires_token() -> None:
    # No token configured -> any non-localhost request refused, read or write.
    assert scp.auth_ok("10.0.0.5", None, None, is_read=False) is False
    assert scp.auth_ok("10.0.0.5", None, None, is_read=True) is False
    assert scp.auth_ok("0.0.0.0", None, None, is_read=True) is False


def test_remote_token_must_match() -> None:
    assert scp.auth_ok("10.0.0.5", "secret", "secret", is_read=False) is True
    assert scp.auth_ok("10.0.0.5", "secret", "wrong", is_read=False) is False
    assert scp.auth_ok("10.0.0.5", "secret", None, is_read=False) is False
    assert scp.auth_ok("10.0.0.5", "secret", "", is_read=False) is False


def test_configured_token_enforced_even_on_localhost_writes() -> None:
    # Once a token is set, writes require it regardless of host.
    assert scp.auth_ok("127.0.0.1", "secret", "secret", is_read=False) is True
    assert scp.auth_ok("127.0.0.1", "secret", "wrong", is_read=False) is False


# --- start refusal: non-localhost host without a token must refuse to start -----------------------
def test_host_requires_token() -> None:
    assert scp.host_requires_token("0.0.0.0") is True
    assert scp.host_requires_token("10.0.0.5") is True
    assert scp.host_requires_token("203.0.113.7") is True
    assert scp.host_requires_token("127.0.0.1") is False
    assert scp.host_requires_token("localhost") is False
    assert scp.host_requires_token("::1") is False


def test_main_refuses_to_start_remote_without_token() -> None:
    # main() must return non-zero (refuse) before ever binding a socket. No socket is bound because
    # the token check fails first.
    rc = scp.main(["--host", "10.0.0.5", "--port", "0"])
    assert rc == 2


# --- public payload never leaks argv -------------------------------------------------------------
def test_public_actions_shape() -> None:
    actions = scp.public_actions()
    ids = {a["id"] for a in actions}
    assert "bridge-status" in ids and "cert-t1" in ids
    for a in actions:
        assert set(a.keys()) == {"id", "label", "gpu"}, f"leaked field in {a}"


# --- selftest mirror -----------------------------------------------------------------------------
def test_selftest_passes() -> None:
    assert scp._selftest() == 0


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"ok {_name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {exc}")
    print("all passed" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
