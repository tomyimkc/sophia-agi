#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cloud-side client for the Spark git bridge (the operator half of bridge/PROTOCOL.md).

The Spark runs ``tools/github_bridge_poll.py`` and talks ONLY to GitHub (the cloud session cannot
reach the Spark's Tailscale Funnel — egress returns ``connect_rejected 502``). This is the cloud
half: it **validates + composes** a command for ``bridge/commands/<id>.json`` and **reads** status +
results from the ``spark-bridge`` branch, so a cloud agent can drive the Spark safely and the same
discipline is reusable across sessions.

Safety, enforced here AND by the poller:
  * ``args`` tokens must ALL be in the allowlist; anything else is rejected before submit.
  * ``--execute`` / ``--run-train`` are GATED: a command carrying them MUST have a non-empty
    ``approvedBy`` (a human handle) or this client refuses to build it — mirroring the poller, which
    runs gated flags only with human approval. An AI does not self-approve a GPU job.
  * ``gpu_is_free(status)`` lets the operator hold the one-GPU-job invariant before an execute.

Reads use ``git show origin/spark-bridge:<path>`` (read-only); the actual commit of a command file
is done by the caller via the GitHub API. Pure validation/compose logic is deterministic + offline
(tested); the git reads are exercised live, not in CI. canClaimAGI stays false.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BRANCH = "spark-bridge"
ALLOWLIST = {"--dry-run", "--bench-a", "--bench-b", "--all", "--execute", "--run-train"}
GATED = {"--execute", "--run-train"}  # require a human approvedBy

# Config env vars a command may carry so cert/train jobs run THROUGH the bridge (results then land
# in bridge/results/ where the cloud reads them — no direct-SSH, no pasting). STRICTLY allowlisted:
# only these run_local_benchmarks.sh / certify_lowram knobs, never arbitrary env (no PATH/LD_*).
ENV_ALLOWLIST = frozenset({
    "SEEDS", "ANSWERS_DIR", "ANSWERS_PREFIX", "JUDGE_OUT_DIR", "JUDGMENTS", "UPLIFT_OUT",
    "JUDGES", "JUDGE_CONFIG", "AUTO_START_JUDGES", "PYTHON",
    "QAT_BASE", "QAT_ADAPTER", "QAT_DATA", "QAT_EPOCHS", "QAT_LAMBDA",
    "KEEP_SUFFIXES", "CERT_NEVAL", "CERT_CALIB", "CERT_OUT",
    "SPARK_HOST", "MAC_HOST", "SPARK_PORT", "MAC_PORT", "SPARK_JUDGE_MODEL", "MAC_JUDGE_MODEL",
    "VIRTUE_JUDGE_A", "VIRTUE_JUDGE_B", "VIRTUE_JUDGE_A_NAME", "VIRTUE_JUDGE_B_NAME",
    "VIRTUE_SUBJECT", "VIRTUE_SEEDS", "THINKING_MODEL", "FAITH_MODEL", "FAITH_SEEDS", "FAITH_BATTERY",
})
# Values are EXPORTED then used inside the script (e.g. ${QAT_ADAPTER}); reject anything that could
# break out of a quoted expansion. Allows model ids, @-urls, paths, comma/space lists, suffixes.
_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@,+= -]{0,200}$")


def parse_env(env_str: str) -> dict:
    """Parse 'K=V,K2=V2' into a dict (values may contain '=' after the first; commas split pairs)."""
    out: dict[str, str] = {}
    for pair in (env_str or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"env entry not K=V: {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def validate_env(env: dict) -> "tuple[bool, list[str]]":
    problems: list[str] = []
    for k, v in (env or {}).items():
        if k not in ENV_ALLOWLIST:
            problems.append(f"env key not allowlisted: {k!r}")
        if not _ENV_VALUE_RE.match(v or ""):
            problems.append(f"env value has unsafe chars or is too long: {k}={v!r}")
    return (not problems), problems


def validate_args(args: str) -> "tuple[bool, list[str]]":
    """Every token must be allowlisted. Returns (ok, problems)."""
    problems: list[str] = []
    tokens = [t for t in (args or "").split() if t]
    if not tokens:
        problems.append("empty args")
    for t in tokens:
        if t not in ALLOWLIST:
            problems.append(f"token not allowlisted: {t!r}")
    return (not problems), problems


def is_gated(args: str) -> bool:
    return any(t in GATED for t in (args or "").split())


def build_command(command_id: str, args: str, *, created_by: str, approved_by: str = "",
                  note: str = "", created_at: str = "", env: "dict | None" = None) -> dict:
    """Validate + compose a bridge command dict. Raises ValueError on an allowlist violation, on a
    GATED command with no human ``approved_by`` (the no-self-approval rule), or on a disallowed/
    unsafe ``env`` entry. ``env`` (config knobs only) lets cert/train run THROUGH the bridge."""
    ok, problems = validate_args(args)
    if not ok:
        raise ValueError(f"args rejected: {problems}")
    if is_gated(args) and not approved_by.strip():
        raise ValueError("gated flag (--execute/--run-train) requires a non-empty human approvedBy "
                         "— an AI does not self-approve a GPU job")
    if not command_id or any(c in command_id for c in "/ \t"):
        raise ValueError("id must be filesystem-safe and non-empty")
    env = env or {}
    env_ok, env_problems = validate_env(env)
    if not env_ok:
        raise ValueError(f"env rejected: {env_problems}")
    cmd = {
        "id": command_id, "args": args, "createdBy": created_by,
        "createdAt": created_at, "approvedBy": approved_by, "note": note,
    }
    if env:
        cmd["env"] = env
    return cmd


def gpu_is_free(status: dict) -> bool:
    """The one-GPU-job invariant guard: free iff nothing running and nothing pending."""
    return (status or {}).get("running") in (None, "", "null") and not (status or {}).get("pendingCommands")


# --- live reads (git, read-only) --------------------------------------------------------------
def _git_show(path: str) -> "str | None":
    try:
        subprocess.run(["git", "fetch", "origin", BRANCH, "--quiet"], cwd=ROOT,
                       check=False, capture_output=True, timeout=60)
        out = subprocess.run(["git", "show", f"origin/{BRANCH}:{path}"], cwd=ROOT,
                             check=True, capture_output=True, text=True, timeout=30)
        return out.stdout
    except Exception:
        return None


def read_status() -> "dict | None":
    raw = _git_show("bridge/STATUS.json")
    return json.loads(raw) if raw else None


def read_result(command_id: str) -> "dict | None":
    raw = _git_show(f"bridge/results/{command_id}.json")
    return json.loads(raw) if raw else None


def _fmt_eta(seconds) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    if s <= 0:
        return "—"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def format_trainwatch(status: "dict | None") -> str:
    """Human ETA/progress view over the bridge's mirrored TrainWatch field.

    Pure + offline (no git, no network) so it is unit-testable; the poller embeds LIVE
    trainwatch each tick (``github_bridge_poll._trainwatch`` -> localhost:8420), so this renders
    real step/ETA/loss during a multi-hour train as seen from the cloud over the git bridge."""
    if not status:
        return "no STATUS.json (bridge unreachable or not fetched)"
    running = status.get("running")
    runs = status.get("trainwatch") or []
    lines = [f"updatedAt: {status.get('updatedAt')}  |  running job: {running or '—'}"]
    active = [r for r in runs if (r.get("status") or "").lower() in ("running", "training", "active")]
    shown = active or runs[:3]  # prefer live runs; else the most recent few
    if not shown:
        lines.append("  trainwatch: (no runs)")
    for r in shown:
        cs, ts = r.get("current_step"), r.get("total_steps")
        pct = f"{100*cs/ts:.0f}%" if (isinstance(cs, (int, float)) and ts) else "?"
        flag = "▶ LIVE" if r in active else r.get("status", "?")
        lines.append(f"  [{flag}] {r.get('name')}  step {cs}/{ts} ({pct})  "
                     f"ETA {_fmt_eta(r.get('eta_seconds'))}  {r.get('latest_metrics') or ''}".rstrip())
    return "\n".join(lines)


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    checks["allow_dryrun"] = validate_args("--dry-run --all")[0]
    checks["reject_unknown"] = not validate_args("--rm -rf /")[0]
    checks["reject_empty"] = not validate_args("")[0]
    checks["gated_detected"] = is_gated("--bench-a --execute") and not is_gated("--dry-run --all")

    # build refuses a gated command with no approval, allows it with one
    refused = False
    try:
        build_command("x", "--bench-a --execute", created_by="claude")
    except ValueError:
        refused = True
    checks["gated_without_approval_refused"] = refused
    ok_cmd = build_command("x", "--bench-a --execute", created_by="claude",
                           approved_by="user: 'run it' (2026)")
    checks["gated_with_approval_built"] = ok_cmd["approvedBy"] != ""
    checks["dryrun_no_approval_ok"] = build_command("y", "--dry-run --all", created_by="claude")["args"] == "--dry-run --all"

    checks["gpu_free_logic"] = gpu_is_free({"running": None, "pendingCommands": []}) and \
        not gpu_is_free({"running": "olmoe", "pendingCommands": []}) and \
        not gpu_is_free({"running": None, "pendingCommands": ["z"]})

    # id safety
    bad_id = False
    try:
        build_command("a/b c", "--dry-run", created_by="claude")
    except ValueError:
        bad_id = True
    checks["unsafe_id_refused"] = bad_id

    # env allowlist: known knob accepted, arbitrary/unsafe env refused
    checks["env_allowlisted_ok"] = build_command(
        "e1", "--bench-b --execute", created_by="claude", approved_by="user: go",
        env={"KEEP_SUFFIXES": "down_proj", "QAT_ADAPTER": "training/lora/checkpoints/olmoe-qat-spark-v3"},
    ).get("env", {}).get("KEEP_SUFFIXES") == "down_proj"
    bad_env_key = bad_env_val = False
    try:
        build_command("e2", "--dry-run", created_by="claude", env={"PATH": "/evil"})
    except ValueError:
        bad_env_key = True
    try:
        build_command("e3", "--dry-run", created_by="claude", env={"QAT_ADAPTER": "x; rm -rf /"})
    except ValueError:
        bad_env_val = True
    checks["env_arbitrary_key_refused"] = bad_env_key
    checks["env_unsafe_value_refused"] = bad_env_val

    return all(checks.values()), {"checks": checks}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--self-test", action="store_true")

    pc = sub.add_parser("compose", help="validate + print a command JSON (does NOT submit)")
    pc.add_argument("--id", required=True)
    pc.add_argument("--args", required=True)
    pc.add_argument("--created-by", default="claude-web")
    pc.add_argument("--approved-by", default="")
    pc.add_argument("--note", default="")
    pc.add_argument("--created-at", default="")
    pc.add_argument("--env", default="", help="config env to carry, 'KEY=VAL,KEY2=VAL2' "
                    "(allowlisted run_local_benchmarks/certify knobs only) so the job runs THROUGH "
                    "the bridge and its result lands in bridge/results/")

    sub.add_parser("status", help="print the live Spark STATUS.json")
    sub.add_parser("trainwatch", help="print the live training ETA/progress (mirrored TrainWatch)")
    pr = sub.add_parser("result", help="print a result by id")
    pr.add_argument("--id", required=True)

    args = ap.parse_args(argv)
    if args.self_test or args.cmd is None and not any(vars(args).get(k) for k in ("cmd",)):
        if args.self_test:
            ok, detail = offline_invariants()
            print("spark_bridge invariants:", "PASS" if ok else "FAIL")
            for k, v in detail["checks"].items():
                print(f"  [{'ok' if v else 'XX'}] {k}")
            return 0 if ok else 1

    if args.cmd == "compose":
        try:
            cmd = build_command(args.id, args.args, created_by=args.created_by,
                                approved_by=args.approved_by, note=args.note, created_at=args.created_at,
                                env=parse_env(getattr(args, "env", "")))
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(cmd, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "status":
        print(json.dumps(read_status(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == "trainwatch":
        # read_status() git-fetches origin/spark-bridge via _git_show, so this is always fresh.
        print(format_trainwatch(read_status()))
        return 0
    if args.cmd == "result":
        print(json.dumps(read_result(args.id), indent=2, ensure_ascii=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
