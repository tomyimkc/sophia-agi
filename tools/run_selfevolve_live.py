#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Wire the self-evolving loop to the LIVE optimizer (RLVR weight update) path (#4).

    python tools/run_selfevolve_live.py [--backend plan|runpod] [--task provenance|math]
                                        [--yes] [--json]

The offline self-evolving loop improves a policy by *verifier-guided selection*. The
live optimizer turns that same verified reward into a real GRPO/QLoRA weight update on
a GPU. That update is CUDA-only and out of scope for CI, so this tool has two backends:

  - ``plan``   (default, CI-safe): run the full offline evolve->gate->retain->commit
               session, assert its invariants, and emit the EXACT RLVR dispatch command
               each committed domain WOULD trigger. No subprocess, no GPU, no network.
  - ``runpod`` (opt-in): actually dispatch ``tools/runpod_rlvr.py`` for each committed
               domain -- dry-run unless ``--yes`` rents a GPU. Requires ssh + a RunPod
               key; never exercised in CI.

The discipline that makes this honest: only domains the agent COMMITTED (cleared the
reward-hack, plasticity, and no-forgetting gates) are ever handed to the optimizer, so
no unverified or reward-hacked update reaches a weight change.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.self_evolving_agent import Experience, SelfEvolvingAgent  # noqa: E402
from okf.page import Page  # noqa: E402

_OBJS = ("the database", "user files", "records", "everything", "the backups",
         "the logs", "all accounts", "the cache", "the index", "the config",
         "the queue", "the secrets")


def _page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def _learnable(token: str) -> tuple:
    out: list = []
    for o in _OBJS:
        out.append((f"{token} {o} now", True))
        out.append((f"read {o} now", False))
    return tuple(out)


def _session() -> "list[Experience]":
    return [
        Experience("danger_intent", _learnable("delete"),
                   (_page("danger_skill", authorConfidence="consensus"),)),
        Experience("question_intent", _learnable("what"),
                   (_page("question_skill", authorConfidence="attributed"),)),
        Experience("noise_domain",
                   tuple((f"alpha{i} beta{i}", i % 2 == 0) for i in range(16)),
                   (_page("noise_skill", authorConfidence="consensus"),)),
    ]


def rlvr_command(domain: str, *, task: str, yes: bool, branch: str = "") -> "list[str]":
    """The exact RLVR dispatch a committed domain would trigger."""
    cmd = ["python", "tools/runpod_rlvr.py", "--task", task, "--name", f"sophia-selfevolve-{domain}"]
    cmd += ["--source", "git"]
    if branch:
        cmd += ["--branch", branch]
    cmd += ["--yes"] if yes else ["--dry-run"]
    return cmd


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["plan", "runpod"], default="plan")
    ap.add_argument("--task", choices=["provenance", "math"], default="provenance")
    ap.add_argument("--branch", default="", help="git branch the GPU pod should clone")
    ap.add_argument("--yes", action="store_true", help="runpod backend: actually rent a GPU (else dry-run)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    # 1. Run the real offline self-evolving loop (CI-gated) and gate the committed set.
    agent = SelfEvolvingAgent()
    report = agent.run_session(_session())
    committed = [o["domain"] for o in report["perRound"] if o["committed"]]
    invariants_ok = all(report["invariants"].values())

    # 2. Compute the RLVR dispatch each committed domain would trigger.
    plans = [
        {"domain": d, "command": rlvr_command(d, task=args.task, yes=args.yes, branch=args.branch)}
        for d in committed
    ]

    out = {
        "schema": "sophia.selfevolve_live_plan.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "backend": args.backend,
        "committedDomains": committed,
        "loopInvariantsHold": invariants_ok,
        "dispatch": plans,
        "note": ("Only committed (gate-clean) domains are handed to the optimizer. "
                 "Live GRPO weight updates run only with --backend runpod --yes."),
    }

    if args.backend == "runpod":
        results = []
        for p in plans:
            proc = subprocess.run(p["command"], cwd=ROOT, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            results.append({"domain": p["domain"], "returncode": proc.returncode,
                            "tail": proc.stdout[-400:] if proc.stdout else ""})
        out["runpodResults"] = results

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"Self-evolving loop -> live optimizer ({args.backend} backend):")
        print(f"  committed domains      : {committed}")
        print(f"  loop invariants hold   : {invariants_ok}")
        print("  RLVR dispatch (gate-clean domains only):")
        for p in plans:
            print(f"    {p['domain']:<16} {' '.join(p['command'])}")
        if args.backend == "plan":
            print("\n  (plan backend: no GPU dispatched. Use --backend runpod [--yes] to run.)")

    # In CI we run the plan backend; success == the offline loop's invariants hold.
    if args.backend == "plan":
        return 0 if invariants_ok else 1
    return 0 if all(r["returncode"] == 0 for r in out.get("runpodResults", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
