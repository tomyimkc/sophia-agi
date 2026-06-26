#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Export the self-evolving agent's COMMITTED rounds as self-distillation data.

    python tools/export_self_evolve_distill.py [--out PATH] [--no-gate] [--json]

Closes the self-distillation loop (#3): the agent's own *verified* skills -- only the
rounds that cleared all four trust gates -- become a student's training data, in the
same `{"messages": [...], "metadata": {...}}` JSONL schema your other distillation
tools emit (consumable by tools/train_lora.py).

Two firewalls keep the dataset honest:
  - only committed rounds are exported (a rejected self-update can't teach the student);
  - every rendered target is re-checked through agent.gate (no gate-dirty text enters).

Offline, deterministic. Default output: training/self_evolve/distill.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402
from agent.self_evolving_agent import Experience, SelfEvolvingAgent  # noqa: E402
from okf.page import Page  # noqa: E402

OUT = ROOT / "training" / "self_evolve" / "distill.jsonl"

_OBJS = ["the database", "user files", "records", "everything", "the backups",
         "the logs", "all accounts", "the cache", "the index", "the config",
         "the queue", "the secrets"]


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
        # rejected (no learnable signal): must NOT contribute any training rows
        Experience("noise_domain",
                   tuple((f"alpha{i} beta{i}", i % 2 == 0) for i in range(16)),
                   (_page("noise_skill", authorConfidence="consensus"),)),
    ]


def _gate_check(target: str, question: str) -> bool:
    """True if the rendered target has gate violations (so it should be dropped)."""
    return bool(check_response(target, mode="advisor", question=question)["violations"])


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-gate", action="store_true", help="skip the gate firewall (not recommended)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    agent = SelfEvolvingAgent()
    agent.run_session(_session())

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    gate = None if args.no_gate else _gate_check
    stats = agent.write_distillation_jsonl(out, gate_check=gate)

    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    if args.json:
        print(json.dumps(stats | {"out": str(shown)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"wrote {stats['rows']} self-distillation rows -> {shown}")
    # A rejected round must contribute nothing: committed rounds < total rounds here.
    return 0 if stats["rows"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
