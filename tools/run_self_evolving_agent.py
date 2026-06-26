#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the self-evolving agent over a multi-round session and print the result.

    python tools/run_self_evolving_agent.py [--json] [--out PATH]

Offline, deterministic. Demonstrates the moat: an agent that self-improves across
rounds, where every committed improvement cleared four independent trust gates
(evolve+promote, no reward-hack, plasticity, no forgetting), rejected rounds left
memory untouched (fail-closed), and across the committed run nothing was forgotten.
Real weight updates stay behind the RunPod GPU path (tools/runpod_rlvr.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.self_evolving_agent import Experience, SelfEvolvingAgent  # noqa: E402
from okf.page import Page  # noqa: E402

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
        # a round with no learnable signal -> must be rejected, fail-closed
        Experience("noise_domain",
                   tuple((f"alpha{i} beta{i}", i % 2 == 0) for i in range(16)),
                   (_page("noise_skill", authorConfidence="consensus"),)),
    ]


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    agent = SelfEvolvingAgent()
    report = agent.run_session(_session())
    if args.out:
        agent.write_report(args.out)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if all(report["invariants"].values()) else 1

    print("Self-evolving agent session:")
    print(f"  rounds                     : {report['rounds']}")
    print(f"  committed (cleared gates)  : {report['committedRounds']}")
    print(f"  knowledge pages in memory  : {report['knowledgePages']}")
    print(f"  forgotten across run       : {report['forgottenGroundedClaimsAcrossRun']}")
    print("  per-round:")
    for r in report["perRound"]:
        flag = "COMMIT" if r["committed"] else "REJECT"
        print(f"    [{flag}] round {r['round']} {r['domain']:<16} "
              f"pre->post {r['preAccuracy']}->{r['postAccuracy']} "
              f"(+{r['improvement']})  hacked={r['rewardHack'].get('hacked')}  "
              f"forgot={r['forgottenGroundedClaims']}  verdict={r['plasticityVerdict']}")
    print("\n  invariants:")
    for k, v in report["invariants"].items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    return 0 if all(report["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
