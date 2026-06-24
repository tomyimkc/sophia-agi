#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Continual learning without catastrophic forgetting — measured, offline.

Learns a stream of "tasks" (batches of OKF pages) one after another and shows that
a neural net's nemesis — catastrophic forgetting — does not occur when declarative
knowledge lives in a provenance-typed belief graph instead of shared weights. Then
it removes a source to show forgetting is *detectable and on purpose*, not silent.

    python scripts/demo_continual_retention.py

No network, no API key. Deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_retention import Snapshot, build_report, run_stream, Task  # noqa: E402
from okf.page import Page  # noqa: E402


def line(title: str) -> None:
    print(f"\n{'─' * 64}\n{title}\n{'─' * 64}")


def page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def main() -> None:
    line("Stream of 4 tasks learned sequentially (purely additive)")
    tasks = [
        Task("philosophy", (page("analects_compiled", authorConfidence="compiled"),
                            page("dao_de_jing_layered", authorConfidence="layered"))),
        Task("psychology", (page("cognitive_dissonance", authorConfidence="attributed"),)),
        Task("history", (page("printing_press_1440", authorConfidence="consensus"),)),
        Task("religion", (page("gospel_layered", authorConfidence="layered"),
                          page("hadith_compiled", derivesFrom=["gospel_layered"], authorConfidence="compiled"))),
    ]
    report = run_stream(tasks)
    for i, tid in enumerate(report["tasks"]):
        row = report["retentionMatrix"][i]
        cells = " ".join("   . " if v is None else f"{v:>4.2f} " for v in row)
        print(f"  task {tid:<11} facts={len(report['factsPerTask'][tid])}  retention→ {cells}")
    print(f"\n  total grounded facts learned : {report['totalGroundedFacts']}")
    print(f"  forgotten grounded claims    : {report['forgottenGroundedClaims']}")
    print(f"  backward transfer            : {report['backwardTransfer']}  (0.0 == no forgetting)")
    print(f"  perfect retention            : {report['perfectRetention']}")

    line("Contrast: the SAME metric detects forgetting when knowledge is removed")
    print("  (a weight model overwrites silently; here it is measured and named)")
    snaps = [
        Snapshot("learn_fact", {"fact_x": 4, "fact_y": 3}, ("fact_x", "fact_y")),
        Snapshot("source_retracted", {"fact_y": 3}, ()),   # fact_x lost its ground
    ]
    contrast = build_report(snaps)
    print(f"  forgotten grounded claims    : {contrast['forgottenGroundedClaims']}")
    for d in contrast["forgottenDetail"]:
        print(f"    - {d['fact']} ({d['reason']}, introduced in '{d['introducedInTask']}')")

    out = ROOT / "agi-proof" / "continual" / "retention_report.json"
    written = __import__("json").dumps(report, indent=2, ensure_ascii=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(written + "\n", encoding="utf-8")
    line("Report written")
    print(f"  {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
