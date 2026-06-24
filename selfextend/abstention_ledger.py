# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Abstention ledger — turns 'I don't know' into a self-directed learning agenda.

Every abstention is a localized gap. Logging them yields the curiosity agenda: the
domains the system most often cannot verify, ranked — exactly where synthesizing a
new verifier (or acquiring grounding) buys the most coverage. JSONL-backed
(inspectable) with an in-memory mode.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class AbstentionLedger:
    def __init__(self, path: "Path | None" = None):
        self.path = Path(path) if path else None
        self.entries: list = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        self.entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def record(self, *, domain: str, query: str = "", reason: str = "no_verifier") -> dict:
        row = {"domain": domain, "query": query, "reason": reason}
        self.entries.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def agenda(self, top_n: int = 5) -> "list[tuple[str, int]]":
        """The most-abstained domains, ranked — what to learn next."""
        return Counter(e["domain"] for e in self.entries).most_common(top_n)

    def gap_domains(self) -> "set[str]":
        return {e["domain"] for e in self.entries}
