# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Promoted-improvement registry — provenance, N-replication, counterfactual revert.

This is SSIL gate G9 made operational and the substrate of *compounding*: promoted
self-modifications are recorded append-only with lineage, and a gain becomes
**canonical** (and thus the new baseline future rounds must beat) only after it has
been independently re-confirmed N times (`no_self_promotion_of_candidates`). The
"recursive" in RSI is literal here — round k+1 measures improvement against the
canonical state produced by round k.

Counterfactual revert ("moral bisect"): rebuild the canonical state with any entry
ids excluded, to find which self-edit a regression came from and revert it.

Generic over the improvement payload: an entry carries a `spec` dict and a scalar
`metric`; the registry does not need to know it is a routing policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _spec_key(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True)


@dataclass
class Registry:
    """Append-only registry of promoted improvements with N-replication promotion."""

    path: Path | None = None
    canonical_n: int = 2
    _log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path = Path(self.path)
            if self.path.exists():
                self._log = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]

    # --- writes ---------------------------------------------------------------
    def record(self, *, entry_id: str, round_idx: int, spec: dict[str, Any], metric: float,
               parent: str | None, gate_verdicts: dict[str, str]) -> dict[str, Any]:
        rec = {
            "id": entry_id, "round": round_idx, "spec": spec, "metric": round(float(metric), 4),
            "parent": parent, "gateVerdicts": gate_verdicts, "specKey": _spec_key(spec),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log.append(rec)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # --- reads ----------------------------------------------------------------
    def entries(self) -> list[dict]:
        return list(self._log)

    def replication_count(self, spec: dict[str, Any], *, exclude: set[str] | None = None) -> int:
        key, exclude = _spec_key(spec), (exclude or set())
        return sum(1 for r in self._log if r["specKey"] == key and r["id"] not in exclude)

    def is_canonical(self, spec: dict[str, Any], *, exclude: set[str] | None = None) -> bool:
        return self.replication_count(spec, exclude=exclude) >= self.canonical_n

    def canonical_best(self, *, exclude: set[str] | None = None) -> dict[str, Any] | None:
        """Highest-metric spec that has reached N independent replications."""
        exclude = exclude or set()
        seen: dict[str, dict] = {}
        for r in self._log:
            if r["id"] in exclude:
                continue
            if self.replication_count(r["spec"], exclude=exclude) >= self.canonical_n:
                cur = seen.get(r["specKey"])
                if cur is None or r["metric"] > cur["metric"]:
                    seen[r["specKey"]] = r
        if not seen:
            return None
        return max(seen.values(), key=lambda r: r["metric"])

    def counterfactual_best(self, exclude_ids: set[str]) -> dict[str, Any] | None:
        """What the canonical best would be if these entries were reverted."""
        return self.canonical_best(exclude=set(exclude_ids))

    def summary(self) -> dict[str, Any]:
        best = self.canonical_best()
        return {
            "schema": "sophia.ssil_registry_summary.v1", "candidateOnly": True, "level3Evidence": False,
            "canonicalN": self.canonical_n, "totalEntries": len(self._log),
            "canonicalBest": best,
            "distinctSpecs": len({r["specKey"] for r in self._log}),
        }
