# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Aggregate multi-seed RLVR adapter runs under the no-overclaim discipline, and
record a replicated adapter into the compounding registry as canonical.

Implements the post-live-run steps:
  - aggregate N independent-seed adapter-evals (mean/min/max/spread, promotes/N);
  - a HARDENED verdict (require a minimum capability delta AND no protected
    regression AND no contamination) so "promote" is not cheap;
  - record each seed run into `agent/ssil_registry.Registry`; the adapter config
    (seed excluded) becomes CANONICAL only after N independent replications
    (`no_self_promotion_of_candidates`), and a future adapter must beat the
    canonical mean to count as a real improvement (compounding on real weights).

Honesty: aggregation over a handful of seeds is an *aggregated gate result*, not a
statistically validated capability claim. Output keeps `candidateOnly: true`,
`level3Evidence: false`, `canClaimAGI: false`.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from agent.ssil_registry import Registry

MIN_CAPABILITY_DELTA = 0.03  # after - before must clear this to be a real gain
MAX_PROTECTED_REGRESSION = 0.0  # integrity (1 - FP rate) may not drop at all


@dataclass(frozen=True)
class SeedRun:
    seed: int
    before: float
    after: float
    protected_before: float
    protected_after: float
    contaminated: bool = False

    @property
    def delta(self) -> float:
        return round(self.after - self.before, 4)

    @property
    def protected_delta(self) -> float:
        return round(self.protected_after - self.protected_before, 4)


def harden_verdict(run: SeedRun, *, min_delta: float = MIN_CAPABILITY_DELTA,
                   baseline_after: float | None = None) -> tuple[str, tuple[str, ...]]:
    """Stricter per-run gate than the raw Layer-1 demo. ``baseline_after`` (the
    current canonical mean) makes the gain be measured against the best prior adapter,
    not the base model — the compounding requirement."""
    reasons: list[str] = []
    ref = baseline_after if baseline_after is not None else run.before
    gain = round(run.after - ref, 4)
    if run.contaminated:
        reasons.append("contaminated (entity intersection non-empty)")
    if run.protected_delta < -MAX_PROTECTED_REGRESSION if MAX_PROTECTED_REGRESSION > 0 else run.protected_delta < 0:
        reasons.append(f"protected integrity regressed: {run.protected_delta}")
    if gain < min_delta:
        ref_name = "canonical" if baseline_after is not None else "base"
        reasons.append(f"capability gain over {ref_name} below floor: {gain} < {min_delta}")
    return ("promote" if not reasons else "reject", tuple(reasons))


@dataclass
class AdapterAggregate:
    adapter_id: str
    config: dict[str, Any]
    runs: list[SeedRun] = field(default_factory=list)
    canonical_n: int = 3
    min_delta: float = MIN_CAPABILITY_DELTA

    def summary(self, *, baseline_after: float | None = None) -> dict[str, Any]:
        befores = [r.before for r in self.runs]
        afters = [r.after for r in self.runs]
        deltas = [r.delta for r in self.runs]
        verdicts = [harden_verdict(r, min_delta=self.min_delta, baseline_after=baseline_after) for r in self.runs]
        promotes = sum(1 for v, _ in verdicts if v == "promote")
        any_regression = any(r.protected_delta < 0 for r in self.runs)
        any_contam = any(r.contaminated for r in self.runs)
        n = len(self.runs)
        mean_delta = round(statistics.mean(deltas), 4) if deltas else 0.0
        stdev_delta = round(statistics.stdev(deltas), 4) if n > 1 else 0.0
        # "Claim-ready" = cleared the repo's multi-run no-overclaim BAR (>=N, all promote,
        # positive mean gain, no regression/contamination). NOT a validated claim.
        claim_ready = (n >= self.canonical_n and promotes == n and mean_delta > 0
                       and not any_regression and not any_contam)
        return {
            "schema": "sophia.rlvr_aggregate.v1",
            "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
            "adapterId": self.adapter_id,
            "config": self.config,
            "seeds": [r.seed for r in self.runs],
            "n": n,
            "perSeed": [
                {"seed": r.seed, "before": r.before, "after": r.after, "delta": r.delta,
                 "protectedDelta": r.protected_delta, "contaminated": r.contaminated,
                 "verdict": v, "reasons": list(rs)}
                for r, (v, rs) in zip(self.runs, verdicts)
            ],
            "capability": {
                "meanBefore": round(statistics.mean(befores), 4) if befores else 0.0,
                "meanAfter": round(statistics.mean(afters), 4) if afters else 0.0,
                "meanDelta": mean_delta, "minDelta": min(deltas) if deltas else 0.0,
                "maxDelta": max(deltas) if deltas else 0.0, "stdevDelta": stdev_delta,
            },
            "promotes": promotes,
            "anyProtectedRegression": any_regression,
            "anyContaminated": any_contam,
            "baselineAfter": baseline_after,
            "capabilityClaimReady": claim_ready,
            "boundary": ("Aggregated gate result over a few seeds, under the no-overclaim "
                         "measurement gate. NOT a validated capability claim; n is small. "
                         "canClaimAGI=false."),
        }

    def record_to_registry(self, registry: Registry, *, baseline_after: float | None = None) -> dict[str, Any]:
        """Record each seed run with its real per-seed verdict; the config canonicalizes
        only if it BOTH replicated N times AND cleared the no-overclaim bar
        (capabilityClaimReady = all seeds promote, no regression/contamination). A 2/3
        result is recorded for provenance but does NOT become canonical — the loop must
        not build the next generation on a foundation the gate rejected on some seeds."""
        summary = self.summary(baseline_after=baseline_after)
        per_verdict = {p["seed"]: p["verdict"] for p in summary["perSeed"]}
        for r in self.runs:
            registry.record(entry_id=f"{self.adapter_id}-seed{r.seed}", round_idx=r.seed,
                            spec=self.config, metric=r.after, parent=None,
                            gate_verdicts={"layer1": per_verdict.get(r.seed, "reject")})
        replications = registry.replication_count(self.config)
        claim_ready = summary["capabilityClaimReady"]
        is_canon = bool(claim_ready and replications >= registry.canonical_n)
        afters = [r.after for r in self.runs]
        canonical_mean_after = round(statistics.mean(afters), 4) if is_canon else None
        return {
            "adapterId": self.adapter_id,
            "replications": replications,
            "capabilityClaimReady": claim_ready,
            "canonical": is_canon,
            "canonicalMeanAfter": canonical_mean_after,
            "nextAdapterMustBeat": canonical_mean_after,
        }


def runs_from_eval_reports(reports: list[dict[str, Any]], adapter_id: str = "sophia-rlvr-v1") -> AdapterAggregate:
    """Build an aggregate from raw eval_rlvr_adapter reports (reuses the ingest mapping)."""
    from tools.ingest_rlvr_eval import map_report

    seed_runs: list[SeedRun] = []
    config: dict[str, Any] = {}
    for i, rep in enumerate(reports):
        m = map_report(rep, adapter_id=adapter_id)
        seed = int(rep.get("split", {}).get("seed", i)) if isinstance(rep.get("split"), dict) else i
        seed_runs.append(SeedRun(seed=seed, before=m["before"], after=m["after"],
                                 protected_before=m["protected_before"], protected_after=m["protected_after"],
                                 contaminated=m["contaminated"]))
        config = {"adapter": adapter_id, "model": rep.get("model"), "kind": "lora_adapter"}
    return AdapterAggregate(adapter_id=adapter_id, config=config, runs=seed_runs)
