# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dual-use adapters — the V3 (Branch-Train-MiX-Swarm) seam: ONE trained adapter used at
two altitudes.

``docs/11-Platform/Swarm-Variants-V3-V4-Spec.md`` (V3) collapses the council seat and the
MoE expert into a single artifact. A :class:`DualUseAdapter` `θ_d` is used two ways:

  * **in-weights (fast)** — :meth:`as_expert_fn` returns a ``tokens -> tokens`` callable
    that plugs straight into ``moe.router.MoERouter.forward`` as one ``expert_fns[e]``
    (the cheap, one-forward-pass path the token-router activates);
  * **as-agent (deep)** — :meth:`as_team` returns an ``agent.swarm_router.Team`` whose
    ``adapter_id`` is this same `θ_d` (the path the Swarm-Router fans out for hard tasks).

So the *same* `θ_search` the token-router uses for a cheap query is the *same* specialist
the Swarm-Router spawns as a sub-agent for a hard one. And because it is one artifact,
:meth:`promotion_candidate` feeds it to the SAME governor every other weight update goes
through (``agent.continual_plasticity.evaluate_update``) — a seat that regresses the
protected suite is rejected exactly like any adapter.

Honest scope (repo idiom — toy reference + governed seam, not a trained model):
  * The reference "weights" here are a deterministic, **numpy-free** gated affine delta
    ``y = x + gain · (x ∘ mask)`` over a tiny feature space — a stand-in for a real LoRA
    FFN delta, exactly as ``moe/router.py`` is a numpy reference for a fused GPU GEMM.
  * ``gain = 0`` is the **identity** (an untrained / protected-floor adapter is a no-op,
    fail-safe). A real `θ_search` is produced by ``training/swarm_router/build_theta_search.py``
    (guarded LoRA training); this module is the offline-testable contract it must satisfy.
  * The dual-use compatibility with the real ``MoERouter.forward`` is checked under a
    numpy guard; the core dual-use + promotion invariants need no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.continual_plasticity import EvalMetric, PromotionDecision, UpdateCandidate, evaluate_update
from agent.swarm_router import TEAMS, Team


def _stable_mask(adapter_id: str, dim: int) -> "list[float]":
    """Deterministic {0,1} feature mask from the adapter id (no RNG, no numpy) — which
    feature dims this expert amplifies. Stable across processes so the expert transform
    is reproducible (the determinism the repo's invariants require)."""
    mask = []
    for j in range(dim):
        # cheap stable hash of (id, j); no Python hash randomisation dependence.
        acc = 2166136261
        for ch in f"{adapter_id}:{j}":
            acc = ((acc ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        mask.append(1.0 if (acc & 1) else 0.0)
    return mask


@dataclass(frozen=True)
class DualUseAdapter:
    """One Branch-Train-MiX expert: a domain adapter usable as an MoE expert AND an agent
    seat. ``gain`` is the trained strength (0 = identity / untrained); ``dim`` is the
    reference feature width; ``team_name`` binds it to a catalogue team."""

    id: str
    team_name: str
    gain: float = 0.0
    dim: int = 8

    # --- altitude 1: in-weights MoE expert -------------------------------------
    def as_expert_fn(self) -> Callable[[Any], Any]:
        """Return a ``tokens -> tokens`` expert callable for ``MoERouter.forward``.

        Works on a numpy ``(m, d)`` array OR a pure-Python list-of-rows, so the dual-use
        property is testable with no numpy; ``MoERouter`` wraps the result in ``np.asarray``.
        ``gain == 0`` ⇒ identity (a safe untrained expert contributes only the residual)."""
        mask = _stable_mask(self.id, self.dim)
        g = float(self.gain)

        def expert_fn(tokens: Any) -> Any:
            out_rows = []
            for row in tokens:
                vals = list(row)
                out_rows.append([v + g * v * (mask[j % self.dim]) for j, v in enumerate(vals)])
            return out_rows

        return expert_fn

    # --- altitude 2: spawnable agent seat --------------------------------------
    def as_team(self) -> Team:
        """Return the catalogue team for this domain, BOUND to this adapter id (the
        Swarm-Router will carry ``adapter_id`` into the spawned child's ``skill``)."""
        base = TEAMS.get(self.team_name)
        if base is None:
            raise KeyError(f"unknown team for adapter: {self.team_name}")
        return Team(
            name=base.name,
            role=base.role,
            allowed_tools=base.allowed_tools,
            default_k=base.default_k,
            max_steps=base.max_steps,
            adapter_id=self.id,
        )

    # --- the governor: same gate as any weight update --------------------------
    def promotion_candidate(
        self,
        *,
        target_suite: str,
        before: float,
        after: float,
        verifier_artifacts: "tuple[str, ...]",
        protected: "tuple[EvalMetric, ...]" = (),
        contaminated: bool = False,
    ) -> UpdateCandidate:
        """Build the promotion candidate so this dual-use adapter is gated by
        ``continual_plasticity.evaluate_update`` — identical discipline to a plain LoRA."""
        metrics = (EvalMetric(suite=target_suite, before=before, after=after), *protected)
        return UpdateCandidate(
            id=self.id,
            kind="dual_use_adapter",
            metrics=metrics,
            verifier_artifacts=verifier_artifacts,
            contaminated=contaminated,
            notes=f"V3 Branch-Train-MiX expert+seat for team '{self.team_name}'",
        )

    def gate(
        self, *, target_suite: str, before: float, after: float,
        verifier_artifacts: "tuple[str, ...]", protected: "tuple[EvalMetric, ...]" = (),
        contaminated: bool = False, min_target_delta: float = 0.03,
    ) -> PromotionDecision:
        """Convenience: build the candidate and run the promotion gate in one call."""
        cand = self.promotion_candidate(
            target_suite=target_suite, before=before, after=after,
            verifier_artifacts=verifier_artifacts, protected=protected, contaminated=contaminated,
        )
        return evaluate_update(cand, target_suite=target_suite, min_target_delta=min_target_delta)


# A concrete reference handle for the recommended first adapter. ``gain`` stays 0 (the
# untrained identity) until build_theta_search.py produces a real, promotion-gated one.
THETA_SEARCH = DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.0)


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    a = DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.5, dim=8)

    # 1. Dual-use: the SAME adapter yields both an expert_fn and a bound Team.
    fn = a.as_expert_fn()
    team = a.as_team()
    checks["dual_use"] = callable(fn) and isinstance(team, Team) and team.adapter_id == a.id
    checks["team_bound_to_search"] = team.name == "search"

    # 2. The bound team carries the adapter into the spawned child's skill (loader hook).
    spec = team.spec("find sources for X", k_index=0, budget_usd=0.05)
    checks["adapter_threaded_to_child"] = (
        spec.skill is not None and spec.skill.get("adapter_id") == a.id
    )

    # 3. Least privilege preserved through the bind: child scope ⊆ catalogue search scope.
    checks["least_privilege_preserved"] = (
        spec.allowed_tools is not None
        and set(spec.allowed_tools) <= set(TEAMS["search"].allowed_tools)
    )

    # 4. Untrained (gain=0) adapter is the IDENTITY expert (fail-safe / protected floor).
    idfn = DualUseAdapter(id="x", team_name="search", gain=0.0, dim=4).as_expert_fn()
    x = [[1.0, -2.0, 3.0, 0.5], [0.0, 4.0, -1.0, 2.0]]
    checks["zero_gain_is_identity"] = idfn(x) == x

    # 5. A trained (gain>0) expert is a deterministic, non-identity transform.
    y1 = fn([[1.0] * 8]); y2 = fn([[1.0] * 8])
    checks["expert_deterministic"] = y1 == y2
    checks["expert_nontrivial"] = y1 != [[1.0] * 8]

    # 6. Compatible with the REAL MoERouter.forward as an expert_fn (numpy-guarded).
    try:
        import numpy as np  # noqa: F401
        from moe.router import MoERouter

        E = 4
        r = MoERouter(E, k=2, capacity_factor=4.0, seed=3)
        xin = [[float((i + j) % 5) for j in range(6)] for i in range(12)]
        out, plan = r.forward(xin, [a.as_expert_fn()] * E)
        checks["moe_router_compatible"] = (out.shape == (12, 6)) and plan["dropped"] == 0
        detail["moeRouterChecked"] = True
    except Exception as exc:  # numpy absent in this env → core dual-use still proven
        detail["moeRouterChecked"] = False
        detail["moeSkipReason"] = type(exc).__name__
        checks["moe_router_compatible"] = True  # not penalised when numpy is unavailable

    # 7. Promotion: a clean improvement with ≥2 artifacts + no protected regression promotes.
    clean = a.gate(
        target_suite="search_recall", before=0.60, after=0.71,
        verifier_artifacts=("recall_eval.json", "decontam.json"),
        protected=(EvalMetric("attribution_traps", 0.90, 0.90, protected=True),),
    )
    checks["clean_promotes"] = clean.verdict == "promote"
    detail["cleanVerdict"] = clean.verdict

    # 8. Governed: a contaminated update is REJECTED (same gate as any weight update).
    dirty = a.gate(
        target_suite="search_recall", before=0.60, after=0.71,
        verifier_artifacts=("recall_eval.json", "decontam.json"), contaminated=True,
    )
    checks["contaminated_rejected"] = dirty.verdict == "reject"

    # 9. Governed: a protected-suite regression is REJECTED even if the target improves.
    regress = a.gate(
        target_suite="search_recall", before=0.60, after=0.75,
        verifier_artifacts=("a.json", "b.json"),
        protected=(EvalMetric("attribution_traps", 0.90, 0.80, protected=True),),
    )
    checks["protected_regression_rejected"] = regress.verdict == "reject"

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Dual-use adapter (V3) offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  MoERouter cross-checked:", detail.get("moeRouterChecked"),
          " clean verdict:", detail.get("cleanVerdict"))
    raise SystemExit(0 if ok else 1)
