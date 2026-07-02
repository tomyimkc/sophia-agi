# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Trained router head — the drop-in the SwarmRouter seam was built for (candidate-only).

``agent/swarm_router.py`` documents its v1 policy as a deterministic foothold whose
``decide`` method a *trained* head may replace without changing the SwarmPlan
contract. This module is that head, kept deliberately small and auditable:

  * **Features are the v1 signals** (:class:`agent.swarm_router.RouteSignals`) plus an
    intent one-hot — transparent, deterministic, CPU-only. No embeddings, no torch.
  * **Model = independent logistic heads** (one for swarm-vs-solo, one per team),
    trained by plain gradient descent with zero init and a fixed iteration order, so
    ``fit`` is bit-deterministic for a given tuple file. Pure stdlib.
  * **Training data = outcome-weighted behavior cloning** of the v1 policy over real
    run traces (``tools/mine_router_tuples.py``): a run that ended ``ok`` reinforces
    the plan v1 chose for it; a failed run down-weights it. This is the v0 recipe —
    it can only learn to *re-weight* v1's choices, not invent new routing structure.
  * **Fail-closed wrapper.** :class:`TrainedSwarmRouter` falls back to the v1 policy
    whenever the head is unsure (below threshold, empty task, or no team clears the
    floor) — the audited hand-authored policy is always the floor, never bypassed.

Honest bound (do not skip): this head is **candidate-only**. It must never become the
default until an ablation (the ``mac-mlx-bench`` claim-router lane is the template)
clears ``evaluate_update()`` with protected suites and answerable-coverage intact.
Nothing in this file flips a default. ``canClaimAGI`` stays false.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # `python agent/router_head.py` (invariants CLI)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.swarm_router import (
    DEFAULT_CHILD_BUDGET_USD,
    TEAMS,
    RouteSignals,
    SwarmPlan,
    SwarmRouter,
    TeamAssignment,
)

SCHEMA = "sophia.router_head.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HEAD_PATH = ROOT / "training" / "swarm_router" / "router_head.json"

_INTENTS = ("factoid", "temporal", "navigational", "comparison")
FEATURE_NAMES = (
    "bias", "difficulty", "multiHop", "quant", "legal", "contested", "entityRisk",
    "nSubQueriesNorm",
) + tuple(f"intent:{i}" for i in _INTENTS)


def featurize(sig: RouteSignals) -> "list[float]":
    return [
        1.0,
        float(sig.difficulty),
        1.0 if sig.multi_hop else 0.0,
        1.0 if sig.quant else 0.0,
        1.0 if sig.legal else 0.0,
        1.0 if sig.contested else 0.0,
        1.0 if sig.entity_risk else 0.0,
        min(float(sig.n_sub_queries), 4.0) / 4.0,
    ] + [1.0 if sig.intent == i else 0.0 for i in _INTENTS]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(w: "list[float]", x: "list[float]") -> float:
    return math.fsum(wi * xi for wi, xi in zip(w, x, strict=True))


@dataclass
class RouterHead:
    """Independent logistic heads over the v1 signal features."""

    swarm_w: "list[float]" = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    team_w: "dict[str, list[float]]" = field(default_factory=lambda: {
        t: [0.0] * len(FEATURE_NAMES) for t in sorted(TEAMS)})
    trained_on: int = 0

    # -- inference ---------------------------------------------------------------
    def predict(self, sig: RouteSignals) -> dict:
        x = featurize(sig)
        return {
            "swarmProb": round(_sigmoid(_dot(self.swarm_w, x)), 6),
            "teamProbs": {t: round(_sigmoid(_dot(w, x)), 6)
                          for t, w in sorted(self.team_w.items())},
        }

    # -- training (deterministic; pure stdlib) ------------------------------------
    def fit(self, examples: "list[dict]", *, epochs: int = 200, lr: float = 0.5) -> dict:
        """Outcome-weighted logistic regression. Each example carries the v1 signals
        dict, the v1 plan (mode + teams), and the run outcome ``ok``. Successful runs
        get weight 1.0, failed runs 0.25 — failure does not flip the label (v1 may
        have routed correctly and the child still failed); it lowers confidence."""
        rows = []
        for ex in examples:
            sig = signals_from_dict(ex["signals"])
            x = featurize(sig)
            weight = 1.0 if ex.get("ok") else 0.25
            y_swarm = 1.0 if ex.get("v1Plan", {}).get("mode") == "swarm" else 0.0
            teams = set(ex.get("v1Plan", {}).get("teams") or [])
            rows.append((x, weight, y_swarm, teams))
        if not rows:
            return {"trained": 0}
        n = float(len(rows))
        for _ in range(epochs):
            g_swarm = [0.0] * len(FEATURE_NAMES)
            g_team = {t: [0.0] * len(FEATURE_NAMES) for t in self.team_w}
            for x, weight, y_swarm, teams in rows:
                err_s = (_sigmoid(_dot(self.swarm_w, x)) - y_swarm) * weight
                for j, xj in enumerate(x):
                    g_swarm[j] += err_s * xj / n
                for t in self.team_w:
                    err_t = (_sigmoid(_dot(self.team_w[t], x)) - (1.0 if t in teams else 0.0)) * weight
                    gt = g_team[t]
                    for j, xj in enumerate(x):
                        gt[j] += err_t * xj / n
            self.swarm_w = [w - lr * g for w, g in zip(self.swarm_w, g_swarm, strict=True)]
            for t in self.team_w:
                self.team_w[t] = [w - lr * g
                                  for w, g in zip(self.team_w[t], g_team[t], strict=True)]
        self.trained_on = len(rows)
        return {"trained": len(rows), "epochs": epochs}

    # -- persistence ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "featureNames": list(FEATURE_NAMES),
                "swarmW": self.swarm_w, "teamW": self.team_w,
                "trainedOn": self.trained_on,
                "claimCeiling": "candidate_only; canClaimAGI:false"}

    def save(self, path: Path = DEFAULT_HEAD_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_HEAD_PATH) -> "RouterHead":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or data.get("featureNames") != list(FEATURE_NAMES):
            raise ValueError(f"incompatible router head file {path} "
                             f"(schema/features mismatch — retrain, do not coerce)")
        head = cls(swarm_w=list(data["swarmW"]),
                   team_w={t: list(w) for t, w in data["teamW"].items()},
                   trained_on=int(data.get("trainedOn", 0)))
        return head


def signals_from_dict(d: dict) -> RouteSignals:
    return RouteSignals(
        difficulty=float(d.get("difficulty", 0.0)),
        multi_hop=bool(d.get("multiHop")),
        quant=bool(d.get("quant")),
        legal=bool(d.get("legal")),
        contested=bool(d.get("contested")),
        entity_risk=bool(d.get("entityRisk")),
        n_sub_queries=int(d.get("nSubQueries", 1)),
        intent=str(d.get("intent", "other")),
    )


class TrainedSwarmRouter(SwarmRouter):
    """SwarmRouter whose ``decide`` consults the trained head — fail-closed to v1.

    The head only ever *narrows or confirms*: when it is confident the task is solo,
    we answer solo; when it is confident about a team set, we dispatch exactly those
    catalogue teams (same tool scopes, budgets, k). In every uncertain case the
    decision is delegated to the hand-authored v1 policy unchanged.
    """

    def __init__(self, head: RouterHead, *, swarm_threshold: float = 0.5,
                 team_threshold: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.head = head
        self.swarm_threshold = swarm_threshold
        self.team_threshold = team_threshold

    def decide(self, task: str) -> SwarmPlan:
        clean = (task or "").strip()
        if not clean:
            return super().decide(task)  # fail-closed empty-input path is v1's
        sig = self.signals(clean)
        pred = self.head.predict(sig)
        if pred["swarmProb"] < self.swarm_threshold:
            # Confident solo — but never below v1's own fail-closed floor semantics:
            # v1 would also answer solo unless a hard signal fires; when v1 disagrees
            # (would swarm), defer to v1 (the audited policy wins ties).
            v1 = super().decide(clean)
            if v1.mode == "swarm":
                return v1
            return SwarmPlan(task=clean, mode="solo", assignments=[], signals=sig,
                             rationale=f"trained head: swarmProb {pred['swarmProb']} < "
                                       f"{self.swarm_threshold}; v1 concurs solo")
        chosen = [t for t, p in pred["teamProbs"].items() if p >= self.team_threshold]
        if not chosen:
            return super().decide(clean)  # confident-swarm but no team clears → v1
        assignments = [TeamAssignment(t, TEAMS[t].default_k, self.child_budget_usd,
                                      f"[head-routed] {clean}")
                       for t in sorted(chosen)]
        return SwarmPlan(task=clean, mode="swarm", assignments=assignments, signals=sig,
                         rationale="trained head (candidate-only): " + ", ".join(
                             f"{t}={pred['teamProbs'][t]}" for t in sorted(chosen)))


def shadow_compare(head_router: TrainedSwarmRouter, tasks: "list[str]",
                   *, v1: "SwarmRouter | None" = None) -> dict:
    """Ablation-lane helper: route the same tasks through v1 and the head, report
    agreement and divergences. This is evidence for the promotion gate — it decides
    nothing itself."""
    v1 = v1 or SwarmRouter()
    rows = []
    agree = 0
    for t in tasks:
        p1, p2 = v1.decide(t), head_router.decide(t)
        same = p1.mode == p2.mode and {a.team for a in p1.assignments} == \
            {a.team for a in p2.assignments}
        agree += int(same)
        rows.append({"task": t, "agree": same,
                     "v1": {"mode": p1.mode, "teams": sorted({a.team for a in p1.assignments})},
                     "head": {"mode": p2.mode, "teams": sorted({a.team for a in p2.assignments})}})
    return {"nTasks": len(tasks), "agreementRate": round(agree / len(tasks), 3) if tasks else 1.0,
            "divergences": [r for r in rows if not r["agree"]],
            "claimCeiling": "candidate_only; promotion requires the ablation gate"}


# --------------------------------------------------------------------------- #
# Offline invariants (CI-gated; deterministic, stdlib-only)
# --------------------------------------------------------------------------- #

def _synthetic_tuples() -> "list[dict]":
    """v1-labelled tuples from the v1 policy itself (behavior cloning fixture)."""
    v1 = SwarmRouter()
    tasks = [
        ("hi", True), ("thanks", True), ("ok great", True),
        ("Compare the epistemology of Kant and the skepticism of Hume in detail", True),
        ("Calculate the probability of at least one six in four rolls and prove it", True),
        ("Which quotes are misattributed to Einstein according to scholars?", True),
        ("Compare the disputed authorship of the Dao De Jing versus the Analects", True),
        ("What does case law say about jurisdiction for the plaintiff here?", False),
    ]
    out = []
    for task, ok in tasks:
        p = v1.decide(task)
        out.append({"task": task, "ok": ok, "signals": p.signals.to_dict(),
                    "v1Plan": {"mode": p.mode,
                               "teams": sorted({a.team for a in p.assignments})}})
    return out


def offline_invariants() -> "tuple[bool, dict]":
    import tempfile

    checks: dict[str, bool] = {}
    head = RouterHead()
    head.fit(_synthetic_tuples())

    # 1) Training is deterministic: refitting from scratch gives identical weights.
    head2 = RouterHead()
    head2.fit(_synthetic_tuples())
    checks["fit_deterministic"] = head.to_dict() == head2.to_dict()

    # 2) The head separates its training poles: an easy task scores lower
    #    swarm-probability than a hard contested one.
    v1 = SwarmRouter()
    p_easy = head.predict(v1.signals("hi"))
    p_hard = head.predict(v1.signals(
        "Compare the disputed authorship of the Dao De Jing versus the Analects"))
    checks["separates_poles"] = p_easy["swarmProb"] < p_hard["swarmProb"]

    # 3) TrainedSwarmRouter preserves the SwarmPlan contract.
    tr = TrainedSwarmRouter(head)
    plan = tr.decide("Compare the disputed authorship of the Dao De Jing versus the Analects")
    d = plan.to_dict()
    checks["contract_preserved"] = {"schema", "task", "mode", "assignments", "reduce",
                                    "signals", "rationale"} <= set(d)

    # 4) Fail-closed: empty input never swarms; and the head can never suppress a
    #    swarm v1 would have run (v1 is the floor).
    checks["empty_is_solo"] = tr.decide("  ").mode == "solo"
    hard_task = "Calculate the probability of at least one six in four rolls and prove it"
    checks["v1_floor_respected"] = not (
        SwarmRouter().decide(hard_task).mode == "swarm" and tr.decide(hard_task).mode == "solo")

    # 5) Head-routed teams come from the catalogue with catalogue scopes (least
    #    privilege can't widen).
    checks["teams_from_catalogue"] = all(a.team in TEAMS for a in plan.assignments)

    # 6) Round-trips through save/load; incompatible schema is refused.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "head.json"
        head.save(p)
        checks["roundtrip"] = RouterHead.load(p).to_dict() == head.to_dict()
        bad = json.loads(p.read_text(encoding="utf-8"))
        bad["schema"] = "other"
        p.write_text(json.dumps(bad), encoding="utf-8")
        try:
            RouterHead.load(p)
            checks["schema_mismatch_refused"] = False
        except ValueError:
            checks["schema_mismatch_refused"] = True

    # 7) shadow_compare reports and never decides.
    rep = shadow_compare(tr, ["hi", hard_task])
    checks["shadow_reports_only"] = "agreementRate" in rep and "candidate_only" in rep["claimCeiling"]

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit/inspect the candidate router head")
    ap.add_argument("--fit", metavar="TUPLES_JSONL",
                    help="train on mined tuples (tools/mine_router_tuples.py output)")
    ap.add_argument("--out", default=str(DEFAULT_HEAD_PATH))
    args = ap.parse_args()
    if args.fit:
        rows = [json.loads(line) for line in
                Path(args.fit).read_text(encoding="utf-8").splitlines() if line.strip()]
        head = RouterHead()
        info = head.fit(rows)
        head.save(Path(args.out))
        print(f"trained on {info.get('trained', 0)} tuples -> {args.out} (candidate-only)")
        raise SystemExit(0)
    ok, detail = offline_invariants()
    for name, passed in detail["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"{'PASS' if ok else 'FAIL'} router_head offline_invariants")
    raise SystemExit(0 if ok else 1)
