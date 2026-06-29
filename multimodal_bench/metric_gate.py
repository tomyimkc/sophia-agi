# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail-closed metric grounding gate — a VLM's physical claim, re-checked.

A VLM that says "the cup is 30 cm from the laptop" or "the box is in front of the
ball" is emitting a *hypothesis* (VISION.md: verification over generation). The
field's metric-blindness failure is that nothing re-checks it. This gate does, in
Sophia's idiom (the metric twin of ``gui_agent.verify_action``):

1. **Region grounding.** The claim must cite a supporting region (a box); that
   region must actually overlap the object(s) the claim is about. A claim about a
   region that doesn't contain its subject is *ungrounded* -> blocked.
2. **Value re-check.** The claimed relation/measure is recomputed by the judge-free
   verifiers (``multimodal_bench/verifiers.py``) over a depth source
   (``depth_backend``): authored z offline, or pixel-derived metric depth when a
   real backend is wired. Mismatch -> blocked. Match -> accepted.

**Fail-closed everywhere:** a missing object, an unresolvable depth, or a depth
source that is itself a blocker (no weights) yields *block + escalate*, never a
silent accept. Confident-but-wrong metric claims are exactly what this stops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multimodal_bench import verifiers
from multimodal_bench.depth_backend import AuthoredDepthSource


@dataclass(frozen=True)
class MetricDecision:
    allowed: bool
    verdict: str            # "accept" | "block"
    reason: str
    claim: dict = field(default_factory=dict)
    escalate: bool = False  # route to a human when blocked


def _overlaps_subject(scene: dict, region, label: str) -> bool:
    """Whether the cited ``region`` box overlaps the named object's box."""
    o = verifiers._find(scene, label)
    if o is None or not region:
        return False
    return verifiers._boxes_overlap(region, o["box"])


def verify_metric_claim(scene: dict, claim: dict, *, depth_source=None, tol: float = 1e-6) -> MetricDecision:
    """Re-check one physical claim against the depth-grounded scene. Fail-closed.

    ``claim`` examples::

        {"kind": "depth_order", "a": "cup", "rel": "in_front_of", "b": "laptop",
         "region": [x, y, w, h], "value": true}
        {"kind": "bigger", "a": "car", "b": "cup", "region": [...], "value": true}
        {"kind": "distance", "a": "car", "b": "house", "region": [...],
         "value": 452.0, "tol": 10.0}

    ``region`` is the model's cited supporting crop; ``value`` is what the model
    asserts (a bool for depth_order/bigger, a number for distance).
    """
    src = depth_source or AuthoredDepthSource()
    if getattr(src, "blocker", None):
        # cannot ground the metric at all -> withhold, never auto-accept
        return MetricDecision(False, "block", f"depth_source_unavailable:{src.blocker}", claim, escalate=True)

    kind = claim.get("kind")
    a, b = claim.get("a"), claim.get("b")
    region = claim.get("region")

    if not a or not b:
        return MetricDecision(False, "block", "missing_subject", claim, escalate=True)
    if verifiers._find(scene, a) is None or verifiers._find(scene, b) is None:
        return MetricDecision(False, "block", "subject_absent", claim, escalate=True)
    # the cited region must actually contain the primary subject (grounding)
    if not _overlaps_subject(scene, region, a):
        return MetricDecision(False, "block", "region_misses_subject", claim, escalate=True)

    # Fail-closed everywhere: any verifier/parse/augment error blocks + escalates,
    # never crashes the gate (e.g. an unknown `rel` raises in verifiers.depth_order).
    try:
        grounded = src.augment(scene)  # fill z/size from the depth source

        if kind == "depth_order":
            truth = verifiers.depth_order(grounded, a, claim.get("rel", "in_front_of"), b)
            ok = bool(claim.get("value")) == truth
            return _verdict(ok, claim, f"depth_order_truth={truth}")
        if kind == "bigger":
            truth = verifiers.bigger_than(grounded, a, b)
            ok = bool(claim.get("value")) == truth
            return _verdict(ok, claim, f"bigger_truth={truth}")
        if kind == "distance":
            truth = verifiers.distance_between(grounded, a, b)
            if truth is None:
                return MetricDecision(False, "block", "distance_unresolvable", claim, escalate=True)
            claimed = claim.get("value")
            if claimed is None:
                return MetricDecision(False, "block", "no_claimed_value", claim, escalate=True)
            ok = abs(float(claimed) - truth) <= float(claim.get("tol", tol))
            return _verdict(ok, claim, f"distance_truth={round(truth, 2)}")
        return MetricDecision(False, "block", f"unknown_claim_kind:{kind}", claim, escalate=True)
    except Exception as exc:  # fail-closed on any verifier/parse error
        return MetricDecision(False, "block", f"verifier_error:{type(exc).__name__}", claim, escalate=True)


def _verdict(ok: bool, claim: dict, detail: str) -> MetricDecision:
    if ok:
        return MetricDecision(True, "accept", f"verified:{detail}", claim)
    return MetricDecision(False, "block", f"contradicted:{detail}", claim, escalate=True)


@dataclass
class MetricGate:
    """Accept only metric claims that re-check against the depth-grounded scene."""
    scene: dict
    depth_source: object = None
    accepted: list = field(default_factory=list)
    blocked: list = field(default_factory=list)

    def step(self, claim: dict) -> MetricDecision:
        d = verify_metric_claim(self.scene, claim, depth_source=self.depth_source)
        (self.accepted if d.allowed else self.blocked).append(d)
        return d

    def run(self, claims: "list[dict]") -> "list[MetricDecision]":
        return [self.step(c) for c in claims]

    def summary(self) -> dict:
        n = len(self.accepted) + len(self.blocked)
        return {
            "proposed": n,
            "accepted": len(self.accepted),
            "blocked": len(self.blocked),
            "blockRate": round(len(self.blocked) / n, 4) if n else 0.0,
            "escalations": sum(1 for d in self.blocked if d.escalate),
            "reasons": sorted({d.reason for d in self.blocked}),
        }


# --- offline demo: grounded claims accepted, hallucinated ones blocked ----- #

DEMO_SCENE = {
    "width": 512, "height": 512,
    "objects": [
        {"label": "cup", "box": [80, 300, 80, 60], "z": 2.0, "size": 0.12},
        {"label": "laptop", "box": [300, 260, 150, 120], "z": 6.0, "size": 0.5},
        {"label": "car", "box": [360, 60, 60, 40], "z": 12.0, "size": 4.5},
    ],
    "texts": [],
}

# Each grounded claim cites a region on its subject and states the true relation.
GROUNDED_CLAIMS = [
    {"kind": "depth_order", "a": "cup", "rel": "in_front_of", "b": "laptop", "region": [80, 300, 80, 60], "value": True},
    {"kind": "bigger", "a": "car", "b": "cup", "region": [360, 60, 60, 40], "value": True},
]

# Hallucinated: a reversed depth order, an apparent-size illusion (cup looks big),
# and a claim whose cited region doesn't even contain its subject.
HALLUCINATED_CLAIMS = [
    {"kind": "depth_order", "a": "cup", "rel": "behind", "b": "laptop", "region": [80, 300, 80, 60], "value": True},
    {"kind": "bigger", "a": "cup", "b": "car", "region": [80, 300, 80, 60], "value": True},
    {"kind": "depth_order", "a": "cup", "rel": "in_front_of", "b": "laptop", "region": [0, 0, 20, 20], "value": True},
]


def demo() -> dict:
    good = MetricGate(DEMO_SCENE)
    good.run(GROUNDED_CLAIMS)
    bad = MetricGate(DEMO_SCENE)
    bad.run(HALLUCINATED_CLAIMS)
    return {"grounded": good.summary(), "hallucinated": bad.summary()}
