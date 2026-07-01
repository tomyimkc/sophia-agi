# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Poisoning-robust ingestion for the OKF belief graph (Sophia discipline).

A retrieval/ingestion pipeline that accepts a claim because it is "well sourced"
is exactly the surface PoisonedRAG (arXiv:2402.07867) attacks: a handful of
injected texts — possibly all minted by one adversary, or one high-"confidence"
source — can flip an answer. RAGDefender (arXiv:2511.01268) answers with
corroboration + filtering. This module implements that defence as an *admission
gate* in front of the OKF graph, reusing the project's existing combiner rather
than inventing a new one.

Three layers, all deterministic and standard-library only:

  1. k-INDEPENDENT corroboration. A claim is admitted ONLY IF at least ``k``
     DISTINCT independence groups (Sybil/duplicate sources collapse to one
     group, per ``agent.corroboration``) clear a trust floor, AND the
     trust-weighted pooled confidence clears a confidence floor. A single
     source — however high its self-reported confidence — can never alone
     satisfy ``k >= 2``.

  2. SOURCE-TRUST modelling. Each source's self-reported confidence is shrunk
     toward the prior in proportion to ``1 - trust`` before pooling, so a
     low-trust source contributes little; unknown sources default to a
     conservative low trust (fail-closed).

  3. POST-RETRIEVAL ADVERSARIAL FILTER. Once a consensus value is established
     for a claim, a later item proposing a CONFLICTING value is flagged as
     suspected poison (outlier-vs-consensus) and quarantined rather than
     admitted. If such a source is later proven malicious, remediation runs
     through ``agent.unlearning.Unlearner.forget`` so the retraction cascade
     un-grounds anything that rested only on it.

This is a candidate gate: every emitted decision carries ``candidateOnly: True``
and the fail-closed default is QUARANTINE, never silent admission. Pooling is NOT
reimplemented here — it is ``agent.corroboration.corroborated_confidence``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from agent.corroboration import Evidence, corroborated_confidence

__all__ = [
    "SourceTrust",
    "assess_item",
    "adversarial_filter",
    "ingest_stream",
    "run_poison_benchmark",
    "SCHEMA",
]

SCHEMA = "sophia.poison_resistant_ingestion.v1"

# Conservative default for an unknown source: trust it little (fail-closed).
DEFAULT_TRUST = 0.2


@dataclass(frozen=True)
class SourceTrust:
    """A read-only ``sourceId -> trust`` map with a conservative default.

    Unknown sources resolve to ``default`` (0.2 by default), so an adversary
    cannot gain credit merely by inventing a fresh, unseen source id.
    """

    scores: Mapping = None  # type: ignore[assignment]
    default: float = DEFAULT_TRUST

    def trust(self, source_id: str) -> float:
        scores = self.scores or {}
        try:
            t = float(scores.get(source_id, self.default))
        except (TypeError, ValueError):
            t = self.default
        # Clamp into [0,1]; non-finite / out-of-range falls back to default-safe.
        if math.isnan(t):  # NaN
            t = self.default
        return min(1.0, max(0.0, t))


def _as_trust(trust) -> SourceTrust:
    if isinstance(trust, SourceTrust):
        return trust
    if trust is None:
        return SourceTrust(scores={})
    if isinstance(trust, Mapping):
        return SourceTrust(scores=trust)
    raise TypeError("trust must be a SourceTrust or a mapping")


def _shrink(confidence: float, trust: float, prior: float) -> float:
    """Shrink a source's self-reported confidence toward ``prior`` by ``1-trust``.

    A fully trusted source (trust=1) keeps its confidence; a fully untrusted
    source (trust=0) collapses to the prior (no information). This downweights
    low-trust sources in the pooled posterior without ever amplifying them.
    """
    c = min(1.0, max(0.0, float(confidence)))
    t = min(1.0, max(0.0, float(trust)))
    return prior + t * (c - prior)


def _evidences(item: dict, st: SourceTrust, prior: float) -> list:
    """Build trust-weighted, group-tagged Evidence from an item's sources."""
    evs = []
    for i, src in enumerate(item.get("sources", []) or []):
        sid = str(src.get("sourceId", f"_src{i}"))
        t = src.get("trust")
        if t is None:
            t = st.trust(sid)
        group = str(src.get("independenceGroup") or sid)
        conf = float(src.get("confidence", prior))
        shrunk = _shrink(conf, float(t), prior)
        evs.append(Evidence(sid, shrunk, independence_group=group))
    return evs


def _independent_group_count(item: dict, st: SourceTrust, trust_floor: float) -> int:
    """Count DISTINCT independence groups that contain >=1 source clearing the
    trust floor. Sybil/duplicate sources sharing a group count once."""
    groups: set = set()
    for i, src in enumerate(item.get("sources", []) or []):
        sid = str(src.get("sourceId", f"_src{i}"))
        t = src.get("trust")
        if t is None:
            t = st.trust(sid)
        if float(t) >= trust_floor:
            groups.add(str(src.get("independenceGroup") or sid))
    return len(groups)


def assess_item(
    item: dict,
    *,
    trust,
    k: int = 2,
    trust_floor: float = 0.3,
    conf_floor: float = 0.6,
    prior: float = 0.5,
) -> dict:
    """Decide whether a single candidate ingestion item may be admitted.

    ADMISSION RULE (k-independent): admit IFF
        independentCorroborations >= k   AND   pooledConfidence >= conf_floor
    where ``independentCorroborations`` is the number of DISTINCT independence
    groups holding at least one source with trust >= ``trust_floor`` (Sybils /
    duplicates collapse to one group), and ``pooledConfidence`` is
    ``corroborated_confidence`` over the trust-shrunk, group-deduped Evidence.
    Otherwise QUARANTINE. Fail-closed: a single source never alone meets k>=2.
    """
    st = _as_trust(trust)
    claim_id = item.get("claimId")
    reasons: list = []

    indep = _independent_group_count(item, st, trust_floor)
    evs = _evidences(item, st, prior)
    pooled = corroborated_confidence(evs, prior=prior) if evs else float(prior)

    enough_independent = indep >= k
    enough_confidence = pooled >= conf_floor

    if not enough_independent:
        reasons.append(
            f"insufficient independent corroboration: {indep} distinct "
            f"trusted group(s) < k={k} (Sybil/duplicate sources collapse to one)"
        )
    if not enough_confidence:
        reasons.append(
            f"pooled confidence {pooled:.3f} < conf_floor={conf_floor} "
            f"(trust-weighted)"
        )

    decision = "admit" if (enough_independent and enough_confidence) else "quarantine"
    if decision == "admit":
        reasons.append(
            f"admitted: {indep} independent trusted group(s) >= k={k}, "
            f"pooled {pooled:.3f} >= {conf_floor}"
        )

    return {
        "claimId": claim_id,
        "independentCorroborations": indep,
        "pooledConfidence": round(pooled, 6),
        "decision": decision,
        "reasons": reasons,
        "candidateOnly": True,
    }


def adversarial_filter(item: dict, consensus_value) -> dict:
    """Flag an item whose value conflicts with an established consensus value.

    This is the post-retrieval defence: once a claim has a corroborated
    consensus, a later item proposing a DIFFERENT value for the same claim is
    an outlier-vs-consensus signal — the classic injected-text poison. We never
    silently overwrite consensus; the conflicting item is flagged for
    quarantine. A matching value (or no established consensus) is not flagged.
    """
    value = item.get("value")
    if consensus_value is None:
        return {
            "suspectedPoison": False,
            "reason": "no established consensus to compare against",
            "candidateOnly": True,
        }
    conflicts = value != consensus_value
    if conflicts:
        reason = (
            f"value {value!r} conflicts with established consensus "
            f"{consensus_value!r} (outlier-vs-consensus suspected poison)"
        )
    else:
        reason = "value agrees with established consensus"
    return {
        "suspectedPoison": bool(conflicts),
        "reason": reason,
        "candidateOnly": True,
    }


def ingest_stream(
    items,
    *,
    trust,
    k: int = 2,
    trust_floor: float = 0.3,
    conf_floor: float = 0.6,
    prior: float = 0.5,
) -> dict:
    """Process a stream of candidate items, tracking per-claim consensus.

    For each item we (1) check the adversarial filter against any consensus
    value already accepted for its claim, and (2) run ``assess_item``. An item
    that conflicts with established consensus is quarantined and recorded as
    suspected poison regardless of its sourcing. Otherwise the k-independent
    admission rule decides; the FIRST admitted value for a claim becomes that
    claim's consensus value, against which later items are filtered.
    """
    st = _as_trust(trust)
    consensus: dict = {}
    admitted: list = []
    quarantined: list = []
    suspected: list = []

    for item in items:
        claim_id = item.get("claimId")
        established = consensus.get(claim_id)

        flag = adversarial_filter(item, established)
        if flag["suspectedPoison"]:
            record = {
                "claimId": claim_id,
                "value": item.get("value"),
                "decision": "quarantine",
                "suspectedPoison": True,
                "reasons": [flag["reason"]],
                "candidateOnly": True,
            }
            quarantined.append(record)
            suspected.append(record)
            continue

        verdict = assess_item(
            item,
            trust=st,
            k=k,
            trust_floor=trust_floor,
            conf_floor=conf_floor,
            prior=prior,
        )
        verdict["value"] = item.get("value")
        if verdict["decision"] == "admit":
            admitted.append(verdict)
            if claim_id not in consensus:
                consensus[claim_id] = item.get("value")
        else:
            quarantined.append(verdict)

    return {
        "admitted": admitted,
        "quarantined": quarantined,
        "suspectedPoison": suspected,
        "consensus": consensus,
        "schema": SCHEMA,
        "candidateOnly": True,
    }


# --------------------------------------------------------------------------- #
# Seeded, deterministic poisoned-stream benchmark.
# --------------------------------------------------------------------------- #


def _build_poison_stream(seed: int = 0) -> tuple:
    """Hand-built-but-seeded fixtures: genuine claims (k+ independent trusted
    sources) plus injected poison (single-group / low-trust / consensus-
    conflicting). Returns ``(items, trust, genuine_ids, poison_ids)``."""
    import random

    rng = random.Random(seed * 7919 + 1)

    trust_map: dict = {}
    items: list = []
    genuine_ids: list = []
    poison_ids: list = []

    n_genuine = 3
    for c in range(n_genuine):
        cid = f"genuine_{c}"
        genuine_ids.append(cid)
        k_src = rng.choice([2, 3])
        sources = []
        for g in range(k_src):
            sid = f"trusted_{c}_{g}"
            t = rng.choice([0.8, 0.9, 0.95])
            trust_map[sid] = t
            sources.append(
                {
                    "sourceId": sid,
                    "trust": t,
                    "independenceGroup": f"{cid}_g{g}",
                    "confidence": rng.choice([0.8, 0.85, 0.9]),
                }
            )
        items.append({"claimId": cid, "value": f"value_{c}", "sources": sources})

    # Poison A: a single high-"confidence" but LOW-trust source (no independence,
    # below trust floor) — cannot fake k-independence.
    cid_a = "poison_single_lowtrust"
    poison_ids.append(cid_a)
    trust_map["liar_a"] = 0.1
    items.append(
        {
            "claimId": cid_a,
            "value": "fabricated_a",
            "sources": [
                {
                    "sourceId": "liar_a",
                    "trust": 0.1,
                    "independenceGroup": "liar_a",
                    "confidence": 0.99,
                }
            ],
        }
    )

    # Poison B: many sources but ALL sharing ONE independence group (Sybil), each
    # high trust/confidence — still only ONE independent group.
    cid_b = "poison_sybil"
    poison_ids.append(cid_b)
    sybil_sources = []
    for s in range(4):
        sid = f"sybil_{s}"
        trust_map[sid] = 0.9
        sybil_sources.append(
            {
                "sourceId": sid,
                "trust": 0.9,
                "independenceGroup": "one_botnet",
                "confidence": 0.97,
            }
        )
    items.append({"claimId": cid_b, "value": "fabricated_b", "sources": sybil_sources})

    # Poison C: conflicts with an already-admitted genuine consensus value. Even
    # with plausible sourcing, the adversarial filter quarantines the outlier.
    cid_c = genuine_ids[0]  # same claim id as a genuine, admitted claim
    poison_ids.append(cid_c + "::conflict")
    items.append(
        {
            "claimId": cid_c,
            "value": "POISON_OVERWRITE",
            "sources": [
                {
                    "sourceId": "attacker_c",
                    "trust": 0.7,
                    "independenceGroup": f"{cid_c}_attack0",
                    "confidence": 0.95,
                },
                {
                    "sourceId": "attacker_c2",
                    "trust": 0.7,
                    "independenceGroup": f"{cid_c}_attack1",
                    "confidence": 0.95,
                },
            ],
        }
    )

    trust = SourceTrust(scores=trust_map)
    return items, trust, genuine_ids, poison_ids


def run_poison_benchmark(seed: int = 0) -> dict:
    """Run the seeded poisoned-stream benchmark.

    Genuine claims (each backed by k+ independent trusted sources) must be
    admitted; injected poison (single low-trust source, Sybil single-group, and
    a consensus-conflicting overwrite) must be quarantined. Deterministic: the
    same ``seed`` yields the same verdicts on every run.
    """
    items, trust, genuine_ids, _poison_ids = _build_poison_stream(seed)
    result = ingest_stream(items, trust=trust, k=2, trust_floor=0.3, conf_floor=0.6)

    admitted_claims = {a["claimId"] for a in result["admitted"]}
    quarantined_values = {
        (q["claimId"], q.get("value")) for q in result["quarantined"]
    }

    genuine_admitted = all(g in admitted_claims for g in genuine_ids)

    # Poison admitted as a NEW consensus would be a failure: none of the poison
    # values may appear in admitted records.
    poison_values = {"fabricated_a", "fabricated_b", "POISON_OVERWRITE"}
    poison_admitted = any(a.get("value") in poison_values for a in result["admitted"])

    # The consensus-conflicting overwrite must be flagged as suspected poison.
    overwrite_flagged = any(
        sp.get("value") == "POISON_OVERWRITE" for sp in result["suspectedPoison"]
    )

    ok = bool(genuine_admitted and not poison_admitted and overwrite_flagged)

    return {
        "ok": ok,
        "schema": SCHEMA,
        "candidateOnly": True,
        "seed": seed,
        "genuineAdmitted": genuine_admitted,
        "poisonAdmitted": poison_admitted,
        "overwriteFlagged": overwrite_flagged,
        "metrics": {
            "nGenuine": len(genuine_ids),
            "nAdmitted": len(result["admitted"]),
            "nQuarantined": len(result["quarantined"]),
            "nSuspectedPoison": len(result["suspectedPoison"]),
        },
        "quarantinedValues": sorted(str(v) for v in quarantined_values),
    }
