# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judged agent-faithfulness benchmark — held-out pack scored by an entailment judge
under the no-overclaim gate.

The deterministic benchmark (``provenance_bench.agent_faithfulness``) proves the
trajectory evaluator's *wiring*. This one measures the part a deterministic lexical
judge CANNOT settle: whether a claim is genuinely entailed by its evidence when the
surface forms diverge (paraphrase, multi-hop) or collide misleadingly (negation /
scope distractors). That decision needs a model, so — exactly like
``agent.legal_faithfulness`` — it is **measured under the no-overclaim gate**:
>=2 independent judge families, Cohen's kappa >= floor, >=3 runs, and a CI above
chance. No headline number is published from a single judge or a mock.

Framing is a **binary certify decision** per case so the gate machinery
(``provenance_bench.consensus.cohen_kappa`` + ``provenance_bench.aggregate``) drops
straight in: ``verdict == "accept" -> 1`` (the evaluator certified the run),
``gold = (expectVerdict == "accept")``. This is precisely the "Agent Data
Evaluation" decision: certify a trajectory or refuse to.

The held-out pack is **sealed** (sha256 commitment) and is first-party held-out,
NOT third-party — see its ``status`` and the failure ledger.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from provenance_bench.aggregate import KAPPA_FLOOR, _ci, _distinct_families
from provenance_bench.consensus import cohen_kappa
from agent.trajectory_eval import (
    Support,
    evaluate_trajectory,
    lexical_support_judge,
    make_entailment_judge,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK = ROOT / "benchmark" / "agent_faithfulness_heldout.json"
SEAL = ROOT / "agi-proof" / "hidden-reviewer-packs" / "agent-faithfulness-heldout.seal.json"


# --------------------------------------------------------------------------- #
# Sealing — tamper-evident commitment over the pack content
# --------------------------------------------------------------------------- #
def content_hash(pack: dict) -> str:
    """sha256 over the canonical JSON of the cases (sorted keys). Stable across
    formatting; changes if any case content changes."""
    canon = json.dumps(pack.get("cases", []), sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_seal(pack: dict) -> dict:
    return {
        "schema": "sophia.agent_faithfulness.heldout.seal.v1",
        "pack": str(DEFAULT_PACK.relative_to(ROOT)),
        "n": len(pack.get("cases", [])),
        "contentHash": content_hash(pack),
        "hashMethod": "sha256(json(cases, sort_keys=True, separators=(',',':')))",
        "note": "Tamper-evident commitment. First-party held-out (NOT third-party). "
                "Held-out = not used to tune the evaluator or its lexical floor.",
    }


def verify_seal(pack: dict, seal: "dict | None" = None) -> bool:
    seal = seal or json.loads(SEAL.read_text(encoding="utf-8"))
    return content_hash(pack) == seal.get("contentHash")


def load_pack(path: "Path | str" = DEFAULT_PACK) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Per-judge / per-run scoring (binary certify decision)
# --------------------------------------------------------------------------- #
def _gold(cases: list[dict]) -> list[int]:
    return [int(c["expectVerdict"] == "accept") for c in cases]


def _certify(trajectory, judge) -> int:
    return int(evaluate_trajectory(trajectory or [], judge=judge)["verdict"] == "accept")


def run_once(cases: list[dict], judges: list) -> dict:
    """One run: every judge certifies (1) or refuses (0) each case. Deterministic
    judges make runs identical; real model judges introduce the variance the gate
    measures."""
    gold = _gold(cases)
    per_judge = [[_certify(c.get("trajectory"), j) for c in cases] for j in judges]
    consensus = _consensus_labels({"n": len(cases), "perJudge": per_judge})
    correct = sum(int(p == g) for p, g in zip(consensus, gold))
    strata = [{"difficulty": c.get("difficulty"), "failureType": c.get("failureType")}
              for c in cases]
    return {"n": len(cases), "gold": gold, "perJudge": per_judge,
            "consensusCorrect": correct, "strata": strata}


def _consensus_labels(run: dict) -> list[int]:
    labels = []
    for idx in range(run["n"]):
        cast = [run["perJudge"][j][idx] for j in range(len(run["perJudge"]))]
        labels.append(1 if cast and sum(cast) > len(cast) / 2 else 0)
    return labels


def _accuracy(labels: list[int], gold: list[int]) -> float:
    return round(sum(int(a == b) for a, b in zip(labels, gold)) / len(gold), 4) if gold else 0.0


def _mean_pairwise_kappa(run: dict) -> "float | None":
    judges = run["perJudge"]
    if len(judges) < 2:
        return None
    kappas = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            kk = cohen_kappa(judges[i], judges[k])
            if kk is not None:
                kappas.append(kk)
    return round(sum(kappas) / len(kappas), 4) if kappas else None


def _kappa_over(run: dict, idxs: list[int]) -> list[float]:
    judges = run["perJudge"]
    out: list[float] = []
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            kk = cohen_kappa([judges[a][i] for i in idxs], [judges[b][i] for i in idxs])
            if kk is not None:
                out.append(kk)
    return out


def _per_judge_acc(runs: list[dict], specs: list[str]) -> dict:
    out = {}
    for i, spec in enumerate(specs):
        accs = [_accuracy(r["perJudge"][i], r["gold"]) for r in runs]
        out[spec] = round(sum(accs) / len(accs), 4)
    return out


def _stratified(runs: list[dict], key: str) -> dict:
    out: dict = {}
    strata = runs[0].get("strata") or []
    labels = {(v.get(key) or "unspecified") for v in strata}
    for lab in sorted(labels):
        idxs = [i for i, v in enumerate(strata) if (v.get(key) or "unspecified") == lab]
        if not idxs:
            continue
        correct = total = 0
        kappas: list[float] = []
        for r in runs:
            cons = _consensus_labels(r)
            for i in idxs:
                correct += int(cons[i] == r["gold"][i])
                total += 1
            kappas += _kappa_over(r, idxs)
        out[lab] = {
            "n": len(idxs),
            "accuracy": round(correct / total, 4) if total else None,
            "meanPairwiseKappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
        }
    return out


def lexical_baseline_accuracy(cases: list[dict]) -> float:
    """Certify-accuracy of the DEFAULT lexical judge — the floor the entailment judge
    must beat. The gap (judged - lexical) is the measured value of the judge."""
    judge = lexical_support_judge()
    gold = _gold(cases)
    labels = [_certify(c.get("trajectory"), judge) for c in cases]
    return _accuracy(labels, gold)


def aggregate(runs: list[dict], *, judge_specs: list[str], cases: "list[dict] | None" = None,
              seed: int = 0, n_boot: int = 2000) -> dict:
    """Aggregate judged runs into gated metrics. Mirrors the legal-faithfulness gate:
    consensus accuracy + bootstrap CI + mean pairwise kappa + the five validated
    checks. ``validated`` is True only when ALL checks pass."""
    gold = runs[0]["gold"]
    per_run_acc = [r["consensusCorrect"] / r["n"] for r in runs]
    pooled = [int(p == g) for r in runs for p, g in zip(_consensus_labels(r), r["gold"])]
    rng = random.Random(seed)
    boot = []
    if pooled:
        for _ in range(n_boot):
            sample = [pooled[rng.randrange(len(pooled))] for _ in range(len(pooled))]
            boot.append(sum(sample) / len(sample))
    ci = _ci(boot) if boot else [0.0, 0.0]
    kappas = [k for k in (_mean_pairwise_kappa(r) for r in runs) if k is not None]
    mean_kappa = round(sum(kappas) / len(kappas), 4) if kappas else None
    acc = round(sum(per_run_acc) / len(per_run_acc), 4)

    checks = {
        "notMock": all(s and "mock" not in s for s in judge_specs) and bool(judge_specs),
        "multiFamilyJudges": _distinct_families(judge_specs) >= 2,
        "kappaAboveFloor": mean_kappa is not None and mean_kappa >= KAPPA_FLOOR,
        "atLeast3Runs": len(runs) >= 3,
        "ciAboveChance": bool(ci) and ci[0] > 0.5,
    }
    report = {
        "benchmark": "agent_faithfulness_judged",
        "schema": "sophia.agent_faithfulness.judged.report.v1",
        "judges": judge_specs,
        "runs": len(runs),
        "n": runs[0]["n"],
        "consensusAccuracy": acc,
        "ci": ci,
        "perRunAccuracy": [round(a, 4) for a in per_run_acc],
        "meanPairwiseKappa": mean_kappa,
        "perJudgeAccuracy": _per_judge_acc(runs, judge_specs),
        "byDifficulty": _stratified(runs, "difficulty"),
        "byFailureType": _stratified(runs, "failureType"),
        "validated": all(checks.values()),
        "validatedChecks": checks,
        "labelProvenance": "first-party held-out, sealed (NOT third-party)",
        "scoring": "model-judged (entailment); validated only under the no-overclaim gate.",
        "candidateOnly": True,
    }
    if cases is not None:
        lex = lexical_baseline_accuracy(cases)
        report["lexicalBaselineAccuracy"] = lex
        report["judgeValueAdd"] = round(acc - lex, 4)
    return report


def _abstain_support_judge(claim: str, evidence: str) -> Support:
    """Offline 'mock' judge: abstains on every pair, with NO model client (so it
    never touches the network and a mock run can never validate)."""
    return Support(abstained=True, reason="mock judge (offline)", method="mock")


def build_judges(specs: list[str]) -> list:
    """Build entailment judges from specs. 'mock' maps to an explicit offline
    abstaining judge — it does NOT resolve a default client, so the mock path stays
    network-free and deterministic in CI (and can never clear the gate)."""
    return [
        _abstain_support_judge if s == "mock" else make_entailment_judge(s)
        for s in specs
    ]


__all__ = [
    "DEFAULT_PACK", "SEAL", "content_hash", "build_seal", "verify_seal", "load_pack",
    "run_once", "aggregate", "lexical_baseline_accuracy", "build_judges",
]
