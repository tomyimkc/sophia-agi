# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 benchmark harness for the verifier-gated real-time grounding loop.

This measures **claim C1 — verifier-as-truth-filter**: does the ingestion gate admit
true live facts and reject false/unknowable ones? It is fully offline and
deterministic (labeled `eval/fact_check/heldout_v1.jsonl` + `FixtureFactBackend`), so
it is a reproducible *floor*, not a live capability claim.

Design follows the repo's no-overclaim contract and the `rlvr-harness-traps` lessons:

  * The load-bearing metrics are admission **precision / recall / fabricationRate**,
    NOT a mean reward. Fine arm deltas are reported through
    ``eval_stats.verdict_or_underpowered`` so an underpowered N (heldout is N~53,
    MDE~0.27) yields "underpowered", never a spurious direction.
  * A **control-sanity** guard (the 0/N-on-both-sides analog): if the verifier admits
    ~no known-true claims AND rejects ~no known-false claims, the backend/harness is
    broken — refuse to read the arms.
  * Arms isolate the verifier's value: ``accept_all`` (no filter), ``raw_rag`` (admit
    on any retrieved evidence, no verifier), ``verdict_only`` (verifier, no conformal),
    ``gated`` (+ conformal), ``full`` (+ decontam + valid-time). The falsifier is
    ``full ~= raw_rag``.

Constructs shipped here = the deterministic verifier + the verdict-level metrics from
``fact_check_eval``. A second, independent LLM-judge family (kappa>=0.40) and a
powered N are Phase-1 (online) requirements — flagged, never faked. Every record is
``candidateOnly=True`` / ``level3Evidence=False``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent import streaming_decontam as sd
from agent.conformal_gate import ConformalPolicy
from agent.fact_check_eval import run_fact_check_eval, wilson_interval
from agent.fact_check_gate import AtomicClaim, classify_claim, fact_check_text, risk_for
from agent.realtime_grounding import DEFAULT_NONCONFORMITY_THRESHOLD, nonconformity
from tools.eval_stats import mcnemar, mde_at_n, required_n_for_mde, verdict_or_underpowered

SCHEMA = "sophia.realtime_benchmark.v1"

# Arm -> human note. Admission booleans are derived from the SAME per-row components
# so every arm is scored on the identical pack/grader (fair contrast).
ARMS = ("accept_all", "raw_rag", "verdict_only", "gated", "full")


def _backend_kwargs(backend: Any) -> dict[str, Any]:
    if backend is None:
        return {}
    return {
        "retriever": getattr(backend, "retriever", None),
        "entailment": getattr(backend, "entailment", None),
        "doi_resolver": getattr(backend, "doi_resolver", None),
        "url_resolver": getattr(backend, "url_resolver", None),
    }


def _components(
    row: dict[str, Any],
    *,
    backend: Any,
    policy: ConformalPolicy | None,
    eval_prompts: set[str] | None,
    eval_cutoff: str | None,
    as_of: str,
) -> dict[str, Any]:
    """Compute every gate signal for one claim, once, so arms are consistent."""
    claim = str(row["claim"])
    decision = fact_check_text(claim, **_backend_kwargs(backend))
    nc = nonconformity(decision)
    conf_answer = (policy.decide(nc)["verdict"] == "answer") if policy is not None else (nc <= DEFAULT_NONCONFORMITY_THRESHOLD)

    has_sources = False
    if backend is not None and getattr(backend, "retriever", None) is not None:
        atom = AtomicClaim(claim, classify_claim(claim), risk_for(claim))
        try:
            has_sources = len(backend.retriever(atom) or []) > 0
        except Exception:  # a backend fault must not crash the benchmark
            has_sources = False

    content_ok = sd.content_decontam(claim, eval_prompts)["ok"]
    temporal_ok = sd.temporal_decontam(row.get("sourceTimestamp", ""), eval_cutoff)["ok"]
    valid_ok = sd.valid_time(row.get("validFrom", ""), row.get("validUntil", ""), as_of)["ok"]

    accepted = decision.verdict == "accepted"
    admits = {
        "accept_all": True,
        "raw_rag": has_sources,
        "verdict_only": accepted,
        "gated": accepted and conf_answer,
        "full": accepted and conf_answer and content_ok and temporal_ok and valid_ok,
    }
    return {
        "id": row.get("id"),
        "label": row["label"],
        "shouldAdmit": row["label"] == "true",
        "verdict": decision.verdict,
        "nonconformity": nc,
        "admits": admits,
    }


def _arm_metrics(comps: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    admitted = [c for c in comps if c["admits"][arm]]
    tp = sum(1 for c in admitted if c["shouldAdmit"])
    fp = len(admitted) - tp
    n_true = sum(1 for c in comps if c["shouldAdmit"])
    precision = round(tp / len(admitted), 4) if admitted else 0.0
    recall = round(tp / n_true, 4) if n_true else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0
    correct = [1 if (c["admits"][arm] == c["shouldAdmit"]) else 0 for c in comps]
    return {
        "nAdmitted": len(admitted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fabricationRate": round(fp / len(admitted), 4) if admitted else 0.0,
        "precisionWilson95": wilson_interval(tp, len(admitted)),
        "admissionAccuracy": round(sum(correct) / len(comps), 4) if comps else 0.0,
        "_correct": correct,  # per-item, for paired tests (stripped before serialize)
    }


def control_sanity(comps: list[dict[str, Any]]) -> dict[str, Any]:
    """The 0/N-on-both-sides guard: is the verifier actually engaging on this pack?"""
    trues = [c for c in comps if c["label"] == "true"]
    falses = [c for c in comps if c["label"] == "false"]
    known_true_admit = round(sum(1 for c in trues if c["verdict"] == "accepted") / len(trues), 4) if trues else 0.0
    known_false_reject = round(sum(1 for c in falses if c["verdict"] != "accepted") / len(falses), 4) if falses else 0.0
    ok = known_true_admit > 0.5 and known_false_reject > 0.5
    return {
        "knownTrueAdmitRate": known_true_admit,
        "knownFalseRejectRate": known_false_reject,
        "ok": ok,
        "note": ("verifier engages on both polarities" if ok else
                 "SUSPECT HARNESS/BACKEND ARTIFACT: verifier near-degenerate on a polarity — do not read the arms"),
    }


def run_c1_benchmark(
    rows: list[dict[str, Any]],
    *,
    backend: Any,
    policy: ConformalPolicy | None = None,
    eval_prompts: set[str] | None = None,
    eval_cutoff: str | None = None,
    as_of: str = "2026-07-01",
    practical_threshold: float = 0.10,
    live_backend: bool = False,
) -> dict[str, Any]:
    """Score admission arms + the verifier-vs-raw-RAG contrast on a labeled pack.

    ``eval_prompts``/``eval_cutoff`` default to None so the content/temporal gates are
    no-ops when benchmarking the verifier ON the held-out itself (the held-out IS the
    eval surface; decontaminating it against itself would veto everything). Pass a real
    eval surface + cutoff only for a distinct live/adversarial pack.
    """
    comps = [
        _components(r, backend=backend, policy=policy, eval_prompts=eval_prompts, eval_cutoff=eval_cutoff, as_of=as_of)
        for r in rows if str(r.get("claim", "")).strip()
    ]
    control = control_sanity(comps)
    arms = {arm: _arm_metrics(comps, arm) for arm in ARMS}

    # Verifier's value over naive retrieval, paired McNemar on per-item correctness.
    full_correct = arms["full"]["_correct"]
    raw_correct = arms["raw_rag"]["_correct"]
    delta_acc = round(arms["full"]["admissionAccuracy"] - arms["raw_rag"]["admissionAccuracy"], 4)
    verifier_vs_rawrag = {
        "deltaAdmissionAccuracy": delta_acc,
        "mcnemar": mcnemar(raw_correct, full_correct),
        "powerVerdict": verdict_or_underpowered(delta_acc, len(comps), tolerance=practical_threshold),
    }
    # Second construct: verdict-level metrics (fabricationRate/ECE/Brier) from fact_check_eval.
    verdict_construct = run_fact_check_eval(rows, **_backend_kwargs(backend), live_backend=live_backend)

    n = len(comps)
    power = {
        "n": n,
        "mdeAtN": round(mde_at_n(n), 4),
        "requiredNForThreshold": required_n_for_mde(practical_threshold),
        "practicalThreshold": practical_threshold,
        "note": "any arm delta smaller than mdeAtN is not resolvable at this N (report underpowered, do not claim a direction)",
    }
    for arm in arms.values():
        arm.pop("_correct", None)

    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "level3Evidence": False,
        "liveBackendUsed": live_backend,
        "n": n,
        "labelCounts": {lab: sum(1 for c in comps if c["label"] == lab) for lab in ("true", "false", "unknowable")},
        "controlSanity": control,
        "arms": arms,
        "verifierVsRawRag": verifier_vs_rawrag,
        "verdictConstruct": {k: verdict_construct[k] for k in ("metrics", "confidenceIntervals", "labelCounts") if k in verdict_construct},
        "power": power,
        "constructs": [
            "deterministic layered verifier (admission precision/recall)",
            "verdict-level metrics (fabricationRate/ECE/Brier via fact_check_eval)",
        ],
        "constructGaps": [
            "second independent LLM-judge family (kappa>=0.40) — Phase-1 (online), not run offline",
            "powered N and live web sources — Phase-1; committed N is a floor/wiring check",
        ],
        "verdict": _honest_verdict(control, verifier_vs_rawrag, power),
    }


def _honest_verdict(control: dict, vvr: dict, power: dict) -> dict[str, Any]:
    if not control["ok"]:
        return {"label": "invalid", "reason": "control-sanity failed; harness/backend artifact suspected"}
    pv = vvr["powerVerdict"]
    if not pv["powered"]:
        return {"label": "candidate-underpowered",
                "reason": f"wiring + verifier floor measured, but N={power['n']} cannot resolve a {power['practicalThreshold']} effect "
                          f"(MDE={power['mdeAtN']}); grow to N>={power['requiredNForThreshold']} in Phase-1 for a GO/NO-GO"}
    return {"label": "candidate-resolvable",
            "reason": f"verifier-vs-rawRAG delta {vvr['deltaAdmissionAccuracy']} is resolvable at N={power['n']}; "
                      f"still candidateOnly (needs >=2 judge families + live pack for promotion)"}


def load_pack(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def write_report(report: dict[str, Any], out: str | Path) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["SCHEMA", "ARMS", "run_c1_benchmark", "control_sanity", "load_pack", "write_report"]
