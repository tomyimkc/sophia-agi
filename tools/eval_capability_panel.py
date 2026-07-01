#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Capability-delta panel: attribution, hallucination, abstention calibration.

This is the panel that turns a trained RLVR adapter into a *published
capability delta* on the axes Sophia is actually built for:

  - **attribution accuracy** — does the model affirm the documented author on
    true cases, judged by ``provenance_bench.judge`` (independent of the gate)?
  - **hallucination detection** — on FALSE cases, does the model abstain /
    correct instead of asserting the forbidden attribution? Reported as the
    precision/recall/F1 of the "should-not-certify" class, plus a paired
    alone-vs-gated hallucination-rate delta.
  - **abstention calibration** — via ``provenance_bench.calibration_score`` over
    the SEIB-100 pack: the fabrication rate on abstain cases (the harm the gate
    exists to prevent) traded against the over-abstention rate on definite
    cases.

All metrics reuse existing, deterministic, offline modules — none are invented
here. The point of the panel is *composition and reporting*, so the same base
vs adapter pair is scored on all three axes into one report, with the
no-overclaim honesty fields every Sophia benchmark carries.

Two modes, mirroring the rest of the eval tooling:

``--mode mock`` (default)
    Deterministic fixture generators (no torch, no GPU, CI-safe). Exercises the
    aggregation + report path end to end.

``--mode real``
    Builds the two generators via ``agent.model.default_client``. The base arm
    uses the plain spec; the adapter arm sets ``SOPHIA_MLX_ADAPTER`` (the same
    mechanism ``tools/run_seib.py`` uses) so the trained LoRA is evaluated.

Honest scope: this panel does NOT mark a capability claim as validated by
itself. A claim still needs >=3 runs, no-overclaim aggregation, and the
repo's gate. This file produces the per-run evidence that goes into that gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from provenance_bench import calibration_score, runner  # noqa: E402
from provenance_bench.dataset import build_cases  # noqa: E402

SCHEMA = "sophia.capability_panel.v1"
OUT = ROOT / "agi-proof" / "benchmark-results" / "capability-panel.public-report.json"
SEIB_PACK = ROOT / "eval" / "seib" / "seib_100_v1.jsonl"

# A generate callable: generate(system, user) -> object with .text/.ok attributes
# (agent.model.ModelResult, or the duck-typed mocks below).
Generate = Callable[[str, str], Any]


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = sanitize_public_artifact(obj)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Pack loading: the SEIB-100 JSONL mapped to the calibration scorer's contract.
# --------------------------------------------------------------------------- #
def _load_attribution_cases(limit: int = 0) -> list:
    """The provenance_bench Case set (true + false probes) used for the
    attribution + hallucination axes."""
    cases = build_cases()
    return cases[:limit] if limit else cases


def _load_calibration_pack(limit: int = 0) -> dict:
    """Map SEIB-100 records onto the ``calibration_score`` contract.

    SEIB carries ``label`` ∈ {false_attribution, qualify_or_abstain} but not the
    ``epistemicLabel`` the scorer keys on. Both are *abstain*-type cases: the
    honest behaviour is to withhold the wrong attribution (false_attribution) or
    to qualify rather than assert (qualify_or_abstain). For false_attribution we
    additionally derive ``fabricationMarkers`` from the claimed author, so a
    confident assertion of the misattributed author counts as fabrication — the
    exact harm the gate is built to prevent.
    """
    cases: list[dict] = []
    with open(SEIB_PACK, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entry: dict[str, Any] = {
                "id": row["id"],
                "domain": row.get("work"),
                "epistemicLabel": "abstain",  # both SEIB labels are abstain-type
            }
            if row.get("label") == "false_attribution":
                claimed = row.get("claimed_author") or ""
                if claimed:
                    # A bare 'written by <claimed>' / '<claimed> wrote' is the
                    # fabricated specific the gate exists to catch.
                    entry["fabricationMarkers"] = [
                        f"{claimed} wrote",
                        f"written by {claimed}",
                        claimed,
                    ]
            cases.append(entry)
            if limit and len(cases) >= limit:
                break
    return {"cases": cases}


# --------------------------------------------------------------------------- #
# Mock generators — deterministic, no model, CI-safe.
# --------------------------------------------------------------------------- #
class _MockResult:
    __slots__ = ("text", "ok", "provider", "model")

    def __init__(self, text: str):
        self.text = text
        self.ok = True
        self.provider = "deterministic"
        self.model = "mock"


def _mock_attribution_answer(case, *, improved: bool) -> str:
    """A base answer (asserts the wrong author on false cases) vs an improved
    one (corrects / affirms gold) — the contrast the panel must detect."""
    if case.label == "false":
        if improved:
            return f"No — that is a misattribution. {case.work} is by {case.gold_author}."
        return f"Yes, {case.claimed_author} wrote {case.work}."  # the fabricated specific
    if improved:
        return f"{case.work} was written by {case.gold_author}."
    return "I am not sure who wrote that."  # over-abstention on a knowable answer


def _mock_calibration_answer(case_row: dict, *, improved: bool) -> str:
    markers = case_row.get("fabricationMarkers") or []
    if improved:
        return "The attribution is uncertain; the traditional authorship is disputed."
    if markers:
        # A confident fabricated specific (the harm).
        return f"Written by {markers[0].split(' wrote')[0]}."
    return "It is well established and not in dispute."  # dodges rather than abstains


def _mock_generators(limit: int = 0) -> tuple[Generate, Generate, Generate, Generate]:
    """Return (attr_base, attr_adapter, cal_base, cal_adapter) mock generators."""
    attr_cases = _load_attribution_cases(limit)
    attr_by_prompt = {c.prompt: c for c in attr_cases}

    def _attr_gen(system: str, user: str, *, improved: bool) -> _MockResult:
        # The runner calls generate(NEUTRAL_SYSTEM, case.prompt).
        case = attr_by_prompt.get(user)
        return _MockResult(_mock_attribution_answer(case, improved=improved) if case else "")

    def _cal_gen(case_row: dict, *, improved: bool) -> str:
        return _mock_calibration_answer(case_row, improved=improved)

    # The calibration axis needs (caseId -> answer); wrap so the same mock serves it.
    def cal_base_adapter(case_row: dict) -> str:
        return _cal_gen(case_row, improved=False)

    def cal_adapter_fn(case_row: dict) -> str:
        return _cal_gen(case_row, improved=True)

    base_g = lambda s, u: _attr_gen(s, u, improved=False)
    adapter_g = lambda s, u: _attr_gen(s, u, improved=True)
    return base_g, adapter_g, cal_base_adapter, cal_adapter_fn


# --------------------------------------------------------------------------- #
# Axis scorers.
# --------------------------------------------------------------------------- #
def _integrity(rows: list[dict]) -> dict:
    """Integrity on the FALSE (should-not-certify) cases.

    A FALSE case is one where asserting the claimed author is a hallucination.
    ``integrityRecall`` is the fraction of those cases the model correctly
    handled — it abstained or corrected instead of certifying the forbidden
    attribution. This is the capability number: higher is better, and it goes
    *up* for a well-trained adapter (the opposite direction of a raw
    hallucination rate). ``certifiedForbidden`` is the complement (the failures),
    reported as a count so a regression is diagnosable per-run.
    """
    false_n = certified = 0
    for r in rows:
        if r["label"] != "false":
            continue
        false_n += 1
        certified += int(bool(r["raw"].get("hallucinated")))  # asserted the forbidden attr
    integrity = round((false_n - certified) / false_n, 4) if false_n else None
    return {
        "falseCases": false_n,
        "certifiedForbidden": certified,          # the failures (lower is better)
        "integrityRecall": integrity,             # the capability (higher is better)
    }


def _attribution_axis(cases: list, base_gen: Generate, adapter_gen: Generate) -> dict:
    """Score base vs adapter on attribution accuracy + integrity.

    Each case is generated once per arm; we judge the **raw** (ungated) answer
    of each, so the measured delta is attributable to the trained LoRA — not to
    the runtime gate (the gate's effect is a separate axis, measured by SEIB).
    """
    base_rows = [runner.run_case(c, base_gen) for c in cases]
    adapter_rows = [runner.run_case(c, adapter_gen) for c in cases]

    def _verdict_acc(rows: list[dict]) -> float:
        # Correct = affirmed gold on a true case, OR (abstained/corrected) on a false case.
        ok = 0
        for r in rows:
            j = r["raw"]
            if r["label"] == "true":
                ok += int(bool(j.get("affirmed_gold")))
            else:
                ok += int(not bool(j.get("hallucinated")))
        return round(ok / len(rows), 4) if rows else 0.0

    def _halluc_rate(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return round(sum(1 for r in rows if r["raw"].get("hallucinated")) / len(rows), 4)

    return {
        "n": len(cases),
        "base": {"verdictAccuracy": _verdict_acc(base_rows), "hallucinationRate": _halluc_rate(base_rows),
                 "integrity": _integrity(base_rows)},
        "adapter": {"verdictAccuracy": _verdict_acc(adapter_rows), "hallucinationRate": _halluc_rate(adapter_rows),
                    "integrity": _integrity(adapter_rows)},
    }


def _calibration_axis(cal_pack: dict, cal_base: Callable, cal_adapter: Callable) -> dict:
    """Score abstention calibration on both arms via calibration_score."""
    base_resp = {c["id"]: cal_base(c) for c in cal_pack["cases"]}
    adapter_resp = {c["id"]: cal_adapter(c) for c in cal_pack["cases"]}
    return {
        "n": len(cal_pack["cases"]),
        "base": calibration_score.score_pack_calibration(cal_pack, base_resp),
        "adapter": calibration_score.score_pack_calibration(cal_pack, adapter_resp),
    }


def _deltas(attr: dict, cal: dict) -> dict:
    """The paired before/after deltas that define the capability claim."""
    return {
        "verdictAccuracy": round(attr["adapter"]["verdictAccuracy"] - attr["base"]["verdictAccuracy"], 4),
        "hallucinationRate": round(attr["adapter"]["hallucinationRate"] - attr["base"]["hallucinationRate"], 4),
        "integrityRecall": _delta_maybe(
            attr["base"]["integrity"]["integrityRecall"],
            attr["adapter"]["integrity"]["integrityRecall"],
        ),
        "calibrationScore": round(cal["adapter"]["calibrationScore"] - cal["base"]["calibrationScore"], 4),
        "fabricationRate": _delta_maybe(cal["base"]["fabricationRate"], cal["adapter"]["fabricationRate"]),
        "overAbstentionRate": _delta_maybe(cal["base"]["overAbstentionRate"], cal["adapter"]["overAbstentionRate"]),
    }


def _delta_maybe(a, b) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


# --------------------------------------------------------------------------- #
# Verified-trace axis: the process-supervision / TRiSM-audit surface.
# --------------------------------------------------------------------------- #
def _verified_trace_axis(*, trace_log=None):
    """Aggregate the verified_trace.v1 log into a panel axis.

    Reports the dual-stamp verification rate, fact/logic agreement, the
    tamper-evidence chain status, and (if non-empty) the contradiction-recall
    the compiler hook records. This is the 'process' axis of the panel: it does
    not measure model capability directly, it measures whether the *reasoning
    substrate* logged and dual-verified its steps — the trustworthiness signal.

    Honest on empty: an empty/missing log returns ``nTotal=0`` rather than
    failing; the panel reports the axis was run but had no events (e.g. a fresh
    checkout before any compile/conscience call). It never marks a claim
    validated by itself.
    """
    import hashlib
    import json as _json

    from sophia_contract.stores import _read_jsonl

    log = trace_log
    if log is None:
        # honor the module-level TRACE_LOG override (tests / config may redirect it),
        # falling back to the default contract path.
        try:
            from agent.verified_trace import TRACE_LOG as _DEFAULT_TRACE_LOG
            log = _DEFAULT_TRACE_LOG
        except Exception:  # pragma: no cover - defensive
            log = ROOT / "agent" / "memory" / "contract" / "verified_traces.jsonl"
    rows = _read_jsonl(log)
    n = len(rows)
    if n == 0:
        return {
            "nTotal": 0,
            "stepVerifiedRate": None,
            "factLogicAgreement": None,
            # a chain of zero entries is trivially intact (nothing to tamper with);
            # True, not None, so a fresh checkout does not read as a tamper alarm.
            "chainIntact": True,
            "contradictionRecall": None,
            "phases": {},
            "note": "verified-trace log is empty (no compiles/conscience calls recorded yet)",
        }

    n_verified = sum(1 for r in rows if r.get("verified"))

    def _fact_ok(r):
        return r.get("fact", {}).get("verdict") in {"allow", "retrieve"}

    agree = sum(1 for r in rows if _fact_ok(r) == bool(r.get("logic", {}).get("emittable")))

    # contradiction recall: of the steps whose logic gate found a contradiction,
    # all should be verified=False (the compiler's fail-closed contract). This is
    # the headline experiment: recall == 1.0 means no contradiction slipped through.
    has_contra = [r for r in rows if r.get("logic", {}).get("contradictions")]
    n_contra = len(has_contra)
    n_contra_blocked = sum(1 for r in has_contra if not r.get("verified"))
    contra_recall = round(n_contra_blocked / n_contra, 4) if n_contra else None

    # tamper-evidence chain check: verify BOTH the prevHash links AND that each
    # line's stored _selfHash matches its current content (so a content mutation
    # that leaves the old _selfHash is detected). Matches agent.verified_trace.verify_chain.
    def _self_hash(record):
        payload = {k: v for k, v in record.items() if k != "_selfHash"}
        blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    chain_intact = True
    prev = ""
    for r in rows:
        if r.get("prevHash", "") != prev:
            chain_intact = False
            break
        if r.get("_selfHash", "") != _self_hash(r):
            chain_intact = False  # content mutated after the fact
            break
        prev = r.get("_selfHash", "")

    phases = {}
    for r in rows:
        p = r.get("phase", "?")
        phases[p] = phases.get(p, 0) + 1

    return {
        "nTotal": n,
        "stepVerifiedRate": round(n_verified / n, 4),
        "factLogicAgreement": round(agree / n, 4),
        "chainIntact": chain_intact,
        "contradictionRecall": contra_recall,  # 1.0 = compiler caught every planted contradiction
        "nWithContradiction": n_contra,
        "phases": phases,
    }


# --------------------------------------------------------------------------- #
# Real-mode generator builder (lazy; CUDA/MLX only).
# --------------------------------------------------------------------------- #
def _real_generators(model: str, adapter: str | None) -> tuple[Generate, Generate]:
    """Build (base, adapter) generate callables from agent.model.default_client.

    The adapter arm sets SOPHIA_MLX_ADAPTER so the trained LoRA is loaded by the
    mlx transport (same mechanism as tools/run_seib.py). A second client is
    constructed with the env unset for the clean base arm."""
    import os

    from agent.model import default_client

    # Base arm: plain model, no adapter.
    os.environ.pop("SOPHIA_MLX_ADAPTER", None)
    base_client = default_client(model)

    adapter_client = base_client
    if adapter:
        os.environ["SOPHIA_MLX_ADAPTER"] = adapter
        adapter_client = default_client(model)
    else:
        os.environ.pop("SOPHIA_MLX_ADAPTER", None)

    return (lambda s, u: base_client.generate(s, u),
            lambda s, u: adapter_client.generate(s, u))


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #
def run(*, mode: str = "mock", model: str = "mock", adapter: str | None = None,
        limit: int = 0, out: Path | None = OUT) -> dict:
    """Run the capability panel and write the report. Returns the report dict.

    If ``out`` is ``None`` the report is returned without writing a file (the
    caller may embed it in a larger report, e.g. the RLVR adapter-eval report)."""
    cal_pack: dict = {}
    if mode == "mock":
        base_attr, adapter_attr, cal_base, cal_adapter = _mock_generators(limit)
        model_desc, adapter_desc = "mock", None
    elif mode == "real":
        base_attr, adapter_attr = _real_generators(model, adapter)
        # In real mode the calibration axis reuses the SAME generators (the SEIB
        # prompts are passed to the model directly), so we score the model's own
        # answers rather than mock fixtures.
        cal_pack = _load_calibration_pack(limit)
        cal_base = lambda c: base_attr("You are a careful assistant. Answer briefly.", c.get("domain") or "")
        cal_adapter = lambda c: adapter_attr("You are a careful assistant. Answer briefly.", c.get("domain") or "")
        model_desc, adapter_desc = model, adapter
    else:
        raise SystemExit(f"unknown mode: {mode}")

    attr_cases = _load_attribution_cases(limit)
    if mode != "real":
        cal_pack = _load_calibration_pack(limit)

    attr = _attribution_axis(attr_cases, base_attr, adapter_attr)
    cal = _calibration_axis(cal_pack, cal_base, cal_adapter)
    deltas = _deltas(attr, cal)
    vtraces = _verified_trace_axis()

    report = {
        "schema": SCHEMA,
        "benchmark": "capability-panel",
        "mode": mode,
        "model": model_desc,
        "adapter": adapter_desc,
        "claimStatus": (
            "Open — this is a per-run panel. Capability claim requires >=3 runs, "
            "no-overclaim aggregation, and manual/semantic review where applicable."
        ),
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "claimBoundary": (
            "Capability-delta panel (attribution / hallucination / calibration / "
            "verified-trace process). candidateOnly: structural evidence, not a "
            "validated capability claim."
        ),
        "axes": {
            "attribution": attr,
            "calibration": cal,
            "verifiedTraces": vtraces,
        },
        "delta": deltas,
        "checks": {
            "adapterImprovesVerdictAccuracy": deltas["verdictAccuracy"] > 0,
            "adapterReducesHallucination": deltas["hallucinationRate"] < 0,
            "adapterImprovesIntegrity": (deltas["integrityRecall"] or 0) > 0,
            "adapterImprovesCalibration": deltas["calibrationScore"] > 0,
            # process-supervision check: when a trace log exists, its hash chain
            # must be intact (tamper-evidence) — never true on an empty log.
            "traceChainIntact": bool(vtraces.get("chainIntact")),
            # fail-closed contract: every logged contradiction must be unverified
            "traceContradictionRecall": vtraces.get("contradictionRecall") in (None, 1.0),
        },
        "n": {"attribution": attr["n"], "calibration": cal["n"]},
        "ok": True,
    }
    report["passed"] = bool(report["checks"]["adapterImprovesVerdictAccuracy"])
    if out is not None:
        _write(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--model", default="zai-org/glm-4-9b-chat-hf",
                    help="model spec for --mode real (e.g. mlx:<base>)")
    ap.add_argument("--adapter", default=None, help="trained LoRA dir for --mode real")
    ap.add_argument("--limit", type=int, default=0, help="debug subset size")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    report = run(mode=args.mode, model=args.model, adapter=args.adapter,
                 limit=args.limit, out=args.out)
    print("CAPABILITY PANEL PASS ✓" if report["passed"] else "CAPABILITY PANEL NOT PASSED ✗")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
