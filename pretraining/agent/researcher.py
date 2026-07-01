# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A role-conditioned *reviewer* agent: the DeepSeek pretraining researcher, as a critic.

Deliberately **NOT an "AGI agent."** Sophia's charter forbids AGI claims (see VISION.md);
every artifact here is candidate-only and carries ``canClaimAGI: false``. What this *is*:
an evaluator that embodies the pretraining (data / algorithm) researcher role as a rubric
and **audits the ``pretraining/`` studies** — checking each against its pre-registered gate,
flagging overclaims, scoring research taste, and proposing the next experiment.

Why it's useful for "future testing":
  * **Regression harness** — re-run it after any change to confirm the studies still clear
    their pre-registration gates (``review_all()`` returns machine-checkable verdicts).
  * **Persona benchmark** — the role embodied as a critic, with a fixed rubric, so the repo
    can be scored the way this employer would score it.

Fail-closed by construction: a missing report yields ``cannot_assess`` (never ``pass``); the
agent flags weaknesses rather than fabricating strengths. The core is deterministic, pure
stdlib, and offline; an optional LLM critique (``llm=True``) is purely additive and degrades
gracefully when no backend/key is present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

PKG = Path(__file__).resolve().parents[1]   # pretraining/

# The role, distilled from the job description into an auditable rubric. Each dimension is
# scored against the repo's pretraining artifacts as strong / partial / absent — honestly.
ROLE = {
    "title": "预训练（数据/算法）研究员 — pretraining researcher (reviewer persona)",
    "canClaimAGI": False,
    "tracks": {
        "algorithm": [
            "model-structure innovation (软硬件协同)",
            "optimizer design + training dynamics/stability",
            "scientific scaling laws (fit / predict / plan)",
            "training & inference acceleration",
            "novel architectures (探索性)",
        ],
        "data": [
            "data pipeline + governance (可追溯/可复现)",
            "data mixing & filtering strategy (配比)",
            "synthetic data + its scaling behaviour",
            "multi-dimensional eval (auto + human)",
            "vertical data (multimodal / agent / feedback / academic)",
        ],
    },
    "values": ["严谨的科学精神", "出色的研究品味", "对模型能力边界有敏锐判断",
               "不过度宣称 (no overclaim)"],
}


def _load(rel: str) -> dict | None:
    p = PKG / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _verdict(passed: bool | None) -> str:
    if passed is None:
        return "cannot_assess"
    return "pass" if passed else "concern"


# -- per-study auditors --------------------------------------------------------
# Each returns (verdict, evidence, critiques, next_experiments). They read the committed
# report and check the property the study claims — overclaim defence, not rubber-stamp.

def _audit_scaling(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no scaling report",
                "critiques": ["run pretraining.scaling.run_scaling"], "next_experiments": []}
    pred = r.get("prediction") or {}
    passed = bool(pred.get("passes_10pct_gate"))
    floor_ok = r.get("recovered_floor_within_15pct")
    crit = []
    if not floor_ok:
        crit.append("irreducible floor E is under-identified away from saturation — "
                    "the report says so, which is the honest call, not a defect")
    if r.get("config", {}).get("seeds", 0) < 3:
        crit.append("seeds<3: report confidence intervals before trusting the exponent")
    return {
        "verdict": _verdict(passed),
        "evidence": f"extrapolation rel_err={pred.get('relative_error')}, "
                    f"r2={r.get('fit_free_floor', {}).get('r2_logspace')}, "
                    f"floor_identified={floor_ok}",
        "critiques": crit,
        "next_experiments": [
            "extend D to saturation so the free-floor fit can recover E",
            "joint L(N,D) fit: sweep hidden width with stabilized optimization, fit a 2D law",
            "Chinchilla-style compute-optimal frontier: trade N vs D at fixed compute budget",
        ],
    }


def _audit_optimizer(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no optimizer report",
                "critiques": ["run pretraining.optimizer_probe.run_probe"], "next_experiments": []}
    summ = r.get("summary", [])
    all_have_stable = all(s.get("max_stable_lr") for s in summ)
    return {
        "verdict": _verdict(all_have_stable),
        "evidence": "; ".join(f"{s['optimizer']}: best_loss={s['best_held_loss']}, "
                              f"max_stable_lr={s['max_stable_lr']}" for s in summ),
        "critiques": ([] if all_have_stable else
                      ["an optimizer diverged at every lr — widen the grid or add warmup"]),
        "next_experiments": [
            "add Muon / Lion / Shampoo and a lr warmup+cosine schedule",
            "measure loss-spike recovery (inject a bad batch; does it recover?)",
            "report gradient-noise scale to predict the critical batch size",
        ],
    }


def _audit_architecture(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no architecture report",
                "critiques": ["run pretraining.architecture.run_arch"], "next_experiments": []}
    moe = r.get("moe", {})
    balanced = bool(moe.get("balanced"))
    crit = []
    if not balanced:
        crit.append("MoE routing collapsed onto few experts — balance penalty too weak")
    if r.get("verdict") == "dense_better":
        crit.append("MoE does not beat dense at this scale — honest; the order-2 source is "
                    "captured by one dense block, so experts have nothing to specialize on")
    return {
        "verdict": _verdict(balanced),   # the GATE is no-collapse, not 'MoE must win'
        "evidence": f"verdict={r.get('verdict')}, load_balance_max_share="
                    f"{moe.get('load_balance_max_share')}, "
                    f"active/token={moe.get('active_params_per_token')} of {moe.get('total_params')}",
        "critiques": crit,
        "next_experiments": [
            "add shared experts + top-k>1 (fine-grained MoE) and a harder higher-order source",
            "implement a nano MLA block: low-rank KV latent, measure cache size vs loss",
            "auxiliary-loss-free load balancing (bias-adjusted routing)",
        ],
    }


def _audit_passport(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no datasheet",
                "critiques": ["run pretraining.data_passport.build_passport on a pack"],
                "next_experiments": []}
    dup = r.get("duplicate_rate", 0.0)
    unlicensed = r.get("by_license", {}).get("unknown", 0)
    crit = []
    if dup > 0.3:
        crit.append(f"duplicate_rate={dup} is high — dedup by cluster before training")
    if unlicensed:
        crit.append(f"{unlicensed} rows unlicensed — backfill license/provenance, fail-closed")
    # the passport WORKING (surfacing these) is the pass condition
    return {
        "verdict": _verdict(r.get("rows", 0) > 0),
        "evidence": f"rows={r.get('rows')}, dup_rate={dup}, "
                    f"mean_quality={r.get('mean_quality')}, flagged={r.get('flagged_rows')}",
        "critiques": crit,
        "next_experiments": [
            "semantic (embedding) dedup to catch paraphrase duplicates MinHash misses",
            "calibrate the quality score against held-out downstream loss",
            "license/source backfill pipeline so no row ships 'unknown'",
        ],
    }


def _audit_mixing(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no mixing report",
                "critiques": ["run pretraining.data_mixing.run_mixing"], "next_experiments": []}
    blend = (r.get("targets", {}) or {}).get("blend", {})
    interior = bool(blend.get("interior_optimum"))
    return {
        "verdict": _verdict(interior),
        "evidence": f"blend best_weight_A={blend.get('best_weight_A')} "
                    f"(interior={interior}); single-source targets skew to their source",
        "critiques": ([] if interior else
                      ["blended target's optimum is at a boundary — expected interior; "
                       "check the two sources are actually distinguishable"]),
        "next_experiments": [
            "scale to >2 sources and search the simplex (not just a 1-D ratio)",
            "curriculum: anneal the mix over training rather than hold it fixed",
            "proxy→target transfer: does the small-model optimum hold at larger scale?",
        ],
    }


def _audit_synthetic(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no synthetic report",
                "critiques": ["run pretraining.synthetic_scaling.run_synthetic"], "next_experiments": []}
    res = r.get("results", {})
    hi = res.get("high_fidelity", {})
    lo = res.get("low_fidelity", {})
    # gate: low-fidelity collapses AND high-fidelity does not
    passed = bool(lo.get("collapsed")) and not bool(hi.get("collapsed"))
    return {
        "verdict": _verdict(passed),
        "evidence": f"high_fidelity collapsed={hi.get('collapsed')} (best_mult={hi.get('best_multiple')}), "
                    f"low_fidelity collapsed={lo.get('collapsed')} (best_mult={lo.get('best_multiple')})",
        "critiques": ([] if passed else
                      ["could not reproduce the fidelity→collapse split — widen the drift gap"]),
        "next_experiments": [
            "sweep drift as a continuum and locate the collapse threshold",
            "early-warning detector: flag collapse from held-out divergence mid-training",
            "real+synthetic ratio sweep at fixed budget (anchor synthetic with real data)",
        ],
    }


def _audit_eval_matrix(r: dict | None) -> dict:
    if not r:
        return {"verdict": "cannot_assess", "evidence": "no eval matrix",
                "critiques": ["run pretraining.eval_matrix.run_matrix"], "next_experiments": []}
    surfaced = len(r.get("uncovered_cells", [])) > 0
    multimodal_gap = any("multimodal" in c for c in r.get("uncovered_cells", []))
    return {
        "verdict": _verdict(surfaced),   # gate: gaps must be SURFACED, not hidden
        "evidence": f"coverage={r.get('covered_cells')}/{r.get('total_cells')} "
                    f"({r.get('coverage_fraction')}), multimodal_uncovered={multimodal_gap}",
        "critiques": [f"low coverage ({r.get('coverage_fraction')}); "
                      "most capability×domain cells are untested — this is the gap to fill"],
        "next_experiments": [
            "stand up the multimodal cell (the one entirely-empty modality)",
            "add a documented human-eval protocol + inter-rater agreement per cell",
            "weight cells by difficulty so coverage isn't just case-count",
        ],
    }


_AUDITS: dict[str, tuple[str, Callable[[dict | None], dict]]] = {
    "scaling": ("scaling/scaling-curve-latest.json", _audit_scaling),
    "optimizer": ("optimizer_probe/optimizer-probe-latest.json", _audit_optimizer),
    "architecture": ("architecture/arch-probe-latest.json", _audit_architecture),
    "data_passport": ("data_passport/datasheet-curriculum.json", _audit_passport),
    "data_mixing": ("data_mixing/mixing-curve-latest.json", _audit_mixing),
    "synthetic_scaling": ("synthetic_scaling/synthetic-scaling-latest.json", _audit_synthetic),
    "eval_matrix": ("eval_matrix/eval-matrix-latest.json", _audit_eval_matrix),
}

# which role-rubric dimensions each study speaks to
_DIMENSION_MAP = {
    "scaling": ("algorithm", "scientific scaling laws (fit / predict / plan)"),
    "optimizer": ("algorithm", "optimizer design + training dynamics/stability"),
    "architecture": ("algorithm", "novel architectures (探索性)"),
    "data_passport": ("data", "data pipeline + governance (可追溯/可复现)"),
    "data_mixing": ("data", "data mixing & filtering strategy (配比)"),
    "synthetic_scaling": ("data", "synthetic data + its scaling behaviour"),
    "eval_matrix": ("data", "multi-dimensional eval (auto + human)"),
}


def role_card() -> dict:
    """The reviewer persona — the role as an auditable rubric (not an AGI claim)."""
    return dict(ROLE)


def review_all(*, llm: bool = False) -> dict:
    """Audit every pretraining study against its pre-registered gate. Returns a structured,
    fail-closed review with per-study verdicts, an overall tally, role-rubric coverage, and
    (optionally) an additive LLM critique."""
    studies = {}
    for name, (rel, fn) in _AUDITS.items():
        result = fn(_load(rel))
        track, dim = _DIMENSION_MAP[name]
        result["track"], result["rubric_dimension"] = track, dim
        studies[name] = result

    tally = {"pass": 0, "concern": 0, "cannot_assess": 0}
    for s in studies.values():
        tally[s["verdict"]] = tally.get(s["verdict"], 0) + 1

    # rubric coverage: which dimensions have a passing study
    rubric = {}
    for track, dims in ROLE["tracks"].items():
        rubric[track] = {}
        for dim in dims:
            studied = [n for n, s in studies.items() if s["rubric_dimension"] == dim]
            best = "absent"
            if studied:
                verdicts = {studies[n]["verdict"] for n in studied}
                best = "strong" if "pass" in verdicts else (
                    "partial" if "concern" in verdicts else "unassessed")
            rubric[track][dim] = {"studies": studied, "status": best}

    review = {
        "agent": "pretraining-researcher reviewer (persona)",
        "canClaimAGI": False,
        "honesty_note": ("This is a role-conditioned critic over toy studies, not an AGI "
                         "agent and not a capability claim. Verdicts check pre-registered "
                         "gates; missing reports are cannot_assess, never pass."),
        "role": role_card(),
        "studies": studies,
        "tally": tally,
        "rubric_coverage": rubric,
        "overall": ("all-gates-pass" if tally["concern"] == 0 and tally["cannot_assess"] == 0
                    else "open-items"),
    }
    if llm:
        review["llm_critique"] = _llm_critique(review)
    return review


def _llm_critique(review: dict) -> dict:
    """Optional, additive LLM critique. Degrades to 'unavailable' offline — never blocks."""
    try:
        from agent.model import complete
        system = ("You are a DeepSeek pretraining researcher reviewing a junior's toy study "
                  "package. Be rigorous and honest; flag any overclaim; do NOT call anything "
                  "AGI. Reply in <=120 words.")
        user = json.dumps({k: review[k] for k in ("tally", "rubric_coverage")},
                          ensure_ascii=False)
        text = complete(system, user, max_tokens=400)
        return {"available": True, "text": text}
    except Exception as exc:  # noqa: BLE001 - LLM is optional
        return {"available": False, "reason": str(exc)}


__all__ = ["ROLE", "role_card", "review_all"]
