#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bundle repo stats and leaderboards into web/data/manifest.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data" / "manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")

# Items mentioning the training method/recipe are kept OFF the public manifest
# (see tools/lint_web_privacy.py for the published-site privacy policy).
_TRAINING_DETAIL_RE = re.compile(r"RLVR|pass@1|reward model|fine[\s\-]?tun", re.I)


def _sanitize_required(items: list) -> list:
    """Drop required-proof items that would reveal the training method."""
    return [item for item in items if not _TRAINING_DETAIL_RE.search(str(item))]


def _rag_chunk_count() -> int:
    chunks_path = ROOT / "rag" / "index" / "chunks.jsonl"
    if not chunks_path.exists():
        return 0
    return sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _models_meta() -> dict:
    manifest_path = ROOT / "models" / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _agi_proof_meta() -> dict:
    manifest_path = ROOT / "agi-proof" / "evidence-manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def _pct(x: float) -> float:
    """Round a 0..1 fraction to a 0..100 percentage with one decimal."""
    return round(x * 100, 1)


def _comparisons() -> dict:
    """Build the public benchmark-comparison charts from the canonical
    published-results.json.

    HONESTY CONTRACT: this surfaces only figures already curated into the
    published results (each cleared its own gate or is labelled as a tradeoff /
    null result), and it deliberately includes Sophia's LOSSES, not just its
    wins. Subject labels are sanitized to satisfy the web privacy guard
    (tools/lint_web_privacy.py) — no base-model identity or training-method
    leaks. The chart values are derived, never hand-typed, so the page cannot
    drift from the source of record.
    """
    src = ROOT / "agi-proof" / "benchmark-results" / "published-results.json"
    if not src.exists():
        return {}
    data = json.loads(src.read_text(encoding="utf-8"))
    charts: list[dict] = []

    # 1. Fabrication on genuine "I don't know" attribution traps (Sophia wins).
    calib = (data.get("calibrationEvals") or [None])[0]
    if calib:
        scorer = calib["judgeCorroboration"]["fabricationRate"]["scorer"]
        charts.append({
            "id": "fabrication-traps",
            "title": "Fabrication on genuine “I don’t know” traps",
            "subtitle": "Unknown-author / unknown-quote questions. DeepSeek subject, 3 runs, "
                        "deterministic scorer corroborated by two independent judge families.",
            "metric": "Fabrication rate",
            "unit": "%",
            "lowerIsBetter": True,
            "max": 30,
            "verdict": "win",
            "verdictLabel": "Sophia wins",
            "bars": [
                {"label": "Sophia (provenance gate)", "value": _pct(scorer["sophia-full"]), "highlight": True},
                {"label": "Raw model", "value": _pct(scorer["raw-model"])},
                {"label": "Raw model + tools", "value": _pct(scorer["raw-model-plus-tools"])},
            ],
            "note": "Sophia abstains rather than invent an attribution: 0% fabrication in all 3 runs "
                    "vs 19.5–25% for the raw model. Two independent judge families (GPT-4o + Claude) "
                    "rank Sophia lowest (inter-judge κ 0.74). Caveat: the trap pack is self-authored.",
        })

    # 2. Hallucinated attributions on a weak local model (Sophia wins; validated headline).
    val = (data.get("validated") or [None])[0]
    if val:
        ci = val.get("deltaCI", [None, None])
        charts.append({
            "id": "hallucinated-attributions",
            "title": "Hallucinated attributions on a weak local model",
            "subtitle": "Headline validated result — 2 independent judge families, 3 runs, "
                        "95% bootstrap CI excludes zero.",
            "metric": "Hallucinated-attribution rate",
            "unit": "%",
            "lowerIsBetter": True,
            "max": 45,
            "verdict": "win",
            "verdictLabel": "Sophia wins",
            "bars": [
                {"label": "Model alone (no gate)", "value": _pct(val["hallucinationAlone"])},
                {"label": "With Sophia gate", "value": _pct(val["hallucinationGated"]), "highlight": True},
            ],
            "note": f"Δ {_pct(val['delta'])} points "
                    f"(95% CI [{_pct(ci[0])}, {_pct(ci[1])}]) at "
                    f"{_pct(val['falsePositiveCost'])}% false-positive cost — the gate never broke a "
                    "correct answer. Honest scope: this is a weak-model effect that decays toward zero "
                    "on a strong, well-aligned model; the pack is self-authored.",
        })

    # 3. External public benchmark: selective-accuracy lift (Sophia wins; validated, non-self-authored).
    ebc = data.get("externalBenchmarkCalibration") or {}
    rows = ebc.get("rows") or []
    if rows:
        bars = []
        for r in rows:
            lift = r.get("liftAt20Coverage", {})
            bars.append({
                "label": f"{r['subject']} · {r['dataset']}",
                "value": _pct(lift.get("mean", 0)),
                "ci": [_pct(lift.get("ciLow", 0)), _pct(lift.get("ciHigh", 0))],
                "highlight": True,
            })
        charts.append({
            "id": "selective-prediction",
            "title": "External public benchmark: selective-accuracy lift",
            "subtitle": "SimpleQA / SimpleQA Verified (OpenAI + Google DeepMind) — public, "
                        "human-authored, external. Graded by 2 independent families.",
            "metric": "Selective-accuracy lift @20% coverage",
            "unit": "pts",
            "lowerIsBetter": False,
            "max": 25,
            "verdict": "win",
            "verdictLabel": "Sophia wins (external)",
            "bars": bars,
            "note": "The first Sophia calibration result on non-self-authored data: knowing when to "
                    "abstain lifts selective accuracy on both subject models, each lift's 95% CI excludes "
                    "zero. The effect depends on the underlying model — larger for the overconfident "
                    "model than the cautious one.",
        })

    # 4. The honest tradeoff (CPQA): grounding buys trap-safety at a recall cost (Sophia LOSES overall).
    cpqa = data.get("continualGroundedEvals") or {}
    g, raw = cpqa.get("grounded"), cpqa.get("raw")
    if g and raw:
        charts.append({
            "id": "grounding-tradeoff",
            "title": "The honest tradeoff: grounded answering vs the raw model",
            "subtitle": "Continual Provenance QA over a 92-page corpus, 3 runs (CANDIDATE — "
                        "self-authored benchmark). Higher is better.",
            "metric": "Pass rate",
            "unit": "%",
            "lowerIsBetter": False,
            "max": 100,
            "verdict": "tradeoff",
            "verdictLabel": "Mixed — raw wins overall",
            "groups": [
                {
                    "label": "Overall accuracy",
                    "bars": [
                        {"label": "Sophia (grounded)", "value": _pct(g["consensus"]), "highlight": True},
                        {"label": "Raw model", "value": _pct(raw["consensus"])},
                    ],
                },
                {
                    "label": "Attribution / abstention traps",
                    "bars": [
                        {"label": "Sophia (grounded)", "value": _pct(g["abstain"]), "highlight": True},
                        {"label": "Raw model", "value": _pct(raw["abstain"])},
                    ],
                },
            ],
            "note": "Published honestly: the raw model wins OVERALL (88.4% vs 52.9%) because answers are "
                    "constrained to a thin, stubby corpus — grounding costs recall. But on attribution "
                    "traps and retractions Sophia is 100% vs 0%: it fails closed where the raw model "
                    "confidently fabricates. The win is trap-safety, not a blanket accuracy lead.",
        })

    return {
        "updated": data.get("lastUpdated"),
        "source": "agi-proof/benchmark-results/published-results.json",
        "charts": charts,
        "honesty": [
            "Every figure here is from the curated published-results set; each cleared its own "
            + "no-overclaim gate or is explicitly labelled a tradeoff / candidate / null result.",
            "Sophia’s anti-fabrication edge is largest on weak or overconfident models and shrinks "
            + "toward zero on strong, well-aligned models — it is a guardrail, not a capability multiplier.",
            "A live test of the multi-agent swarm structure found NO measured benefit over a single "
            + "well-prompted pass, and sometimes degraded it — published as a null result.",
            "Most packs are self-authored and keys are held by one operator; the SimpleQA selective-"
            + "prediction result is the first on fully external, human-authored data.",
        ],
    }


def _patreon_supporters_summary() -> dict:
    """Lightweight public summary for the web manifest.
    Only count + tier names (no individual names) to keep it tasteful.
    """
    supporters_path = ROOT / "data" / "patreon" / "supporters.json"
    tiers_config_path = ROOT / "data" / "patreon" / "tiers.json"

    if not supporters_path.exists():
        return {}

    try:
        data = json.loads(supporters_path.read_text(encoding="utf-8"))
        tiers = data.get("tiers", {}) or {}
        total = data.get("count", 0) or sum(len(v) for v in tiers.values())

        # Prefer canonical order + nice names from config
        tier_order = data.get("tier_order") or []
        if not tier_order and tiers_config_path.exists():
            cfg = json.loads(tiers_config_path.read_text(encoding="utf-8"))
            tier_order = [t["patreon_title"] for t in cfg.get("tiers", [])]

        # Only include tiers that actually have members, in preferred order
        active_tiers = [t for t in tier_order if t in tiers and tiers[t]]
        if not active_tiers:
            active_tiers = [t for t, names in tiers.items() if names]

        return {
            "count": int(total),
            "tiers": active_tiers,
            "lastSync": data.get("synced_at"),
            "patreonUrl": "https://www.patreon.com/c/aideveloper_tomyim",
        }
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        # Malformed or missing supporters data should not crash the web build, but
        # a bare pass would hide a real bug — log to stderr so it is debuggable.
        import sys
        print(f"[build_web_data] warning: patreon supporters summary skipped: {exc}", file=sys.stderr)
        return {}


def main() -> int:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    examples = len(list((ROOT / "training" / "examples").glob("*.json")))
    leaderboards = {}
    for domain in DOMAINS:
        path = ROOT / "benchmark" / "results" / f"leaderboard-{domain}.json"
        if path.exists():
            leaderboards[domain] = json.loads(path.read_text(encoding="utf-8"))

    models = _models_meta()
    agi_proof = _agi_proof_meta()
    # Public-safe manifest only. Intentionally EXCLUDES training/architecture
    # details (base model, adapter/LoRA path, internal module map / artifactIndex,
    # training-pipeline runners). The site renders results + the claim boundary,
    # not how the system is built. The web privacy guard enforces this.
    score_pct = models.get("benchmarkScore", {}).get("scorePct")
    payload = {
        "version": version,
        "trainingExamples": examples,
        "domains": list(DOMAINS),
        "leaderboards": leaderboards,
        "comparisons": _comparisons(),
        "rag": {
            "indexChunks": _rag_chunk_count(),
        },
        "localModel": {
            # score only — no base model, adapter path, or model repo
            "benchmark": {"scorePct": score_pct} if score_pct is not None else {},
        },
        "agiProof": {
            "claimBoundary": agi_proof.get(
                "claimBoundary",
                "Sophia is an AGI-candidate proof package; true AGI is not claimed.",
            ),
            "proofLadder": agi_proof.get("proofLadder", []),
            "externalBenchmarks": agi_proof.get("externalBenchmarks", []),
            "requiredProofData": _sanitize_required(agi_proof.get("requiredProofData", [])),
            "docs": "agi-proof/README.md",
        },
        "links": {
            "github": "https://github.com/tomyimkc/sophia-agi",
            "huggingface": "https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus",
            "patreon": "https://www.patreon.com/c/aideveloper_tomyim",
        },
        "supporters": _patreon_supporters_summary(),
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {WEB_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
