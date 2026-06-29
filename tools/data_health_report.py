#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data Health Index (DHI) — a deterministic scorecard for the data-management process.

This is the *instrument* called for in
``docs/11-Platform/Data-Analysis-Agent-Strategy.md`` (Phase 0): a single,
offline-computed, version-stamped scorecard over the COMMITTED data artifacts so
that "how good is our data process" becomes a measured number and any regression
fails CI the way ``pipeline/quality_regression.py`` does for a single shard.

Seven deterministic dimensions, each scored 0..1, combined by a transparent
weighted mean (weights are declared in the output — no hidden knobs):

  coverage · mixBalance · decontamStrength · dedupHealth ·
  provenanceCompleteness · lineage · reproducibility

Honest scope (load-bearing): the DHI is an **operational / illustrative** internal
metric. It is NOT a no-overclaim result, is never promoted to
``published-results.json``, and ``canClaimAGI`` stays false. It measures the *data
process*, not model capability.

    python tools/data_health_report.py            # write the report JSON
    python tools/data_health_report.py --print     # write + pretty-print to stdout
    python tools/data_health_report.py --check      # CI: exit 1 if the committed report is stale

Determinism: pure stdlib, sorted iteration, floats rounded to 4 dp, no timestamps —
so a re-run on the same committed tree reproduces a byte-identical report (the
contract the ``--check`` drift gate relies on).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agi-proof" / "data-health" / "report.json"
MANIFEST = ROOT / "training" / "local_sophia_v3" / "manifest.json"
CORPUS = ROOT / "training" / "corpus.jsonl"
REGISTRY = ROOT / "agi-proof" / "data-health" / "registry.json"
DATA_MANIFEST_GLOB = "data/*/manifest.json"

SCHEMA = "sophia.data_health.v1"

# --- targets / thresholds (auditable knobs; mirror the strategy doc §5) -------
RECORD_TARGET = 500        # structured ground-truth records to aim for
ROW_TARGET = 10_000        # M2 volume target (sophia-wisdom-4b-m2-volume-below-target)
ROWS_PER_RECORD_CEILING = 8.0   # above this, templating inflation (Goodhart)
HASH_FIELDS = ("contentHash", "embeddingsSha256", "sha256", "embeddingsSha")

# Transparent weights (sum to 1.0). Declared in the report, never hidden.
WEIGHTS = {
    "coverage": 0.20,
    "mixBalance": 0.15,
    "decontamStrength": 0.20,
    "dedupHealth": 0.10,
    "provenanceCompleteness": 0.15,
    "lineage": 0.10,
    "reproducibility": 0.10,
}


def _round(x: float) -> float:
    return round(float(x), 4)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# --- dimension scorers -------------------------------------------------------
def score_coverage(manifest: dict) -> dict:
    recs = (manifest.get("records") or {})
    n_records = int(recs.get("_total", 0))
    rows = int((manifest.get("totals") or {}).get("accepted", 0))
    rpr = float(recs.get("rowsPerRecord", 0.0))
    inflation = bool(recs.get("inflationFlag", rpr > ROWS_PER_RECORD_CEILING))
    record_term = min(n_records / RECORD_TARGET, 1.0) if RECORD_TARGET else 0.0
    row_term = min(rows / ROW_TARGET, 1.0) if ROW_TARGET else 0.0
    score = 0.6 * record_term + 0.4 * row_term
    if inflation:               # templating masquerading as coverage → cap
        score = min(score, 0.5)
    return {
        "score": _round(score),
        "records": n_records,
        "recordTarget": RECORD_TARGET,
        "rows": rows,
        "rowTarget": ROW_TARGET,
        "rowsPerRecord": _round(rpr),
        "inflationFlag": inflation,
    }


def score_mix_balance(manifest: dict) -> dict:
    by_family = manifest.get("byFamily") or {}
    total = sum(int(v.get("rows", 0)) for v in by_family.values()) or 1
    # L1 distance between actual family fraction and the midpoint of its target band.
    l1 = 0.0
    worst: list[tuple[str, float]] = []
    for fam in sorted(by_family):
        info = by_family[fam]
        actual = int(info.get("rows", 0)) / total
        band = info.get("targetPct") or [0.0, 0.0]
        target_mid = (float(band[0]) + float(band[1])) / 2.0 / 100.0
        diff = abs(actual - target_mid)
        l1 += diff
        worst.append((fam, _round(diff)))
    worst.sort(key=lambda kv: kv[1], reverse=True)
    score = max(0.0, 1.0 - l1)   # L1 over fractions: 0 = perfect, 1+ = badly skewed
    return {
        "score": _round(score),
        "l1Distance": _round(l1),
        "worstOffenders": worst[:3],
        "nFamilies": len(by_family),
    }


def score_decontam_strength(manifest: dict) -> dict:
    # Reward which decontamination LAYERS are actually wired (the dedicated assert_*
    # tools are the real gates; the DHI records coverage of layers, deterministically).
    exact_shingle = (ROOT / "tools" / "assert_decontam.py").exists()
    entity = (ROOT / "tools" / "assert_entity_decontam.py").exists()
    layers = {
        "exact": exact_shingle,
        "shingle": exact_shingle,
        "entity": entity,
    }
    present = sum(1 for v in layers.values() if v)
    score = present / len(layers)
    drops = int((manifest.get("totals") or {}).get("decontaminationDrops", 0))
    return {
        "score": _round(score),
        "layers": layers,
        "buildDecontaminationDrops": drops,
    }


def score_dedup_health(manifest: dict) -> dict:
    totals = manifest.get("totals") or {}
    candidates = int(totals.get("candidates", 0)) or 1
    drops = int(totals.get("decontaminationDrops", 0))
    dup_rate = drops / candidates
    # Thresholds pinned into the manifest? (dedup Jaccard / quality keep-threshold)
    thresholds_pinned = bool(manifest.get("dedupThresholds") or manifest.get("thresholds"))
    score = max(0.0, 1.0 - dup_rate)
    if not thresholds_pinned:    # implicit thresholds are a reproducibility risk
        score = min(score, 0.85)
    return {
        "score": _round(score),
        "duplicateRate": _round(dup_rate),
        "thresholdsPinned": thresholds_pinned,
    }


def score_provenance_completeness(corpus: list[dict]) -> dict:
    n = len(corpus) or 1
    with_source = 0
    with_license = 0
    for r in corpus:
        m = r.get("metadata") or {}
        if m.get("source"):
            with_source += 1
        if m.get("license"):
            with_license += 1
    src_frac = with_source / n
    lic_frac = with_license / n
    score = 0.5 * src_frac + 0.5 * lic_frac
    return {
        "score": _round(score),
        "rows": len(corpus),
        "withSource": _round(src_frac),
        "withLicense": _round(lic_frac),
    }


def _data_manifests() -> list[Path]:
    return sorted(ROOT.glob(DATA_MANIFEST_GLOB))


def score_lineage() -> dict:
    manifests = _data_manifests()
    total = len(manifests) or 1
    covered = 0
    registry_present = REGISTRY.exists()
    if registry_present:
        reg = _load_json(REGISTRY)
        covered_paths = {a.get("path") for a in (reg.get("assets") or [])}
        for mp in manifests:
            rel = mp.parent.relative_to(ROOT).as_posix()
            if rel in covered_paths or mp.relative_to(ROOT).as_posix() in covered_paths:
                covered += 1
    score = covered / total if registry_present else 0.0
    return {
        "score": _round(score),
        "registryPresent": registry_present,
        "manifestsCovered": covered,
        "manifestsTotal": len(manifests),
    }


def score_reproducibility() -> dict:
    manifests = _data_manifests()
    rag_meta = ROOT / "rag" / "index" / "embeddings.meta.json"
    if rag_meta.exists():
        manifests = manifests + [rag_meta]
    total = len(manifests) or 1
    hashed = 0
    for mp in manifests:
        try:
            doc = _load_json(mp)
        except (json.JSONDecodeError, OSError):
            continue
        if any(doc.get(f) for f in HASH_FIELDS):
            hashed += 1
    return {
        "score": _round(hashed / total),
        "artifactsWithHash": hashed,
        "artifactsTotal": len(manifests),
    }


# --- assembly ----------------------------------------------------------------
def compute_report() -> dict:
    """Compute the full DHI report from committed artifacts. Deterministic."""
    manifest = _load_json(MANIFEST) if MANIFEST.exists() else {}
    corpus = _load_jsonl(CORPUS)

    dims = {
        "coverage": score_coverage(manifest),
        "mixBalance": score_mix_balance(manifest),
        "decontamStrength": score_decontam_strength(manifest),
        "dedupHealth": score_dedup_health(manifest),
        "provenanceCompleteness": score_provenance_completeness(corpus),
        "lineage": score_lineage(),
        "reproducibility": score_reproducibility(),
    }
    dhi = sum(WEIGHTS[k] * dims[k]["score"] for k in WEIGHTS)
    return {
        "schema": SCHEMA,
        "label": "operational/illustrative — NOT a no-overclaim result; never promoted to published-results.json",
        "canClaimAGI": False,
        "weights": {k: _round(v) for k, v in sorted(WEIGHTS.items())},
        "dimensions": {k: dims[k] for k in sorted(dims)},
        "dhi": _round(dhi),
        "source": {
            "manifest": MANIFEST.relative_to(ROOT).as_posix(),
            "corpus": CORPUS.relative_to(ROOT).as_posix(),
            "registry": REGISTRY.relative_to(ROOT).as_posix(),
        },
    }


def serialize(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed report is stale")
    ap.add_argument("--print", dest="do_print", action="store_true", help="also pretty-print to stdout")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    report = compute_report()
    rendered = serialize(report)

    if args.check:
        if not args.out.exists():
            print(f"DATA HEALTH: FAIL — {args.out.relative_to(ROOT)} missing; run tools/data_health_report.py")
            return 1
        current = args.out.read_text(encoding="utf-8")
        if current != rendered:
            print("DATA HEALTH: FAIL — committed report is stale; re-run tools/data_health_report.py")
            return 1
        print(f"DATA HEALTH: OK — report current (DHI={report['dhi']})")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"DATA HEALTH: wrote {args.out.relative_to(ROOT)} — DHI={report['dhi']}")
    if args.do_print:
        for k in sorted(report["dimensions"]):
            print(f"  {k:24s} {report['dimensions'][k]['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
