#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data asset registry — the missing catalog over all committed data manifests.

Phase 1 of docs/11-Platform/Data-Analysis-Agent-Strategy.md. The repo already emits
per-shard / per-benchmark manifests, but there is no single queryable index of "what
corpora exist, their size, sealed/disjoint status, and a content anchor". This builds
that catalog deterministically from the committed manifests, so humans and the Data
Analysis Agent stop doing git archaeology, and so the DHI ``lineage`` dimension has a
real source.

Each asset carries a ``manifestSha256`` (sha256 of the manifest file bytes) — the
lineage/reproducibility anchor: if a manifest changes without the registry being
rebuilt, ``--check`` fails in CI.

A ``lineage`` block adds the first real edges: provenance edges derived ONLY from upstream
fields a manifest explicitly declares (``baseModel`` / ``teacher`` / ``distill_model`` /
``decontaminated_against`` / ``corpus``), plus a ``registryVersion`` content anchor over all
manifest hashes so an eval report can pin "ran against registry version X". Coverage is
reported honestly — most assets declare no upstream yet, so the full
source->shard->checkpoint->eval graph is still partial (failure-ledger
``data-lineage-graph-partial``).

    python tools/build_data_registry.py            # write the registry JSON
    python tools/build_data_registry.py --check     # CI: exit 1 if the registry is stale

Determinism: pure stdlib, sorted iteration, no timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agi-proof" / "data-health" / "registry.json"
MANIFEST_GLOBS = ("data/*/manifest.json", "training/*/manifest.json")
SCHEMA = "sophia.data_registry.v1"
HASH_FIELDS = ("contentHash", "embeddingsSha256", "sha256")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_of(doc: dict) -> int:
    """Best-effort row/case count from heterogeneous manifest shapes."""
    if "nCases" in doc:
        return int(doc.get("nCases", 0))
    totals = doc.get("totals") or {}
    if "accepted" in totals:
        return int(totals.get("accepted", 0))
    mlx = doc.get("mlx") or {}
    if "trainRows" in mlx:
        return int(mlx.get("trainRows", 0)) + int(mlx.get("validRows", 0))
    return 0


def _kind_of(rel_parent: str, doc: dict) -> str:
    if doc.get("sealed") or rel_parent.endswith("_benchmark"):
        return "benchmark"
    if rel_parent.startswith("training/"):
        return "training"
    return "dataset"


# Upstream-declaring manifest fields -> edge relation. Edges are derived ONLY from what a
# manifest explicitly declares; an absent edge means "no declared upstream", not "no upstream".
_LINEAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("derivedFrom", "derivedFrom"),
    ("baseModel", "fineTunedFrom"),
    ("teacher", "teacher"),
    ("teacherRun", "teacher"),
    ("distill_model", "distilledFrom"),
    ("author_model", "distilledFrom"),
    ("decontaminated_against", "decontaminatedAgainst"),
    ("corpus", "corpus"),
)


def _lineage_edges(path: str, doc: dict) -> list[dict]:
    """Deterministic provenance edges from explicitly-declared manifest fields."""
    edges: list[dict] = []
    for field, rel in _LINEAGE_FIELDS:
        val = doc.get(field)
        if not val:
            continue
        targets = val if isinstance(val, list) else [val]
        for t in targets:
            if isinstance(t, str) and t.strip():
                edges.append({"from": path, "rel": rel, "to": t.strip()})
    edges.sort(key=lambda e: (e["from"], e["rel"], e["to"]))
    return edges


def build_registry() -> dict:
    manifests: list[Path] = []
    for g in MANIFEST_GLOBS:
        manifests.extend(ROOT.glob(g))
    assets: list[dict] = []
    for mp in sorted(set(manifests)):
        try:
            doc = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rel_parent = mp.parent.relative_to(ROOT).as_posix()
        content_hash = next((doc.get(f) for f in HASH_FIELDS if doc.get(f)), None)
        assets.append({
            "path": rel_parent,
            "manifest": mp.relative_to(ROOT).as_posix(),
            "schema": doc.get("schema"),
            "id": doc.get("datasetId") or doc.get("benchmarkId") or rel_parent.split("/")[-1],
            "kind": _kind_of(rel_parent, doc),
            "rows": _rows_of(doc),
            "sealed": bool(doc.get("sealed", False)),
            "trainingDisjoint": bool(doc.get("trainingDisjoint", False)),
            "candidateOnly": bool(doc.get("candidateOnly", False)),
            "contentHash": content_hash,
            "manifestSha256": _sha256_file(mp),
        })
    assets.sort(key=lambda a: a["path"])

    # --- lineage: declared-edge graph + a version anchor for pinning eval runs ----------
    edges: list[dict] = []
    for a in assets:
        try:
            doc = json.loads((ROOT / a["manifest"]).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        edges.extend(_lineage_edges(a["path"], doc))
    edges.sort(key=lambda e: (e["from"], e["rel"], e["to"]))
    with_upstream = sorted({e["from"] for e in edges})
    # registryVersion: a single content anchor over every asset's manifest hash, so an eval
    # report can pin "ran against registry version X" and drift is detectable.
    anchor = "\n".join(f"{a['path']}={a['manifestSha256']}" for a in assets)
    registry_version = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
    lineage = {
        "registryVersion": registry_version,
        "edges": edges,
        "nEdges": len(edges),
        "assetsWithDeclaredUpstream": len(with_upstream),
        "assetsWithoutDeclaredUpstream": len(assets) - len(with_upstream),
        "upstreamCoverage": round(len(with_upstream) / len(assets), 4) if assets else 0.0,
        "note": ("Edges are derived ONLY from upstream fields DECLARED in manifests "
                 "(baseModel/teacher/distill/decontaminated_against/corpus). An absent edge "
                 "means the manifest declares no upstream, NOT that none exists — the full "
                 "source-document -> shard -> checkpoint -> eval-result graph is still partial."),
    }

    summary = {
        "nAssets": len(assets),
        "nSealed": sum(1 for a in assets if a["sealed"]),
        "nWithContentHash": sum(1 for a in assets if a["contentHash"]),
        "byKind": {
            k: sum(1 for a in assets if a["kind"] == k)
            for k in sorted({a["kind"] for a in assets})
        },
    }
    return {
        "schema": SCHEMA,
        "note": "Generated by tools/build_data_registry.py — do not edit by hand.",
        "canClaimAGI": False,
        "summary": summary,
        "lineage": lineage,
        "assets": assets,
    }


def serialize(reg: dict) -> str:
    return json.dumps(reg, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed registry is stale")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    reg = build_registry()
    rendered = serialize(reg)

    if args.check:
        if not args.out.exists():
            print(f"DATA REGISTRY: FAIL — {args.out.relative_to(ROOT)} missing; run tools/build_data_registry.py")
            return 1
        if args.out.read_text(encoding="utf-8") != rendered:
            print("DATA REGISTRY: FAIL — committed registry is stale; re-run tools/build_data_registry.py")
            return 1
        print(f"DATA REGISTRY: OK — {reg['summary']['nAssets']} assets current")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"DATA REGISTRY: wrote {args.out.relative_to(ROOT)} — {reg['summary']['nAssets']} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
