# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Semantic-entropy utilities for hallucination/confabulation detection.

Modes:
- exact: normalized exact clusters (deterministic CI baseline)
- lexical: token-overlap clusters approximating semantic equivalence
- entailment: caller-injected equivalence function for NLI/backends
"""
from __future__ import annotations

import json, math, re
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def tokens(text: Any) -> set[str]:
    stop = {"the","a","an","is","are","was","were","of","to","in","and","or","for","with","that","this","it"}
    return {t for t in re.findall(r"[a-z0-9]+", normalize(text)) if len(t) > 2 and t not in stop}


def lexical_equivalent(a: Any, b: Any, *, threshold: float = 0.82) -> bool:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return True
    if not ta or not tb:
        return False
    j = len(ta & tb) / len(ta | tb)
    return j >= threshold


def cluster_meanings(samples: list[Any], *, mode: str = "lexical", equivalence_fn: Callable[[Any, Any], bool] | None = None) -> list[list[str]]:
    eq = equivalence_fn or ((lambda a, b: normalize(a) == normalize(b)) if mode == "exact" else lexical_equivalent)
    clusters: list[list[str]] = []
    for s in [str(x) for x in samples if normalize(x)]:
        for cluster in clusters:
            if eq(s, cluster[0]):
                cluster.append(s); break
        else:
            clusters.append([s])
    return clusters


def semantic_entropy(samples: list[Any], *, mode: str = "lexical", equivalence_fn: Callable[[Any, Any], bool] | None = None) -> dict[str, Any]:
    clusters = cluster_meanings(samples, mode=mode, equivalence_fn=equivalence_fn)
    n = sum(len(c) for c in clusters)
    if n == 0:
        h = 1.0
    else:
        probs = [len(c) / n for c in clusters]
        raw = -sum(p * math.log(p, 2) for p in probs)
        h = raw / max(1.0, math.log(max(2, len(clusters)), 2))
    return {"schema": "sophia.semantic_entropy.v1", "mode": mode, "n": n, "clusterCount": len(clusters), "entropy": round(h, 4), "clusters": clusters, "candidateOnly": True, "level3Evidence": False}


def write_semantic_entropy_report(samples_path: str | Path, out: str | Path, *, mode: str = "lexical") -> dict[str, Any]:
    data = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    report = semantic_entropy(data.get("samples", data if isinstance(data, list) else []), mode=mode)
    p=Path(out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


__all__ = ["normalize", "tokens", "lexical_equivalent", "cluster_meanings", "semantic_entropy", "write_semantic_entropy_report"]
