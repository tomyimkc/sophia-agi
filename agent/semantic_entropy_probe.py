# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Single-pass semantic-entropy probe placeholder.

This deterministic linear probe operates over transparent text features so CI can
exercise the contract without torch/MLX. The same API can later wrap an MPS/MLX
hidden-state probe.
"""
from __future__ import annotations

import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EntropyProbe:
    weights: dict[str, float]
    bias: float = 0.0

    def features(self, text: str) -> dict[str, float]:
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {
            "length": min(1.0, len(words) / 80),
            "hedges": min(1.0, sum(w in {"maybe","perhaps","unclear","unknown","possibly"} for w in words) / 3),
            "specifics": min(1.0, len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or "")) / 5),
            "citations": 1.0 if re.search(r"https?://|doi:|10\.\d{4,9}/", text or "", re.I) else 0.0,
        }

    def predict(self, text: str) -> dict[str, Any]:
        feats = self.features(text)
        score = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in feats.items())
        score = max(0.0, min(1.0, score))
        return {"schema":"sophia.semantic_entropy_probe.v1", "predictedEntropy": round(score,4), "features": feats, "candidateOnly": True, "level3Evidence": False}

    def to_dict(self) -> dict[str, Any]:
        return {"schema":"sophia.semantic_entropy_probe_model.v1", "weights": self.weights, "bias": self.bias, "candidateOnly": True, "level3Evidence": False}


def default_probe() -> EntropyProbe:
    return EntropyProbe(weights={"length":0.15,"hedges":0.45,"specifics":0.25,"citations":-0.20}, bias=0.25)


def write_probe_report(texts: list[str], out: str | Path) -> dict[str, Any]:
    probe = default_probe()
    rows = [{"text": t, **probe.predict(t)} for t in texts]
    report = {"schema":"sophia.semantic_entropy_probe_report.v1", "model": probe.to_dict(), "rows": rows, "candidateOnly": True, "level3Evidence": False}
    p=Path(out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return report


__all__ = ["EntropyProbe", "default_probe", "write_probe_report"]
