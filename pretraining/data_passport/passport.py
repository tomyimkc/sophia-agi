# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-row data passports for training packs — provenance for pretraining data.

DeepSeek's data direction wants "高质量、可追溯、可复现的数据资产" — high-quality, traceable,
reproducible data assets. Sophia already enforces provenance on *answers*; this extends the
same discipline to *training rows*. Every row gets a passport recording:

  * content_hash  — sha256 of the normalized text (exact-dup key, reproducibility anchor)
  * source        — where the row came from (declared or "unknown")
  * license       — license/usage tag (declared or "unknown" -> flagged)
  * quality_score — a 0..1 heuristic (length, non-degeneracy, declared verifier oracle)
  * minhash       — signature for near-duplicate detection (Jaccard over char shingles)
  * dedup_cluster — id grouping near-duplicate rows (so a recipe can keep one per cluster)

``stamp_pack`` returns the rows with ``_passport`` attached plus a ``datasheet`` summary
(counts by source/license, quality histogram, exact/near dup rates, unlicensed/low-quality
flags). Fail-closed posture: unknown license and below-threshold quality are flagged, not
silently accepted. Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# Stable, dependency-free MinHash: N hash "permutations" via salted sha1.
_NUM_HASHES = 16
_SHINGLE = 4
_MASK = (1 << 32) - 1


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def row_text(row: dict) -> str:
    """Concatenate the user/assistant text of an SFT (messages) or DPO/plain row."""
    if isinstance(row.get("messages"), list):
        return " ".join(str(m.get("content", "")) for m in row["messages"]
                        if isinstance(m, dict))
    parts = [str(row.get(k, "")) for k in ("prompt", "completion", "chosen", "text")]
    return " ".join(p for p in parts if p)


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def _shingles(text: str) -> "set[str]":
    t = normalize(text)
    if len(t) < _SHINGLE:
        return {t} if t else set()
    return {t[i:i + _SHINGLE] for i in range(len(t) - _SHINGLE + 1)}


def minhash(text: str) -> "list[int]":
    """Deterministic MinHash signature over char shingles."""
    sh = _shingles(text)
    if not sh:
        return [0] * _NUM_HASHES
    sig = []
    for k in range(_NUM_HASHES):
        best = _MASK
        salt = str(k).encode()
        for s in sh:
            hv = int.from_bytes(hashlib.sha1(salt + s.encode("utf-8")).digest()[:4], "big")
            if hv < best:
                best = hv
        sig.append(best)
    return sig


def estimate_jaccard(a: "list[int]", b: "list[int]") -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def quality_score(row: dict, text: str) -> float:
    """Heuristic 0..1 quality. Rewards adequate length and a declared verifier oracle;
    penalizes degeneracy (very short, or low character diversity / repetition)."""
    t = normalize(text)
    if not t:
        return 0.0
    n = len(t)
    length_score = min(1.0, n / 200.0)                 # saturates around 200 chars
    diversity = len(set(t)) / n if n else 0.0          # char-level non-degeneracy
    diversity_score = min(1.0, diversity * 6.0)
    # declared, machine-checkable provenance bumps quality (e.g. exec/sympy-verified rows)
    verified_bonus = 0.15 if (row.get("verifier") or row.get("oracle")
                              or row.get("verified")) else 0.0
    score = 0.5 * length_score + 0.4 * diversity_score + verified_bonus
    return round(min(1.0, score), 4)


def fair_assessment(stamped: "list[dict]") -> "dict[str, Any]":
    """A FAIR self-assessment over already-stamped rows (Findable, Accessible,
    Interoperable, Reusable). The Leiden Declaration asks researchers to adhere to FAIR
    and UNESCO open-science principles as work becomes data-dependent; this turns that ask
    into a per-pack, machine-checkable score rather than a claim.

    Heuristic, deterministic, fail-closed (a missing source/license lowers the score, it is
    not silently treated as present):
      * findable     — every row carries a stable content_hash id (reproducibility anchor)
      * accessible   — fraction of rows whose origin (source) is declared, not "unknown"
      * interoperable— fraction of rows carrying a structured passport (machine-readable)
      * reusable     — fraction of rows whose license is declared, not "unknown"
    """
    n = len(stamped)
    if not n:
        return {"findable": 0.0, "accessible": 0.0, "interoperable": 0.0, "reusable": 0.0,
                "rows": 0, "note": "empty pack"}
    has_id = sum(1 for r in stamped if r.get("_passport", {}).get("content_hash"))
    known_source = sum(1 for r in stamped
                       if r.get("_passport", {}).get("source", "unknown") != "unknown")
    structured = sum(1 for r in stamped if isinstance(r.get("_passport"), dict))
    known_license = sum(1 for r in stamped
                        if r.get("_passport", {}).get("license", "unknown") != "unknown")
    return {
        "findable": round(has_id / n, 4),
        "accessible": round(known_source / n, 4),
        "interoperable": round(structured / n, 4),
        "reusable": round(known_license / n, 4),
        "rows": n,
        "note": ("FAIR self-assessment (Leiden/UNESCO open-science ask); fail-closed — "
                 "unknown source/license lowers accessible/reusable, not silently passed."),
    }


def stamp_pack(rows: "list[dict]", *, near_dup_threshold: float = 0.8,
               quality_floor: float = 0.35) -> "dict[str, Any]":
    """Attach ``_passport`` to each row and return {rows, datasheet}."""
    stamped = []
    sigs: list[list[int]] = []
    seen_hash: dict[str, int] = {}
    cluster_of: list[int] = []

    for i, row in enumerate(rows):
        text = row_text(row)
        ch = content_hash(text)
        sig = minhash(text)
        # exact dup -> same cluster as the first occurrence
        if ch in seen_hash:
            cluster = cluster_of[seen_hash[ch]]
            exact_dup = True
        else:
            # near-dup against existing cluster representatives
            cluster = i
            exact_dup = False
            for j, prev_sig in enumerate(sigs):
                if estimate_jaccard(sig, prev_sig) >= near_dup_threshold:
                    cluster = cluster_of[j]
                    break
            seen_hash[ch] = i
        sigs.append(sig)
        cluster_of.append(cluster)

        q = quality_score(row, text)
        passport = {
            "content_hash": ch,
            "source": row.get("source", "unknown"),
            "license": row.get("license", "unknown"),
            "quality_score": q,
            "minhash": sig,
            "dedup_cluster": cluster,
            "exact_duplicate": exact_dup,
            "flags": [],
        }
        if passport["license"] == "unknown":
            passport["flags"].append("unlicensed")
        if q < quality_floor:
            passport["flags"].append("low_quality")
        new_row = dict(row)
        new_row["_passport"] = passport
        stamped.append(new_row)

    n = len(stamped)
    n_clusters = len(set(cluster_of))
    by_source: dict[str, int] = {}
    by_license: dict[str, int] = {}
    flagged = 0
    qsum = 0.0
    for r in stamped:
        p = r["_passport"]
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
        by_license[p["license"]] = by_license.get(p["license"], 0) + 1
        flagged += 1 if p["flags"] else 0
        qsum += p["quality_score"]

    datasheet = {
        "rows": n,
        "unique_clusters": n_clusters,
        "duplicate_rate": round(1 - n_clusters / n, 4) if n else 0.0,
        "by_source": by_source,
        "by_license": by_license,
        "mean_quality": round(qsum / n, 4) if n else 0.0,
        "flagged_rows": flagged,
        "near_dup_threshold": near_dup_threshold,
        "quality_floor": quality_floor,
        "fail_closed": ("rows flagged unlicensed/low_quality are surfaced for review; "
                        "a recipe should keep one row per dedup_cluster and drop flagged"),
        "fair": fair_assessment(stamped),
    }
    return {"rows": stamped, "datasheet": datasheet}


__all__ = [
    "normalize", "row_text", "content_hash", "minhash", "estimate_jaccard",
    "quality_score", "fair_assessment", "stamp_pack",
]
