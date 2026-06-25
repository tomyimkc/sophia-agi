# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-confirmed failure memory — separate from OKF wiki belief.

Stores ONLY gate/verifier/eval-oracle confirmed errors with grounded corrections.
Append-only, versioned, decontaminated against held-out eval prompts. Past errors
are NEVER belief; they are auditable guard-rail evidence for inference-time RAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.config import ROOT
from provenance_bench.dataset_guard import eval_prompt_set, normalize

_LOG = logging.getLogger("sophia.failure_memory")

SCHEMA = "sophia.failure_memory.v1"
DEFAULT_STORE_PATH = ROOT / "training" / "feedback" / "failure_memory" / "nodes.jsonl"

# Verifier sources that may create a node (hard oracle only — never model self-judgment).
ALLOWED_VERIFIER_KINDS = frozenset({
    "provenance_faithful",
    "gate",
    "eval_label",
    "formal_verifier",
    "hard_oracle",
})

_TOKEN_RE = re.compile(r"[a-zA-Z\u4e00-\u9fff]{3,}")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(str(text or ""))}


def stable_key(question: str, wrong_claim: str) -> str:
    """Dedupe key: normalized question + wrong claim."""
    return f"{normalize(question)}||{normalize(wrong_claim)}"


def stable_id(stable_key_value: str) -> str:
    return hashlib.sha256(stable_key_value.encode("utf-8")).hexdigest()[:16]


def heldout_prompt_hash(*, root: Path = ROOT) -> str:
    """Stable hash of the sealed held-out eval prompt set (decontam guard)."""
    prompts = sorted(eval_prompt_set(root=root))
    payload = "|".join(prompts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def error_memory_heldout_hash(path: Path | None = None) -> str:
    """Stable hash of the error-memory held-out pack (stored outside eval/ globs)."""
    target = path or (ROOT / "data" / "error_memory_heldout_v1.jsonl")
    if not target.exists():
        return ""
    ids: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            ids.append(str(row.get("id", "")))
    return hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]


def deterministic_embed(text: str, *, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Hash-seeded bag-of-token embedding — offline, reproducible."""
    tokens = sorted(_tokenize(text))
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        digest = hashlib.sha256(f"{seed}:{tok}".encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) or 1.0
    return float(np.dot(a, b) / denom)


@dataclass
class IngestResult:
    ok: bool
    node_id: str | None = None
    version: str | None = None
    rejected: bool = False
    reasons: list[str] = field(default_factory=list)
    deduped: bool = False


def _required_str(obj: dict, key: str) -> str | None:
    val = obj.get(key)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def validate_node(node: dict) -> tuple[bool, list[str]]:
    """Enforce ACCURATE + TRACEABLE schema invariants."""
    reasons: list[str] = []

    if node.get("schema") != SCHEMA:
        reasons.append(f"schema must be {SCHEMA}")
    if node.get("candidateOnly") is not True:
        reasons.append("candidateOnly must be true")
    if node.get("level3Evidence") is not False:
        reasons.append("level3Evidence must be false")

    for key in ("id", "version", "createdAt", "stableKey"):
        if not _required_str(node, key):
            reasons.append(f"missing {key}")

    source = node.get("sourceEvent")
    if not isinstance(source, dict):
        reasons.append("sourceEvent must be an object")
    else:
        if not _required_str(source, "question"):
            reasons.append("sourceEvent.question required")
        if not _required_str(source, "runId"):
            reasons.append("sourceEvent.runId required")

    verifier = node.get("verifier")
    if not isinstance(verifier, dict):
        reasons.append("verifier must be an object")
    else:
        kind = verifier.get("name")
        if kind not in ALLOWED_VERIFIER_KINDS:
            reasons.append(f"verifier.name must be one of {sorted(ALLOWED_VERIFIER_KINDS)}")
        if not _required_str(verifier, "verdict"):
            reasons.append("verifier.verdict required")

    if not _required_str(node, "wrongClaim"):
        reasons.append("wrongClaim required")

    correction = node.get("correction")
    if not isinstance(correction, dict):
        reasons.append("correction required with grounded claim + citation")
    else:
        if not _required_str(correction, "claim"):
            reasons.append("correction.claim required (fail-closed without grounded correction)")
        if not _required_str(correction, "citation"):
            reasons.append("correction.citation required (provenance citation)")
        if not _required_str(correction, "source"):
            reasons.append("correction.source required")

    sk = node.get("stableKey")
    if isinstance(sk, str) and node.get("id"):
        expected = stable_id(sk)
        if node["id"] != expected:
            reasons.append("id must equal stable_id(stableKey)")

    return (not reasons, reasons)


def has_grounded_correction(node: dict) -> bool:
    ok, _ = validate_node(node)
    return ok


def overlaps_heldout(question: str, *, root: Path = ROOT) -> bool:
    return normalize(question) in eval_prompt_set(root=root)


def build_contradiction_edge(
    belief_id: str,
    *,
    wrong_claim: str,
    created_at: str,
    run_id: str,
) -> dict[str, Any]:
    """Audit record linking a failure node to the belief/claim it contradicts."""
    return {
        "beliefId": belief_id,
        "kind": "contradicts",
        "wrongClaim": wrong_claim,
        "recordedAt": created_at,
        "sourceRunId": run_id,
        "note": "failure-memory node; NOT a wiki belief — guard-rail evidence only",
    }


def enrich_contradiction_ledger(
    belief_pages: list,
    belief_id: str,
) -> dict[str, Any] | None:
    """Optional: attach live contradiction_ledger slice when wiki graph is available."""
    try:
        from okf import build_graph, contradiction_ledger
        from okf.linker import load_dnm_by_tradition

        graph = build_graph(belief_pages)
        ledger = contradiction_ledger(graph, dnm_by_tradition=load_dnm_by_tradition())
        pairs = ledger.get("pairs") or ledger.get("contradictions") or []
        related = [p for p in pairs if belief_id in str(p)]
        if related:
            return {"beliefId": belief_id, "ledgerExcerpt": related[:5]}
    except Exception as exc:  # noqa: BLE001 — optional enrichment only
        _LOG.debug("contradiction_ledger enrichment skipped: %s", exc)
    return None


@dataclass
class FailureMemoryStore:
    path: Path = field(default_factory=lambda: DEFAULT_STORE_PATH)
    embed_seed: int = 0
    embed_dim: int = 64

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _LOG.warning("skipping malformed failure-memory line")
        return rows

    def list_nodes(self, *, latest_only: bool = True) -> list[dict]:
        rows = self._read_all()
        if not latest_only:
            return rows
        by_id: dict[str, dict] = {}
        for row in rows:
            nid = row.get("id")
            if not nid:
                continue
            prev = by_id.get(nid)
            if prev is None or str(row.get("version", "")) > str(prev.get("version", "")):
                by_id[nid] = row
        return sorted(by_id.values(), key=lambda r: r.get("id", ""))

    def versions_of(self, node_id: str) -> list[dict]:
        return sorted(
            [r for r in self._read_all() if r.get("id") == node_id],
            key=lambda r: str(r.get("version", "")),
        )

    def ingest(
        self,
        *,
        question: str,
        wrong_claim: str,
        correction_claim: str,
        correction_citation: str,
        correction_source: str,
        verifier_name: str,
        verifier_verdict: str,
        run_id: str,
        created_at: str,
        eval_id: str | None = None,
        input_text: str | None = None,
        contradicts_belief_id: str | None = None,
        belief_pages: list | None = None,
        force_new_version: bool = False,
    ) -> IngestResult:
        """Ingest a verifier-confirmed error. Fail-closed on missing correction or held-out overlap."""
        reasons: list[str] = []

        if verifier_name not in ALLOWED_VERIFIER_KINDS:
            reasons.append(f"verifier {verifier_name!r} not in allowed hard-oracle set")
        if overlaps_heldout(question):
            reasons.append("question overlaps sealed held-out eval prompt set")
        if not correction_claim.strip():
            reasons.append("correction.claim required")
        if not correction_citation.strip():
            reasons.append("correction.citation required")

        sk = stable_key(question, wrong_claim)
        nid = stable_id(sk)

        existing = self.versions_of(nid)
        if existing and not force_new_version:
            latest = existing[-1]
            if (
                latest.get("correction", {}).get("claim") == correction_claim.strip()
                and latest.get("correction", {}).get("citation") == correction_citation.strip()
            ):
                return IngestResult(ok=True, node_id=nid, version=latest.get("version"), deduped=True)

        if reasons:
            return IngestResult(ok=False, rejected=True, reasons=reasons)

        version_num = len(existing) + 1
        version = f"v{version_num}"

        node: dict[str, Any] = {
            "schema": SCHEMA,
            "candidateOnly": True,
            "level3Evidence": False,
            "id": nid,
            "version": version,
            "createdAt": created_at,
            "stableKey": sk,
            "sourceEvent": {
                "runId": run_id,
                "evalId": eval_id,
                "question": question.strip(),
                "input": (input_text or question).strip(),
            },
            "verifier": {
                "name": verifier_name,
                "verdict": verifier_verdict,
            },
            "wrongClaim": wrong_claim.strip(),
            "correction": {
                "claim": correction_claim.strip(),
                "citation": correction_citation.strip(),
                "source": correction_source.strip(),
            },
        }

        if contradicts_belief_id:
            edge = build_contradiction_edge(
                contradicts_belief_id,
                wrong_claim=wrong_claim.strip(),
                created_at=created_at,
                run_id=run_id,
            )
            if belief_pages:
                excerpt = enrich_contradiction_ledger(belief_pages, contradicts_belief_id)
                if excerpt:
                    edge["ledgerExcerpt"] = excerpt.get("ledgerExcerpt")
            node["contradictionEdge"] = edge

        ok, val_reasons = validate_node(node)
        if not ok:
            return IngestResult(ok=False, rejected=True, reasons=val_reasons)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")

        return IngestResult(ok=True, node_id=nid, version=version)

    def ingest_from_eval_result(
        self,
        result: dict,
        *,
        created_at: str,
        run_id: str,
        correction_claim: str,
        correction_citation: str,
        correction_source: str,
        verifier_name: str = "eval_label",
    ) -> IngestResult:
        """Builder: ingest from a scored eval row with oracle label false + grounded correction."""
        label = str(result.get("label", "")).lower()
        if label not in ("false", "wrong", "incorrect"):
            return IngestResult(
                ok=False,
                rejected=True,
                reasons=["eval label must be false/wrong for verifier-confirmed error"],
            )

        question = (
            result.get("question")
            or result.get("prompt")
            or (result.get("work") and f"Who wrote {result.get('work')}?")
        )
        wrong_claim = (
            result.get("wrong_claim")
            or result.get("claimed_author")
            or result.get("answer")
            or ""
        )
        if not question or not wrong_claim:
            return IngestResult(ok=False, rejected=True, reasons=["missing question or wrong claim"])

        verdict = result.get("verifier_verdict") or "label:false"
        return self.ingest(
            question=str(question),
            wrong_claim=str(wrong_claim),
            correction_claim=correction_claim,
            correction_citation=correction_citation,
            correction_source=correction_source,
            verifier_name=verifier_name,
            verifier_verdict=str(verdict),
            run_id=run_id,
            created_at=created_at,
            eval_id=result.get("case_id") or result.get("id"),
            contradicts_belief_id=result.get("contradicts_belief_id"),
        )

    def embed_node(self, node: dict) -> np.ndarray | None:
        if not has_grounded_correction(node):
            return None
        question = (node.get("sourceEvent") or {}).get("question", "")
        wrong = node.get("wrongClaim", "")
        text = f"{question} {wrong}"
        return deterministic_embed(text, dim=self.embed_dim, seed=self.embed_seed)

    def retrieve_similar(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.05,
    ) -> list[tuple[float, dict]]:
        """Deterministic retrieval: cosine sim, stable tie-breaks on node id."""
        q_emb = deterministic_embed(query, dim=self.embed_dim, seed=self.embed_seed)
        scored: list[tuple[float, dict]] = []
        for node in self.list_nodes(latest_only=True):
            if not has_grounded_correction(node):
                continue
            n_emb = self.embed_node(node)
            if n_emb is None:
                continue
            score = _cosine(q_emb, n_emb)
            if score >= min_score:
                scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1].get("id", ""), item[1].get("version", "")))
        return scored[:top_k]


def check_store_decontamination(store: FailureMemoryStore, *, root: Path = ROOT) -> dict:
    """Assert every stored question is disjoint from sealed eval/holdout prompts."""
    forbidden = eval_prompt_set(root=root)
    overlaps: list[str] = []
    for node in store.list_nodes():
        q = normalize((node.get("sourceEvent") or {}).get("question", ""))
        if q and q in forbidden:
            overlaps.append(node.get("id", ""))
    return {
        "clean": not overlaps,
        "overlapCount": len(overlaps),
        "overlappingNodeIds": overlaps,
        "heldoutPromptHash": heldout_prompt_hash(root=root),
    }
