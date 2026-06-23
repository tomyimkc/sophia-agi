"""Keyless live/fixture source adapters for the out-of-wiki fact-check gate.

The fact-check gate is deliberately backend-injected: CI must stay deterministic
and offline, while production can choose live Wikidata/Crossref/URL resolvers.
This module supplies both sides of that contract:

- fixture adapters for committed/offline held-out evaluation;
- keyless Crossref DOI resolution;
- keyless Wikidata authorship retrieval for simple ``author wrote work`` claims;
- source ranking + structured fixture/Wikidata entailment helpers.

No source here is treated as Sophia-internal wiki evidence. Live and fixture
records are external-source observations that must still pass the gate's normal
independence, entailment, confidence-floor, and learning-candidate rules.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from agent.fact_check_gate import AtomicClaim, EvidenceSource

DEFAULT_TIMEOUT = 8.0
USER_AGENT = "sophia-agi fact-check gate (keyless; contact: github.com/tomyimkc/sophia-agi)"

_AUTHOR_WROTE_RE = re.compile(
    r"^(?P<author>[A-Z][A-Za-z0-9 .,'’\-]+?)\s+(?:wrote|authored|penned|composed)\s+(?P<work>.+)$",
    re.I,
)
_WORK_BY_AUTHOR_RE = re.compile(
    r"^(?P<work>.+?)\s+(?:was|is)\s+(?:written|authored|penned|composed)\s+by\s+(?P<author>.+)$",
    re.I,
)


def normalize_text(value: str) -> str:
    """Lowercase alnum normalization for evidence/entity comparisons."""
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def extract_authorship_claim(text: str) -> dict[str, str] | None:
    """Return ``{"author": ..., "work": ...}`` for simple authorship claims."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip(" .!?。！？")
    for rx in (_AUTHOR_WROTE_RE, _WORK_BY_AUTHOR_RE):
        m = rx.match(cleaned)
        if m:
            return {
                "author": m.group("author").strip(" .,'’\""),
                "work": m.group("work").strip(" .,'’\""),
            }
    return None


def ranked_sources(sources: list[EvidenceSource]) -> list[EvidenceSource]:
    """Deterministically rank source observations by authority and specificity.

    Ranking is not a verdict. It only orders the evidence supplied to the gate so
    authoritative structured sources are considered before generic snippets.
    """
    priority = {
        "wikidata": 100,
        "crossref": 95,
        "doi": 92,
        "scholarly": 88,
        "official_data": 86,
        "wikipedia": 75,
        "web": 40,
        "fixture": 30,
    }

    def score(src: EvidenceSource) -> tuple[int, int, str]:
        st = (src.source_type or "web").lower()
        text_len = len((src.title or "") + " " + (src.snippet or ""))
        return (-priority.get(st, 40), -text_len, src.id)

    return sorted(sources, key=score)


@dataclass
class FixtureFactBackend:
    """Offline deterministic source backend loaded from a JSON fixture."""

    doi_exists: dict[str, bool]
    url_exists: dict[str, bool]
    claims: dict[str, list[dict[str, Any]]]

    @classmethod
    def from_file(cls, path: str | Path) -> "FixtureFactBackend":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            doi_exists={k.lower(): bool(v) for k, v in data.get("doiExists", {}).items()},
            url_exists={k: bool(v) for k, v in data.get("urlExists", {}).items()},
            claims={str(k): list(v) for k, v in data.get("claims", {}).items()},
        )

    def doi_resolver(self, doi: str) -> bool:
        return bool(self.doi_exists.get((doi or "").lower(), False))

    def url_resolver(self, url: str) -> bool:
        return bool(self.url_exists.get(url or "", False))

    def retriever(self, claim: AtomicClaim) -> list[EvidenceSource]:
        rows = self.claims.get(claim.text, [])
        return ranked_sources([_source_from_fixture(row) for row in rows])

    def entailment(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        # Fixture relation is explicit and deterministic; it represents a cached
        # held-out source/claim annotation, not a model vote.
        marker = (source.id or "").split("#rel=", 1)
        if len(marker) == 2 and marker[1] in {"entails", "contradicts", "irrelevant"}:
            return marker[1]
        text = f"{source.title} {source.snippet}".lower()
        if "[contradicts]" in text:
            return "contradicts"
        if "[entails]" in text or "[supports]" in text:
            return "entails"
        return structured_entailment(claim, source)


def _source_from_fixture(row: dict[str, Any]) -> EvidenceSource:
    relation = row.get("relation")
    sid = str(row.get("id", "fixture"))
    if relation and "#rel=" not in sid:
        sid = f"{sid}#rel={relation}"
    return EvidenceSource(
        id=sid,
        url=str(row.get("url", "")),
        title=str(row.get("title", "")),
        snippet=str(row.get("snippet", "")),
        publisher=str(row.get("publisher", "")),
        retrieved_at=str(row.get("retrieved_at", "fixture")),
        source_type=str(row.get("source_type", "fixture")),
    )


class LiveFactBackend:
    """Keyless live resolver/retriever backend.

    Network use is opt-in at the CLI. Failures return ``False``/``[]`` so the
    gate holds fail-closed rather than accepting from a broken backend.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT, sleep_s: float = 0.05):
        self.timeout = timeout
        self.sleep_s = sleep_s
        self._label_cache: dict[str, str] = {}

    def doi_resolver(self, doi: str) -> bool:
        doi = (doi or "").strip().lower()
        if not doi:
            return False
        url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        try:
            data = _get_json(url, timeout=self.timeout)
            return data.get("status") == "ok" and bool(data.get("message", {}).get("DOI"))
        except Exception:
            return False

    def url_resolver(self, url: str) -> bool:
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - operator-provided URL resolver
                return 200 <= int(resp.status) < 400
        except HTTPError as exc:
            # Some servers reject HEAD but the URL exists enough to redirect/deny.
            return 200 <= int(exc.code) < 500
        except Exception:
            return False

    def retriever(self, claim: AtomicClaim) -> list[EvidenceSource]:
        out: list[EvidenceSource] = []
        out.extend(self.wikidata_authorship(claim))
        return ranked_sources(out)

    def entailment(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        return structured_entailment(claim, source)

    def wikidata_authorship(self, claim: AtomicClaim) -> list[EvidenceSource]:
        parsed = extract_authorship_claim(claim.text)
        if not parsed:
            return []
        work = parsed["work"]
        try:
            qids = self._wikidata_search(work, limit=3)
        except Exception:
            return []
        sources: list[EvidenceSource] = []
        for qid in qids:
            try:
                entity = self._wikidata_entity(qid)
                authors = self._wikidata_authors(entity)
            except Exception:
                continue
            if not authors:
                continue
            label = self._entity_label(entity, qid)
            if not _title_match(work, label):
                continue
            author_text = ", ".join(authors)
            sources.append(EvidenceSource(
                id=f"wikidata:{qid}",
                url=f"https://www.wikidata.org/wiki/{qid}",
                title=f"Wikidata authorship record for {label}",
                snippet=f"Wikidata records {label} author(s): {author_text}.",
                publisher="Wikidata",
                retrieved_at=_utc_now(),
                source_type="wikidata",
            ))
            time.sleep(self.sleep_s)
        return sources

    def _wikidata_search(self, query: str, *, limit: int = 3) -> list[str]:
        params = urlencode({
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": str(limit),
            "type": "item",
        })
        data = _get_json(f"https://www.wikidata.org/w/api.php?{params}", timeout=self.timeout)
        return [row["id"] for row in data.get("search", []) if row.get("id")]

    def _wikidata_entity(self, qid: str) -> dict[str, Any]:
        data = _get_json(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json", timeout=self.timeout)
        return data.get("entities", {}).get(qid, {})

    def _wikidata_authors(self, entity: dict[str, Any]) -> list[str]:
        claims = entity.get("claims", {}).get("P50", [])  # P50 = author
        labels: list[str] = []
        for claim in claims:
            value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
            qid = value.get("id")
            if qid:
                labels.append(self._label(qid))
        return [x for x in labels if x]

    def _entity_label(self, entity: dict[str, Any], fallback_qid: str) -> str:
        labels = entity.get("labels", {}) or {}
        return ((labels.get("en") or labels.get("mul") or {}).get("value")) or fallback_qid

    def _label(self, qid: str) -> str:
        if qid in self._label_cache:
            return self._label_cache[qid]
        # Special:EntityData reliably includes labels even when wbgetentities is
        # served with sparse labels by some unauthenticated mirrors/caches.
        try:
            entity = self._wikidata_entity(qid)
            label = self._entity_label(entity, qid)
        except Exception:
            label = qid
        self._label_cache[qid] = label
        return label


def structured_entailment(claim: AtomicClaim, source: EvidenceSource) -> str:
    """Deterministic source-aware entailment for structured external records.

    This is intentionally narrow: it handles authorship records whose snippet has
    the canonical shape emitted by the Wikidata/fixture adapters. Everything else
    returns ``irrelevant`` and lets the gate hold or use another entailment layer.
    """
    parsed = extract_authorship_claim(claim.text)
    if not parsed:
        return "irrelevant"
    text = f"{source.title} {source.snippet}"
    m = re.search(r"author\(s\):\s*(?P<authors>.+?)(?:\.|$)", text, re.I)
    if not m:
        return "irrelevant"
    # Do not let a broad/wrong Wikidata search result contradict a claim about a
    # different work. Require substantial title overlap before support/contradict.
    if not _title_match(parsed["work"], text):
        return "irrelevant"
    wanted = normalize_text(parsed["author"])
    authors = [normalize_text(a) for a in re.split(r",|;|\band\b", m.group("authors"))]
    if wanted and any(wanted == a or wanted in a or a in wanted for a in authors if a):
        return "entails"
    if authors:
        return "contradicts"
    return "irrelevant"



def _title_match(query: str, candidate_text: str) -> bool:
    q = set(normalize_text(query).split()) - {"the", "a", "an"}
    c = set(normalize_text(candidate_text).split())
    if not q:
        return False
    return (len(q & c) / len(q)) >= 0.75


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed keyless public endpoints
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "FixtureFactBackend", "LiveFactBackend", "extract_authorship_claim", "normalize_text",
    "ranked_sources", "structured_entailment",
]
