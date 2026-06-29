# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Keyless live/fixture source adapters for the out-of-wiki fact-check gate.

The fact-check gate is deliberately backend-injected: CI must stay deterministic
and offline, while production can choose live Wikidata/Crossref/URL resolvers.
This module supplies both sides of that contract:

- fixture adapters for committed/offline held-out evaluation;
- keyless Crossref DOI resolution;
- keyless Wikidata authorship retrieval for simple ``author wrote work`` claims;
- keyless macro/economics retrieval from World Bank, FRED CSV, and BLS;
- keyless scholarly search from Crossref/OpenAlex for evidence discovery;
- source ranking + structured fixture/Wikidata entailment helpers.

No source here is treated as Sophia-internal wiki evidence. Live and fixture
records are external-source observations that must still pass the gate's normal
independence, entailment, confidence-floor, and learning-candidate rules.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
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
DEFAULT_WIKIDATA_CACHE_DIR = Path(__file__).resolve().parents[1] / "eval" / "external" / ".cache" / "wikidata"
_RETRYABLE_HTTP = {429, 503}

_AUTHOR_WROTE_RE = re.compile(
    r"^(?P<author>[A-Z][A-Za-z0-9 .,'’\-]+?)\s+(?:wrote|authored|penned|composed)\s+(?P<work>.+)$",
    re.I,
)
_WORK_BY_AUTHOR_RE = re.compile(
    r"^(?P<work>.+?)\s+(?:was|is)\s+(?:written|authored|penned|composed)\s+by\s+(?P<author>.+)$",
    re.I,
)

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_DIRECTION_RE = re.compile(r"\b(?P<dir>increased|rose|rising|grew|decreased|declined|fell|falling|dropped)\b", re.I)
_COUNTRIES = {
    "united states": {"aliases": {"us", "u.s", "u.s.", "usa", "united states", "america"}, "wb": "USA", "name": "United States"},
    "china": {"aliases": {"china", "prc", "people s republic of china"}, "wb": "CHN", "name": "China"},
    "hong kong": {"aliases": {"hong kong", "hong kong sar"}, "wb": "HKG", "name": "Hong Kong SAR, China"},
    "united kingdom": {"aliases": {"uk", "u.k", "u.k.", "britain", "united kingdom"}, "wb": "GBR", "name": "United Kingdom"},
}
_INDICATORS = {
    "inflation": {"aliases": {"inflation", "cpi", "consumer price"}, "wb": "FP.CPI.TOTL.ZG", "fred": "CPIAUCSL", "bls": "CUUR0000SA0", "name": "inflation"},
    "unemployment": {"aliases": {"unemployment", "jobless"}, "wb": "SL.UEM.TOTL.ZS", "fred": "UNRATE", "bls": "LNS14000000", "name": "unemployment"},
    "gdp": {"aliases": {"gdp", "gross domestic product"}, "wb": "NY.GDP.MKTP.CD", "fred": "GDP", "bls": None, "name": "gdp"},
}
_POSITIVE_DIRS = {"increased", "rose", "rising", "grew"}
_SCHOLARLY_RE = re.compile(r"\b(?:paper|study|doi|journal|research|scholarly|arxiv|economics|political economy|agi|deployment incentives?|safety evidence|regulatory capture|rent seeking)\b", re.I)


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
        "world_bank": 91,
        "fred": 90,
        "bls": 89,
        "scholarly": 88,
        "official_data": 86,
        "openalex": 84,
        "google_factcheck": 83,
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

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        sleep_s: float = 0.05,
        cache_dir: str | Path | None = DEFAULT_WIKIDATA_CACHE_DIR,
        opener: Any | None = None,
        sleep_fn: Any | None = None,
    ):
        self.timeout = timeout
        self.sleep_s = sleep_s
        self._label_cache: dict[str, str] = {}
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._opener = opener
        self._sleep_fn = sleep_fn or time.sleep

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
        out.extend(self.macro_economics(claim))
        out.extend(self.crossref_scholarly(claim))
        out.extend(self.openalex_scholarly(claim))
        return ranked_sources(out)

    def entailment(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        # Structured source records can decide entail/contradict deterministically.
        # Generic scholarly snippets remain evidence for later NLI/judge layers,
        # but are not accepted from keyword overlap alone.
        label = structured_entailment(claim, source)
        if label != "irrelevant":
            return label
        return macro_structured_entailment(claim, source)

    def wikidata_authorship(self, claim: AtomicClaim) -> list[EvidenceSource]:
        parsed = extract_authorship_claim(claim.text)
        if not parsed:
            return []
        work = parsed["work"]
        try:
            qids = self._wikidata_search(work, limit=3)
        except (RateLimited, URLError, TimeoutError):
            raise
        except Exception:
            return []
        sources: list[EvidenceSource] = []
        for qid in qids:
            try:
                entity = self._wikidata_entity(qid)
                authors = self._wikidata_authors(entity)
            except (RateLimited, URLError, TimeoutError):
                raise
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
            self._sleep_fn(self.sleep_s)
        return sources

    def macro_economics(self, claim: AtomicClaim) -> list[EvidenceSource]:
        parsed = extract_macro_claim(claim.text)
        if not parsed:
            return []
        sources: list[EvidenceSource] = []
        for fn in (self.world_bank_macro, self.fred_macro, self.bls_macro):
            try:
                src = fn(parsed)
            except Exception:
                src = None
            if src is not None:
                sources.append(src)
                time.sleep(self.sleep_s)
        return sources

    def world_bank_macro(self, parsed: dict[str, Any]) -> EvidenceSource | None:
        ind = _INDICATORS[parsed["indicator"]]["wb"]
        code = parsed["country"]["wb"]
        year = int(parsed["year"])
        url = f"https://api.worldbank.org/v2/country/{code}/indicator/{ind}?{urlencode({'format': 'json', 'date': f'{year-1}:{year}', 'per_page': '1000'})}"
        data = _get_json(url, timeout=self.timeout)
        rows = data[1] if isinstance(data, list) and len(data) > 1 else []
        values = {int(r["date"]): float(r["value"]) for r in rows if r.get("value") is not None and str(r.get("date", "")).isdigit()}
        if year not in values or (year - 1) not in values:
            return None
        return _macro_source("World Bank", "world_bank", url, parsed, values[year - 1], values[year])

    def fred_macro(self, parsed: dict[str, Any]) -> EvidenceSource | None:
        # FRED graph CSV is keyless and stable for public series downloads.
        series = _INDICATORS[parsed["indicator"]].get("fred")
        if not series or parsed["country_key"] != "united states":
            return None
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - fixed public FRED CSV endpoint
            raw = resp.read().decode("utf-8")
        year = int(parsed["year"])
        values = _annual_averages_from_fred_csv(raw, series)
        if parsed["indicator"] == "inflation":
            rates = _annual_percent_change(values)
            if year not in rates or (year - 1) not in rates:
                return None
            return _macro_source("FRED", "fred", f"https://fred.stlouisfed.org/series/{series}", parsed, rates[year - 1], rates[year])
        if year not in values or (year - 1) not in values:
            return None
        return _macro_source("FRED", "fred", f"https://fred.stlouisfed.org/series/{series}", parsed, values[year - 1], values[year])

    def bls_macro(self, parsed: dict[str, Any]) -> EvidenceSource | None:
        series = _INDICATORS[parsed["indicator"]].get("bls")
        if not series or parsed["country_key"] != "united states":
            return None
        year = int(parsed["year"])
        start_year = year - 2 if parsed["indicator"] == "inflation" else year - 1
        params = urlencode({"startyear": str(start_year), "endyear": str(year)})
        url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series}?{params}"
        data = _get_json(url, timeout=self.timeout)
        series_rows = (((data.get("Results") or {}).get("series") or [{}])[0].get("data") or [])
        values_by_year: dict[int, list[float]] = {}
        for row in series_rows:
            period = str(row.get("period", ""))
            if period == "M13":
                continue
            try:
                y = int(row.get("year"))
                v = float(row.get("value"))
            except (TypeError, ValueError):
                continue
            values_by_year.setdefault(y, []).append(v)
        values = {y: sum(vs) / len(vs) for y, vs in values_by_year.items() if vs}
        if parsed["indicator"] == "inflation":
            rates = _annual_percent_change(values)
            if year not in rates or (year - 1) not in rates:
                return None
            return _macro_source("BLS", "bls", url, parsed, rates[year - 1], rates[year])
        if year not in values or (year - 1) not in values:
            return None
        return _macro_source("BLS", "bls", url, parsed, values[year - 1], values[year])

    def crossref_scholarly(self, claim: AtomicClaim) -> list[EvidenceSource]:
        if not _SCHOLARLY_RE.search(claim.text):
            return []
        query = claim.text[:220]
        params = urlencode({"query.bibliographic": query, "rows": "3"})
        try:
            data = _get_json(f"https://api.crossref.org/works?{params}", timeout=self.timeout)
        except Exception:
            return []
        out: list[EvidenceSource] = []
        for item in (data.get("message", {}).get("items") or [])[:3]:
            title = " ".join(item.get("title") or [])
            doi = item.get("DOI", "")
            if not title:
                continue
            out.append(EvidenceSource(
                id=f"crossref:{doi or normalize_text(title)[:32]}",
                url=f"https://doi.org/{doi}" if doi else str(item.get("URL", "")),
                title=f"Crossref work: {title}",
                snippet=f"Crossref bibliographic record title={title}; DOI={doi}; published={item.get('published-print') or item.get('published-online') or item.get('published') or {}}.",
                publisher="Crossref",
                retrieved_at=_utc_now(),
                source_type="crossref",
            ))
        return out

    def openalex_scholarly(self, claim: AtomicClaim) -> list[EvidenceSource]:
        if not _SCHOLARLY_RE.search(claim.text):
            return []
        params = urlencode({"search": claim.text[:220], "per-page": "3"})
        try:
            data = _get_json(f"https://api.openalex.org/works?{params}", timeout=self.timeout)
        except Exception:
            return []
        out: list[EvidenceSource] = []
        for item in (data.get("results") or [])[:3]:
            title = item.get("display_name") or ""
            if not title:
                continue
            abstract = _openalex_abstract(item.get("abstract_inverted_index") or {})
            out.append(EvidenceSource(
                id=f"openalex:{item.get('id', normalize_text(title)[:32])}",
                url=str(item.get("doi") or item.get("id") or ""),
                title=f"OpenAlex work: {title}",
                snippet=f"OpenAlex scholarly record title={title}. {abstract[:600]}",
                publisher="OpenAlex",
                retrieved_at=_utc_now(),
                source_type="openalex",
            ))
        return out

    def _wikidata_search(self, query: str, *, limit: int = 3) -> list[str]:
        def fetch() -> Any:
            params = urlencode({
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": str(limit),
                "type": "item",
            })
            return _get_json(
                f"https://www.wikidata.org/w/api.php?{params}",
                timeout=self.timeout,
                opener=self._opener,
                sleep_fn=self._sleep_fn,
            )

        data = self._cached_json("search", {"query": query, "limit": limit}, fetch)
        return [row["id"] for row in data.get("search", []) if row.get("id")]

    def _wikidata_entity(self, qid: str) -> dict[str, Any]:
        def fetch() -> Any:
            return _get_json(
                f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
                timeout=self.timeout,
                opener=self._opener,
                sleep_fn=self._sleep_fn,
            )

        data = self._cached_json("entity", {"qid": qid}, fetch)
        return data.get("entities", {}).get(qid, {})

    def _wikidata_authors(self, entity: dict[str, Any]) -> list[str]:
        cache_key = {"entity": str(entity.get("id") or _cache_digest(entity))}
        cached = self._cache_read("authors", cache_key)
        if isinstance(cached, list):
            return [str(x) for x in cached if x]
        claims = entity.get("claims", {}).get("P50", [])  # P50 = author
        labels: list[str] = []
        for claim in claims:
            value = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
            qid = value.get("id")
            if qid:
                labels.append(self._label(qid))
        authors = [x for x in labels if x]
        self._cache_write("authors", cache_key, authors)
        return authors

    def _cached_json(self, prefix: str, key: Any, fetch: Any) -> Any:
        cached = self._cache_read(prefix, key)
        if cached is not None:
            return cached
        data = fetch()
        self._cache_write(prefix, key, data)
        return data

    def _cache_read(self, prefix: str, key: Any) -> Any | None:
        path = self._cache_path(prefix, key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, prefix: str, key: Any, data: Any) -> None:
        path = self._cache_path(prefix, key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return

    def _cache_path(self, prefix: str, key: Any) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{prefix}-{_cache_digest(key)}.json"

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


# --------------------------------------------------------------------------- #
# Google Fact Check Tools backend (ClaimReview) — the first backend that reads
# GOOGLE_FACTCHECK_API_KEY. Keyed because Google's Fact Check Tools API requires
# an enabled API key (see .env.example). Offline-deterministic via an injectable
# fetcher so CI never needs the key or the network.
# --------------------------------------------------------------------------- #

# Clean-negative ClaimReview ratings. We normalize ONLY unambiguous false-style
# verdicts (which CONTRADICT a claim asserting the rated proposition) and a small
# set of true-style verdicts (which ENTAIL it). Everything else — "Misleading",
# "Not the Whole Story", prose like "We have abundant evidence..." — is dropped
# (treated as "irrelevant") rather than guessed, because fact-checker rating
# vocabularies are heterogeneous prose and a wrong normalization would silently
# flip the gate. This is the fail-closed rule the README's plan specified.
_NEGATIVE_RATINGS = {
    # universal clear-false across publishers
    "false", "incorrect", "wrong", "untrue", "fabrication", "fake",
    # AFP / PolitiFact / Snopes
    "pants on fire", "pants fire", "mostly false", "false!", "scam", "hoax",
    # Washington Post (publisher-scoped below — "Pinocchios" is WaPo-only)
}
_POSITIVE_RATINGS = {"true", "correct", "accurate", "mostly true", "half true"}
# Publisher-scoped ratings: a string that is a clear verdict from ONE publisher but
# ambiguous from another. Keyed by lowercased publisher name substring.
_PUBLISHER_RATINGS = {
    "washington post": {"four pinocchios": "false", "three pinocchios": "false", "two pinocchios": "false"},
    "snopes": {"false": "false", "mixture": "irrelevant", "mostly false": "false", "true": "true"},
    "politifact": {"pants on fire!": "false", "pants on fire": "false", "mostly false": "false"},
    "fullfact": {"incorrect": "false", "false": "false", "true": "true"},
}


def normalize_claimreview_rating(rating: str, publisher: str) -> str:
    """Map a ClaimReview ``textualRating`` to ``true`` | ``false`` | ``irrelevant``.

    Conservative by design: returns ``irrelevant`` (drop) for anything not a clean
    binary verdict. A wrong normalization would silently flip the gate; an unknown
    rating leaves the gate to hold on its other evidence, which is fail-closed."""
    r = (rating or "").strip().lower()
    p = (publisher or "").lower()
    if not r:
        return "irrelevant"
    # 1) Publisher-scoped vocabulary first (most specific).
    for pub_fragment, mapping in _PUBLISHER_RATINGS.items():
        if pub_fragment in p:
            mapped = mapping.get(r)
            if mapped:
                return mapped
    # 2) Universal clean verdicts.
    if r in _POSITIVE_RATINGS:
        return "true"
    if r in _NEGATIVE_RATINGS:
        return "false"
    return "irrelevant"


class GoogleFactCheckBackend:
    """Google Fact Check Tools (ClaimReview) backend for the out-of-wiki gate.

    Queries ``factchecktools.googleapis.com`` for professional fact-checker
    verdicts on a claim, maps each ClaimReview to an :class:`EvidenceSource`, and
    derives entailment from the normalized rating. Fail-closed: a missing key, a
    network error, or an unnormalizable rating yields ``[]`` / ``irrelevant`` so
    the gate holds rather than accepting from a broken or ambiguous backend.

    The API key is read from ``GOOGLE_FACTCHECK_API_KEY`` (or passed explicitly).
    It must NEVER be committed; this class only holds it in memory. Offline-
    testable via the injected ``fetcher`` (a callable taking the full URL and
    returning a parsed dict) so CI runs without a key or network.
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_pages: int = 2,
        fetcher: Any | None = None,
        page_size: int = 20,
        language_code: str = "en",
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
        self.timeout = timeout
        self.max_pages = max(1, max_pages)
        self.page_size = page_size
        self.language_code = language_code
        # fetcher(url) -> parsed-JSON dict. Default hits the network via _get_json.
        self._fetcher = fetcher

    def _fetch(self, url: str) -> dict[str, Any]:
        if self._fetcher is not None:
            return self._fetcher(url) or {}
        return _get_json(url, timeout=self.timeout)

    def _search(self, query: str) -> list[dict[str, Any]]:
        """Page through ClaimReview claims for ``query`` up to ``max_pages``."""
        claims: list[dict[str, Any]] = []
        page_token: str | None = None
        for _ in range(self.max_pages):
            params = {
                "key": self.api_key,
                "query": query,
                "languageCode": self.language_code,
                "pageSize": str(self.page_size),
            }
            if page_token:
                params["pageToken"] = page_token
            url = f"{self.BASE_URL}?{urlencode(params)}"
            try:
                data = self._fetch(url)
            except Exception:
                break
            claims.extend(data.get("claims") or [])
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return claims

    def retriever(self, claim: AtomicClaim) -> list[EvidenceSource]:
        if not self.api_key:
            return []  # fail-closed: no key => no Google evidence (not an error)
        out: list[EvidenceSource] = []
        for entry in self._search(claim.text):
            claim_text = entry.get("text") or claim.text
            for cr in entry.get("claimReview") or []:
                publisher = (cr.get("publisher") or {}).get("name", "")
                rating = cr.get("textualRating", "")
                normalized = normalize_claimreview_rating(rating, publisher)
                # Fail-closed: DROP ClaimReviews whose rating we cannot map to a clean
                # true/false verdict (per the README spec). An unmappable rating is NOT
                # weak evidence — it is no signal, so the gate holds on its other
                # evidence rather than acting on a guessed entailment.
                if normalized == "irrelevant":
                    continue
                relation = "contradicts" if normalized == "false" else "entails"
                # Encode the normalized relation in the id so entailment() recovers
                # it deterministically (same convention as FixtureFactBackend).
                sid = f"google_factcheck:{normalize_text(publisher)}:{_cache_digest({'url': cr.get('url', ''), 'rating': rating})}"
                out.append(EvidenceSource(
                    id=f"{sid}#rel={relation}",
                    url=str(cr.get("url", "")),
                    title=str(cr.get("title", "") or f"{publisher} fact check"),
                    snippet=f"ClaimReview by {publisher}: rating='{rating}' (normalized: {normalized}). Reviewed claim: \"{(claim_text or '')[:200]}\".",
                    publisher=publisher,
                    retrieved_at=_utc_now(),
                    source_type="google_factcheck",
                ))
        return ranked_sources(out)

    def entailment(self, claim: AtomicClaim, source: EvidenceSource) -> str:
        # The retriever encoded the normalized rating as #rel=... in the id.
        marker = (source.id or "").split("#rel=", 1)
        if len(marker) == 2 and marker[1] in {"entails", "contradicts", "irrelevant"}:
            return marker[1]
        return "irrelevant"

    # A ClaimReview backend resolves neither DOIs nor arbitrary URLs; delegate to
    # the keyless resolvers so this backend can stand alone in the CLI wiring.
    def doi_resolver(self, doi: str) -> bool:
        return LiveFactBackend().doi_resolver(doi)

    def url_resolver(self, url: str) -> bool:
        return LiveFactBackend().url_resolver(url)


def extract_macro_claim(text: str) -> dict[str, Any] | None:
    """Parse simple macro-direction claims, e.g. ``US inflation increased in 2021``."""
    cleaned = normalize_text(text)
    year_match = _YEAR_RE.search(text or "")
    dir_match = _DIRECTION_RE.search(text or "")
    if not year_match or not dir_match:
        return None
    direction_word = dir_match.group("dir").lower()
    direction = "increased" if direction_word in _POSITIVE_DIRS else "decreased"
    country_key = None
    for key, meta in _COUNTRIES.items():
        if any(alias in cleaned for alias in meta["aliases"]):
            country_key = key
            break
    if country_key is None:
        # Default to US only when the claim explicitly starts with US/U.S./USA.
        return None
    indicator_key = None
    for key, meta in _INDICATORS.items():
        if any(alias in cleaned for alias in meta["aliases"]):
            indicator_key = key
            break
    if indicator_key is None:
        return None
    return {
        "country_key": country_key,
        "country": _COUNTRIES[country_key],
        "indicator": indicator_key,
        "indicator_name": _INDICATORS[indicator_key]["name"],
        "year": int(year_match.group(1)),
        "direction": direction,
    }


def macro_structured_entailment(claim: AtomicClaim, source: EvidenceSource) -> str:
    parsed = extract_macro_claim(claim.text)
    if not parsed:
        return "irrelevant"
    text = f"{source.title} {source.snippet}"
    m = re.search(
        r"macro record:\s*country=(?P<country>[^;]+);\s*indicator=(?P<indicator>[^;]+);\s*year=(?P<year>\d{4});.*?direction=(?P<direction>increased|decreased)",
        text, re.I,
    )
    if not m:
        return "irrelevant"
    country_ok = normalize_text(parsed["country"]["name"]) == normalize_text(m.group("country"))
    indicator_ok = parsed["indicator"] == normalize_text(m.group("indicator"))
    year_ok = int(parsed["year"]) == int(m.group("year"))
    if not (country_ok and indicator_ok and year_ok):
        return "irrelevant"
    observed = m.group("direction").lower()
    return "entails" if observed == parsed["direction"] else "contradicts"


def _macro_source(publisher: str, source_type: str, url: str, parsed: dict[str, Any], prev: float, cur: float) -> EvidenceSource:
    observed = "increased" if cur > prev else "decreased" if cur < prev else "unchanged"
    year = int(parsed["year"])
    country = parsed["country"]["name"]
    indicator = parsed["indicator"]
    return EvidenceSource(
        id=f"{source_type}:{parsed['country']['wb']}:{indicator}:{year}",
        url=url,
        title=f"{publisher} macro record for {country} {indicator} in {year}",
        snippet=(
            f"{publisher} macro record: country={country}; indicator={indicator}; year={year}; "
            f"previousYear={year - 1}; previousValue={prev:.6g}; currentValue={cur:.6g}; "
            f"direction={observed}."
        ),
        publisher=publisher,
        retrieved_at=_utc_now(),
        source_type=source_type,
    )


def _annual_averages_from_fred_csv(raw: str, series: str) -> dict[int, float]:
    reader = csv.DictReader(io.StringIO(raw))
    buckets: dict[int, list[float]] = {}
    for row in reader:
        date = row.get("observation_date") or row.get("DATE") or ""
        try:
            year = int(date[:4])
            value = float(row.get(series, ""))
        except (TypeError, ValueError):
            continue
        buckets.setdefault(year, []).append(value)
    return {year: sum(values) / len(values) for year, values in buckets.items() if values}


def _annual_percent_change(values: dict[int, float]) -> dict[int, float]:
    rates: dict[int, float] = {}
    for year, value in values.items():
        prev = values.get(year - 1)
        if prev not in (None, 0):
            rates[year] = ((value - prev) / prev) * 100.0
    return rates


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    if not isinstance(index, dict) or not index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in index.items():
        for idx in idxs:
            positions[int(idx)] = word
    return " ".join(positions[i] for i in sorted(positions))


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


class RateLimited(RuntimeError):
    """Raised when a live source keeps returning retryable throttling errors."""


def _cache_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _retry_after_seconds(exc: HTTPError, fallback: float) -> float:
    value = ""
    try:
        value = str(exc.headers.get("Retry-After", "") or "")
    except AttributeError:
        value = ""
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            return fallback
    return fallback


def _get_json(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    opener: Any | None = None,
    sleep_fn: Any | None = None,
    max_attempts: int = 5,
) -> dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    open_fn = opener or urlopen
    sleep = sleep_fn or time.sleep
    last: HTTPError | None = None
    for attempt in range(max_attempts):
        try:
            with open_fn(req, timeout=timeout) as resp:  # noqa: S310 - fixed keyless public endpoints
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except HTTPError as exc:
            if exc.code not in _RETRYABLE_HTTP:
                raise
            last = exc
            if attempt == max_attempts - 1:
                break
            delay = _retry_after_seconds(exc, float(2 ** attempt))
            sleep(delay)
    code = getattr(last, "code", "unknown")
    raise RateLimited(f"HTTP {code} persisted after {max_attempts} attempts for {url[:120]}")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "FixtureFactBackend", "LiveFactBackend", "GoogleFactCheckBackend", "RateLimited",
    "extract_authorship_claim", "extract_macro_claim", "macro_structured_entailment",
    "normalize_claimreview_rating", "normalize_text", "ranked_sources", "structured_entailment",
]
