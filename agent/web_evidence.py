"""Web and local evidence search helpers for Sophia.

The online providers are opt-in because hidden benchmark prompts and private
review materials should not be sent to third-party search APIs by accident.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from agent.retrieval import retrieve

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
PROVIDERS = {"off", "auto", "brave", "tavily", "serpapi"}


@dataclass
class EvidenceSource:
    sourceType: str
    provider: str
    title: str
    url: str
    snippet: str
    score: float | None = None
    quality: str = "unknown"
    retrievedAt: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["score"] is None:
            data.pop("score")
        return data


def provider_from_env(provider: str | None = None) -> str:
    requested = (provider or os.environ.get("SOPHIA_WEB_SEARCH_PROVIDER") or "off").strip().lower()
    if requested == "auto":
        if _env("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
            return "brave"
        if _env("TAVILY_API_KEY"):
            return "tavily"
        if _env("SERPAPI_API_KEY"):
            return "serpapi"
        return "off"
    return requested if requested in PROVIDERS else "off"


def gather_evidence(
    query: str,
    *,
    local_top_k: int = 3,
    web_top_k: int = 5,
    online: bool = False,
    provider: str | None = None,
    source_profile: str = "academic",
    timeout_sec: int = 20,
) -> dict[str, Any]:
    """Collect local RAG evidence plus optional online search evidence."""
    local_sources = local_evidence(query, top_k=local_top_k) if local_top_k else []
    web_result = web_search(
        query,
        top_k=web_top_k,
        online=online,
        provider=provider,
        source_profile=source_profile,
        timeout_sec=timeout_sec,
    )
    return {
        "query": query,
        "generatedAt": _now(),
        "localSources": [source.to_dict() for source in local_sources],
        "web": web_result,
        "warnings": _evidence_warnings(online=online, web_result=web_result),
    }


def local_evidence(query: str, *, top_k: int = 3) -> list[EvidenceSource]:
    sources: list[EvidenceSource] = []
    for chunk in retrieve(query, top_k=top_k):
        sources.append(
            EvidenceSource(
                sourceType="local",
                provider="sophia-rag",
                title=chunk.title,
                url=chunk.path,
                snippet=chunk.excerpt,
                score=chunk.score,
                quality="curated-local",
                retrievedAt=_now(),
            )
        )
    return sources


def web_search(
    query: str,
    *,
    top_k: int = 5,
    online: bool = False,
    provider: str | None = None,
    source_profile: str = "academic",
    timeout_sec: int = 20,
) -> dict[str, Any]:
    selected = provider_from_env(provider)
    from agent.dataflow.firewall import egress_blocked

    if online and egress_blocked():
        return {
            "ok": False,
            "online": True,
            "provider": selected,
            "sources": [],
            "reason": "airgap profile blocks network egress (web_search)",
        }
    if not online:
        return {
            "ok": False,
            "online": False,
            "provider": selected,
            "sources": [],
            "reason": "online search disabled; pass --web-evidence or set online=True to query external APIs",
        }
    if selected == "off":
        return {
            "ok": False,
            "online": True,
            "provider": "off",
            "sources": [],
            "error": "no enabled provider; set SOPHIA_WEB_SEARCH_PROVIDER=auto|brave|tavily|serpapi and an API key",
        }
    submitted_query = evidence_query(query, source_profile=source_profile)
    try:
        if selected == "brave":
            sources = _search_brave(submitted_query, top_k=top_k, timeout_sec=timeout_sec)
        elif selected == "tavily":
            sources = _search_tavily(submitted_query, top_k=top_k, timeout_sec=timeout_sec)
        elif selected == "serpapi":
            sources = _search_serpapi(submitted_query, top_k=top_k, timeout_sec=timeout_sec)
        else:
            return {"ok": False, "online": True, "provider": selected, "sources": [], "error": "unsupported provider"}
    except Exception as exc:
        return {
            "ok": False,
            "online": True,
            "provider": selected,
            "submittedQuery": submitted_query,
            "sources": [],
            "error": _redact(repr(exc)),
        }
    return {
        "ok": True,
        "online": True,
        "provider": selected,
        "submittedQuery": submitted_query,
        "sources": [source.to_dict() for source in sources],
    }


def evidence_query(query: str, *, source_profile: str = "academic") -> str:
    profile = (source_profile or "general").strip().lower()
    if profile == "academic":
        return (
            f"{query} "
            "(site:arxiv.org OR site:plato.stanford.edu OR site:iep.utm.edu "
            "OR site:nih.gov OR site:edu OR filetype:pdf)"
        )
    if profile == "primary":
        return f"{query} (site:docs.* OR site:github.com OR site:gov OR site:edu)"
    return query


def format_evidence_context(evidence: dict[str, Any]) -> str:
    local_sources = evidence.get("localSources", [])
    web = evidence.get("web", {})
    web_sources = web.get("sources", [])
    warnings = evidence.get("warnings", [])
    if not local_sources and not web_sources and not warnings:
        return ""

    parts = ["## Evidence search context"]
    if warnings:
        parts.append("### Evidence warnings")
        parts.extend(f"- {warning}" for warning in warnings)

    if local_sources:
        parts.append("### Local Sophia RAG evidence")
        for index, source in enumerate(local_sources, 1):
            parts.append(
                "- "
                f"[local {index}] {source.get('url')} / {source.get('title')} "
                f"(relevance {float(source.get('score', 0)):.2f}; {source.get('quality')})\n"
                f"  {_clip(source.get('snippet', ''), 500)}"
            )

    if web_sources:
        parts.append(f"### Web evidence ({web.get('provider')})")
        for index, source in enumerate(web_sources, 1):
            parts.append(
                "- "
                f"[web {index}] {source.get('title')} - {source.get('url')} "
                f"({source.get('quality')})\n"
                f"  {_clip(source.get('snippet', ''), 500)}"
            )
    elif web.get("online"):
        parts.append(f"### Web evidence\n- Provider {web.get('provider')} returned no usable sources.")

    parts.append(
        "### Evidence use rules\n"
        "- Cite exact local/web labels when you rely on them.\n"
        "- Prefer primary, academic, official, or curated-local sources over general summaries.\n"
        "- If online evidence conflicts with the curated Sophia corpus, flag the conflict instead of overwriting memory."
    )
    return "\n".join(parts)


def classify_source(url: str) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if not host and url.startswith(("data/", "docs/", "training/", "benchmark/", "agi-proof/")):
        return "curated-local"
    academic_hosts = (
        "arxiv.org",
        "plato.stanford.edu",
        "iep.utm.edu",
        "philpapers.org",
        "jstor.org",
        "doi.org",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "nih.gov",
        "apa.org",
    )
    official_hosts = (
        "github.com",
        "docs.python.org",
        "developer.mozilla.org",
        "learn.microsoft.com",
        "developer.apple.com",
        "w3.org",
        "loc.gov",
        "archives.gov",
        "openai.com",
        "google.com",
    )
    if any(host.endswith(name) for name in academic_hosts) or host.endswith(".edu"):
        return "academic"
    if any(host.endswith(name) for name in official_hosts) or host.endswith(".gov"):
        return "official-primary"
    if any(host.endswith(name) for name in ("britannica.com", "stanford.edu", "cambridge.org", "oup.com")):
        return "reference"
    return "general-web"


def _search_brave(query: str, *, top_k: int, timeout_sec: int) -> list[EvidenceSource]:
    key = _env("BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY")
    if not key:
        raise RuntimeError("missing BRAVE_SEARCH_API_KEY")
    params = urllib.parse.urlencode({"q": query, "count": _count(top_k, 20), "extra_snippets": "true"})
    request = urllib.request.Request(
        f"{BRAVE_ENDPOINT}?{params}",
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    )
    data = _request_json(request, timeout_sec=timeout_sec)
    results = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
    sources: list[EvidenceSource] = []
    for item in results[:top_k]:
        url = str(item.get("url", ""))
        snippets = [str(item.get("description", ""))]
        snippets.extend(str(value) for value in item.get("extra_snippets", []) if value)
        sources.append(_source("web", "brave", item.get("title", ""), url, " ".join(snippets)))
    return sources


def _search_tavily(query: str, *, top_k: int, timeout_sec: int) -> list[EvidenceSource]:
    key = _env("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("missing TAVILY_API_KEY")
    payload = {
        "query": query,
        "max_results": _count(top_k, 20),
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_usage": True,
    }
    request = urllib.request.Request(
        TAVILY_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    data = _request_json(request, timeout_sec=timeout_sec)
    results = data.get("results", []) if isinstance(data, dict) else []
    sources: list[EvidenceSource] = []
    for item in results[:top_k]:
        url = str(item.get("url", ""))
        score = item.get("score")
        sources.append(_source("web", "tavily", item.get("title", ""), url, item.get("content", ""), score=score))
    return sources


def _search_serpapi(query: str, *, top_k: int, timeout_sec: int) -> list[EvidenceSource]:
    key = _env("SERPAPI_API_KEY")
    if not key:
        raise RuntimeError("missing SERPAPI_API_KEY")
    params = urllib.parse.urlencode(
        {
            "engine": "google",
            "q": query,
            "api_key": key,
            "num": _count(top_k, 10),
            "hl": "en",
            "safe": "active",
            "output": "json",
        }
    )
    request = urllib.request.Request(f"{SERPAPI_ENDPOINT}?{params}", headers={"Accept": "application/json"})
    data = _request_json(request, timeout_sec=timeout_sec)
    results = data.get("organic_results", []) if isinstance(data, dict) else []
    sources: list[EvidenceSource] = []
    for item in results[:top_k]:
        url = str(item.get("link", ""))
        sources.append(_source("web", "serpapi", item.get("title", ""), url, item.get("snippet", "")))
    return sources


def _source(
    source_type: str,
    provider: str,
    title: Any,
    url: str,
    snippet: Any,
    *,
    score: Any = None,
) -> EvidenceSource:
    parsed_score: float | None
    try:
        parsed_score = float(score) if score is not None else None
    except (TypeError, ValueError):
        parsed_score = None
    return EvidenceSource(
        sourceType=source_type,
        provider=provider,
        title=str(title or url or "untitled"),
        url=url,
        snippet=str(snippet or ""),
        score=parsed_score,
        quality=classify_source(url),
        retrievedAt=_now(),
    )


def _request_json(request: urllib.request.Request, *, timeout_sec: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"HTTP {exc.code}: {_redact(body)}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(_redact(repr(exc))) from exc
    return json.loads(raw)


def _evidence_warnings(*, online: bool, web_result: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not online:
        warnings.append("Online search is disabled by default to avoid leaking hidden/private prompts.")
    elif not web_result.get("ok"):
        warnings.append(f"Online evidence unavailable: {web_result.get('error') or web_result.get('reason')}")
    return warnings


def _env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _count(value: int, maximum: int) -> int:
    return max(1, min(int(value or 1), maximum))


def _clip(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(text: str) -> str:
    redacted = str(text)
    for name in (
        "BRAVE_SEARCH_API_KEY",
        "BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "SERPAPI_API_KEY",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted
