# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Link / site prioritization from quality feedback (Phase 1).

Turns per-document quality scores (`pipeline.quality_score`) into per-site crawl priorities
and fetch quotas — the closed loop the JD describes: cleaning/scoring signals decide where to
spend the next crawl budget ("属性聚合、站点抓取配额、数据优先级"). A site that keeps
producing high-quality, well-sourced documents earns priority and quota; a low-quality or
spammy site is deprioritized.

Registered-domain extraction is a deterministic, dependency-free approximation (eTLD+1 over a
small known multi-label suffix set) — good enough for aggregation without pulling in the
public-suffix list. Phase 2's URL canonicalizer will supersede the host parsing here.
"""

from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlsplit

#: Multi-label public suffixes we special-case so "bbc.co.uk" aggregates as one site.
_MULTI_LABEL_SUFFIXES = frozenset(
    {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.cn", "org.cn", "net.cn", "com.au", "co.jp"}
)


def registered_domain(url: str) -> str:
    """Best-effort eTLD+1 for ``url`` (e.g. 'a.b.bbc.co.uk' -> 'bbc.co.uk'). Empty on failure."""
    host = (urlsplit(url).hostname or "").lower().strip(".")
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    last_two = ".".join(labels[-2:])
    if last_two in _MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def _quota_for(priority: float, volume: int, *, base_quota: int) -> int:
    """Suggested next-crawl quota: scales with priority, dampened sub-linearly by volume."""
    # High-priority sites get more budget; very large sites are mildly dampened to keep
    # crawl diversity (the JD's 保证数据多样性 / 最大化数据覆盖).
    diversity_damp = 1.0 / (1.0 + max(0, volume - 1) * 0.1)
    return max(1, round(base_quota * priority * (0.5 + 0.5 * diversity_damp)))


def prioritize(docs, *, base_quota: int = 100) -> list[dict]:
    """Aggregate scored documents by registered domain into a priority-ranked site list.

    Each ``doc`` must already carry a ``doc['quality']`` block (from ``quality_score``).
    Returns a list of site dicts sorted by descending priority, each with:
    ``domain``, ``docs``, ``kept``, ``meanQuality``, ``keepRate``, ``priority``, ``suggestedQuota``.
    """
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        dom = registered_domain(doc.get("url", ""))
        if dom:
            by_domain[dom].append(doc)

    sites: list[dict] = []
    for dom, group in by_domain.items():
        scores = [float((d.get("quality") or {}).get("score", 0.0)) for d in group]
        kept = sum(1 for d in group if (d.get("quality") or {}).get("keep"))
        volume = len(group)
        mean_quality = round(sum(scores) / volume, 6) if volume else 0.0
        keep_rate = round(kept / volume, 6) if volume else 0.0
        # Priority blends how good the content is with how much of it survives filtering.
        priority = round(0.7 * mean_quality + 0.3 * keep_rate, 6)
        sites.append(
            {
                "domain": dom,
                "docs": volume,
                "kept": kept,
                "meanQuality": mean_quality,
                "keepRate": keep_rate,
                "priority": priority,
                "suggestedQuota": _quota_for(priority, volume, base_quota=base_quota),
            }
        )

    sites.sort(key=lambda s: (-s["priority"], s["domain"]))
    return sites


__all__ = ["registered_domain", "prioritize"]
