# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Draw real auto-research experiments from the OKF wiki corpus.

Instead of synthetic signal-token domains, this builds one experiment per OKF
``domain`` (philosophy, history, science, ...): the hypothesis is *"within this
domain, whether a claim's attribution is grounded is predictable from the claim
text"*, the labeled examples are real page claims, and the knowledge committed on
success is the domain's real OKF pages (so the retention gate measures forgetting over
actual corpus knowledge).

The label is a verifiable provenance property already in the frontmatter:

  - GROUNDED  (positive): authorConfidence in {consensus, attributed, compiled, layered}
  - CONTESTED (negative): authorConfidence in {disputed, legendary, anachronism_risk,
              none_extant}

A domain whose verifier cannot be held-out-validated is *refuted* and logged -- not
every real domain is lexically separable, and the loop honestly says so.

    from agent.auto_research import AutoResearcher
    from agent.okf_research_source import okf_experiments
    report = AutoResearcher().run_experiments(okf_experiments("wiki"))
"""

from __future__ import annotations

from collections import defaultdict

from agent.auto_research import Hypothesis
from agent.self_evolving_agent import Experience
from okf import load_pages

GROUNDED = frozenset({"consensus", "attributed", "compiled", "layered"})
CONTESTED = frozenset({"disputed", "legendary", "anachronism_risk", "none_extant"})


def _claim_text(page) -> str:
    """A page's canonical title plus its lead claim, lowercased and clipped."""
    title = page.meta.get("canonicalTitleEn") or page.id
    body = " ".join(
        line.strip()
        for line in (page.body or "").splitlines()
        if line.strip() and not line.startswith(("#", "-", ">", "_"))
    )
    return f"{title} . {body}"[:300].lower()


def okf_experiments(wiki_root: str = "wiki", *, min_per_class: int = 2):
    """Build ``[(Hypothesis, Experience)]`` from the OKF corpus, one per domain.

    Only domains with at least ``min_per_class`` grounded AND contested examples are
    included (a held-out split needs both classes). Examples carry the claim text +
    label; the experiment's committed knowledge is the domain's real OKF pages.
    """
    pages = load_pages(wiki_root)
    by_domain: dict = defaultdict(list)
    for p in pages:
        dom = p.meta.get("domain")
        if dom:
            by_domain[dom].append(p)

    pairs: list = []
    for dom in sorted(by_domain):
        domain_pages = by_domain[dom]
        examples: list = []
        for p in domain_pages:
            ac = p.meta.get("authorConfidence")
            if ac in GROUNDED:
                examples.append((_claim_text(p), True))
            elif ac in CONTESTED:
                examples.append((_claim_text(p), False))
        pos = sum(1 for _, lab in examples if lab)
        neg = len(examples) - pos
        if pos < min_per_class or neg < min_per_class:
            continue  # not enough of both classes to validate a verifier
        # Commit the GROUNDED pages as the domain's knowledge (contested pages are not
        # assertable facts, so they would not count as retained anyway).
        grounded_pages = tuple(p for p in domain_pages if p.meta.get("authorConfidence") in GROUNDED)
        hyp = Hypothesis(
            id=f"OKF:{dom}",
            domain=f"okf_{dom}",
            signal=f"okf:{dom}",
            prereg={"minImprovement": 0.05, "requireCommit": True},
            rationale=(f"within OKF domain '{dom}', a claim's grounding status is "
                       f"predictable from claim text ({pos} grounded / {neg} contested)"),
        )
        exp = Experience(domain=f"okf_{dom}", examples=tuple(examples), pages=grounded_pages)
        pairs.append((hyp, exp))
    return pairs


__all__ = ["okf_experiments", "GROUNDED", "CONTESTED"]
