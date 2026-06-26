# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Medical citation discipline — does a cited reference EXIST, and does it SUPPORT
the proposition it is cited for?

This is the medicine sibling of ``agent.legal_faithfulness``, built because the
market is explicitly hiring specialized-domain data discipline for *medicine and
law*. The two failure modes are the same shape as in law:

- **Existence (deterministic).** A model invents ``PMID 99999999`` or a plausible
  DOI for a paper that does not exist — the medical analogue of *Mata v. Avianca*.
  ``medical_citation_exists`` answers "is this reference real?" against a register
  (a bundled snapshot here; a live PubMed/Crossref/guideline resolver in
  production). This tier is deterministic and always runs.
- **Faithfulness (judged, measured).** A model cites a *real* trial for a claim its
  result does not establish — e.g. citing a secondary-prevention statin trial to
  recommend a statin for an asymptomatic low-risk adult. Deciding whether a
  recommendation supports a proposition needs a judge, which reintroduces
  hallucination risk, so this tier is **measured under the no-overclaim gate**
  (>=2 judge families + CIs), never asserted from a single judge.

Honest design, faithful to the repo's thesis and to the domain's stakes:

- **Fail-closed / abstaining.** No authoritative recommendation text, no judge, or
  a judge error -> the pair is **abstained** (reported unchecked), never silently
  passed. With ``require_support`` an abstention is a hard fail too.
- **Not clinical advice.** This verifies *citations*, not medicine. It never tells
  a user what to do; it flags when a cited reference is fake or misused so a human
  clinician is not handed a fabricated warrant.

The judge is pluggable: the default abstains (so wiring is provable offline);
inject ``make_llm_judge(spec)`` for a measured run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from agent.config import DATA_DIR

_BUNDLED_REGISTER = "medical_register.json"

# A judge maps (proposition, recommendation) -> Verdict. MUST NOT raise; on any
# failure return an abstaining verdict (the caller treats abstain as "unchecked").
Judge = Callable[[str, str], "Verdict"]


@dataclass
class Verdict:
    supports: bool = False
    abstained: bool = True
    reason: str = ""
    method: str = ""


_JUDGE_SYSTEM = (
    "You assess whether a medical reference's finding SUPPORTS the proposition it "
    "is cited for. Reply with ONLY a JSON object: "
    '{"supports": bool, "abstained": bool, "reason": string}. '
    "supports=true ONLY if the reference's finding actually establishes the "
    "proposition (same population, same intervention, same outcome direction). "
    "abstained=true if the finding text is insufficient to decide. Be strict: a "
    "real trial cited for a population or claim it did not study is supports=false "
    "(the misstated-evidence failure mode). This is citation review, not advice."
)

# PMID: 1-8 digits. DOI: 10.<registrant>/<suffix>. Guideline IDs: NICE NG/CG/QS<n>.
_PMID = re.compile(r"\bPMID[:\s]*?(\d{1,8})\b", re.IGNORECASE)
_DOI = re.compile(r"\b(?:doi[:\s]*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)
_NICE = re.compile(r"\bNICE\s+((?:NG|CG|QS|TA)\d{1,4})\b", re.IGNORECASE)


def normalize_citation(raw: str) -> str:
    """Canonical key for register lookup: ``PMID <n>`` / ``DOI <id>`` / ``NICE <id>``.
    DOIs are lowercased (case-insensitive by spec); PMIDs/guideline IDs upper-cased."""
    s = (raw or "").strip()
    m = _PMID.search(s)
    if m:
        return f"PMID {m.group(1)}"
    m = _DOI.search(s)
    if m:
        return f"DOI {m.group(1).lower()}"
    m = _NICE.search(s)
    if m:
        return f"NICE {m.group(1).upper()}"
    return s


def _citation_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for rx in (_PMID, _DOI, _NICE):
        for m in rx.finditer(text or ""):
            spans.append((m.start(), m.end(), normalize_citation(m.group(0))))
    return sorted(set(spans))


def extract_citations(text: str) -> list[str]:
    """Every distinct medical citation in ``text``, order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for _, _, cite in _citation_spans(text):
        if cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


def _load_register() -> dict:
    path = DATA_DIR / _BUNDLED_REGISTER
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def register_citations(resolver: "Callable[[str], bool] | None" = None) -> Callable[[str], bool]:
    """Return ``exists(citation) -> bool``. Default checks the bundled register; pass
    a ``resolver`` (e.g. a live PubMed/Crossref lookup) to override. Fail-closed:
    an unknown citation is reported as NOT existing."""
    if resolver is not None:
        def _exists(citation: str) -> bool:
            try:
                return bool(resolver(normalize_citation(citation)))
            except Exception:  # noqa: BLE001 - a broken resolver verifies nothing
                return False

        return _exists

    known = {
        normalize_citation(str(e.get("citation", "")))
        for e in _load_register().get("authorities", [])
        if e.get("citation")
    }

    def _exists(citation: str) -> bool:
        return normalize_citation(citation) in known

    return _exists


def register_holdings(holdings: "dict | None" = None) -> Callable[[str], "str | None"]:
    """Return ``recommendation_for(citation) -> str | None`` from register entries
    carrying a ``holding`` field (the authoritative finding to judge against)."""
    if holdings is None:
        holdings = {}
        for e in _load_register().get("authorities", []):
            cite, holding = e.get("citation"), e.get("holding")
            if cite and holding:
                holdings[normalize_citation(str(cite))] = str(holding)

    def _get(citation: str) -> "str | None":
        return holdings.get(normalize_citation(citation))

    return _get


def medical_citation_exists(*, resolver=None) -> Callable[..., dict]:
    """Return a deterministic existence-verifier closure in the repo's verifier
    shape: ``verify(text, _, _) -> {passed, reasons, detail}``. (This factory
    returns the *callable*, not a verdict; call the returned function on text.)
    Fail-closed: any unresolvable citation fails."""
    exists = register_citations(resolver)

    def verify(text: str, _records=None, _ctx=None) -> dict:
        cites = extract_citations(text or "")
        missing = [c for c in cites if not exists(c)]
        return {
            "passed": not missing,
            "reasons": [f"unverifiable medical citation: {c}" for c in missing],
            "detail": {"checked": cites, "missing": missing},
        }

    return verify


def make_llm_judge(spec: "str | None" = None) -> Judge:
    """An LLM-backed faithfulness judge. Fail-closed: abstains on any model failure."""
    try:
        from agent.model import default_client

        client = default_client(spec)
    except Exception:  # noqa: BLE001
        return abstain_judge(reason="no model client configured")

    def judge(proposition: str, recommendation: str) -> Verdict:
        user = (
            f"Proposition (as cited):\n'''{proposition}'''\n\n"
            f"Reference finding:\n'''{recommendation}'''"
        )
        try:
            res = client.generate(_JUDGE_SYSTEM, user)
        except Exception:  # noqa: BLE001
            return Verdict(abstained=True, reason="judge call failed", method=f"llm:{spec}")
        if not getattr(res, "ok", False):
            return Verdict(abstained=True, reason="judge unavailable", method=f"llm:{spec}")
        m = re.search(r"\{.*\}", getattr(res, "text", "") or "", re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except (ValueError, AttributeError):
            return Verdict(abstained=True, reason="unparseable judge output", method=f"llm:{spec}")
        return Verdict(
            supports=bool(data.get("supports")),
            abstained=bool(data.get("abstained")),
            reason=str(data.get("reason", ""))[:200],
            method=f"llm:{spec}",
        )

    return judge


def abstain_judge(reason: str = "no judge") -> Judge:
    def judge(proposition: str, recommendation: str) -> Verdict:
        return Verdict(abstained=True, reason=reason, method="abstain")

    return judge


def _safe_judge(judge: Judge, proposition: str, recommendation: str) -> Verdict:
    try:
        v = judge(proposition, recommendation)
    except Exception:  # noqa: BLE001 - a broken judge verifies nothing
        return Verdict(abstained=True, reason="judge raised")
    return v if isinstance(v, Verdict) else Verdict(abstained=True, reason="bad judge return")


def claim_citation_pairs(text: str) -> list[tuple[str, list[str]]]:
    """Split ``text`` into (proposition, citations) for each sentence carrying a
    medical citation. Citations are masked before sentence-splitting so periods in
    ``10.1056/...`` do not fracture a citation across sentences."""
    text = text or ""
    parts: list[str] = []
    mapping: dict[str, str] = {}
    last = 0
    for idx, (s, e, cite) in enumerate(_citation_spans(text)):
        if s < last:  # overlapping match -> skip
            continue
        token = f"{idx}"  # control sentinels: never in clinical text
        mapping[token] = cite
        parts.append(text[last:s])
        parts.append(token)
        last = e
    parts.append(text[last:])
    masked = "".join(parts)

    pairs: list[tuple[str, list[str]]] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", masked):
        toks = [t for t in mapping if t in sentence]
        if not toks:
            continue
        prop = sentence
        for t in toks:
            prop = prop.replace(t, mapping[t])
        seen: set[str] = set()
        cites = [mapping[t] for t in toks if not (mapping[t] in seen or seen.add(mapping[t]))]
        pairs.append((prop.strip(), cites))
    return pairs


def assess_text(text: str, *, holding_for=None, judge: "Judge | None" = None,
                resolver=None) -> dict:
    """Classify each (proposition, citation) pair: fabricated / supported /
    contradicted / abstained. Fail-closed — anything not affirmatively supported is
    not 'supported', and a non-existent citation is 'fabricated' (the worst case)."""
    holding_for = holding_for or register_holdings()
    judge = judge or abstain_judge()
    exists = register_citations(resolver)

    fabricated: list[str] = []
    supported: list[str] = []
    contradicted: list[dict] = []
    abstained: list[dict] = []
    for proposition, cites in claim_citation_pairs(text):
        for c in cites:
            if not exists(c):
                fabricated.append(c)
                continue
            holding = holding_for(c)
            if not holding:
                abstained.append({"citation": c, "why": "no authoritative finding text"})
                continue
            v = _safe_judge(judge, proposition, holding)
            if v.abstained:
                abstained.append({"citation": c, "why": v.reason or "judge abstained"})
            elif v.supports:
                supported.append(c)
            else:
                contradicted.append({"citation": c, "claim": proposition[:90], "reason": v.reason})
    return {
        "fabricated": fabricated,
        "supported": supported,
        "contradicted": contradicted,
        "abstained": abstained,
    }


__all__ = [
    "Verdict",
    "normalize_citation",
    "extract_citations",
    "register_citations",
    "register_holdings",
    "medical_citation_exists",
    "make_llm_judge",
    "abstain_judge",
    "claim_citation_pairs",
    "assess_text",
]
