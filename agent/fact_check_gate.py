"""Layered out-of-wiki fact-check gate.

Problem addressed
-----------------
The internal wiki/OKF gate is intentionally fail-closed, but it can only verify
claims the internal corpus already covers. This module handles claims **not
covered by the internal wiki** by attempting active verification in layers before
abstaining:

  0. atomic claim decomposition + typing
  1. deterministic type-verifiers (math, dates, DOI/URL syntax/live resolvers,
     Python syntax, local code specs)
  2. live/pluggable external grounding (retrieval sources + entailment)
  3. consensus-by-verification (model judges may only pass with cited evidence;
     unsupported majority is never enough)
  4. calibrated abstention
  5. provenance-preserving learning candidate

All network/model backends are optional and injected. The default path is pure
stdlib and deterministic/offline so CI remains stable. Verdicts are fail-closed:
only ``accepted`` may surface a claim; ``held`` means abstain/defer/escalate;
``rejected`` means active contradiction or deterministic impossibility.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from agent.claim_router import split_claims

Verdict = str  # accepted | held | rejected

# Calibrated-abstention floors (documented and ENFORCED). An ``accepted`` whose
# confidence falls below the floor is demoted to ``held`` — this is the precise
# condition that prevents over-confidence (passing a plausible-but-weak specific)
# while deterministic certainties (math/code/DOI/URL, confidence ~1.0/0.85+) and
# subjective passes stay above the floor and surface normally.
CONF_FLOOR_NORMAL = 0.70
CONF_FLOOR_HIGH = 0.82


def _floor_for(risk: str) -> float:
    return CONF_FLOOR_HIGH if risk == "high" else CONF_FLOOR_NORMAL


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    type: str
    risk: str = "normal"


@dataclass(frozen=True)
class EvidenceSource:
    id: str
    url: str = ""
    title: str = ""
    snippet: str = ""
    publisher: str = ""
    retrieved_at: str = ""
    source_type: str = "web"

    @property
    def domain(self) -> str:
        host = urlparse(self.url or self.id).hostname or self.publisher or self.id
        return host.lower().removeprefix("www.")


@dataclass(frozen=True)
class LayerResult:
    layer: str
    verdict: Verdict
    reason: str
    evidence: tuple[dict, ...] = ()
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimDecision:
    claim: AtomicClaim
    verdict: Verdict
    reason: str
    confidence: float
    layers: tuple[LayerResult, ...]
    learning_candidate: dict[str, Any] | None = None


@dataclass(frozen=True)
class GateDecision:
    verdict: Verdict
    reason: str
    claims: tuple[ClaimDecision, ...]


# Optional backend protocols. Keep them as duck-typed callables/classes so tests
# can use tiny deterministic fakes with no network.
Retriever = Callable[[AtomicClaim], list[EvidenceSource]]
EntailmentFn = Callable[[AtomicClaim, EvidenceSource], str]  # entails|contradicts|irrelevant
JudgeFn = Callable[[AtomicClaim, list[EvidenceSource]], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Layer 0: claim typing
# --------------------------------------------------------------------------- #
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.I)
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")
_DATE_ORDER_RE = re.compile(r"\b(before|after|since|by|from|until)\b", re.I)
_ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")
_CODE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.I | re.S)
_ECON_RE = re.compile(
    r"\b(?:gdp|inflation|cpi|interest rate|central bank|unemployment|tariff|subsidy|"
    r"rent[- ]seeking|regulatory capture|monopoly|productivit\w*|wages?|median income|"
    r"political economy|incentives?|principal-agent|agi\s+labs?|agi\s+deployment|"
    r"deployment incentives?|narrative capture|monopoly rents?)\b",
    re.I,
)
_CAUSAL_RE = re.compile(r"\b(?:caused|causes|because|led to|increased|decreased|driven by|due to)\b", re.I)

# Non-checkable / subjective / meta text. These are NOT factual claims, so the
# fail-closed rule does not apply to them: passing an opinion is not the same as
# surfacing an unverified fact. Restoring the original claim_router principle
# ("a soundness gate, not a presence requirement") is what stops the gate from
# going silent on every answer that contains a hedge, a question, a transition
# sentence, or a code-fence artifact. Conservative by construction: a sentence is
# only treated as subjective when it carries an opinion/meta marker AND has no
# checkable signal (no number/year/DOI/URL/econ term/causal cue/authorship verb).
_SUBJECTIVE_RE = re.compile(
    r"\b(?:i\s+(?:think|believe|recommend|suggest|feel|would)|in\s+my\s+opinion|"
    r"we\s+should|you\s+should|you\s+could|let'?s|let\s+us|consider|it\s+depends|"
    r"here\s+is|here'?s|as\s+follows|the\s+following|for\s+example|note\s+that|"
    r"in\s+summary|to\s+summari[sz]e|overall|arguably|it\s+seems|perhaps|maybe)\b",
    re.I,
)
# A code-fence artifact line (``` or ```lang) left over after fenced blocks are
# extracted; must never be treated as an open factual claim.
_FENCE_ARTIFACT_RE = re.compile(r"^\s*`{3,}\s*[a-z0-9_+-]*\s*$", re.I)
_AUTHORSHIP_VERB_RE = re.compile(r"\b(?:wrote|authored|penned|composed|author of)\b", re.I)
# Interrogative opener (sentence splitting strips the trailing '?', so detect the
# leading question word instead). A question asserts nothing factual.
_QUESTION_RE = re.compile(r"^\s*(?:who|what|when|where|why|how|which|should|can|could|would|do|does|is|are)\b", re.I)


def _has_checkable_signal(claim: str) -> bool:
    """True if the sentence carries any signal a non-wiki verifier can act on."""
    return bool(
        _ARITH_RE.search(claim) or _DOI_RE.search(claim) or _URL_RE.search(claim)
        or _YEAR_RE.search(claim) or _ECON_RE.search(claim) or _CAUSAL_RE.search(claim)
        or _AUTHORSHIP_VERB_RE.search(claim) or re.search(r"\d", claim)
    )


def decompose_and_type(text: str) -> list[AtomicClaim]:
    """Break a generated statement into atomic, typed claims.

    Rule: split with ``agent.claim_router.split_claims`` first; then route each
    atom to the most specific non-wiki verifier type. Code blocks are kept as a
    separate atom because sentence splitting would shatter them.
    """
    # Pull fenced Python blocks out FIRST and verify them as single atoms; remove
    # them from the prose so their lines (``x = 1``) and fence markers (backticks)
    # are never shattered into bogus held ``open_empirical`` claims.
    code_blocks = _CODE_RE.findall(text or "")
    prose = _CODE_RE.sub(" ", text or "")
    claims = [
        AtomicClaim(c, classify_claim(c), risk=risk_for(c))
        for c in split_claims(prose)
        if c.strip() and not _FENCE_ARTIFACT_RE.match(c.strip())
    ]
    for block in code_blocks:
        if block.strip():
            claims.append(AtomicClaim(block.strip(), "code_python", risk="normal"))
    return [c for c in claims if c.text.strip()]


def classify_claim(claim: str) -> str:
    if _CODE_RE.search(claim):
        return "code_python"
    if _ARITH_RE.search(claim):
        return "math"
    if _DOI_RE.search(claim):
        return "doi"
    if _URL_RE.search(claim):
        return "url"
    if _YEAR_RE.search(claim) and _DATE_ORDER_RE.search(claim):
        return "date_temporal"
    if _ECON_RE.search(claim):
        if _CAUSAL_RE.search(claim):
            return "econ_causal"
        return "econ_empirical"
    if _CAUSAL_RE.search(claim):
        return "causal_empirical"
    # Pure opinion/meta/question with no checkable signal: non-factual, so it
    # passes (does not force a hold). This is the over-abstention fix.
    if not _has_checkable_signal(claim) and (
        _SUBJECTIVE_RE.search(claim) or claim.strip().endswith("?") or _QUESTION_RE.match(claim)
    ):
        return "subjective"
    return "open_empirical"


def risk_for(claim: str) -> str:
    if _ECON_RE.search(claim) or re.search(
        r"\b(?:medical|legal|financial|election|war|agi\s+(?:safety|risk|deployment|lab)|incentives?)\b",
        claim, re.I,
    ):
        return "high"
    return "normal"


# --------------------------------------------------------------------------- #
# Layer 1: deterministic type verifiers
# --------------------------------------------------------------------------- #
def deterministic_verify(
    claim: AtomicClaim,
    *,
    url_resolver: Callable[[str], bool] | None = None,
    doi_resolver: Callable[[str], bool] | None = None,
) -> LayerResult:
    """Run deterministic/computational checks that need no internal wiki.

    Accept means the claim is fully resolved by computation/resolution. Hold
    means this deterministic layer cannot decide and the claim should advance to
    active grounding. Reject means contradiction/impossibility.
    """
    text = claim.text
    if claim.type == "math":
        return _verify_math(text)
    if claim.type == "subjective":
        # Non-factual (opinion/meta/question): not subject to the fail-closed
        # factual rule. Passes WITHOUT manufacturing evidence.
        return LayerResult("deterministic", "accepted", "non-factual/subjective; no factual claim to verify", confidence=1.0)
    if claim.type == "code_python":
        return _verify_python_syntax(text)
    if claim.type == "date_temporal":
        return _verify_date_order(text)
    if claim.type == "url":
        urls = _URL_RE.findall(text)
        if not urls:
            return LayerResult("deterministic", "held", "no URL extracted", confidence=0.0)
        if url_resolver is None:
            return LayerResult("deterministic", "held", "URL syntax valid but live resolver unavailable", confidence=0.55,
                               details={"urls": urls, "offline": True})
        ok = [u for u in urls if url_resolver(u)]
        if len(ok) == len(urls):
            return LayerResult("deterministic", "accepted", "all URLs resolved", confidence=0.95,
                               evidence=tuple({"id": u, "url": u, "kind": "url-exists"} for u in ok))
        return LayerResult("deterministic", "rejected", "one or more URLs do not resolve", confidence=0.9,
                           details={"urls": urls, "resolved": ok})
    if claim.type == "doi":
        dois = _DOI_RE.findall(text)
        if doi_resolver is None:
            return LayerResult("deterministic", "held", "DOI syntax valid but resolver unavailable", confidence=0.55,
                               details={"dois": dois, "offline": True})
        ok = [d for d in dois if doi_resolver(d)]
        if len(ok) == len(dois):
            return LayerResult("deterministic", "accepted", "all DOIs resolved", confidence=0.96,
                               evidence=tuple({"id": f"doi:{d}", "url": f"https://doi.org/{d}", "kind": "doi-exists"} for d in ok))
        return LayerResult("deterministic", "rejected", "one or more DOIs do not resolve", confidence=0.9,
                           details={"dois": dois, "resolved": ok})
    return LayerResult("deterministic", "held", "claim type requires external grounding", confidence=0.0)


def _verify_math(text: str) -> LayerResult:
    failures = []
    for a, op, b, c in _ARITH_RE.findall(text):
        x, y, z = float(a), float(b), float(c)
        val = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y != 0 else math.inf}[op]
        if abs(val - z) > 1e-9:
            failures.append(f"{a} {op} {b} = {c} is false; computed {val:g}")
    if failures:
        return LayerResult("deterministic", "rejected", "; ".join(failures), confidence=1.0)
    return LayerResult("deterministic", "accepted", "arithmetic equality verified by computation", confidence=1.0)


def _verify_python_syntax(code: str) -> LayerResult:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return LayerResult("deterministic", "rejected", f"python syntax error: {exc}", confidence=1.0)
    return LayerResult("deterministic", "accepted", "python syntax parses", confidence=0.85)


def _verify_date_order(text: str) -> LayerResult:
    # Only resolve explicit year ordering patterns; otherwise hold for grounding.
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if len(years) < 2:
        return LayerResult("deterministic", "held", "not enough explicit years for temporal check", confidence=0.0)
    low = text.lower()
    if "before" in low and not years[0] < years[1]:
        return LayerResult("deterministic", "rejected", f"temporal order false: {years[0]} is not before {years[1]}", confidence=0.9)
    if "after" in low and not years[0] > years[1]:
        return LayerResult("deterministic", "rejected", f"temporal order false: {years[0]} is not after {years[1]}", confidence=0.9)
    if "before" in low or "after" in low:
        return LayerResult("deterministic", "accepted", "explicit year ordering is internally consistent", confidence=0.8)
    return LayerResult("deterministic", "held", "date claim requires external grounding", confidence=0.0)


# --------------------------------------------------------------------------- #
# Layer 2: external grounding
# --------------------------------------------------------------------------- #
def external_ground(
    claim: AtomicClaim,
    retriever: Retriever | None,
    entailment: EntailmentFn | None = None,
    *,
    min_sources_normal: int = 2,
    min_sources_high: int = 3,
) -> LayerResult:
    if retriever is None:
        return LayerResult("external_grounding", "held", "no live/offline retriever configured", confidence=0.0)
    sources = retriever(claim) or []
    if not sources:
        return LayerResult("external_grounding", "held", "active retrieval returned no evidence; abstain/defer", confidence=0.0)
    # When no real entailment backend (NLI / model) is injected, we fall back to a
    # LEXICAL SCREEN. A lexical screen can detect overlap and obvious contradiction
    # but CANNOT prove entailment (the user's "vs mere keyword overlap" concern).
    # So lexical-only support is capped below the high-risk floor: high-risk claims
    # then HOLD until a real entailment backend confirms them, while low-stakes
    # claims may pass. This is the precise over-confidence guard for Layer 2.
    is_lexical_screen = entailment is None
    entailment = entailment or lexical_entailment
    entailed: list[EvidenceSource] = []
    contradicted: list[EvidenceSource] = []
    irrelevant: list[EvidenceSource] = []
    for src in sources:
        label = entailment(claim, src)
        if label == "entails":
            entailed.append(src)
        elif label == "contradicts":
            contradicted.append(src)
        else:
            irrelevant.append(src)
    if contradicted:
        return LayerResult("external_grounding", "rejected", "one or more independent sources contradict the claim", confidence=0.88,
                           evidence=tuple(_src_dict(s, "contradicts") for s in contradicted),
                           details={"entailed": len(entailed), "irrelevant": len(irrelevant)})
    independent = _independent_domains(entailed)
    required = min_sources_high if claim.risk == "high" else min_sources_normal
    if len(independent) >= required:
        # Real entailment backend → full confidence. Lexical screen → capped at
        # 0.78 so the high-risk floor (0.82) demotes it to HOLD; normal-risk
        # (floor 0.70) may still pass on a lexical screen for low-stakes claims.
        if is_lexical_screen:
            confidence = 0.78
            basis = "lexical screen (not proven entailment)"
        else:
            confidence = 0.86 if claim.risk == "normal" else 0.84
            basis = "entailment backend"
        return LayerResult("external_grounding", "accepted",
                           f"{len(independent)} independent sources entail claim (required {required}; {basis})",
                           confidence=confidence,
                           evidence=tuple(_src_dict(s, "entails") for s in entailed),
                           details={"independentDomains": sorted(independent), "requiredSources": required,
                                    "entailmentBasis": basis})
    return LayerResult("external_grounding", "held", f"insufficient independent entailing sources: {len(independent)}/{required}", confidence=0.35,
                       evidence=tuple(_src_dict(s, "entails") for s in entailed),
                       details={"irrelevant": len(irrelevant), "requiredSources": required})


def lexical_entailment(claim: AtomicClaim, src: EvidenceSource) -> str:
    """Deterministic offline entailment screen.

    It is intentionally conservative: source text must cover at least 70% of the
    claim's content tokens and include all numbers/years from the claim. It can
    reject obvious contradictions via negation/antonym cues; otherwise irrelevant.
    A real live backend should replace/augment this with NLI/model entailment.
    """
    claim_tokens = _content_tokens(claim.text)
    text = f"{src.title} {src.snippet}".lower()
    source_tokens = set(_content_tokens(text))
    if not claim_tokens:
        return "irrelevant"
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim.text)
    if any(n.lower() not in text for n in numbers):
        return "irrelevant"
    overlap = sum(1 for t in claim_tokens if t in source_tokens) / max(1, len(claim_tokens))
    if overlap >= 0.50 and re.search(r"\b(?:not|false|myth|incorrect|no evidence|did not|does not)\b", text) and not re.search(r"\bnot\b", claim.text.lower()):
        return "contradicts"
    if overlap >= 0.70:
        return "entails"
    return "irrelevant"


def _content_tokens(text: str) -> list[str]:
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and", "or", "for", "with", "by", "that", "this"}
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in stop]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _independent_domains(sources: Iterable[EvidenceSource]) -> set[str]:
    return {s.domain for s in sources if s.domain}


def _src_dict(s: EvidenceSource, relation: str) -> dict[str, Any]:
    return {"id": s.id, "url": s.url, "title": s.title, "publisher": s.publisher, "domain": s.domain, "relation": relation}


# --------------------------------------------------------------------------- #
# Layer 3: consensus by verification (not vote)
# --------------------------------------------------------------------------- #
def consensus_by_verification(
    claim: AtomicClaim,
    judges: list[JudgeFn] | None,
    evidence: list[EvidenceSource] | None = None,
    *,
    min_competent_families: int = 2,
) -> LayerResult:
    if not judges:
        return LayerResult("consensus_by_verification", "held", "no judge families configured", confidence=0.0)
    results = []
    for judge in judges:
        r = judge(claim, evidence or {}) or {}
        results.append(r)
    competent = [r for r in results if _judge_is_competent(r)]
    support = [r for r in competent if r.get("verdict") == "supports" and r.get("evidenceIds")]
    contradict = [r for r in competent if r.get("verdict") == "contradicts" and r.get("evidenceIds")]
    if contradict:
        return LayerResult("consensus_by_verification", "rejected", "competent judge found evidence-backed contradiction", confidence=0.75,
                           details={"judgeResults": results})
    families = {r.get("family") for r in support if r.get("family")}
    if len(families) >= min_competent_families:
        return LayerResult("consensus_by_verification", "accepted", "multiple competent judge families support with cited evidence", confidence=0.72,
                           details={"judgeResults": results, "families": sorted(families)})
    return LayerResult("consensus_by_verification", "held", "no evidence-backed competent multi-family support; majority vote is insufficient", confidence=0.25,
                       details={"judgeResults": results, "competentCount": len(competent)})


def _judge_is_competent(r: dict[str, Any]) -> bool:
    """Competence rule: judge must not rubber-stamp, must cite evidence, must be
    calibrated. In production the calibration fields come from a held-out judge
    suite. Defaults fail closed.
    """
    if not r.get("family"):
        return False
    if float(r.get("calibrationEce", 1.0)) > 0.12:
        return False
    if float(r.get("rubberStampRate", 1.0)) > 0.80:
        return False
    if int(r.get("heldoutN", 0)) < 40:
        return False
    if r.get("verdict") in {"supports", "contradicts"} and not r.get("evidenceIds"):
        return False
    return True


# --------------------------------------------------------------------------- #
# Layer 4/5: decision + learning candidate
# --------------------------------------------------------------------------- #
def fact_check_claim(
    claim: AtomicClaim,
    *,
    retriever: Retriever | None = None,
    entailment: EntailmentFn | None = None,
    judges: list[JudgeFn] | None = None,
    url_resolver: Callable[[str], bool] | None = None,
    doi_resolver: Callable[[str], bool] | None = None,
    learn: bool = True,
) -> ClaimDecision:
    def _finalize(verdict: str, reason: str, confidence: float, layers: list[LayerResult], src_layer: LayerResult) -> ClaimDecision:
        # Enforce the calibrated-abstention floor: an accept below the risk floor
        # is demoted to held (over-confidence guard). Reject is never demoted.
        if verdict == "accepted" and confidence < _floor_for(claim.risk):
            return ClaimDecision(
                claim, "held",
                f"below calibrated-abstention floor ({confidence:.2f} < {_floor_for(claim.risk):.2f} for {claim.risk}-risk); abstain",
                confidence, tuple(layers), None,
            )
        learning = _learning_candidate(claim, src_layer) if (learn and verdict == "accepted") else None
        return ClaimDecision(claim, verdict, reason, confidence, tuple(layers), learning)

    layers: list[LayerResult] = []
    det = deterministic_verify(claim, url_resolver=url_resolver, doi_resolver=doi_resolver)
    layers.append(det)
    if det.verdict in {"accepted", "rejected"}:
        return _finalize(det.verdict, det.reason, det.confidence, layers, det)

    ext = external_ground(claim, retriever, entailment)
    layers.append(ext)
    if ext.verdict in {"accepted", "rejected"}:
        return _finalize(ext.verdict, ext.reason, ext.confidence, layers, ext)

    # Judges may only use the evidence already retrieved; if no evidence exists,
    # they cannot create support by vote.
    judge_evidence = [
        EvidenceSource(
            id=str(e.get("id", "")),
            url=str(e.get("url", "")),
            title=str(e.get("title", "")),
            publisher=str(e.get("publisher", "")),
        )
        for e in ext.evidence
    ]
    # Judges receive only retrieved evidence. They may not create support from
    # memory/vote; competence rules require cited evidenceIds for support.
    judge = consensus_by_verification(claim, judges, judge_evidence)
    layers.append(judge)
    if judge.verdict in {"accepted", "rejected"}:
        return _finalize(judge.verdict, judge.reason, judge.confidence, layers, judge)

    return ClaimDecision(
        claim,
        "held",
        "abstain/defer: deterministic checks inconclusive, external grounding insufficient, and no evidence-backed competent consensus",
        min(0.49, max((l.confidence for l in layers), default=0.0)),
        tuple(layers),
        None,
    )


def fact_check_text(text: str, **kwargs: Any) -> GateDecision:
    claims = decompose_and_type(text)
    if not claims:
        return GateDecision("held", "no atomic claims extracted", ())
    decisions = tuple(fact_check_claim(c, **kwargs) for c in claims)
    if any(d.verdict == "rejected" for d in decisions):
        return GateDecision("rejected", "one or more atomic claims were contradicted/impossible", decisions)
    if all(d.verdict == "accepted" for d in decisions):
        return GateDecision("accepted", "all atomic claims verified", decisions)
    return GateDecision("held", "one or more atomic claims remain unverified; fail-closed hold", decisions)


def _learning_candidate(claim: AtomicClaim, layer: LayerResult) -> dict[str, Any]:
    return {
        "schema": "sophia.fact_check.learning_candidate.v1",
        "claimId": "fc_" + hashlib.sha256(claim.text.encode("utf-8")).hexdigest()[:16],
        "claim": claim.text,
        "type": claim.type,
        "risk": claim.risk,
        "verifiedBy": layer.layer,
        "confidence": layer.confidence,
        "evidence": list(layer.evidence),
        "promotionState": "pending_quarantine",
        "promotionRules": [
            "never use this candidate as evidence for itself",
            "require source freshness TTL before reuse",
            "require independent recheck before promotion to curated wiki",
            "store as external/provisional, not canonical, until review",
        ],
    }


def decision_to_dict(decision: GateDecision) -> dict[str, Any]:
    return {
        "verdict": decision.verdict,
        "reason": decision.reason,
        "claims": [
            {
                "claim": d.claim.text,
                "type": d.claim.type,
                "risk": d.claim.risk,
                "verdict": d.verdict,
                "reason": d.reason,
                "confidence": d.confidence,
                "learningCandidate": d.learning_candidate,
                "layers": [
                    {"layer": l.layer, "verdict": l.verdict, "reason": l.reason, "confidence": l.confidence,
                     "evidence": list(l.evidence), "details": l.details}
                    for l in d.layers
                ],
            }
            for d in decision.claims
        ],
    }


__all__ = [
    "AtomicClaim", "EvidenceSource", "LayerResult", "ClaimDecision", "GateDecision",
    "decompose_and_type", "classify_claim", "deterministic_verify", "external_ground",
    "lexical_entailment", "consensus_by_verification", "fact_check_claim", "fact_check_text",
    "decision_to_dict",
]
