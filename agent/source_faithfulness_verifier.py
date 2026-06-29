# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Source-faithfulness verifier — the misstated-conclusion check (semantic frontier).

The existence + attribution verifiers cover *structural* fabrications (a study that does not
exist; a real work credited to the wrong creator). They do NOT cover the last slice: a REAL,
correctly-attributed source whose *finding/conclusion* is misstated (the Ayinde misstated-
authority failure mode, generalized beyond law). That is irreducibly a *semantic* question —
"does the source actually support this claim?" — which needs an entailment judge.

To keep it as trustworthy as a model-based check can be, this verifier:
  - judges the answer's claim against an **independent real source** (retrieved, not the
    possibly-contaminated grounding source), so the source is independent of the answer model;
  - uses a **multi-judge panel** (the repo's no-overclaim posture) and acts only on CONSENSUS;
  - is **fail-open on insufficiency** — it rejects ONLY when the independent source is judged to
    CONTRADICT the claim, never on "insufficient", so it does not over-block.

Independence is **medium** and labelled as such: the source and judges are independent of the
answer, but the support verdict is a model judgment (not a deterministic record). It complements
the high-independence layers; it does not replace them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "Verdict", "assess_support", "make_llm_support_judge",
    "make_faithfulness_corroborate_fn",
]

# A judge maps (claim, source_text) -> Verdict. MUST NOT raise; on failure it abstains.
SupportJudge = Callable[[str, str], "Verdict"]


@dataclass
class Verdict:
    supports: bool = False
    abstained: bool = True
    reason: str = ""
    method: str = ""


_JUDGE_SYSTEM = (
    "You assess whether a SOURCE supports a CLAIM. Reply with ONLY a JSON object: "
    '{"supports": bool, "abstained": bool, "reason": string}. '
    "supports=true ONLY if the source actually establishes the claim. "
    "abstained=true if the source is insufficient to decide. "
    "Be strict: a claim that misstates what the source says is supports=false."
)


def assess_support(claim: str, source_text: str, judges: "list[SupportJudge]") -> "dict[str, Any]":
    """Run a panel of judges and return a CONSENSUS verdict.

    Returns ``{"verdict": "supports"|"contradicts"|"insufficient", "supports": n, "contradicts": n,
    "abstained": n, "n": N}``. ``contradicts`` (misstated) only when a majority of judges decide
    NOT-supports AND not-abstained; ``supports`` only on a majority of supports; otherwise
    ``insufficient`` (fail-open).
    """
    verdicts = []
    for j in judges:
        try:
            verdicts.append(j(claim, source_text))
        except Exception:  # noqa: BLE001 — a judge error is an abstention, never a pass
            verdicts.append(Verdict(abstained=True, reason="judge-error"))
    n = len(verdicts)
    sup = sum(1 for v in verdicts if v.supports and not v.abstained)
    con = sum(1 for v in verdicts if (not v.supports) and (not v.abstained))
    abst = sum(1 for v in verdicts if v.abstained)
    # STRICT majority of ALL judges (not just a plurality of non-abstainers) so a single
    # over-strict judge cannot trigger a rejection on a correct claim (avoids over-block).
    maj = (n // 2) + 1 if n else 1
    if con >= maj and con > sup:
        verdict = "contradicts"
    elif sup >= maj and sup > con:
        verdict = "supports"
    else:
        verdict = "insufficient"
    return {"verdict": verdict, "supports": sup, "contradicts": con, "abstained": abst, "n": n}


def make_llm_support_judge(spec: str) -> "SupportJudge":
    """Build an LLM entailment judge from a model spec (lazy import; used live, not in tests)."""
    from agent.model import complete  # noqa: PLC0415

    def judge(claim: str, source_text: str) -> Verdict:
        try:
            raw = complete(_JUDGE_SYSTEM,
                           f"CLAIM: {claim}\n\nSOURCE: {source_text[:1500]}",
                           spec=spec, max_tokens=160) or ""
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            return Verdict(supports=bool(data.get("supports")),
                           abstained=bool(data.get("abstained", not m)),
                           reason=str(data.get("reason", ""))[:200], method=f"llm:{spec}")
        except Exception:  # noqa: BLE001 — fail-closed: unparseable -> abstain
            return Verdict(abstained=True, reason="parse-error", method=f"llm:{spec}")

    return judge


def make_faithfulness_corroborate_fn(
    retrieve_source_fn: "Callable[[str, str], str]",
    judges: "list[SupportJudge]",
    *,
    extractor_fn: "Callable[[str, str], str] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn for the contamination gate.

    Returns False (REJECT -> misstatement caught) iff an INDEPENDENT retrieved source is judged by
    a CONSENSUS panel to CONTRADICT the answer's core claim. Accepts otherwise (supports /
    insufficient -> fail-open). ``retrieve_source_fn(question, answer) -> source_text`` fetches an
    independent source (e.g. a Wikipedia summary for the topic). Exposes ``.last_result``.
    """
    from agent.core_claim_source_verifier import extract_core_claim  # noqa: PLC0415

    holder: "dict[str, Any]" = {}

    def verify(question: str, answer: str) -> bool:
        if not answer or not answer.strip():
            return True
        claim = (extractor_fn or extract_core_claim)(question, answer)
        try:
            source = retrieve_source_fn(question, answer) or ""
        except Exception:  # noqa: BLE001 — no independent source -> cannot judge -> fail-open
            source = ""
        if not source.strip():
            holder.clear(); holder.update({"verdict": "no_source", "independence": "medium"})
            return True
        res = assess_support(claim, source, judges)
        res["independence"] = "medium"
        holder.clear(); holder.update(res)
        return res["verdict"] != "contradicts"

    verify.last_result = holder  # type: ignore[attr-defined]
    return verify
