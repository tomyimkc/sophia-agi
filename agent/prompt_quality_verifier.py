# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prompt-quality verifier — the machine metric behind a verifier-gated prompt skill.

A "best prompt" in this repo is not a stylistic judgement; it is a *checkable* one, scored by the
same discipline Sophia applies to claims: a good task/PR prompt states how done is **verified**,
**bounds** its scope, gives an **abstention / failure path**, **grounds** itself in concrete
artifacts, and does **not overclaim**. This module turns that into a deterministic predicate so a
prompt generator can be gated — only prompts that clear the bar get promoted (Layer-C of the skill
taxonomy, via ``tools/sophia_skill_forge.py``). It is the metric DSPy-style prompt optimisation
needs, but machine-checked rather than an LLM judge.

Dimensions (all offline, deterministic):
  * success_criterion — names how completion is verified (test / gate / metric / CI / expected output)
  * bounded_scope     — constrains what is in/out (only, must not, a single deliverable, a branch/path)
  * abstention_path   — says what to do when blocked/uncertain (report, abstain, NO-GO, stays candidate)
  * no_overclaim      — carries no unqualified superlative / AGI / safety claim (reuses lint_claims)
  * grounding         — references concrete artifacts (files, paths, PR/#, branches) not "the thing"

``passed`` requires the four load-bearing dimensions (success_criterion, bounded_scope,
abstention_path, no_overclaim); grounding is scored and reported but not fail-closed on its own, so a
short well-formed prompt is not rejected for brevity. canClaimAGI is irrelevant here — this makes no
capability claim; it scores prose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the repo's AUTHORITATIVE overclaim patterns so the prompt verifier and the claims linter
# can never drift apart. Fall back to a small subset if the linter cannot be imported.
try:
    from tools.lint_claims import ALLOW_MARKER, FORBIDDEN  # noqa: E402

    _OVERCLAIM = [(re.compile(p, re.I), why) for p, why in FORBIDDEN]
    _ALLOW = ALLOW_MARKER
except Exception:  # pragma: no cover - defensive
    _ALLOW = "claim-ok"
    _OVERCLAIM = [(re.compile(p, re.I), "overclaim") for p in (
        r"\bbreakthrough\b", r"\bworld'?s first\b", r"\bproven agi\b", r"\bis agi\b",
        r"\bmakes ai safe\b", r"\bthe only open project\b")]

_SUCCESS_CRITERION = re.compile(
    r"\b(test|gate|claim-?check|\bCI\b|metric|pass(?:es|ed)?|expected|verif(?:y|ied|ication)|"
    r"κ|kappa|confidence interval|excludes? zero|benchmark|assert|receipt|GO|NO-?GO|"
    r"≥|>=|\d+\s?%|seeds?)\b", re.I)

_BOUNDED_SCOPE = re.compile(
    r"\b(only|do not|don'?t|must not|never|in scope|out of scope|scope|bounded|limit(?:ed)?|"
    r"single|exactly one|a single|branch|do NOT (?:touch|edit|push))\b", re.I)

_ABSTENTION_PATH = re.compile(
    r"\b(if (?:blocked|unsure|uncertain|stuck|it fails|you can'?t)|abstain|report (?:back|the|any)|"
    r"escalate|NO-?GO|candidate|OPEN|fail[- ]closed|stays? (?:false|candidate)|"
    r"do not (?:proceed|push|merge)|ask (?:me|the|first))\b", re.I)

_GROUNDING = re.compile(
    r"(/[\w./-]+|\b\w+\.(?:py|md|jsonl|json|ya?ml)\b|#\d+|\bPR\b|\bbranch\b|"
    r"\bclaude/|\bglm/|tools/|agent/|tests/|docs/|provenance_bench/)", re.I)

REQUIRED = ("success_criterion", "bounded_scope", "abstention_path", "no_overclaim")
SCORED = REQUIRED + ("grounding",)


def score_prompt(text: str) -> dict:
    """Score a prompt across the five dimensions. Deterministic; no model, no network."""
    text = text or ""
    allow = _ALLOW in text
    overclaims = [] if allow else [why for rx, why in _OVERCLAIM if rx.search(text)]

    dims = {
        "success_criterion": bool(_SUCCESS_CRITERION.search(text)),
        "bounded_scope": bool(_BOUNDED_SCOPE.search(text)),
        "abstention_path": bool(_ABSTENTION_PATH.search(text)),
        "no_overclaim": len(overclaims) == 0,
        "grounding": bool(_GROUNDING.search(text)),
    }
    reasons: list[str] = []
    for d in REQUIRED:
        if not dims[d]:
            reasons.append(f"missing {d}")
    if overclaims:
        reasons.append(f"overclaim: {overclaims[:3]}")
    if not dims["grounding"]:
        reasons.append("weak grounding (no file/path/PR reference) — recommended, not required")

    passed = all(dims[d] for d in REQUIRED)
    n_scored = sum(1 for d in SCORED if dims[d])
    return {
        "passed": passed,
        "score": round(n_scored / len(SCORED), 3),
        "dimensions": dims,
        "reasons": reasons,
        "overclaims": overclaims,
    }


def prompt_quality_ok(text: str) -> bool:
    """Boolean predicate for the skill forge / autoresearch firewall: is this prompt promotable?"""
    return score_prompt(text)["passed"]


def prompt_quality():
    """Verifier-style callable matching the repo convention: ``v(text, record, ctx) -> dict``."""

    def _v(text, _record=None, _ctx=None) -> dict:
        s = score_prompt(text)
        return {"passed": s["passed"], "reasons": s["reasons"], "detail": s}

    return _v


# Deterministic fixtures — a well-formed task prompt, a vague one, an overclaiming one.
_GOOD = (
    "On branch glm/firewall-redteam ONLY, find bypasses of the reward-hacking firewall in "
    "tools/sophia_autoresearch.py. For each hole: a failing test in tests/test_sophia_autoresearch.py "
    "plus the minimal decide() hardening. Run `make claim-check` (must be GO); canClaimAGI stays false. "
    "If you cannot reproduce a hole, report it as NO-GO rather than guess."
)
_VAGUE = "Make the repo better and improve the model as much as you can."
_OVERCLAIMING = "Build the world's first proven AGI and make AI safe — this is a breakthrough."


def offline_invariants() -> "tuple[bool, dict]":
    checks = {
        "good_prompt_passes": score_prompt(_GOOD)["passed"] is True,
        "vague_prompt_fails": score_prompt(_VAGUE)["passed"] is False,
        "overclaiming_prompt_fails": score_prompt(_OVERCLAIMING)["dimensions"]["no_overclaim"] is False,
        "allow_marker_exempts_overclaim": score_prompt(_OVERCLAIMING + " claim-ok")["dimensions"]["no_overclaim"] is True,
        "deterministic": score_prompt(_GOOD) == score_prompt(_GOOD),
        "good_scores_higher_than_vague": score_prompt(_GOOD)["score"] > score_prompt(_VAGUE)["score"],
    }
    return all(checks.values()), {"checks": checks,
                                  "goodScore": score_prompt(_GOOD)["score"],
                                  "vagueScore": score_prompt(_VAGUE)["score"]}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Prompt-quality verifier invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  good/vague score:", detail["goodScore"], "/", detail["vagueScore"])
    raise SystemExit(0 if ok else 1)
