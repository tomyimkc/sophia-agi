"""Reflexive no-overclaim gate for Sophia's own AGI claims.

This is intentionally narrower than the general fact-check gate. It scans project
text for self-referential AGI status claims and enforces the invariant that the
repo may describe Sophia as an AGI-candidate / proof package, but must not assert
that Sophia is proven AGI unless the machine-readable evidence says so.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

AGI_RE = re.compile(r"\bAGI\b|artificial general intelligence", re.I)
PROVEN_RE = re.compile(r"\b(?:proven|achieved|is|are|certified|verified)\s+(?:an?\s+)?AGI\b|\bAGI\s+(?:has\s+been\s+)?(?:proven|achieved|certified|verified)\b", re.I)
SAFE_RE = re.compile(
    r"\b(?:do\s+not\s+use|disallowed\s+public\s+wording|disallowed|forbidden|not\s+that|not\s+(?:a\s+)?claim(?:ed)?\s+(?:as\s+)?(?:of\s+)?AGI|not\s+claimed\s+as\s+proven\s+AGI|"
    r"not\s+proven\s+AGI|AGI-candidate|candidate\s+proof|canClaimAGI\s*[=:]\s*false|"
    r"thresholds\s+are\s+not\s+met|cannot\s+claim\s+AGI|not\s+machine-assertable|not\s+met)",
    re.I,
)
NEGATION_WINDOW_RE = re.compile(r"\b(?:not|never|no|without|cannot|can't|isn't|not yet)\b.{0,80}\b(?:proven|claim|claimed|AGI)\b", re.I)


@dataclass(frozen=True)
class SelfClaim:
    path: str
    line: int
    text: str
    verdict: str
    reason: str


def scan_paths(paths: Iterable[str | Path], *, repo_root: str | Path = ".") -> dict:
    root = Path(repo_root)
    claims: list[SelfClaim] = []
    can_claim_agi_values: list[dict] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if _should_scan_file(child):
                    claims.extend(scan_file(child, repo_root=root))
        elif _should_scan_file(path):
            claims.extend(scan_file(path, repo_root=root))
            if path.suffix == ".json":
                can_claim_agi_values.extend(_extract_can_claim_agi(path, root))
    claims.extend(_check_can_claim_agi_values(can_claim_agi_values))
    if any(c.verdict == "rejected" for c in claims):
        verdict = "rejected"
        reason = "one or more self-claims overstate Sophia's AGI status"
    elif any(c.verdict == "held" for c in claims):
        verdict = "held"
        reason = "one or more AGI self-claims need evidence or clearer no-overclaim wording"
    else:
        verdict = "accepted"
        reason = "all scanned AGI self-claims respect candidate/no-overclaim boundary"
    return {
        "schema": "sophia.reflexive_self_gate.v1",
        "verdict": verdict,
        "reason": reason,
        "canClaimAGI": False,
        "claims": [c.__dict__ for c in claims],
        "summary": {
            "accepted": sum(c.verdict == "accepted" for c in claims),
            "held": sum(c.verdict == "held" for c in claims),
            "rejected": sum(c.verdict == "rejected" for c in claims),
        },
    }


def scan_file(path: Path, *, repo_root: str | Path = ".") -> list[SelfClaim]:
    root = Path(repo_root)
    rel = str(path.relative_to(root)) if path.is_absolute() and path.is_relative_to(root) else str(path)
    out: list[SelfClaim] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    for i, line in enumerate(lines, 1):
        text = line.strip()
        if not text or not AGI_RE.search(text):
            continue
        context = " ".join(x.strip() for x in lines[max(0, i - 3): i])
        out.append(classify_self_claim(rel, i, text, context=context))
    return out


def classify_self_claim(path: str, line: int, text: str, *, context: str | None = None) -> SelfClaim:
    ctx = f"{context or ''} {text}"
    # Evaluation/example rows may contain intentionally false claims; they are not
    # repo self-assertions. They remain useful adversarial fixtures for the gate.
    if '"claim"' in text or "'claim'" in text:
        return SelfClaim(path, line, text, "accepted", "benchmark/example claim, not public self-assertion")
    if SAFE_RE.search(ctx) or NEGATION_WINDOW_RE.search(ctx):
        return SelfClaim(path, line, text, "accepted", "explicit candidate/no-overclaim wording")
    if PROVEN_RE.search(text):
        return SelfClaim(path, line, text, "rejected", "asserts or implies proven/achieved AGI without machine-cleared evidence")
    # Mere discussion of AGI is allowed but not proof-bearing.
    return SelfClaim(path, line, text, "accepted", "AGI mention without status overclaim")


def _extract_can_claim_agi(path: Path, root: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    found: list[dict] = []

    def walk(obj, pointer=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{pointer}/{k}"
                if k == "canClaimAGI":
                    found.append({"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "pointer": p, "value": v})
                walk(v, p)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                walk(v, f"{pointer}/{idx}")

    walk(data)
    return found


def _check_can_claim_agi_values(values: list[dict]) -> list[SelfClaim]:
    out: list[SelfClaim] = []
    for row in values:
        ok = row.get("value") is False
        out.append(SelfClaim(
            row["path"], 0, f"{row['pointer']}={row.get('value')!r}",
            "accepted" if ok else "rejected",
            "machine manifest keeps canClaimAGI=false" if ok else "machine manifest permits AGI claim",
        ))
    return out


def _should_scan_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    # Generated gate/eval reports can contain intentionally bad example claims or
    # copies of prior scan findings; scanning them recursively creates false
    # self-overclaim alarms. The source docs/manifests remain scanned.
    if any(part in {"self-gate", "fact-check-live"} for part in path.parts):
        return False
    return path.suffix.lower() in {".md", ".json", ".txt"}


__all__ = ["scan_paths", "scan_file", "classify_self_claim"]
