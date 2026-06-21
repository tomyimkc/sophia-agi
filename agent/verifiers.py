"""Pluggable verifiers for the agent harness, eval, and RL loops.

A verifier is ``(text, task, step) -> {"passed": bool, "reasons": [...], "detail": {...}}``
matching agent.harness. These turn "looks good" into "verified": exact answers,
regex, keywords, unit tests, rubric scoring, and citation checks — plus
combinators. Quality follows verifiability.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from glob import glob as _glob
from pathlib import Path
from typing import Any, Callable

from agent.config import DATA_DIR, ROOT

Verifier = Callable[[str, Any, dict], dict]

# Attribution verbs/markers that signal an *assertion* of authorship (any language).
_ATTR_VERBS = [
    r"\bwrote\b", r"\bwritten\b", r"\bauthored?\b", r"\bauthor of\b", r"\bpenned\b",
    r"\bcomposed\b", r"\battribut", r"\bby\b", r"'s\b", r"’s\b", r"著", r"作者", r"撰", r"所著",
]
# Domain files whose records carry doNotAttributeTo lists. The last entry is the
# active-learning sink: records promoted from verified gate misses
# (tools/promote_pending.py) land here, so the live gate fires on them on the next
# run — closing the feedback loop without hand-editing the seed domain files.
_PROVENANCE_FILES = (
    "attributions.json",
    "psychology_concepts.json",
    "religion_concepts.json",
    "history_events.json",
    "learned_attributions.json",
)


def _ok(detail: dict | None = None) -> dict:
    return {"passed": True, "reasons": [], "detail": detail or {}}


def _fail(reasons: list[str], detail: dict | None = None) -> dict:
    return {"passed": False, "reasons": reasons, "detail": detail or {}}


def exact_match(expected: str, *, normalize: bool = True) -> Verifier:
    """Pass if the answer equals (or, by default, contains) the expected string."""

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower() if normalize else s

    target = norm(expected)

    def _verify(text: str, task: Any, step: dict) -> dict:
        got = norm(text)
        if target == got or target in got:
            return _ok({"expected": expected})
        return _fail([f"expected '{expected}' not found"], {"expected": expected})

    return _verify


def regex_match(pattern: str, *, flags: int = re.IGNORECASE) -> Verifier:
    compiled = re.compile(pattern, flags)

    def _verify(text: str, task: Any, step: dict) -> dict:
        if compiled.search(text):
            return _ok({"pattern": pattern})
        return _fail([f"pattern /{pattern}/ did not match"], {"pattern": pattern})

    return _verify


def keyword(must_include: list[str] | None = None, must_avoid: list[str] | None = None) -> Verifier:
    must_include = must_include or []
    must_avoid = must_avoid or []

    def _verify(text: str, task: Any, step: dict) -> dict:
        lowered = text.lower()
        missing = [k for k in must_include if k.lower() not in lowered]
        present_forbidden = [k for k in must_avoid if k.lower() in lowered]
        reasons = [f"missing: {k}" for k in missing] + [f"forbidden present: {k}" for k in present_forbidden]
        return _ok() if not reasons else _fail(reasons, {"missing": missing, "forbidden": present_forbidden})

    return _verify


def unit_test(command: list[str], *, cwd: Path = ROOT, timeout_sec: int = 300) -> Verifier:
    """Pass iff a command (pytest/build/lint) exits 0. The answer text is ignored;
    the world state is the verifier (the strongest signal for code tasks)."""

    def _verify(text: str, task: Any, step: dict) -> dict:
        try:
            proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return _fail([f"command timed out: {' '.join(command)}"])
        except FileNotFoundError as exc:
            return _fail([f"command not found: {exc}"])
        if proc.returncode == 0:
            return _ok({"cmd": " ".join(command), "returncode": 0})
        return _fail([f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"], {"returncode": proc.returncode})

    return _verify


def score_pack_case(case: dict, *, threshold: float = 1.0) -> Verifier:
    """Wrap hidden_eval_protocol.score_case as a verifier (rubric/operational)."""
    from tools.hidden_eval_protocol import score_case

    def _verify(text: str, task: Any, step: dict) -> dict:
        result = score_case(case, text)
        ratio = (result["score"] / result["maxPoints"]) if result.get("maxPoints") else 0.0
        passed = ratio >= threshold and result.get("passed", False)
        reasons = [m.get("label", str(m)) for m in result.get("missedRubric", [])]
        return {"passed": passed, "reasons": reasons, "detail": {"score": result["score"], "maxPoints": result["maxPoints"]}}

    return _verify


def citation_present(sources: list[str]) -> Verifier:
    """Pass if the answer cites at least one provided source label/path token."""
    tokens = [s for s in (Path(str(x)).name for x in sources) if s]

    def _verify(text: str, task: Any, step: dict) -> dict:
        lowered = text.lower()
        hit = any(tok.lower() in lowered for tok in tokens) or bool(re.search(r"\[(?:local|web)\s*\d+\]|source", lowered))
        return _ok() if hit else _fail(["no citation to a provided source"], {"sources": tokens})

    return _verify


# --------------------------------------------------------------------------- #
# General machine-checked verifiers — generalize the verifier-gated loop beyond
# provenance: citation faithfulness, executable code, and arithmetic soundness.
# Each is deterministic (no model call) so "verified" means verified.
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "was", "were", "are",
    "be", "by", "for", "with", "that", "this", "it", "as", "at", "from", "his", "her",
    "their", "its", "which", "who", "whom", "they", "he", "she", "but", "not", "no",
}


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def citation_faithful(sources: list[str], *, min_overlap: float = 0.35, require_citation: bool = False) -> Verifier:
    """Fail if a cited sentence is NOT supported by its source.

    Sources are 1-indexed for ``[1]`` / ``[local 1]`` / ``[web 1]`` markers. For
    each sentence carrying a citation, the cited source must share at least
    ``min_overlap`` of the sentence's content words (a deterministic support
    check, no model). With ``require_citation`` a declarative factual sentence
    that cites nothing is also flagged. This is RAG citation-faithfulness — the
    gate against "grounded-looking but unsupported" answers.

    Scope (honest): lexical overlap reliably catches fabricated / topically
    mismatched / out-of-range citations. It does NOT catch a subtle wrong
    predicate when the subject still matches the source — that needs a model
    judge (compose with an LLM-judge for that tier).
    """
    src_words = [(_content_words(s), s) for s in sources]

    def _verify(text: str, task: Any, step: dict) -> dict:
        violations: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            s = sentence.strip()
            if len(s) < 12:
                continue
            cites = [int(n) for n in re.findall(r"\[(?:local|web|source)?\s*(\d+)\]", s.lower())]
            words = _content_words(s)
            if not words:
                continue
            if cites:
                best = 0.0
                scored = False
                for idx in cites:
                    if 1 <= idx <= len(src_words):
                        sw = src_words[idx - 1][0]
                        if words:
                            best = max(best, len(words & sw) / len(words))
                        scored = True
                    else:
                        violations.append(f"citation [{idx}] out of range")
                if scored and best < min_overlap:
                    violations.append(f"unsupported citation (overlap {best:.0%} < {min_overlap:.0%}): {s[:60]}")
            elif require_citation and re.search(r"\b(?:is|was|were|are|wrote|invented|discovered|founded|caused)\b", s.lower()):
                violations.append(f"uncited claim: {s[:60]}")
        if violations:
            return _fail([f"citation not faithful: {v}" for v in violations], {"violations": violations})
        return _ok({"sourcesChecked": len(sources)})

    return _verify


def _softmax(xs: list) -> list:
    import math

    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _default_nli():
    """A lazy cross-encoder NLI scorer ``(premise, hypothesis) -> entailment prob``.

    Opt-in: needs ``sentence-transformers`` + a model (``$SOPHIA_NLI_MODEL``,
    default ``cross-encoder/nli-deberta-v3-small``). Returns None when unavailable
    so callers FAIL CLOSED rather than silently pass an unchecked claim. Not run in
    CI (no model); the unit tests inject a mock scorer.
    """
    name = os.environ.get("SOPHIA_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None
    try:
        model = CrossEncoder(name)
    except Exception:
        return None

    # Resolve the entailment index from the model's OWN label map — never assume a
    # fixed order (MNLI models use {0:contradiction,1:neutral,2:entailment}, deberta
    # uses {…,1:entailment,…}). If we can't identify it, FAIL CLOSED (return None)
    # rather than silently mis-score in the unsafe direction.
    ent_idx = None
    try:
        id2label = model.config.id2label
        for i, label in id2label.items():
            if str(label).lower().startswith("entail"):
                ent_idx = int(i)
                break
    except Exception:
        ent_idx = None
    if ent_idx is None:
        return None

    def score(premise: str, hypothesis: str) -> float:
        try:
            logits = list(model.predict([(premise, hypothesis)])[0])
        except Exception:
            return 0.0
        probs = _softmax([float(x) for x in logits])
        return float(probs[ent_idx]) if ent_idx < len(probs) else 0.0

    return score


def claim_supported(sources: list[str], *, nli=None, threshold: float = 0.5,
                    require_citation: bool = False) -> Verifier:
    """Fail if a cited sentence is not ENTAILED by its cited source.

    The semantic tier above :func:`citation_faithful`: it catches a *wrong
    predicate even when the subject matches the source* (e.g. "Marie Curie invented
    the telephone [1]" against a source about her radioactivity work) — the lexical
    blind spot the red-team flagged. ``nli(premise, hypothesis) -> float`` in
    ``[0,1]``; defaults to a cross-encoder (opt-in). If no scorer is available the
    verifier FAILS CLOSED (it refuses to vouch) rather than silently passing.

    Scope (honest): entailment detection is only as good as the supplied NLI model
    and ``threshold`` — a weak model or a mis-set threshold can miss a wrong
    predicate or false-positive on a valid claim. Compose it as a tier, not a
    guarantee. The default model is resolved with the correct entailment label;
    a custom ``$SOPHIA_NLI_MODEL`` must expose an "entailment" label or it fails closed.
    """
    scorer = nli or _default_nli()

    def _verify(text: str, task: Any, step: dict) -> dict:
        if scorer is None:
            return _fail(["NLI scorer unavailable — semantic faithfulness not checked"],
                         {"nli": "unavailable"})
        violations: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            s = sentence.strip()
            if len(s) < 12:
                continue
            cites = [int(n) for n in re.findall(r"\[(?:local|web|source)?\s*(\d+)\]", s.lower())]
            if cites:
                best = 0.0
                scored = False
                for idx in cites:
                    if 1 <= idx <= len(sources):
                        best = max(best, float(scorer(sources[idx - 1], s)))
                        scored = True
                    else:
                        violations.append(f"citation [{idx}] out of range")
                if scored and best < threshold:
                    violations.append(f"claim not entailed by source (entail {best:.0%} < {threshold:.0%}): {s[:60]}")
            elif require_citation and re.search(r"\b(?:is|was|were|are|wrote|invented|discovered|founded|caused)\b", s.lower()):
                violations.append(f"uncited claim: {s[:60]}")
        if violations:
            return _fail([f"claim unsupported: {v}" for v in violations], {"violations": violations})
        return _ok({"sourcesChecked": len(sources)})

    return _verify


def _extract_code(text: str, language: str = "python") -> str:
    """Concatenate fenced code blocks (``` ```), preferring language-tagged ones."""
    blocks = re.findall(r"```(?:" + re.escape(language) + r"|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return "\n\n".join(b.rstrip() for b in blocks)


def code_tests_pass(*, timeout_sec: int = 30, allow_execution: bool = True) -> Verifier:
    """Extract code from the answer and EXECUTE it; pass iff it exits 0.

    The strongest machine check: the produced code (typically asserts / a test
    harness the model wrote) is run in an isolated temp dir. ``allow_execution``
    (or env ``SOPHIA_ALLOW_CODE_EXEC=0``) gates running untrusted code; when off,
    falls back to a safe syntax-only check (``compile``). Times out and never
    touches the repo tree.
    """
    import sys
    import tempfile

    env_off = os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "1").strip() in ("0", "false", "no")

    def _verify(text: str, task: Any, step: dict) -> dict:
        code = _extract_code(text)
        if not code.strip():
            return _fail(["no python code block found"])
        if env_off or not allow_execution:
            try:
                compile(code, "<answer>", "exec")
                return _ok({"mode": "syntax-only", "executed": False})
            except SyntaxError as exc:
                return _fail([f"syntax error: {exc}"], {"executed": False})
        with tempfile.TemporaryDirectory() as tmp:
            sol = Path(tmp) / "solution.py"
            sol.write_text(code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(sol)], cwd=tmp, text=True,
                    capture_output=True, timeout=timeout_sec, check=False,
                )
            except subprocess.TimeoutExpired:
                return _fail([f"code timed out after {timeout_sec}s"], {"executed": True})
        if proc.returncode == 0:
            return _ok({"executed": True, "returncode": 0})
        return _fail([f"code exited {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"],
                     {"executed": True, "returncode": proc.returncode})

    return _verify


_ARITH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×x])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def arithmetic_sound(*, tol: float = 1e-6) -> Verifier:
    """Fail if the text states a FALSE arithmetic equality (e.g. '2 + 2 = 5').

    Parses ``a OP b = c`` for + - * / (and × / x), recomputes, and flags any
    mismatch. No model, no eval of arbitrary expressions — just the stated
    binary equalities. If the text contains no checkable arithmetic it passes
    (this is a soundness check, not a presence requirement).
    """

    def _verify(text: str, task: Any, step: dict) -> dict:
        wrong: list[str] = []
        checked = 0
        for a, op, b, c in _ARITH.findall(text):
            x, y, z = float(a), float(b), float(c)
            if op in ("*", "×", "x"):
                got = x * y
            elif op == "/":
                if y == 0:
                    continue
                got = x / y
            elif op == "+":
                got = x + y
            else:
                got = x - y
            checked += 1
            if abs(got - z) > tol:
                wrong.append(f"{a} {op} {b} = {c} (actual {got:g})")
        if wrong:
            return _fail([f"false arithmetic: {w}" for w in wrong], {"wrong": wrong, "checked": checked})
        return _ok({"checked": checked})

    return _verify


def no_secret_leak(secrets: "list[str]", *, mask: bool = True) -> Verifier:
    """Fail if any classified/canary value appears verbatim in the text.

    A deterministic confidentiality tripwire: exact-substring match against a list
    of secret values/canaries — no model, no heuristics, so "blocked" means
    blocked. This is the minimal egress check the injection red-team motivates.

    Scope (honest): it catches *verbatim* leakage only. It does NOT understand
    paraphrase, partial disclosure, or encoding — which is exactly why the roadmap
    pairs it with out-of-prompt taint tracking (CaMeL-style). Use it as a tripwire
    in depth, not as a standalone guarantee. The detail never echoes the secret.
    """
    toks = [s for s in (secrets or []) if s]

    def _verify(text: str, task: Any, step: dict) -> dict:
        body = text or ""
        leaked = [s for s in toks if s in body]
        if leaked:
            shown = [(s[:4] + "…") if mask else s for s in leaked]
            return _fail([f"secret value leaked: {m}" for m in shown], {"leakedCount": len(leaked)})
        return _ok({"secretsChecked": len(toks)})

    return _verify


def legal_citation_exists(
    known_authorities: "set | None" = None,
    *,
    require_citation: bool = False,
    resolver: "Callable[[str], Any] | None" = None,
) -> Verifier:
    """Fail if the answer cites a legal authority that is NOT in a trusted source.

    This is the *Mata v. Avianca* / *Ayinde v Haringey* killer: every common-law
    **neutral citation** (``[2025] HKCFI 808``) and HK **ordinance** ref
    (``Cap. 614``) in the text is checked against a register of authorities that
    actually exist. An unrecognized citation is treated as fabricated and fails —
    **fail-closed**, so an empty/missing register flags every citation rather than
    waving fabrications through.

    A citation absent from the static register is given a second chance via the
    optional ``resolver`` (``agent.legal_sources.make_resolver``) — a cache-first
    HKLII / e-Legislation lookup that returns a ``Resolution``; only a
    ``verified=True`` result clears it. Still fail-closed: any non-verified
    resolution (not found / offline / network error) remains a violation.

    With ``require_citation`` an answer that makes a case-law-style assertion
    ("the court held ...") while citing nothing is also flagged.

    Scope (honest, mirrors the legal-AI literature): this verifies *existence*
    only. It does NOT confirm the authority is in the right jurisdiction, that the
    holding supports the proposition, or that the law is current — the three-part
    check a human must still perform. Populate the register from an authoritative
    primary source via ``SOPHIA_LEGAL_AUTHORITIES`` / the live resolver; the
    bundled snapshot is illustrative only.
    """
    from agent.legal_citations import extract_citations, load_known_authorities

    known = known_authorities if known_authorities is not None else load_known_authorities()

    def _resolves(citation: str) -> bool:
        if resolver is None:
            return False
        try:
            res = resolver(citation)
        except Exception:  # fail-closed: a broken resolver never passes a citation
            return False
        return bool(getattr(res, "verified", False) or (isinstance(res, dict) and res.get("verified")))

    def _verify(text: str, task: Any, step: dict) -> dict:
        cites = extract_citations(text or "")
        violations = [c for c in cites if c not in known and not _resolves(c)]
        if require_citation and not cites and (
            re.search(r"\b(?:the court (?:held|found|ruled)|it was held)\b", text or "", re.IGNORECASE)
            or re.search(r"[A-Z][a-z]+ v\.? [A-Z]", text or "")
        ):
            violations.append("legal assertion with no citation")
        if violations:
            return _fail(
                [f"unverified/fabricated legal citation: {v}" for v in violations],
                {"violations": violations, "checked": len(cites), "registerSize": len(known)},
            )
        return _ok({"checked": len(cites), "registerSize": len(known)})

    return _verify


def legal_holding_faithful(*, holding_for=None, judge=None, require_support: bool = False) -> Verifier:
    """Fail if a cited authority is MISSTATED — its holding does not support the
    proposition it is cited for (the *Ayinde* misstated-authority failure).

    This is the semantic tier above ``legal_citation_exists``: it judges support,
    not mere existence, so it needs an LLM ``judge`` (``agent.legal_faithfulness``).
    Fail-closed and abstaining — a pair with no authoritative holding text, or that
    the judge cannot decide, is reported as **abstained** (unchecked), never passed.
    A ``contradicted`` verdict (real authority, wrong proposition) is the violation.

    By default ``passed`` is True iff nothing is contradicted (it flags affirmative
    misuse and surfaces what it could not check). With ``require_support`` an
    abstention is also a hard fail (full fail-closed).

    Honest scope: the support judgment is a model call — measure it under the
    no-overclaim gate (≥2 independent judges + CIs); a single judge is illustrative,
    never a headline. Tests inject a deterministic stub judge to verify wiring only.
    """
    from agent.legal_faithfulness import assess_text

    def _verify(text: str, task: Any, step: dict) -> dict:
        result = assess_text(text, holding_for=holding_for, judge=judge)
        contradicted = result["contradicted"]
        abstained = result["abstained"]
        reasons = [f"misstated authority {c['citation']}: {c.get('reason', '')}".strip() for c in contradicted]
        if require_support:
            reasons += [f"unverifiable support for {a['citation']}: {a.get('why', '')}".strip() for a in abstained]
        detail = {
            "supported": result["supported"],
            "contradicted": contradicted,
            "abstained": abstained,
        }
        return _ok(detail) if not reasons else _fail(reasons, detail)

    return _verify


# --------------------------------------------------------------------------- #
# OKF / provenance verifiers — encode "don't merge lineages" as a hard gate.
# --------------------------------------------------------------------------- #


# Env var letting a user enforce their OWN attribution rules (legal/corporate/code
# provenance) through the same machine-checked gate, beyond the seeded domains.
# Value is a directory (its *.json), a glob, a single file, or several of these
# joined by the OS path separator.
_DISCIPLINE_RECORDS_ENV = "SOPHIA_DISCIPLINE_RECORDS"


def _merge_records(into: dict, data: object, *, source: str, warn_skips: bool = False) -> None:
    """Merge a ``{key: record}`` mapping into ``into``, keeping only records that
    carry a non-empty ``doNotAttributeTo`` list (the one field this gate enforces).
    With ``warn_skips`` (user-supplied files), emit a warning for each skip so a
    malformed record is visible rather than silently ignored."""
    if not isinstance(data, dict):
        warnings.warn(f"discipline records in {source} are not a JSON object; skipped")
        return
    for key, record in data.items():
        if isinstance(record, dict) and record.get("doNotAttributeTo"):
            into[record.get("recordId") or record.get("textId") or key] = record
        elif warn_skips:
            warnings.warn(f"{source}: record '{key}' has no doNotAttributeTo; skipped")


def _user_record_paths() -> list[Path]:
    """Resolve ``SOPHIA_DISCIPLINE_RECORDS`` to a list of JSON files. Each entry
    may be a directory (its ``*.json``), a glob, or a single file."""
    spec = os.environ.get(_DISCIPLINE_RECORDS_ENV, "").strip()
    if not spec:
        return []
    paths: list[Path] = []
    for part in spec.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif p.exists():
            paths.append(p)
        else:
            matched = sorted(Path(m) for m in _glob(os.path.expanduser(part)))
            if matched:
                paths.extend(matched)
            else:
                warnings.warn(f"{_DISCIPLINE_RECORDS_ENV} path not found: {part}")
    return paths


def _load_provenance_records() -> dict:
    records: dict = {}
    for filename in _PROVENANCE_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            continue
        _merge_records(records, json.loads(path.read_text(encoding="utf-8")), source=str(path))
    for path in _user_record_paths():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"could not load discipline records from {path}: {exc}")
            continue
        _merge_records(records, data, source=str(path), warn_skips=True)
    return records


# Clause boundaries for SCOPING the negation/correction carve-out. We split a
# sentence into clauses on contrastive connectors and a leading subordinate clause,
# but NOT on bare commas — commas hold appositives the gate must keep matching
# ("Enoch, the great-grandson of Adam, wrote ..."). This closes the negation-
# evasion the red-team found: a correction clause ("it is a myth, but ...") can no
# longer shield an assertion clause ("... Confucius wrote the Dao De Jing").
_CLAUSE_SPLIT = re.compile(
    r"\bbut\b|\bhowever\b|\byet\b|\bnevertheless\b|\bnonetheless\b|\bin truth\b|"
    r"\bin fact\b|\bin reality\b|\bactually\b|\bindeed\b|\brather\b|\binstead\b|"
    r"\btruth is\b|;|—|–"
)
_LEAD_SUBORD = re.compile(
    r"^\s*(?:contrary to|despite|although|though|while|whereas|unlike|far from|"
    r"notwithstanding|regardless of)\b[^,]{0,100},"
)


def _carveout_clauses(low: str) -> list:
    """Split a lowercased sentence into carve-out scopes: peel a leading
    subordinate clause (``Contrary to ... that he did not, X wrote Y``) then split
    on contrastive connectors. Commas inside the body are preserved so appositive
    author→title patterns still match."""
    head: list = []
    m = _LEAD_SUBORD.match(low)
    if m:
        head = [low[: m.end()]]
        low = low[m.end():]
    return [c for c in (head + _CLAUSE_SPLIT.split(low)) if c and c.strip()]


def provenance_faithful(records: "dict | None" = None) -> Verifier:
    """Fail if the text asserts an attribution forbidden by a record's
    doNotAttributeTo — Sophia's core "don't merge lineages" rule, machine-checked.

    Sentence-scoped with a negation/contrast carve-out (reusing the benchmark
    DENY/MYTH markers), so a page that CORRECTLY says "Confucius did not write the
    Dao De Jing" passes while "Confucius wrote the Dao De Jing" fails. Works on
    agent answers and on wiki page bodies alike.
    """
    from agent.benchmark_checks import DENY_PATTERNS, MYTH_PATTERNS, author_markers, matches_any

    records = records if records is not None else _load_provenance_records()
    the = r"(?:the\s+)?"
    q = r"[\"“”'’«»]?"                                     # an optional opening/closing quote
    the_q = q + r"\s*" + the + q + r"\s*"                  # quote/"the"-tolerant lead-in to a title
    attr = r"(?:wrote|authored|penned|composed)"          # active
    attr_p = r"(?:written|authored|penned|composed)"      # passive participle
    # Bounded honorific/article lead-in to a name ("to the prophet Daniel").
    nm = r"(?:(?:the|a|an|prophet|apostle|king|saint|st\.?|emperor|biblical|tyrant)\s+){0,3}"
    # Optional bounded appositive/parenthetical between an author and the verb:
    #   "Enoch, the great-grandson of Adam, wrote ...", "Lie Yukou (also known as
    #   Liezi) wrote ...". Length/punctuation-bounded and contrast-free (no "not"/
    #   "but") so it never bridges a correction.
    app = (
        r"(?:\s*\([^)]{0,50}\)"                                        # ( ... )
        r"|\s*,(?:(?!,|\bnot\b|\bbut\b)[^.;]){1,60},"                  # , ... ,
        r"|\s+(?:the|a|an)\s+(?!not\b|but\b)(?:[\w'’-]+\s+){1,5})?"    # the X of Y
    )
    # Skip non-assertions: instructions ("do not attribute X to Y"), reported/hedged
    # speech ("summaries say ...", "often said"), scare-quoted verbs ("wrote"), and
    # scholarly hedges that *correctly* flag a traditional/spurious attribution.
    extra_deny = [
        r"do not attribut", r"not\s+\w*\s*attribut", r"never\s+\w*\s*attribut",
        r"misattribut", r"wrongly", r"falsely", r"erroneous", r"並非", r"勿",
        r"summaries say", r'say[s]?\s+["“\']', r"often\s+said", r"commonly\s+(said|believed|attributed)",
        r"\bpopular", r"\bmisread", r"\bmistaken", r"\bloosely\b", r"\bas if\b", r"main speaker",
        r'["“\'](?:wrote|written|authored|composed|penned|attributed)["”\']',
        r"traditionally", r"\bspuri", r"pseudo[\s-]?", r"apocryphal", r"\bforg(?:ed|ery)",
        r"\bdisputed\b", r"\bdoubtful\b", r"\bdebated\b", r"scholars?\s+(?:doubt|reject|dispute|question)",
    ]

    specs: list[tuple] = []
    for rid, record in records.items():
        titles = [
            str(t).lower()
            for t in (
                record.get("canonicalTitleEn"),
                record.get("canonicalTitleZh"),
                rid.replace("_", " "),
                *(record.get("altTitlesEn") or []),
            )
            if t and len(str(t)) >= 3
        ]
        if not titles:
            continue
        # longest titles first so "constitution of the athenians" wins over a substring
        titles = sorted(dict.fromkeys(titles), key=len, reverse=True)
        title_alt = "(?:" + "|".join(re.escape(t) for t in titles) + ")"
        for author in record.get("doNotAttributeTo", []):
            a = "(?:" + "|".join(re.escape(m.lower()) for m in author_markers(author)) + r")\b"
            t = title_alt + r"\b"
            # Explicit grammatical constructions of *asserted authorship*, in both
            # orders, with tight connectors only (no wildcard gap) so a comparison
            # ("Plato wrote Republic — not Socrates") or a possessive of a different
            # noun ("from Epictetus's teachings") does not cross-match.
            patterns = [
                re.compile(a + app + r"\s*" + attr + r"\s+" + the_q + t),                                # X (the …) wrote (the) "Y"
                re.compile(a + r"(?:\s+(?:is|was|being))?\s+" + the + r"(?:author|writer|composer)\s+of\s+" + the_q + t),  # X is the author of Y
                re.compile(a + app + r"\s*(?:is|was)\s+credited\s+with\s+\w+ing\s+" + the_q + t),        # X (the …) is credited with writing Y
                re.compile(a + r"['’]s\s+" + the_q + t),                                                 # X's (the) Y
                re.compile(t + r"(?:\s*,)?(?:\s+(?:was|is|were|been))?\s+(?:" + attr_p + r"|attributed)\s+by\s+" + nm + a),  # Y (was) written by X
                re.compile(t + r"(?:\s*,)?\s+(?:is|was|are|were|been|being)?\s*attributed\s+to\s+" + nm + a),  # Y is attributed to (the prophet) X
                re.compile(t + r"(?:\s*,)?\s+(?:a\s+|the\s+)?(?:work|text|book|composition|treatise)\s+(?:by|of)\s+" + a),  # Y, a work by X
                re.compile(t + r"\s+(?:was\s+|is\s+)?" + a + r"['’]s\s+(?:\w+\s+)?(?:work|text|masterpiece|composition|book|treatise)\b"),  # Y was X's work
                re.compile(a + r"\s*著\s*" + title_alt),                                                 # X 著 Y
            ]
            specs.append((rid, author, patterns))

    def _verify(text: str, task: Any, step: dict) -> dict:
        violations: list[str] = []
        for sentence in re.split(r"[.!?。！？\n]+", text):
            low = sentence.lower()
            if not low.strip():
                continue
            # Carve-out is CLAUSE-scoped: a correction/negation only excuses the
            # clause it lives in, so it cannot shield an asserting clause in the
            # same sentence (the negation-evasion the red-team surfaced).
            for clause in _carveout_clauses(low):
                if matches_any(clause, DENY_PATTERNS) or matches_any(clause, MYTH_PATTERNS) or matches_any(clause, extra_deny):
                    continue  # the correction/negation/instruction case — allowed
                for rid, author, patterns in specs:
                    if any(p.search(clause) for p in patterns):
                        violations.append(f"{author} -> {rid}")
        violations = sorted(set(violations))
        if violations:
            return _fail([f"forbidden attribution asserted: {v}" for v in violations], {"violations": violations})
        return _ok({"recordsChecked": len(records)})

    return _verify


# Alias used in the brainstorm/docs.
source_discipline = provenance_faithful


def frontmatter_schema_valid() -> Verifier:
    """Pass if the text parses as an OKF page with schema-valid frontmatter."""

    def _verify(text: str, task: Any, step: dict) -> dict:
        from okf import frontmatter, schema

        meta, _ = frontmatter.parse(text)
        if not meta:
            return _fail(["no OKF frontmatter found"])
        errors = schema.validate_meta(meta)
        return _ok({"id": meta.get("id")}) if not errors else _fail(errors, {"frontmatter": meta})

    return _verify


def no_broken_wikilink(known_ids: list[str]) -> Verifier:
    """Pass if every [[wikilink]] in the text resolves to a known page id."""
    from okf import wikilinks

    known = {wikilinks.normalize_target(k) for k in known_ids}

    def _verify(text: str, task: Any, step: dict) -> dict:
        broken = [t for t in wikilinks.extract_links(text) if t not in known]
        return _ok() if not broken else _fail([f"broken wikilink: [[{t}]]" for t in sorted(set(broken))])

    return _verify


def wiki_consistent(roots: "list | None" = None) -> Verifier:
    """World-state verifier: the OKF wiki must be a clean provenance graph.

    Ignores `text` (like unit_test) — it grades the repository, not the answer.
    """

    def _verify(text: str, task: Any, step: dict) -> dict:
        from okf import linker

        paths = roots or [ROOT / "wiki", ROOT / "docs" / "04-Disputes"]
        existing = [p for p in paths if Path(p).exists()]
        report = linker.link_report(*existing)
        if report["ok"]:
            return _ok({"pages": report["pages"]})
        reasons = (
            [f"schema {e['page']}" for e in report["schemaErrors"]]
            + [f"dangling {d['page']}->[[{d['target']}]]" for d in report["danglingLinks"]]
            + [f"lineage-merge {m['page']}" for m in report["contradictions"]["selfMerges"]]
        )
        return _fail(reasons or ["wiki inconsistent"], {"pages": report["pages"]})

    return _verify


def all_of(*verifiers: Verifier) -> Verifier:
    def _verify(text: str, task: Any, step: dict) -> dict:
        reasons: list[str] = []
        detail: dict[str, Any] = {}
        passed = True
        for i, v in enumerate(verifiers):
            r = v(text, task, step)
            detail[f"v{i}"] = r.get("detail", {})
            if not r["passed"]:
                passed = False
                reasons.extend(r["reasons"])
        return {"passed": passed, "reasons": reasons, "detail": detail}

    return _verify


def any_of(*verifiers: Verifier) -> Verifier:
    def _verify(text: str, task: Any, step: dict) -> dict:
        all_reasons: list[str] = []
        for v in verifiers:
            r = v(text, task, step)
            if r["passed"]:
                return _ok(r.get("detail", {}))
            all_reasons.extend(r["reasons"])
        return _fail(["none of the alternatives passed", *all_reasons])

    return _verify


def _temporal_consistent() -> Verifier:
    """Lazy wrapper so the registry stays import-cheap (loads the dated-facts table
    only when used). Catches author-died-before-work impossibilities. See
    agent/temporal_verifier.py."""
    from agent.temporal_verifier import temporal_consistent

    return temporal_consistent()


# Parameterless verifiers usable directly by name (CLI / harness registry).
VERIFIERS: dict[str, Callable[[], Verifier]] = {
    "arithmetic_sound": arithmetic_sound,
    "code_tests_pass": code_tests_pass,
    "provenance_faithful": provenance_faithful,
    "temporal_consistent": _temporal_consistent,
    "frontmatter_schema_valid": frontmatter_schema_valid,
    "legal_citation_exists": legal_citation_exists,
}


def check_text(name: str, text: str) -> dict:
    """Run a registered parameterless verifier over ``text``. CLI/MCP surface."""
    if name not in VERIFIERS:
        raise KeyError(f"unknown verifier {name!r}; known: {', '.join(sorted(VERIFIERS))}")
    return VERIFIERS[name]()(text, None, {})
