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
# Domain files whose records carry doNotAttributeTo lists.
_PROVENANCE_FILES = (
    "attributions.json",
    "psychology_concepts.json",
    "religion_concepts.json",
    "history_events.json",
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
            if matches_any(low, DENY_PATTERNS) or matches_any(low, MYTH_PATTERNS) or matches_any(low, extra_deny):
                continue  # the correction/negation/instruction case — allowed
            for rid, author, patterns in specs:
                if any(p.search(low) for p in patterns):
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
