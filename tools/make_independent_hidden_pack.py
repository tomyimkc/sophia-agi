#!/usr/bin/env python3
"""
make_independent_hidden_pack.py — build a schema-valid, decontaminated, THIRD-PARTY
hidden-evaluation pack for tools/run_hidden_eval_sophia.py.

WHAT IT DOES
  Takes a reviewer-authored input file (JSON or YAML) of (id, domain, prompt, rubric,
  mustInclude/mustAvoid, ...) items across >=4 domains, and:
    1. validates every item against agi-proof/hidden-reviewer-packs/schema.json,
    2. runs a decontamination guard against the training corpus / wiki (shingle-Jaccard),
       refusing any item that overlaps known training text,
    3. stamps provenance reviewer.status = "third-party" + a salt,
    4. writes a schema-valid pack.json + a .checksums.sha256 sidecar.

  FAIL-CLOSED: if schema validation OR decontamination fails for ANY case, the tool
  REFUSES to write the pack (exit non-zero). It never emits a partial or tainted pack,
  and it never downgrades a contaminated item to "pass".

HONEST BOUND
  This tool produces the *instrument* for an independent hidden eval; it does NOT run the
  eval and produces NO scores. Independence is only real if the (question, rubric) items
  are authored by someone other than the model/agent under test — the tool enforces the
  provenance STAMP and the decontam FLOOR, not the sociology of who typed the questions.
  candidateOnly:true, level3Evidence:false, canClaimAGI:false.

USAGE
  python3 tools/make_independent_hidden_pack.py \
      --input examples/reviewer_pack_input.json \
      --schema agi-proof/hidden-reviewer-packs/schema.json \
      --corpus wiki/ \
      --out agi-proof/hidden-reviewer-packs/pack-<reviewer>-<date>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN_DOMAINS = 4
# Jaccard over word-5-shingles; >= this against any corpus doc => contaminated.
DECONTAM_JACCARD_THRESHOLD = 0.20
SHINGLE_N = 5

VALID_DOMAINS = {
    "philosophy", "psychology", "history", "logic",
    "coding", "planning", "tool_use", "learning",
}


# ---------------------------------------------------------------- input load
def _load_any(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # opt-in; abstain cleanly if absent
        except ImportError:
            sys.exit(
                "FAIL-CLOSED: PyYAML not installed but a YAML input was given. "
                "Install pyyaml or provide JSON input."
            )
        return yaml.safe_load(text)
    return json.loads(text)


# ---------------------------------------------------------------- runner-contract check
def _validate_pack(pack: dict[str, Any]) -> list[str]:
    """Validate against the RUNNER'S OWN contract, not schema.json.

    Review D3: tools/run_hidden_eval_sophia.py never loads schema.json — it calls
    tools.hidden_eval_protocol.validate_pack(), a hand-written validator that
    enforces rules JSON Schema cannot express (id-uniqueness, domain coupling,
    maxPoints>0). A schema-derived check diverges in BOTH directions (accepts
    duplicate ids / maxPoints<=0; rejects createdAt-less packs the runner accepts).
    So we bind to the runner's validator directly: a pack that passes here passes
    the runner by construction. If the runner isn't importable (running outside the
    tree), we FAIL CLOSED rather than fall back to a divergent structural check.
    """
    try:
        from tools.hidden_eval_protocol import validate_pack as runner_validate
    except Exception as e:
        return [f"cannot import the runner's validate_pack ({type(e).__name__}: {e}); "
                "run inside the sophia-agi tree with PYTHONPATH=. so validation matches "
                "the real contract (fail-closed — no divergent fallback)"]
    return list(runner_validate(pack))


# ---------------------------------------------------------------- decontam guard
def _shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _iter_corpus_texts(corpus_dir: Path):
    """Yield text of every corpus doc. Best-effort over .md/.json/.txt/.jsonl."""
    for p in sorted(corpus_dir.rglob("*")):
        if p.suffix.lower() in {".md", ".txt"}:
            yield p.name, p.read_text(encoding="utf-8", errors="replace")
        elif p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                yield p.name, json.dumps(obj)
            except Exception:
                continue
        elif p.suffix.lower() == ".jsonl":
            yield p.name, p.read_text(encoding="utf-8", errors="replace")


def decontam_check(pack: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Return a list of contamination findings (empty => CLEAN).

    NOTE: if the corpus dir is missing/empty we FAIL CLOSED — we cannot prove a
    pack is decontaminated against a corpus we cannot read, so we refuse rather
    than silently pass.
    """
    if not corpus_dir.exists():
        return [f"corpus dir {corpus_dir} does not exist — cannot prove decontamination (fail-closed)"]
    corpus = list(_iter_corpus_texts(corpus_dir))
    if not corpus:
        return [f"corpus dir {corpus_dir} is empty — cannot prove decontamination (fail-closed)"]

    corpus_shingles = [(name, _shingles(text)) for name, text in corpus]
    findings: list[str] = []
    for c in pack["cases"]:
        probe = _shingles(c.get("prompt", "") + " " + " ".join(
            (m if isinstance(m, str) else m.get("match", ""))
            for m in c.get("scoring", {}).get("mustInclude", [])
        ))
        worst = 0.0
        worst_doc = ""
        for name, sh in corpus_shingles:
            j = _jaccard(probe, sh)
            if j > worst:
                worst, worst_doc = j, name
        if worst >= DECONTAM_JACCARD_THRESHOLD:
            findings.append(
                f"case '{c['id']}' contaminated: Jaccard {worst:.3f} vs {worst_doc} "
                f"(>= {DECONTAM_JACCARD_THRESHOLD})"
            )
    return findings


# ---------------------------------------------------------------- build
def build_pack(items: list[dict[str, Any]], *, reviewer: str, salt: str | None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    salt = salt or hashlib.sha256(f"{reviewer}{now}".encode()).hexdigest()[:16]
    return {
        "packId": f"independent-{reviewer}-{now[:10]}",
        "createdAt": now,
        "visibility": "private-hidden",
        "salt": salt,
        "reviewer": {
            "status": "third-party",
            "note": f"Authored by {reviewer}; provenance stamped by make_independent_hidden_pack.py. "
                    "Independence is a claim about authorship, verified out-of-band by signature.",
        },
        "cases": items,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="reviewer-authored items (JSON or YAML)")
    ap.add_argument("--schema", default=None,
                    help="(deprecated / ignored) validation now binds to the runner's "
                         "hidden_eval_protocol.validate_pack, not schema.json — see review D3")
    ap.add_argument("--corpus", required=True, help="training corpus / wiki dir for decontam")
    ap.add_argument("--reviewer", default="anon-reviewer", help="reviewer id to stamp")
    ap.add_argument("--salt", default=None)
    ap.add_argument("--out", required=True, help="output pack path")
    args = ap.parse_args()

    raw = _load_any(Path(args.input))
    items = raw["cases"] if isinstance(raw, dict) and "cases" in raw else raw
    if not isinstance(items, list) or not items:
        sys.exit("FAIL-CLOSED: input has no cases")

    domains = {i.get("domain") for i in items}
    if len(domains & VALID_DOMAINS) < MIN_DOMAINS:
        sys.exit(
            f"FAIL-CLOSED: pack spans {len(domains & VALID_DOMAINS)} valid domains "
            f"({sorted(domains)}); need >= {MIN_DOMAINS}."
        )

    pack = build_pack(items, reviewer=args.reviewer, salt=args.salt)

    schema_errors = _validate_pack(pack)
    if schema_errors:
        print("FAIL-CLOSED: runner-contract validation failed:", file=sys.stderr)
        for e in schema_errors:
            print("  -", e, file=sys.stderr)
        return 2

    contam = decontam_check(pack, Path(args.corpus))
    if contam:
        print("FAIL-CLOSED: decontamination failed (pack NOT written):", file=sys.stderr)
        for e in contam:
            print("  -", e, file=sys.stderr)
        return 3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(pack, indent=2, ensure_ascii=False)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    Path(str(out) + ".checksums.sha256").write_text(
        f"{digest}  {out.name}\n", encoding="utf-8"
    )
    print(f"OK: wrote {out} ({len(pack['cases'])} cases, {len(domains & VALID_DOMAINS)} domains)")
    print(f"    sha256 {digest}")
    print(f"    decontam CLEAN vs {args.corpus}; provenance third-party ({args.reviewer})")
    print("    candidateOnly:true level3Evidence:false canClaimAGI:false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
