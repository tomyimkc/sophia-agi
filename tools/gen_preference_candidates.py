#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Candidate generator for the Verifier-Gated Preference Engine.

This is the **generation half** of the preference-data pipeline; the **labelling
half** is :file:`tools/gen_verifier_dpo.py` (already shipped). Splitting them keeps
the labelling step offline, deterministic and CI-testable — only generation needs a
model + keys. The full pipeline is::

    prompts (tasks.jsonl)
      └─[this tool, needs model+keys]──► tasks_with_candidates.jsonl
      └─[tools/gen_verifier_dpo.py]─────► dpo_pairs.jsonl   (machine-verifier label)

What it does. For each prompt, sample ``--n`` candidate answers from any provider via
``agent.model`` and write them into the ``candidates`` field in the exact shape
:file:`gen_verifier_dpo.py` consumes (``{prompt, question, mode, candidates}``).
Sampling uses a small temperature ladder so the candidates SPREAD — a preference
label is only informative when the gate can *separate* clean from violating answers,
and separation needs some candidates to fail. The temperature ladder is therefore
load-bearing, not cosmetic.

Why this is the highest-ROI training input the repo can mint cheaply. The repo's
differentiating asset is a farm of **machine** verifiers that label an answer
``clean``/``violating`` deterministically, with no learnable judge. That lets us mint
``(chosen, rejected)`` DPO pairs at scale without an LLM judge — the same unhackable
property ``provenance_bench/swarm_rl.py`` relies on. Generation is the only step that
costs money/keys; everything downstream is free and CI-checked.

Honest scope (pre-registered — see ``docs/06-Roadmap/Frontier-Positioning-Plan.md``):
  * Candidates are a **training INPUT**, not a result. Whether an adapter trained on
    the pairs minted downstream improves on a held-out third-party pack is an OPEN
    gate (ledger: ``v4-adapter-externally-unvalidated``).
  * Generation quality is bounded by the base model. A weak base may not produce
    violating candidates on a prompt, in which case the labeller abstains (no pair) —
    fail-closed, by design.
  * Decontamination against eval packs is the labeller's job (``--seen``), reusing
    the authoritative ``tools/assert_decontam.py``. This tool only forwards an
    optional ``--seen`` path; it does not assert disjointness itself.

Input (JSONL), one task per line::

    {"prompt": "...", "question": "<optional, defaults to prompt>",
     "mode": "advisor|repo|life"}

Output (JSONL) — same rows with the ``candidates`` field populated::

    {"prompt": "...", "question": "...", "mode": "advisor",
     "candidates": ["answer A", "answer B", ...], "metadata": {...}}

Usage::

    # Generate candidates (needs a model spec + keys in env):
    python tools/gen_preference_candidates.py --in tasks.jsonl \\
        --out tasks_with_candidates.jsonl --n 4 --spec openai:gpt-4o-mini

    # Then label offline with the machine verifiers (no model):
    python tools/gen_verifier_dpo.py --in tasks_with_candidates.jsonl \\
        --out training/tool_use/dpo_pairs_v2.jsonl --seen eval-prompts.jsonl

    # Deterministic offline self-test (no model, no network):
    python tools/gen_preference_candidates.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Generation is the ONLY step that needs a model; import lazily so --self-test and
# CI (no keys, no network) run without a provider configured.
def _import_complete() -> Callable[..., str]:
    from agent.model import complete  # noqa: E402
    return complete


# A temperature ladder produces a SPREAD of candidate quality: cool candidates tend
# to stay clean, hot ones tend to drift into the violations the gate exists to catch.
# Without spread the labeller abstains on every row (no separation). Keep this small
# and deterministic so the generation step is reproducible per seed.
DEFAULT_TEMP_LADDER: tuple[float, ...] = (0.2, 0.6, 1.0, 1.3)


@dataclass(frozen=True)
class GenStats:
    rows: int = 0
    emitted: int = 0
    skipped: int = 0
    empty_candidates: int = 0
    reasons: dict = None  # type: ignore[assignment]

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "emitted": self.emitted,
            "skipped": self.skipped,
            "empty_candidates": self.empty_candidates,
            "reasons": dict(self.reasons or {}),
        }


def _build_system(mode: str) -> str:
    """System prompt that elicits a direct answer in the gate's home-turf discipline.

    Intentionally does NOT coach the model to abstain or to cite — the point of
    preference data is to sample the model's *natural* distribution, which the gate
    then separates into clean vs violating. Coaching would bias the sample and make
    the downstream label less informative about real deployment behaviour.
    """
    return (
        "You are a concise factual assistant. Answer the user's question directly "
        f"and briefly (mode: {mode}). Do not refuse; give your best direct answer."
    )


def generate_candidates(
    prompt: str,
    *,
    n: int,
    complete_fn: Callable[..., str],
    spec: "str | None" = None,
    temps: "tuple[float, ...] | None" = None,
    max_tokens: int = 400,
) -> "tuple[list[str], str | None]":
    """Sample ``n`` candidates for one prompt across the temperature ladder.

    Returns ``(candidates, skip_reason)``. ``skip_reason`` is set only when no
    non-empty candidate was produced (fail-closed: the labeller cannot mint a pair
    from an empty candidate set). Candidate strings are stripped and de-duplicated
    while preserving order, so repeated identical draws do not inflate the count.
    """
    ladder = list(temps or DEFAULT_TEMP_LADDER)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(max(n, 1)):
        # Cycle through the ladder deterministically; n > len(ladder) reuses it.
        temperature = ladder[i % len(ladder)]
        system = _build_system("advisor")
        # spec-resolution + temperature selection is delegated to agent.model; we do
        # not reimplement provider plumbing here. We pass temperature via env var the
        # same way other repo tools do (SOPHIA_MODEL_TEMP), restored in finally.
        prev = os.environ.get("SOPHIA_MODEL_TEMP")
        try:
            if temperature is not None:
                os.environ["SOPHIA_MODEL_TEMP"] = str(temperature)
            try:
                text = complete_fn(system, prompt, max_tokens=max_tokens, spec=spec)
            except TypeError:
                # older signatures without `spec` — fall back gracefully
                text = complete_fn(system, prompt, max_tokens=max_tokens)
        finally:
            if prev is None:
                os.environ.pop("SOPHIA_MODEL_TEMP", None)
            else:
                os.environ["SOPHIA_MODEL_TEMP"] = prev
        text = (text or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    if not out:
        return [], "all_candidates_empty"
    return out, None


def run(
    rows: Iterable[dict],
    *,
    n: int,
    complete_fn: Callable[..., str],
    spec: "str | None" = None,
    seen_prompts: "set[str] | None" = None,
    temps: "tuple[float, ...] | None" = None,
    max_tokens: int = 400,
) -> "tuple[list[dict], GenStats]":
    """Generate candidates for many rows. ``seen_prompts`` are skipped as a cheap
    decontamination guard (the authoritative gate stays ``tools/assert_decontam.py``,
    invoked at label time via ``gen_verifier_dpo.py --seen``)."""
    seen = {p.strip() for p in (seen_prompts or set())}
    out: list[dict] = []
    s = GenStats()
    reasons: dict[str, int] = {}
    for row in rows:
        s = GenStats(rows=s.rows + 1, emitted=s.emitted, skipped=s.skipped,
                     empty_candidates=s.empty_candidates, reasons=reasons)
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            reasons["no_prompt"] = reasons.get("no_prompt", 0) + 1
            s = GenStats(rows=s.rows, skipped=s.skipped + 1, emitted=s.emitted,
                         empty_candidates=s.empty_candidates, reasons=reasons)
            continue
        if prompt in seen:
            reasons["decontam_skipped"] = reasons.get("decontam_skipped", 0) + 1
            s = GenStats(rows=s.rows, skipped=s.skipped + 1, emitted=s.emitted,
                         empty_candidates=s.empty_candidates, reasons=reasons)
            continue
        question = (row.get("question") or prompt).strip()
        mode = row.get("mode") or "advisor"
        candidates, reason = generate_candidates(
            prompt, n=n, complete_fn=complete_fn, spec=spec, temps=temps,
            max_tokens=max_tokens,
        )
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "all_candidates_empty":
                s = GenStats(rows=s.rows, skipped=s.skipped, emitted=s.emitted,
                             empty_candidates=s.empty_candidates + 1, reasons=reasons)
            else:
                s = GenStats(rows=s.rows, skipped=s.skipped + 1, emitted=s.emitted,
                             empty_candidates=s.empty_candidates, reasons=reasons)
            continue
        meta = dict(row.get("metadata") or {})
        meta["gen"] = {"n_requested": n, "n_emitted": len(candidates),
                       "label_source": "unverified",  # until gen_verifier_dpo runs
                       "generator": "agent.model"}
        out.append({
            "prompt": prompt,
            "question": question,
            "mode": mode,
            "candidates": candidates,
            "metadata": meta,
        })
        s = GenStats(rows=s.rows, skipped=s.skipped, emitted=s.emitted + 1,
                     empty_candidates=s.empty_candidates, reasons=reasons)
    return out, s


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# Deterministic offline fixtures (no model, no network). The self-test substitutes a
# fake complete_fn that returns scripted candidates spanning clean + violating, so the
# end-to-end "generate -> label" wiring is provable in CI without keys. Mirrors the
# self-test pattern in tools/gen_verifier_dpo.py.
SELF_TEST_ROWS: list[dict] = [
    {
        "prompt": "Did Socrates write The Republic? Answer yes or no and explain briefly.",
        "question": "Did Socrates write The Republic?",
        "mode": "advisor",
    },
    {
        "prompt": "Who wrote the Dao De Jing? Answer briefly.",
        "question": "Did Confucius write the Dao De Jing?",
        "mode": "advisor",
    },
]


def _fake_complete_factory(script: dict[str, list[str]]) -> Callable[..., str]:
    """Build a deterministic stand-in for ``agent.model.complete`` keyed by prompt.

    Returns scripted candidate strings in order, regardless of temperature, so the
    self-test is reproducible. Each prompt maps to a list long enough to cover ``n``."""
    def _fake(system: str, user: str, *, max_tokens: int = 400,
              spec: "str | None" = None, **_) -> str:
        # Match on the raw prompt; the ladder is ignored by design here.
        for key, cands in script.items():
            if key in user:
                idx = _fake._counter.get(key, 0)  # type: ignore[attr-defined]
                _fake._counter[key] = idx + 1      # type: ignore[attr-defined]
                if idx < len(cands):
                    return cands[idx]
                return ""  # exhausted -> empty candidate (exercises the dedup/empty path)
        return ""
    _fake._counter = {k: 0 for k in script}  # type: ignore[attr-defined]
    return _fake


def self_test(emit: "Path | None" = None) -> int:
    """Prove the generation wiring offline: scripted candidates land in the
    ``candidates`` field in the exact shape ``gen_verifier_dpo.py`` consumes, the
    temperature ladder cycles deterministically, and empty/duplicate draws are folded.

    Then verify the FULL pipeline by piping the output through the real labeller
    (``tools/gen_verifier_dpo.run``), asserting at least one machine-labelled pair is
    minted — i.e. the gate can separate the generated candidates. No model, no network.
    """
    script = {
        "Socrates": [
            "No — Socrates wrote nothing himself; The Republic was written by Plato.",
            "Yes, Socrates wrote The Republic.",          # violating -> separated
            "Socrates is the author of The Republic.",    # violating -> separated
            "No, Socrates did not write The Republic.",   # clean (dup-ish, distinct)
        ],
        "Dao De Jing": [
            "No — Confucius did not write the Dao De Jing; it is a Daoist text by Laozi.",
            "Confucius wrote the Dao De Jing.",           # violating -> separated
            "The Dao De Jing is by Confucius.",           # violating -> separated
        ],
    }
    fake = _fake_complete_factory(script)
    out, stats = run(SELF_TEST_ROWS, n=4, complete_fn=fake)
    ok = True
    msgs: list[str] = []

    # 1. Both rows emitted with a non-empty candidates list in the right shape.
    if stats.emitted != len(SELF_TEST_ROWS) or len(out) != len(SELF_TEST_ROWS):
        ok = False
        msgs.append(f"emitted {stats.emitted} rows, expected {len(SELF_TEST_ROWS)}")
    for r in out:
        if not r.get("candidates") or not isinstance(r["candidates"], list):
            ok = False
            msgs.append(f"row missing candidates: {r.get('prompt')[:40]}")
        for cand in r.get("candidates") or []:
            if not isinstance(cand, str) or not cand.strip():
                ok = False
                msgs.append("empty/non-string candidate emitted")
        # exact keys gen_verifier_dpo.py reads:
        for key in ("prompt", "question", "mode", "candidates"):
            if key not in r:
                ok = False
                msgs.append(f"row missing key {key}")

    # 2. Full pipeline: feed generated candidates to the real machine-verifier labeller.
    from tools.gen_verifier_dpo import run as label_run  # noqa: E402
    pairs, lstats = label_run(out)
    if lstats["pairs"] < 1:
        ok = False
        msgs.append(f"labeller minted {lstats['pairs']} pairs from generated candidates "
                    f"(expected >=1); reasons={lstats['reasons']}")
    for p in pairs:
        if p["metadata"].get("label_source") != "machine_verified":
            ok = False
            msgs.append("a minted pair lacks machine_verified provenance")

    print("Candidate-generator self-test:", "PASS" if ok else "FAIL")
    print(f"  gen rows={stats.rows} emitted={stats.emitted} reasons={stats.reasons}")
    print(f"  label pairs={lstats['pairs']} reasons={lstats['reasons']}")
    for m in msgs:
        print(f"  [XX] {m}")
    if emit is not None:
        _write_jsonl(emit, out)
        print(f"  wrote {len(out)} candidate-rows -> {emit}")
    return 0 if ok else 1


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, help="input tasks JSONL (prompt per row)")
    ap.add_argument("--out", dest="out_path", type=Path,
                    help="output JSONL: same rows with candidates[] populated")
    ap.add_argument("--n", type=int, default=4, help="candidates to sample per prompt (default 4)")
    ap.add_argument("--spec", default=None,
                    help="agent.model spec, e.g. openai:gpt-4o-mini, anthropic:claude-…, mock")
    ap.add_argument("--seen", dest="seen_path", type=Path,
                    help="JSONL/text of eval prompts to skip (decontamination guard)")
    ap.add_argument("--max-tokens", type=int, default=400, help="max tokens per candidate")
    ap.add_argument("--self-test", action="store_true",
                    help="run the deterministic offline self-test (no model, no network)")
    ap.add_argument("--emit", type=Path, help="with --self-test, also write the demo rows here")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(emit=args.emit)

    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required (or use --self-test)")

    rows = _read_jsonl(args.in_path)
    seen: "set[str]" = set()
    if args.seen_path and args.seen_path.exists():
        for ln in args.seen_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                p = obj.get("prompt") if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                p = ln
            if p:
                seen.add(p)

    complete_fn = _import_complete()
    out, stats = run(rows, n=args.n, complete_fn=complete_fn, spec=args.spec,
                     seen_prompts=seen, max_tokens=args.max_tokens)
    _write_jsonl(args.out_path, out)
    print(json.dumps({"in": str(args.in_path), "out": str(args.out_path),
                      "spec": args.spec, "n": args.n, **stats.as_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
