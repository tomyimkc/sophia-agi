#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Entity/concept-level train/eval decontamination — the layer above shingles.

``tools/assert_decontam.py`` catches *lexical* overlap (exact prompt + word-shingle
near-duplicates). It does NOT catch the contamination that actually bit the SEIB-100
split (failure-ledger ``seib-generalization-split-not-validated``): a contested
ENTITY ("Socrates", "Dao De Jing") appearing in both train and eval — possibly with
different attributions — while every shingle differs. This tool closes that hole.

It resolves a canonical entity vocabulary from the committed provenance sources
(``data/attributions.json`` + ``provenance_bench/data/wikidata_snapshot.json``), then
reports which entities appear in BOTH the training surfaces and the eval surfaces, and
which eval prompts are *fully covered* by training entities (the strongest
contamination signal). Matching is whole-word (token n-gram), normalized, deterministic.

    python tools/assert_entity_decontam.py                  # report (exit 0; diagnostic)
    python tools/assert_entity_decontam.py --json           # machine-readable report
    python tools/assert_entity_decontam.py --fail-covered 0 # CI gate: fail if any eval prompt is fully entity-covered

Default is report-only (exit 0): the known SEIB contamination is documented, not a
build error, so this runs as a *diagnostic the Data Analysis Agent consumes* rather
than a hard gate — until a clean entity-disjoint split exists (Phase 3), at which point
``--fail-covered 0`` can be wired into CI on that split.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_bench.dataset_guard import (  # noqa: E402
    eval_prompt_set, prompt_of, _load_jsonl)

ATTRIBUTIONS = ROOT / "data" / "attributions.json"
WIKIDATA = ROOT / "provenance_bench" / "data" / "wikidata_snapshot.json"
TRAIN_SOURCES = [
    "training/corpus.jsonl",
    "training/moral_gate_sft.jsonl",
    "training/local_sophia_v3/mlx/train.jsonl",   # present only after a local build
    "training/local_sophia_v3/sft_*.jsonl",
]
MAX_ENTITY_TOKENS = 5
_STOP = {"the", "a", "an", "of", "on", "in", "to", "and", "school"}
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Punctuation-robust, unicode word tokens (normalize() only lowercases/strips ws,
    so 'jing?' would otherwise never match 'jing'). CJK is preserved by \\w."""
    return _TOKEN_RE.findall(str(text or "").lower())


def build_entity_vocab() -> dict[str, str]:
    """Canonical entity -> category ('author' | 'work'), normalized, deterministic."""
    vocab: dict[str, str] = {}

    def add(name, cat: str) -> None:
        if not name or not isinstance(name, str):
            return
        toks = _tokens(name)
        if not toks or len(toks) > MAX_ENTITY_TOKENS:
            return
        if len(toks) == 1 and (toks[0] in _STOP or len(toks[0]) < 4):
            return                       # skip bare stopwords / tiny tokens
        vocab.setdefault(" ".join(toks), cat)

    if ATTRIBUTIONS.exists():
        rec = json.loads(ATTRIBUTIONS.read_text(encoding="utf-8"))
        for v in (rec.values() if isinstance(rec, dict) else rec):
            if not isinstance(v, dict):
                continue
            add(v.get("attributedAuthor"), "author")
            add(v.get("canonicalTitleEn"), "work")
            for bad in (v.get("doNotAttributeTo") or []):
                add(bad, "author")
    if WIKIDATA.exists():
        wd = json.loads(WIKIDATA.read_text(encoding="utf-8"))
        for v in (wd.get("attributions") or []):
            if isinstance(v, dict):
                add(v.get("gold_author"), "author")
                add(v.get("work"), "work")
    return vocab


def _ngrams(text: str, max_n: int) -> set[str]:
    toks = _tokens(text)
    grams: set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            grams.add(" ".join(toks[i:i + n]))
    return grams


def entities_in(text: str, vocab: dict[str, str]) -> set[str]:
    grams = _ngrams(text, MAX_ENTITY_TOKENS)
    return {e for e in vocab if e in grams}


def _train_prompts() -> list[str]:
    prompts: list[str] = []
    for g in TRAIN_SOURCES:
        for p in sorted(ROOT.glob(g)) if "*" in g else [ROOT / g]:
            if p.exists():
                for row in _load_jsonl(p):
                    pr = prompt_of(row)
                    if pr:
                        prompts.append(pr)
    return prompts


def audit() -> dict:
    vocab = build_entity_vocab()
    train_prompts = _train_prompts()
    eval_prompts = sorted(eval_prompt_set(root=ROOT))

    train_entities: set[str] = set()
    for pr in train_prompts:
        train_entities |= entities_in(pr, vocab)

    eval_entities: set[str] = set()
    fully_covered: list[str] = []
    partially: int = 0
    for pr in eval_prompts:
        ents = entities_in(pr, vocab)
        if not ents:
            continue
        eval_entities |= ents
        shared = ents & train_entities
        if shared == ents:
            fully_covered.append(pr[:90])
        elif shared:
            partially += 1

    shared_entities = sorted(train_entities & eval_entities)
    return {
        "schema": "sophia.entity_decontam.v1",
        "vocabSize": len(vocab),
        "nTrainPrompts": len(train_prompts),
        "nEvalPrompts": len(eval_prompts),
        "nTrainEntities": len(train_entities),
        "nEvalEntities": len(eval_entities),
        "nSharedEntities": len(shared_entities),
        "sharedEntities": shared_entities[:25],
        "nEvalPromptsFullyCovered": len(fully_covered),
        "nEvalPromptsPartiallyCovered": partially,
        "sampleFullyCovered": sorted(fully_covered)[:10],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    ap.add_argument("--fail-covered", type=int, default=None,
                    help="exit 1 if more than N eval prompts are FULLY entity-covered by train")
    ap.add_argument("--fail-shared", type=int, default=None,
                    help="exit 1 if more than N entities are shared between train and eval")
    args = ap.parse_args(argv)

    rep = audit()
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(f"ENTITY DECONTAM: vocab={rep['vocabSize']} "
              f"trainEntities={rep['nTrainEntities']} evalEntities={rep['nEvalEntities']} "
              f"shared={rep['nSharedEntities']} | "
              f"evalFullyCovered={rep['nEvalPromptsFullyCovered']} partially={rep['nEvalPromptsPartiallyCovered']}")
        if rep["sharedEntities"]:
            print("  shared entities (sample): " + ", ".join(rep["sharedEntities"][:15]))
        for ex in rep["sampleFullyCovered"]:
            print(f"  FULLY-COVERED eval: «{ex}»")

    rc = 0
    if args.fail_covered is not None and rep["nEvalPromptsFullyCovered"] > args.fail_covered:
        print(f"FAIL — {rep['nEvalPromptsFullyCovered']} eval prompts fully entity-covered (> {args.fail_covered})")
        rc = 1
    if args.fail_shared is not None and rep["nSharedEntities"] > args.fail_shared:
        print(f"FAIL — {rep['nSharedEntities']} shared entities (> {args.fail_shared})")
        rc = 1
    if rc == 0 and (args.fail_covered is not None or args.fail_shared is not None):
        print("OK — within entity-contamination thresholds.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
