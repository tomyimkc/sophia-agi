#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Third-party multi-hop QA recall harness: OKF entity-graph recall vs vector-only.

This is the gated benchmark that would let Sophia make a SAG-comparable retrieval claim:
on the standard multi-hop QA datasets (HotpotQA / 2WikiMultiHop / MuSiQue), does the
OKF entity-graph expansion (okf.extract.multi_hop_recall) recall the gold supporting
paragraphs better than a vector-only baseline (agent.lexical_embed cosine) over the SAME
candidate pool?

IMPORTANT — what this harness can and cannot claim:
  * Third-party datasets are plain Wikipedia paragraphs with NO provenance labels, so the
    provenance-faithfulness metric (the wiki/-only contribution, see tools/eval_okf_recall.py)
    is NOT computable here. This harness measures the RECALL LIFT only.
  * The real datasets are NOT committed and there is no network here. Point --data at a
    farm-downloaded file. With no --data, it runs the committed SYNTHETIC fixture
    (agi-proof/benchmark-results/okf-multihop/fixtures/mini_multihop.jsonl) purely to
    validate the harness wiring end-to-end — fixture numbers are NOT a result.
  * Entity extraction on raw paragraphs uses a transparent deterministic proper-noun /
    title extractor by default. A real NER/LLM backend is the farm upgrade (--ner-backend),
    and is what SAG uses; the deterministic default is a floor, not the ceiling.

Pre-registration: agi-proof/benchmark-results/okf-multihop/measurement_spec.json +
the byte-stable not-run artifact okf-multihop-qa.PENDING.public-report.json. The gates
decide validity; this script never sets canClaimAGI.

    python tools/eval_okf_multihop_qa.py                          # fixture self-test
    python tools/eval_okf_multihop_qa.py --data hotpot_dev.json --dataset hotpot --json
    python tools/eval_okf_multihop_qa.py --dataset musique --data musique_dev.jsonl --k 2 5 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lexical_embed  # noqa: E402
from okf import extract  # noqa: E402
from okf.wikilinks import normalize_target  # noqa: E402
from tools.assert_decontam import TRAIN_GLOBS, _jaccard, _shingles  # noqa: E402
from provenance_bench.dataset_guard import _load_jsonl as _guard_load_jsonl, normalize, prompt_of  # noqa: E402

FIXTURE = ROOT / "agi-proof" / "benchmark-results" / "okf-multihop" / "fixtures" / "mini_multihop.jsonl"
DEFAULT_KS = (2, 5, 10)

# Deterministic proper-noun extractor: capitalized word runs (the offline NER floor).
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_STOP_PROPER = {"the", "a", "an", "in", "on", "of", "and"}


# --------------------------------------------------------------------------- loaders

def _load_jsonl(path: Path) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def normalize_hotpot(raw) -> "list[dict]":
    """HotpotQA / 2WikiMultiHop distractor format -> normalized items.

    Each raw item: {question, context: [[title, [sent,...]], ...], supporting_facts:
    [[title, sent_idx], ...]}. Gold paragraphs = titles appearing in supporting_facts.
    """
    items = raw if isinstance(raw, list) else raw.get("data", raw)
    out = []
    for it in items:
        support = {t for t, _ in it.get("supporting_facts", [])}
        paras = []
        for title, sents in it.get("context", []):
            paras.append({"title": title, "text": " ".join(sents), "gold": title in support})
        out.append({"id": it.get("_id") or it.get("id"), "question": it["question"],
                    "answer": it.get("answer"), "paragraphs": paras})
    return out


def normalize_musique(raw) -> "list[dict]":
    """MuSiQue format -> normalized items (paragraphs carry an is_supporting flag)."""
    items = raw if isinstance(raw, list) else raw.get("data", raw)
    out = []
    for it in items:
        paras = []
        for p in it.get("paragraphs", []):
            title = p.get("title", "")
            text = p.get("paragraph_text", p.get("text", ""))
            paras.append({"title": title, "text": text, "gold": bool(p.get("is_supporting"))})
        out.append({"id": it.get("id"), "question": it["question"],
                    "answer": it.get("answer"), "paragraphs": paras})
    return out


def normalize_fixture(raw) -> "list[dict]":
    """The committed synthetic fixture is already in normalized schema."""
    return raw


NORMALIZERS = {"hotpot": normalize_hotpot, "2wiki": normalize_hotpot,
               "musique": normalize_musique, "fixture": normalize_fixture}


# ----------------------------------------------------------------- entity extraction

def _deterministic_entities(title: str, text: str) -> "list[str]":
    """The offline floor: the paragraph title + capitalized proper-noun runs."""
    ents = [normalize_target(title)] if title else []
    for m in _PROPER.finditer(text or ""):
        phrase = m.group(1)
        if phrase.lower() in _STOP_PROPER:
            continue
        slug = normalize_target(phrase)
        if slug and slug not in ents:
            ents.append(slug)
    return ents


def _extract_json_array(s: str) -> "list":
    """Pull the first JSON array out of an LLM reply (tolerant of prose/code fences)."""
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        arr = json.loads(s[start:end + 1])
        return [str(x) for x in arr] if isinstance(arr, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


_LLM_NER_SYSTEM = (
    "You are a named-entity extractor. Given a paragraph, return ONLY a JSON array of the "
    "salient entities (people, places, organizations, works, dates, concepts) as short "
    "strings. No prose, no keys — just the array."
)


def _llm_entities(title: str, text: str, *, model: str) -> "list[str]":
    """Real NER via the repo LLM client (farm path). Fail-closed without an API key.

    A per-paragraph parse failure falls back to the deterministic floor for THAT paragraph
    (logged), so one bad reply cannot silently tank a long run; absence of a key raises
    (fail-closed) rather than degrading the whole benchmark to the floor unannounced.
    """
    from agent import llm  # lazy: never imported on the deterministic path / in CI

    import os
    os.environ.setdefault("ANTHROPIC_MODEL", model)
    reply = llm.complete(_LLM_NER_SYSTEM, f"Title: {title}\n\n{text}", max_tokens=400)
    raw = _extract_json_array(reply)
    if not raw:
        print(f"[ner] empty/unparsable LLM reply for {title!r}; using deterministic floor",
              file=sys.stderr)
        return _deterministic_entities(title, text)
    ents = [normalize_target(title)] if title else []
    for phrase in raw:
        slug = normalize_target(phrase)
        if slug and slug not in ents:
            ents.append(slug)
    return ents


# Backend registry. Unknown names are treated as a model id for the LLM backend, so
# `--ner-backend claude-sonnet-4-6` works without a code change.
_ENTITY_BACKENDS = {"deterministic": _deterministic_entities}


def extract_entities(title: str, text: str, backend: str = "deterministic") -> "list[str]":
    """Entity proxies for a raw paragraph via the named backend (default: offline floor)."""
    fn = _ENTITY_BACKENDS.get(backend)
    if fn is not None:
        return fn(title, text)
    return _llm_entities(title, text, model=backend)


def _events_for_item(item, ner_backend: str = "deterministic") -> "list[extract.EventUnit]":
    """One EventUnit per candidate paragraph (rank neutral — no provenance on web text)."""
    events = []
    for i, p in enumerate(item["paragraphs"]):
        ents = extract_entities(p["title"], p["text"], backend=ner_backend)
        events.append(extract.EventUnit(
            id=f"{item['id']}::p{i}", page_id=f"{item['id']}::p{i}",
            text=f"{p['title']}. {p['text']}", entities=tuple(ents),
            author_confidence=None, confidence_rank=2, tradition=None))
    return events


# ------------------------------------------------------------------------- retrieval

def _gold_idx(item) -> "set[int]":
    return {i for i, p in enumerate(item["paragraphs"]) if p["gold"]}


def _vector_rank(item) -> "list[int]":
    """Vector-only baseline: rank paragraph indices by lexical_embed cosine to the question."""
    docs = [(str(i), f"{p['title']}. {p['text']}") for i, p in enumerate(item["paragraphs"])]
    ranked = lexical_embed.rank(item["question"], docs, top_k=len(docs))
    return [int(doc_id) for doc_id, _ in ranked]


def _graph_rank(item, ner_backend: str = "deterministic") -> "list[int]":
    """OKF entity-graph recall: multi_hop_recall over the paragraph entity index."""
    events = _events_for_item(item, ner_backend)
    hits = extract.multi_hop_recall(item["question"], events, max_hops=2, top_k=len(events))
    order = [int(h.event.page_id.split("::p")[1]) for h in hits]
    # append any paragraph the recall never surfaced, preserving determinism
    for i in range(len(item["paragraphs"])):
        if i not in order:
            order.append(i)
    return order


def _recall_at(order, gold, k) -> float:
    if not gold:
        return 0.0
    topk = set(order[:k])
    return len(topk & gold) / len(gold)


def check_decontam(items, *, jaccard: float = 0.6, shingle: int = 5) -> dict:
    """Assert dataset questions are disjoint from every committed training corpus.

    Reuses the repo's shared decontam primitives (tools.assert_decontam TRAIN_GLOBS +
    provenance_bench.dataset_guard). Layer 1: exact/normalized question overlap. Layer 2:
    content-shingle near-duplicate (Jaccard). A GO requires `clean == true`.
    """
    train_prompts: list[str] = []
    for g in TRAIN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            for row in _guard_load_jsonl(p):
                pr = prompt_of(row)
                if pr:
                    train_prompts.append(pr)
    train_norm = {normalize(p) for p in train_prompts}
    train_sh = [_shingles(p, shingle) for p in train_prompts]

    exact: list[str] = []
    near: list[dict] = []
    for it in items:
        q = it["question"]
        if normalize(q) in train_norm:
            exact.append(it.get("id"))
            continue
        qsh = _shingles(q, shingle)
        worst = max((_jaccard(qsh, ts) for ts in train_sh), default=0.0)
        if worst >= jaccard:
            near.append({"id": it.get("id"), "jaccard": round(worst, 3)})
    # A scan over zero training prompts (corpora absent / git-crypt-locked in this checkout)
    # is VACUOUS — no leak can be found, so it must not be read as a real decontam pass.
    vacuous = len(train_prompts) == 0
    return {"trainPromptsScanned": len(train_prompts), "exactLeaks": exact,
            "nearLeaks": near, "jaccardThreshold": jaccard, "shingleK": shingle,
            "vacuous": vacuous, "clean": (not exact and not near and not vacuous)}


def evaluate(items, ks=DEFAULT_KS, ner_backend: str = "deterministic") -> dict:
    arms = {"vector_only": _vector_rank,
            "graph_multihop": lambda it: _graph_rank(it, ner_backend)}
    sums = {a: {k: 0.0 for k in ks} for a in arms}
    n = len(items)
    for item in items:
        gold = _gold_idx(item)
        for a, ranker in arms.items():
            order = ranker(item)
            for k in ks:
                sums[a][k] += _recall_at(order, gold, k)
    metrics = {a: {f"recall@{k}": round(sums[a][k] / n, 4) for k in ks} for a in arms} if n else {}
    lift = {f"recall@{k}": round((sums["graph_multihop"][k] - sums["vector_only"][k]) / n, 4)
            for k in ks} if n else {}
    return {"items": n, "ks": list(ks), "arms": metrics, "graphMinusVector": lift}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to a farm-downloaded dataset file (else: fixture)")
    ap.add_argument("--dataset", choices=sorted(NORMALIZERS), default="fixture")
    ap.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS), help="recall depths")
    ap.add_argument("--ner-backend", default="deterministic",
                    help="entity backend (farm: a real NER/LLM; default: deterministic floor)")
    ap.add_argument("--check-decontam", action="store_true",
                    help="assert questions are disjoint from training corpora (fail-closed)")
    ap.add_argument("--out", help="write the JSON report to this path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.ner_backend != "deterministic":
        print(f"[ner] using LLM entity backend model={args.ner_backend!r} "
              "(fail-closed without ANTHROPIC_API_KEY/CLAUDE_API_KEY).", file=sys.stderr)

    is_fixture = args.data is None
    path = FIXTURE if is_fixture else Path(args.data)
    raw = _load_jsonl(path) if path.suffix in {".jsonl", ".ndjson"} else json.loads(path.read_text("utf-8"))
    dataset = "fixture" if is_fixture else args.dataset
    items = NORMALIZERS[dataset](raw)

    try:
        result = evaluate(items, ks=tuple(args.k), ner_backend=args.ner_backend)
    except RuntimeError as exc:
        # Fail-closed: the LLM NER backend needs an API key. Clean message, no traceback.
        print(f"[fail-closed] NER backend {args.ner_backend!r}: {exc}", file=sys.stderr)
        return 2
    result["dataset"] = dataset
    result["isFixture"] = is_fixture
    result["nerBackend"] = args.ner_backend
    result["canClaimAGI"] = False
    result["claimCeiling"] = "candidate_only; canClaimAGI:false"
    result["honestBound"] = (
        "FIXTURE self-test — wiring only, NOT a result." if is_fixture else
        "third-party recall lift; provenance-faithfulness is NOT measurable on unlabeled web text.")

    decontam_dirty = False
    if args.check_decontam:
        result["decontam"] = check_decontam(items)
        decontam_dirty = not result["decontam"]["clean"]

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote report -> {out_path}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        banner = "SYNTHETIC FIXTURE (wiring self-test — not a result)" if is_fixture else f"dataset={dataset}"
        print(f"OKF multi-hop QA recall — {banner}")
        print("=" * 64)
        for arm, m in result["arms"].items():
            print(f"  {arm:16} " + "  ".join(f"{k}={v}" for k, v in m.items()))
        print("  " + "-" * 50)
        print("  graph - vector   " + "  ".join(f"{k}={v}" for k, v in result["graphMinusVector"].items()))
        if args.check_decontam:
            dc = result["decontam"]
            print(f"  decontam: {'CLEAN' if dc['clean'] else 'DIRTY ' + str(dc['exactLeaks'] + dc['nearLeaks'])}")
        print("-" * 64)
        print(result["honestBound"])
    # Fail-closed: a decontamination leak on a real dataset is a hard stop.
    return 2 if decontam_dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
