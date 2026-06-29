# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build the sealed semantic-grounding datasets (D1 + D2), deterministically.

D1 (definition-faithfulness) is projected from the committed OKF concept pages
under ``wiki/concept/`` — each page's gloss + ``doNotAttributeTo`` becomes one
word-sense case with lexically-near distractor definitions. D2 (compositional
derivation) is generated from a small, FROZEN, sourced concept-TBox seed (generic
consensus taxonomy — no protected religion/history identity) plus claims whose
gold verdict is *computed* by the Datalog engine (``score.reference_verdict``),
never hand-labelled.

Source of truth = ``wiki/concept/*`` + the ``AXIOM_WORLDS`` seed below. The
emitted ``data/*.jsonl`` are generated artifacts; CI checks they have not drifted.

    python -m eval.semantic_grounding.build_dataset --emit    # (re)write data/*.jsonl
    python -m eval.semantic_grounding.build_dataset --check    # fail on drift (CI)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.semantic_grounding.score import reference_verdict  # noqa: E402
from okf.page import load_pages  # noqa: E402
from okf.schema import as_list  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CONCEPT_DIR = _REPO_ROOT / "wiki" / "concept"
N_DISTRACTORS = 3


def _stable_key(*parts: str) -> str:
    """Deterministic, cross-process ordering key (builtin hash() is salted)."""
    return hashlib.md5("\x1f".join(parts).encode("utf-8")).hexdigest()


def fold_of(*parts: str) -> str:
    """Deterministic ~70/30 train/eval split. The training generator draws only the
    'train' fold and the Phase-2 uplift is measured only on the 'eval' fold, so they
    are disjoint BY CONSTRUCTION — no train/eval leakage to argue about."""
    return "eval" if int(_stable_key("fold", *parts), 16) % 10 < 3 else "train"


# ----------------------------------------------------------------- D1 from wiki
def _gloss(body: str) -> str:
    """The substantive definition paragraph of a concept page.

    Pages are templated: an H1, a stock "X is a <domain> concept ..." line, two
    bullets, then the real gloss paragraph, then a `>` do-not-attribute footer and
    a `_generated_` line. We take the first prose paragraph AFTER the bullet block
    and BEFORE the footer; if none, fall back to the stock intro line.
    """
    lines = [ln.rstrip() for ln in body.splitlines()]
    intro = ""
    seen_bullet = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith(("- ", "* ")):
            seen_bullet = True
            continue
        if s.startswith((">", "_")):
            continue
        if not intro and not seen_bullet:
            intro = s  # stock intro, kept as fallback
            continue
        if seen_bullet:
            return s
    return intro


def build_d1() -> list[dict]:
    pages = [p for p in load_pages(str(CONCEPT_DIR)) if p.page_type == "concept"]
    pages = sorted(pages, key=lambda p: p.id)
    glosses = {p.id: _gloss(p.body) for p in pages}
    ids = [p.id for p in pages]
    cases: list[dict] = []
    for i, p in enumerate(pages):
        term = str(p.meta.get("canonicalTitleEn") or p.id.replace("_", " "))
        forbidden = [str(a) for a in as_list(p.meta.get("doNotAttributeTo"))]
        # Deterministic, spread-out distractors (offsets across the corpus so the
        # gold is not clustered with its neighbours), then a per-case pseudo-shuffle
        # so candidate ORDER never leaks the gold's position.
        offsets = (1, 1 + len(ids) // 3, 1 + 2 * len(ids) // 3)
        distractor_ids = []
        for off in offsets:
            cid = ids[(i + off) % len(ids)]
            if cid != p.id and cid not in distractor_ids:
                distractor_ids.append(cid)
        cand_ids = [p.id, *distractor_ids[:N_DISTRACTORS]]
        cand_ids.sort(key=lambda cid: _stable_key(p.id, cid))
        candidates = [{"conceptId": cid, "gloss": glosses[cid]} for cid in cand_ids]
        # CLOSED-BOOK prompt: the candidate definitions are NOT shown to the model — they
        # are a scoring-only inventory (the scorer lexically matches the free-form answer to
        # them). This makes A0 a true distributional baseline; A1 adds the retrieved OKF
        # gloss, A2 adds the provenance constraint. (See the runner's build_prompt.)
        prompt = (
            f"Define the term \"{term}\" in one sentence (its established meaning), then "
            "state any author it must NOT be attributed to (write 'none' if there is none)."
        )
        cases.append({
            "id": f"d1-{p.id}",
            "task": "definition",
            "fold": fold_of("d1", p.id),
            "domain": p.meta.get("domain"),
            "term": term,
            "goldConceptId": p.id,
            "doNotAttributeTo": forbidden,
            "candidates": candidates,
            "prompt": prompt,
        })
    return cases


# ------------------------------------------------------ D2 from a frozen TBox seed
# Small, sourced, consensus taxonomy. These are the CLOSED-WORLD axioms of the
# benchmark (each an untrusted, sourced claim — see Ontology-Claim-Boundary.md),
# chosen to be uncontroversial and free of protected religion/history identity.
AXIOM_WORLDS: list[dict] = [
    {
        "world": "biology",
        "source": "consensus Linnaean taxonomy (genus-differentia subsumption)",
        "axioms": [
            ["subClassOf", "dog", "mammal"], ["subClassOf", "cat", "mammal"],
            ["subClassOf", "mammal", "vertebrate"], ["subClassOf", "vertebrate", "animal"],
            ["subClassOf", "sparrow", "bird"], ["subClassOf", "bird", "vertebrate"],
            ["disjointWith", "mammal", "bird"],
        ],
        "claims": [
            ["subClassOf", "dog", "animal"],        # entailed (3-hop transitive)
            ["subClassOf", "sparrow", "vertebrate"],  # entailed (2-hop)
            ["subClassOf", "dog", "bird"],          # violation (mammal⊥bird)
            ["subClassOf", "cat", "reptile"],       # abstain (silent)
            ["disjointWith", "mammal", "bird"],     # entailed (asserted)
            ["disjointWith", "dog", "cat"],         # abstain (silent)
        ],
    },
    {
        "world": "geometry",
        "source": "consensus Euclidean quadrilateral hierarchy",
        "axioms": [
            ["subClassOf", "square", "rectangle"], ["subClassOf", "rectangle", "parallelogram"],
            ["subClassOf", "parallelogram", "quadrilateral"], ["subClassOf", "rhombus", "parallelogram"],
            ["disjointWith", "quadrilateral", "triangle"],
        ],
        "claims": [
            ["subClassOf", "square", "quadrilateral"],   # entailed (3-hop)
            ["subClassOf", "rhombus", "quadrilateral"],   # entailed (2-hop)
            ["subClassOf", "square", "triangle"],         # violation (quad⊥triangle)
            ["subClassOf", "square", "rhombus"],          # abstain (silent)
        ],
    },
    {
        "world": "language",
        "source": "consensus part-of-speech / lexical category subsumption",
        "axioms": [
            ["subClassOf", "noun", "open_class_word"], ["subClassOf", "verb", "open_class_word"],
            ["subClassOf", "open_class_word", "word"], ["subClassOf", "preposition", "closed_class_word"],
            ["subClassOf", "closed_class_word", "word"],
            ["disjointWith", "open_class_word", "closed_class_word"],
        ],
        "claims": [
            ["subClassOf", "noun", "word"],                 # entailed (2-hop)
            ["subClassOf", "preposition", "word"],          # entailed (2-hop)
            ["subClassOf", "noun", "closed_class_word"],    # violation (open⊥closed)
            ["subClassOf", "verb", "preposition"],          # abstain (silent)
        ],
    },
    {
        "world": "chemistry",
        "source": "consensus classification of matter",
        "axioms": [
            ["subClassOf", "metal", "element"], ["subClassOf", "nonmetal", "element"],
            ["subClassOf", "element", "pure_substance"], ["subClassOf", "compound", "pure_substance"],
            ["subClassOf", "pure_substance", "matter"], ["subClassOf", "mixture", "matter"],
            ["disjointWith", "element", "compound"], ["disjointWith", "pure_substance", "mixture"],
            ["disjointWith", "metal", "nonmetal"],
        ],
        "claims": [
            ["subClassOf", "metal", "matter"],          # entailed (3-hop)
            ["subClassOf", "compound", "matter"],       # entailed (2-hop)
            ["subClassOf", "metal", "nonmetal"],        # violation (metal⊥nonmetal)
            ["subClassOf", "element", "compound"],      # violation (element⊥compound)
            ["subClassOf", "gas", "matter"],            # abstain (silent)
        ],
    },
    {
        "world": "music",
        "source": "consensus Western music notation taxonomy",
        "axioms": [
            ["subClassOf", "quarter_note", "note"], ["subClassOf", "half_note", "note"],
            ["subClassOf", "note", "musical_symbol"], ["subClassOf", "rest", "musical_symbol"],
            ["disjointWith", "note", "rest"],
        ],
        "claims": [
            ["subClassOf", "quarter_note", "musical_symbol"],  # entailed (2-hop)
            ["subClassOf", "quarter_note", "rest"],            # violation (note⊥rest)
            ["subClassOf", "note", "clef"],                    # abstain (silent)
        ],
    },
    {
        "world": "cs_types",
        "source": "consensus programming-language type hierarchy",
        "axioms": [
            ["subClassOf", "integer", "number"], ["subClassOf", "float", "number"],
            ["subClassOf", "number", "scalar"], ["subClassOf", "scalar", "value"],
            ["subClassOf", "string", "value"], ["disjointWith", "number", "string"],
        ],
        "claims": [
            ["subClassOf", "integer", "value"],    # entailed (3-hop)
            ["subClassOf", "float", "scalar"],     # entailed (2-hop)
            ["subClassOf", "integer", "string"],   # violation (number⊥string)
            ["subClassOf", "boolean", "value"],    # abstain (silent)
        ],
    },
    {
        "world": "plants",
        "source": "consensus botanical subsumption",
        "axioms": [
            ["subClassOf", "rose", "flowering_plant"], ["subClassOf", "flowering_plant", "plant"],
            ["subClassOf", "fern", "plant"], ["subClassOf", "moss", "plant"],
            ["disjointWith", "flowering_plant", "fern"],
        ],
        "claims": [
            ["subClassOf", "rose", "plant"],   # entailed (2-hop)
            ["subClassOf", "rose", "fern"],    # violation (flowering⊥fern)
            ["subClassOf", "rose", "moss"],    # abstain (silent)
        ],
    },
    {
        "world": "kinship",
        "source": "consensus kinship relation subsumption",
        "axioms": [
            ["subClassOf", "parent", "ancestor"], ["subClassOf", "grandparent", "ancestor"],
            ["subClassOf", "child", "descendant"], ["subClassOf", "ancestor", "relative"],
            ["subClassOf", "descendant", "relative"], ["disjointWith", "ancestor", "descendant"],
        ],
        "claims": [
            ["subClassOf", "parent", "relative"],     # entailed (2-hop)
            ["subClassOf", "child", "relative"],      # entailed (2-hop)
            ["subClassOf", "parent", "descendant"],   # violation (ancestor⊥descendant)
            ["subClassOf", "sibling", "relative"],    # abstain (silent)
        ],
    },
    {
        "world": "food",
        "source": "consensus culinary classification",
        "axioms": [
            ["subClassOf", "apple", "fruit"], ["subClassOf", "carrot", "vegetable"],
            ["subClassOf", "fruit", "food"], ["subClassOf", "vegetable", "food"],
            ["disjointWith", "fruit", "vegetable"],
        ],
        "claims": [
            ["subClassOf", "apple", "food"],       # entailed (2-hop)
            ["subClassOf", "carrot", "food"],      # entailed (2-hop)
            ["subClassOf", "apple", "vegetable"],  # violation (fruit⊥vegetable)
            ["subClassOf", "apple", "grain"],      # abstain (silent)
        ],
    },
    {
        "world": "polygons",
        "source": "consensus Euclidean polygon hierarchy",
        "axioms": [
            ["subClassOf", "equilateral", "triangle"], ["subClassOf", "isosceles", "triangle"],
            ["subClassOf", "triangle", "polygon"], ["subClassOf", "polygon", "shape"],
            ["disjointWith", "triangle", "circle"],
        ],
        "claims": [
            ["subClassOf", "equilateral", "shape"],      # entailed (3-hop)
            ["subClassOf", "equilateral", "polygon"],    # entailed (2-hop)
            ["subClassOf", "triangle", "circle"],        # violation (triangle⊥circle)
            ["subClassOf", "equilateral", "isosceles"],  # abstain (true in reality, silent here)
        ],
    },
]


def build_d2() -> list[dict]:
    cases: list[dict] = []
    for w in AXIOM_WORLDS:
        for j, claim in enumerate(w["claims"]):
            gold = reference_verdict(w["axioms"], claim)
            rel, x, y = claim
            prompt = (
                "Closed world — treat ONLY these facts as known:\n"
                + "\n".join(f"  {a[1]} {a[0]} {a[2]}" for a in w["axioms"])
                + f"\n\nClaim: {x} {rel} {y}\n"
                "Answer exactly one of: entailed (derivable), violation (contradicts a "
                "disjointness), abstain (the facts are silent). Give the derivation."
            )
            cases.append({
                "id": f"d2-{w['world']}-{j}",
                "task": "composition",
                "fold": fold_of("d2", w["world"]),  # split by world (axioms stay together)
                "world": w["world"],
                "source": w["source"],
                "axioms": w["axioms"],
                "claim": claim,
                "goldVerdict": gold,
                "prompt": prompt,
            })
    return cases


# ------------------------------------------------------------------------- emit
SPLITS = {
    "d1_definition_faithfulness.jsonl": build_d1,
    "d2_compositional_derivation.jsonl": build_d2,
}


def _dump(cases: list[dict]) -> str:
    return "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)


def emit() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fname, builder in SPLITS.items():
        cases = builder()
        (DATA / fname).write_text(_dump(cases), encoding="utf-8")
        counts[fname] = len(cases)
    return counts


def check() -> int:
    drift = []
    for fname, builder in SPLITS.items():
        want = _dump(builder())
        path = DATA / fname
        have = path.read_text(encoding="utf-8") if path.exists() else ""
        if want != have:
            drift.append(fname)
    if drift:
        print("DRIFT — committed datasets differ from their source of truth:", file=sys.stderr)
        for f in drift:
            print(f"  {f}  (run: python -m eval.semantic_grounding.build_dataset --emit)", file=sys.stderr)
        return 1
    print(f"OK — semantic_grounding datasets match source ({', '.join(SPLITS)}).")
    return 0


def _main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build/check the semantic-grounding datasets.")
    p.add_argument("--emit", action="store_true", help="(re)write data/*.jsonl")
    p.add_argument("--check", action="store_true", help="fail on drift (CI)")
    args = p.parse_args(argv)
    if args.emit:
        counts = emit()
        print(json.dumps({"emitted": counts}, indent=2))
        return 0
    if args.check:
        return check()
    p.error("pass --emit or --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
