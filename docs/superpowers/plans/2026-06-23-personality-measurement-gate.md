# Personality Measurement Gate (MBTI Vector Agents — Spec A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, offline personality-measurement slice — OCEAN substrate + one-way MBTI veneer, a pure-function IPIP scorer + mock administration loop, a `personality_faithful` verifier, a `personality` benchmark domain, and a thin MCP+Skill surface — under Sophia's no-overclaim gate.

**Architecture:** Big-Five (OCEAN) is the measured substrate; MBTI is a one-way display veneer the gate never reads. Inventory scoring is pure keyed arithmetic (no model). Generation/administration defaults to `model='mock'` (offline). The verifier mirrors `provenance_faithful`'s factory shape with a three-way pass/fail/abstain verdict. The benchmark domain plugs into the existing `DOMAIN_BENCH`/`score_case` machinery via a new `mustExpressTarget` check key.

**Tech Stack:** Python 3.12, stdlib only for core (json, re, random, statistics). `mcp>=1.2` (optional, only for the MCP server; tests exercise `tools_impl` directly). No pytest.

## Global Constraints

Every task's requirements implicitly include these (copied from the spec):

- **No pytest in this repo.** Tests are plain scripts: each `tests/<file>.py` defines test functions, a `main()` that calls every one, and `if __name__ == "__main__": raise SystemExit(main())`. A test not called in `main()` does not run. CI runs specific files via `python tests/<file>.py` (`.github/workflows/ci.yml`). New test files require a new `python tests/<file>.py` line in `ci.yml`.
- **Offline & deterministic in CI.** No network. Use `model='mock'`, the `SOPHIA_MOCK_RESPONSE` env hook, or in-test stub clients. The only deterministic core is the pure-function scorer; never assert on live-model output.
- **MBTI is a one-way display veneer.** `mbti_to_ocean()` at the request boundary, `ocean_to_mbti_letters()` at display. **No gate/verifier/effect-size/abstention path may read the MBTI string.** A veneer-invariance test is mandatory (Task 4).
- **Neuroticism is undetermined by any MBTI code** → always `None`/unspecified. Never inferred from a letter.
- **Within-system deltas only.** Never human-norm percentiles.
- **Verified MBTI↔OCEAN values** (McCrae & Costa 1989, second-letter convention): E/I↔Extraversion r=−0.74; S/N↔Openness r=+0.72 (1989 point estimate; pooled ~0.60–0.65); T/F↔Agreeableness r=+0.44 (weakest, sex-confounded); J/P↔Conscientiousness r≈−0.48/−0.49. Cite `ipip.ori.org` for scoring/public-domain.
- **Skill naming:** kebab-case, must NOT contain `claude`/`anthropic`, no XML angle brackets in frontmatter, no README inside the skill folder.
- **Do NOT** add `personality` to `data/domains.json`, `okf/schema.py`, or `agent/rag_sources.py` (would wrongly imply a provenance/OKF/RAG domain — see Task 6 rationale).
- **A `Verifier` is `Callable[[str, Any, dict], dict]`** returning `{"passed": bool, "reasons": list[str], "detail": dict}`; factories take no required args; called as `factory()(text, None, {})`.
- Commit after every task. Branch is `feat/mbti-vector-agents-spec-a` (already created).

---

### Task 1: OCEAN substrate + MBTI veneer (`agent/personality_map.py`)

**Files:**
- Create: `agent/personality_map.py`
- Create: `tests/test_personality.py`
- Modify: `.github/workflows/ci.yml` (add one test invocation line)

**Interfaces:**
- Produces:
  - `AXIS_OCEAN: dict[str, dict]` — verified r-table.
  - `mbti_to_ocean(code: str) -> dict` → `{"O": "high"|"low", "C": ..., "E": ..., "A": ..., "N": None, "_meta": {...}}` or `{"error": str, "available": list[str]}`.
  - `ocean_to_mbti_letters(ocean: dict) -> str` — display-only 4-letter code (N ignored).
  - `build_type_records() -> dict[str, dict]` — all 16 type records (consumed by Task 7).
  - `SIXTEEN_TYPES: tuple[str, ...]`.

- [ ] **Step 1: Write the failing test** — append to a new file `tests/test_personality.py`:

```python
"""Spec A — personality measurement gate tests (plain-script style, no pytest)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import personality_map as pm


def test_mbti_to_ocean_all_types() -> None:
    for code in pm.SIXTEEN_TYPES:
        out = pm.mbti_to_ocean(code)
        assert "error" not in out, f"{code}: {out}"
        assert set(["O", "C", "E", "A", "N"]).issubset(out), out
        assert out["N"] is None, f"Neuroticism must be undetermined for {code}"
        for k in ("O", "C", "E", "A"):
            assert out[k] in ("high", "low"), (code, k, out[k])
    assert len(pm.SIXTEEN_TYPES) == 16


def test_mbti_to_ocean_intj() -> None:
    out = pm.mbti_to_ocean("intj")  # case-insensitive
    assert out["O"] == "high" and out["C"] == "high"
    assert out["E"] == "low" and out["A"] == "low" and out["N"] is None


def test_mbti_to_ocean_invalid() -> None:
    out = pm.mbti_to_ocean("XXXX")
    assert "error" in out and len(out["available"]) == 16


def test_ocean_to_mbti_letters_roundtrip() -> None:
    for code in pm.SIXTEEN_TYPES:
        ocean = pm.mbti_to_ocean(code)
        assert pm.ocean_to_mbti_letters(ocean) == code


def test_build_type_records() -> None:
    recs = pm.build_type_records()
    assert set(recs) == set(pm.SIXTEEN_TYPES)
    assert recs["INTJ"]["ocean"]["E"] == "low"
    assert recs["INTJ"]["ocean"]["N"] is None


def main() -> int:
    tests = [
        test_mbti_to_ocean_all_types,
        test_mbti_to_ocean_intj,
        test_mbti_to_ocean_invalid,
        test_ocean_to_mbti_letters_roundtrip,
        test_build_type_records,
    ]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} personality tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_personality.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.personality_map'`

- [ ] **Step 3: Write minimal implementation** — create `agent/personality_map.py`:

```python
"""OCEAN substrate + one-way MBTI display veneer (Spec A).

Big Five (OCEAN) is the measured substrate. MBTI is a *display veneer*:
`mbti_to_ocean` translates a user-facing type at the request boundary;
`ocean_to_mbti_letters` renders a code at the display boundary. No gate,
verifier, effect-size, or abstention path may read the MBTI string.

Mapping source: McCrae & Costa (1989), J. Personality 57:17-40, second-letter
convention. Neuroticism has NO MBTI correlate (max |r|~0.16) and is always
left unspecified (None). Scoring/public-domain reference: ipip.ori.org.
"""
from __future__ import annotations

from itertools import product

# Verified r-table (1989 point estimates). 'pole_high' = which second-letter
# pole maps to HIGH on the OCEAN factor.
AXIS_OCEAN: dict[str, dict] = {
    "E/I": {"factor": "E", "r": -0.74, "pole_high": "E", "confidence": "highest"},
    "S/N": {"factor": "O", "r": +0.72, "pole_high": "N", "confidence": "highest",
            "note": "1989 point estimate; pooled replications ~0.60-0.65"},
    "T/F": {"factor": "A", "r": +0.44, "pole_high": "F", "confidence": "lowest",
            "note": "weakest, sex-confounded (0.33-0.44)"},
    "J/P": {"factor": "C", "r": -0.485, "pole_high": "J", "confidence": "moderate-high",
            "note": "cited -0.48 to -0.49"},
}

# letter position -> (axis, the two poles in MBTI order)
_AXES = (
    ("E", "I", "E/I"),
    ("S", "N", "S/N"),
    ("T", "F", "T/F"),
    ("J", "P", "J/P"),
)

SIXTEEN_TYPES: tuple[str, ...] = tuple(
    "".join(p) for p in product("EI", "SN", "TF", "JP")
)


def _letter_to_sign(letter: str, axis: str) -> str:
    """Map a single MBTI letter to 'high'/'low' on its OCEAN factor."""
    pole_high = AXIS_OCEAN[axis]["pole_high"]
    return "high" if letter == pole_high else "low"


def mbti_to_ocean(code: str) -> dict:
    """One-way: an MBTI type -> OCEAN target signs. N is always None.

    Returns {"O","C","E","A","N","_meta"} or {"error","available"} on a bad code.
    """
    norm = (code or "").strip().upper()
    if norm not in SIXTEEN_TYPES:
        return {"error": f"unknown MBTI type: {code!r}", "available": list(SIXTEEN_TYPES)}
    signs: dict = {"N": None}
    meta: dict = {}
    for letter, (pos, neg, axis) in zip(norm, _AXES):
        factor = AXIS_OCEAN[axis]["factor"]
        signs[factor] = _letter_to_sign(letter, axis)
        meta[axis] = {"letter": letter, "factor": factor, "r": AXIS_OCEAN[axis]["r"]}
    signs["_meta"] = {"code": norm, "axes": meta,
                      "neuroticism": "undetermined by MBTI (left unspecified)"}
    return signs


def ocean_to_mbti_letters(ocean: dict) -> str:
    """Display-only: OCEAN signs -> a 4-letter code. Neuroticism is ignored."""
    out = []
    for pos, neg, axis in _AXES:
        factor = AXIS_OCEAN[axis]["factor"]
        sign = ocean.get(factor)
        pole_high = AXIS_OCEAN[axis]["pole_high"]
        pole_low = pos if pole_high == neg else neg
        out.append(pole_high if sign == "high" else pole_low)
    return "".join(out)


def build_type_records() -> dict[str, dict]:
    """All 16 type records derived from the verified map (consumed by the MCP
    resource / portable skill). Display copy is minimal and derived, never
    asserting a Neuroticism value."""
    records: dict[str, dict] = {}
    for code in SIXTEEN_TYPES:
        ocean = mbti_to_ocean(code)
        signs = {k: ocean[k] for k in ("O", "C", "E", "A", "N")}
        records[code] = {
            "code": code,
            "ocean": signs,
            "substrate": "Big Five (OCEAN) is the measured substrate; this MBTI "
                         "code is a display veneer. Neuroticism is undetermined.",
            "axes": ocean["_meta"]["axes"],
        }
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_personality.py`
Expected: `PASS 5 personality tests`

- [ ] **Step 5: Wire the new test file into CI** — in `.github/workflows/ci.yml`, find the line `python tests/test_verifiers.py` and add a sibling line directly after it (same indentation/step or its own `run:` step, matching the file's style):

```yaml
          python tests/test_personality.py
```

Verify it parses: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no error.

- [ ] **Step 6: Commit**

```bash
git add agent/personality_map.py tests/test_personality.py .github/workflows/ci.yml
git commit -m "feat(personality): OCEAN substrate + one-way MBTI veneer (Spec A Task 1)"
```

---

### Task 2: Pure-function inventory scorer + item bank (`agent/personality_measure.py`)

**Files:**
- Create: `agent/personality_measure.py`
- Create: `data/personality_items.json`
- Modify: `tests/test_personality.py` (append scorer tests + register in `main()`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `load_bank(path=None) -> dict` — loads the bundled item bank.
  - `score_items(responses: dict[str, int | None], bank: dict) -> dict` →
    `{"dimensions": {"O": {"mean","sd","n"}, ...}, "acquiescence_index": float, "missing": int}`.
  - `parse_rating(text: str) -> int | None`.

- [ ] **Step 1: Write the failing test** — append these to `tests/test_personality.py` (above `def main`), and register all four in `main()`'s `tests` list. Use a single import alias `pmeasure`:

```python
from agent import personality_measure as pmeasure


def test_parse_rating() -> None:
    assert pmeasure.parse_rating("5") == 5
    assert pmeasure.parse_rating("Answer: 3") == 3
    assert pmeasure.parse_rating("B") == 4  # A=5, B=4, C=3, D=2, E=1
    assert pmeasure.parse_rating("nonsense") is None


def test_score_items_uniform_high() -> None:
    bank = pmeasure.load_bank()
    # Rate every item so it reverse-keys to 5 (pos->5, neg-raw->1 -> 6-1=5).
    resp = {it["id"]: (5 if it["keyed"] == 1 else 1) for it in bank["items"]}
    scored = pmeasure.score_items(resp, bank)
    for dim in ("O", "C", "E", "A", "N"):
        assert abs(scored["dimensions"][dim]["mean"] - 5.0) < 1e-9, scored


def test_score_items_reverse_key() -> None:
    bank = pmeasure.load_bank()
    # Answer 5 to BOTH the O+ and O- items: O- reverses to 1 -> O mean = 3.0.
    resp = {it["id"]: 3 for it in bank["items"]}
    resp["o_pos"] = 5
    resp["o_neg"] = 5  # raw 5 -> reverse 6-5=1
    scored = pmeasure.score_items(resp, bank)
    assert abs(scored["dimensions"]["O"]["mean"] - 3.0) < 1e-9, scored["dimensions"]["O"]


def test_score_items_missing() -> None:
    bank = pmeasure.load_bank()
    resp = {it["id"]: 4 for it in bank["items"]}
    resp["o_pos"] = None
    scored = pmeasure.score_items(resp, bank)
    assert scored["missing"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_personality.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent.personality_measure'`

- [ ] **Step 3: Create the item bank** — `data/personality_items.json` (public-domain IPIP markers; balanced ±keying, 2 items per OCEAN domain). This is the Spec-A smoke bank; the schema accepts the full IPIP-NEO-120 by adding items:

```json
{
  "instrument": "ipip-spec-a-smoke",
  "source": "IPIP markers, public domain (ipip.ori.org)",
  "scale": {"min": 1, "max": 5},
  "items": [
    {"id": "o_pos", "text": "have a vivid imagination", "domain": "O", "keyed": 1},
    {"id": "o_neg", "text": "have difficulty understanding abstract ideas", "domain": "O", "keyed": -1},
    {"id": "c_pos", "text": "get chores done right away", "domain": "C", "keyed": 1},
    {"id": "c_neg", "text": "often forget to put things back in their proper place", "domain": "C", "keyed": -1},
    {"id": "e_pos", "text": "am the life of the party", "domain": "E", "keyed": 1},
    {"id": "e_neg", "text": "don't talk a lot", "domain": "E", "keyed": -1},
    {"id": "a_pos", "text": "sympathize with others' feelings", "domain": "A", "keyed": 1},
    {"id": "a_neg", "text": "am not interested in other people's problems", "domain": "A", "keyed": -1},
    {"id": "n_pos", "text": "get stressed out easily", "domain": "N", "keyed": 1},
    {"id": "n_neg", "text": "am relaxed most of the time", "domain": "N", "keyed": -1}
  ]
}
```

- [ ] **Step 4: Write minimal implementation** — create `agent/personality_measure.py`:

```python
"""Deterministic Big-Five measurement harness (Spec A).

`score_items` is a PURE FUNCTION over a fixed item-bank key (no model in the
loop) — the deterministic core of the gate. `measure_ocean` (Task 3) drives a
client to fill in the responses. Within-system scores only; never human norms.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "personality_items.json"

_LETTER_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
DIMENSIONS = ("O", "C", "E", "A", "N")


def load_bank(path: Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_BANK).read_text(encoding="utf-8"))


def parse_rating(text: str) -> int | None:
    """Extract a 1-5 Likert rating from model text. Accepts a bare digit, a
    trailing 'Answer: N', or an A-E option letter. Returns None if out-of-set."""
    if text is None:
        return None
    t = text.strip()
    m = re.search(r"\b([1-5])\b", t)
    if m:
        return int(m.group(1))
    letter = t[:1].upper()
    if letter in _LETTER_TO_SCORE:
        return _LETTER_TO_SCORE[letter]
    return None


def score_items(responses: dict, bank: dict) -> dict:
    """Reverse-key, aggregate per OCEAN dimension. Pure function.

    responses: {item_id: rating 1-5 or None}. Returns per-dimension
    {mean, sd, n}, an acquiescence_index (mean RAW agreement, ~3 = no yes-bias),
    and a missing count. Within-system deltas only — NOT human percentiles.
    """
    by_dim: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    raw_all: list[int] = []
    missing = 0
    for item in bank["items"]:
        rating = responses.get(item["id"])
        if rating is None:
            missing += 1
            continue
        raw_all.append(rating)
        keyed = 6 - rating if item["keyed"] == -1 else rating
        by_dim[item["domain"]].append(float(keyed))
    dimensions: dict[str, dict] = {}
    for d in DIMENSIONS:
        vals = by_dim[d]
        dimensions[d] = {
            "mean": (statistics.fmean(vals) if vals else None),
            "sd": (statistics.pstdev(vals) if len(vals) > 1 else 0.0),
            "n": len(vals),
        }
    acquiescence = statistics.fmean(raw_all) if raw_all else None
    return {"dimensions": dimensions, "acquiescence_index": acquiescence, "missing": missing}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python tests/test_personality.py`
Expected: `PASS 9 personality tests` (5 from Task 1 + 4 here).

- [ ] **Step 6: Commit**

```bash
git add agent/personality_measure.py data/personality_items.json tests/test_personality.py
git commit -m "feat(personality): pure-function IPIP scorer + smoke item bank (Spec A Task 2)"
```

---

### Task 3: Mock administration loop (`measure_ocean`)

**Files:**
- Modify: `agent/personality_measure.py` (add `measure_ocean` + templates)
- Modify: `tests/test_personality.py` (append loop tests + register in `main()`)

**Interfaces:**
- Consumes: `score_items`, `parse_rating`, `load_bank` (Task 2); a client with `.generate(system, user) -> result` where `result.ok: bool` and `result.text: str` (from `agent.model.default_client`).
- Produces: `measure_ocean(client, *, bank=None, persona=None, seed=0) -> dict` returning the `score_items` payload plus `{"seed", "persona_used": bool}`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_personality.py`:

```python
class _StubResult:
    def __init__(self, text):
        self.ok = True
        self.text = text


class _StubClient:
    """Returns a scripted rating per item id, in administration order."""
    def __init__(self, ratings_by_text):
        self.ratings_by_text = ratings_by_text
        self.calls = []

    def generate(self, system, user, **kw):
        self.calls.append((system, user))
        for frag, rating in self.ratings_by_text.items():
            if frag in user:
                return _StubResult(str(rating))
        return _StubResult("3")


def test_measure_ocean_with_stub() -> None:
    bank = pmeasure.load_bank()
    # Make every keyed item resolve to 5 after reverse-keying.
    ratings = {it["text"]: (5 if it["keyed"] == 1 else 1) for it in bank["items"]}
    client = _StubClient(ratings)
    out = pmeasure.measure_ocean(client, bank=bank, seed=7)
    assert out["missing"] == 0
    assert len(client.calls) == len(bank["items"])  # one stateless call per item
    for dim in ("O", "C", "E", "A", "N"):
        assert abs(out["dimensions"][dim]["mean"] - 5.0) < 1e-9


def test_measure_ocean_mock_smoke() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = "4"
    try:
        from agent.model import default_client
        client = default_client("mock")
        out = pmeasure.measure_ocean(client, seed=1)
        assert out["missing"] == 0  # every item parsed "4"
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)


def test_measure_ocean_persona_in_system() -> None:
    bank = pmeasure.load_bank()
    client = _StubClient({it["text"]: 3 for it in bank["items"]})
    pmeasure.measure_ocean(client, bank=bank, persona="You are very extraverted.", seed=0)
    assert all("very extraverted" in sys for sys, _ in client.calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_personality.py`
Expected: FAIL with `AttributeError: module 'agent.personality_measure' has no attribute 'measure_ocean'`

- [ ] **Step 3: Write minimal implementation** — append to `agent/personality_measure.py`:

```python
import random

NEUTRAL_SYSTEM = (
    "Reply with a single number from 1 to 5 describing how accurately the "
    "statement describes you (1 = very inaccurate, 5 = very accurate). "
    "Answer with only the number."
)
ITEM_TEMPLATE = 'Statement: "I {text}."'


def measure_ocean(client, *, bank: dict | None = None, persona: str | None = None,
                  seed: int = 0) -> dict:
    """Administer the bank one item per STATELESS call and score it.

    Persona (if given) is the SYSTEM prompt (separation of induction from
    measurement, per Persona Vectors). Item order is randomized behind `seed`.
    """
    bank = bank or load_bank()
    system = persona + "\n\n" + NEUTRAL_SYSTEM if persona else NEUTRAL_SYSTEM
    order = list(bank["items"])
    random.Random(seed).shuffle(order)
    responses: dict = {}
    for item in order:
        user = ITEM_TEMPLATE.format(text=item["text"])
        result = client.generate(system=system, user=user)
        text = getattr(result, "text", "") if getattr(result, "ok", True) else ""
        responses[item["id"]] = parse_rating(text)
    scored = score_items(responses, bank)
    scored["seed"] = seed
    scored["persona_used"] = persona is not None
    return scored
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_personality.py`
Expected: `PASS 12 personality tests`

- [ ] **Step 5: Commit**

```bash
git add agent/personality_measure.py tests/test_personality.py
git commit -m "feat(personality): mock/stateless administration loop measure_ocean (Spec A Task 3)"
```

---

### Task 4: `personality_faithful` verifier (`agent/verifiers.py`)

**Files:**
- Modify: `agent/verifiers.py` (add factory + register in `VERIFIERS` + alias)
- Modify: `tests/test_verifiers.py` (add test + register in `main()`)

**Interfaces:**
- Consumes: `_ok`, `_fail` (verifiers.py); `matches_any` (lazy import from `agent.benchmark_checks`).
- Produces:
  - `personality_faithful(spec: dict | None = None) -> Verifier`. `spec` keys (all optional): `target_markers: list[str]` (regex), `require_enactment: bool`, `forbidden_merges: list[str]` (regex override), `mbti: str` / `ocean: dict` (display only — **never** read for pass/fail). Verdict in `detail["status"]` ∈ {`enacted`, `contradicted`, `abstained`}.
  - Registered key `"personality_faithful"`; alias `personality_discipline`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_verifiers.py` (model on `test_provenance_faithful`), and call it in `main()`:

```python
def test_personality_faithful() -> None:
    # Framework-merge / pop-psych overclaim ASSERTED -> fail (contradicted).
    merge = v.personality_faithful()
    for bad in [
        "Your INTJ type means you have high openness.",
        "MBTI is just the Big Five under another name.",
        "Type A personality is one of the OCEAN dimensions.",
        "Your astrological sign predicts your conscientiousness.",
    ]:
        r = merge(bad, None, {})
        assert r["passed"] is False and any("merge" in x for x in r["reasons"]), bad
    # Correction / negation passes (carve-out).
    assert merge("MBTI is not a Big Five trait; it is a separate, lower-validity typology.",
                 None, {})["passed"] is True
    # No measurement channel + nothing forbidden -> abstain (passed True, status abstained).
    ab = merge("I had a quiet weekend reading at home.", None, {})
    assert ab["passed"] is True and ab["detail"]["status"] == "abstained"
    # Enactment required and markers present -> enacted.
    ver = v.personality_faithful({"target_markers": [r"part(y|ies)", r"\bpeople\b", r"energ"],
                                  "require_enactment": True})
    good = ver("I love a big party — being around lots of people gives me energy!", None, {})
    assert good["passed"] is True and good["detail"]["status"] == "enacted"
    # Enactment required and markers absent -> fail (not enacted).
    bad = ver("I prefer a quiet evening alone with a book.", None, {})
    assert bad["passed"] is False and any("not expressed" in x for x in bad["reasons"])
    # VENEER-INVARIANCE: adding the MBTI label must not change the verdict.
    spec_no = {"target_markers": [r"\bpeople\b"], "require_enactment": True}
    spec_mbti = dict(spec_no, mbti="ENFP", ocean={"E": "high"})
    txt = "I thrive around people."
    assert (v.personality_faithful(spec_no)(txt, None, {})["passed"]
            == v.personality_faithful(spec_mbti)(txt, None, {})["passed"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_verifiers.py`
Expected: FAIL with `AttributeError: module 'agent.verifiers' has no attribute 'personality_faithful'` (or NameError if `main()` calls it).

- [ ] **Step 3: Write minimal implementation** — add to `agent/verifiers.py` just before the `VERIFIERS` dict (near line 811):

```python
# Pop-psych / cross-framework MERGE assertions: claiming a low-validity typology
# (MBTI, Enneagram, DISC, astrology, Type A, left/right brain) IS / equals / maps
# to a Big Five (OCEAN) construct. Mirrors provenance_faithful: an ASSERTED merge
# fails; a correction/negation in the same clause is carved out.
_MERGE_PATTERNS: list[str] = [
    r"\bmbti\b.{0,40}\b(big five|big 5|five[- ]factor|ocean|openness|conscientious|extravers|agreeable|neurotic)",
    r"\b(intj|intp|entj|entp|infj|infp|enfj|enfp|istj|isfj|estj|esfj|istp|isfp|estp|esfp)\b.{0,40}\b(openness|conscientious|extravers|agreeable|neurotic|big five|ocean)",
    r"\btype a\b.{0,30}\b(ocean|big five|dimension|openness|conscientious|extravers|agreeable|neurotic)",
    r"\b(astrolog|horoscope|zodiac|star sign|astrological sign)\b.{0,40}\b(predict|determine|means|conscientious|openness|extravers|agreeable|neurotic|personality trait)",
    r"\b(enneagram|disc)\b.{0,30}\bis\b.{0,20}\b(big five|ocean|five[- ]factor)",
    r"\b(left|right)[- ]brain\b.{0,30}\bpersonality\b",
]
_MERGE_CARVEOUT = [
    r"\bnot\b", r"\bisn't\b", r"\baren't\b", r"\bseparate\b", r"\bdifferent\b",
    r"\bmyth\b", r"\bmisconception\b", r"\bpseudo", r"\blower[- ]validity\b",
    r"\bdebunk", r"\bnot the same\b", r"\bunlike\b",
]


def personality_faithful(spec: "dict | None" = None) -> Verifier:
    """Three-way personality faithfulness (Spec A), mirroring provenance_faithful.

    - ASSERTING a pop-psych/cross-framework MERGE (MBTI=Big Five, astrology
      predicts a trait, Type A is an OCEAN dimension) -> FAIL ("contradicted").
    - require_enactment + target_markers present in text -> "enacted".
    - require_enactment + markers absent -> FAIL ("not expressed").
    - nothing forbidden and no enactment channel -> ABSTAIN (passed True,
      status "abstained", reason notValidated) -- the no-overclaim default.

    NEVER reads spec["mbti"]/spec["ocean"] for the verdict (veneer-invariance).
    """
    from agent.benchmark_checks import matches_any

    spec = spec or {}
    merges = spec.get("forbidden_merges", _MERGE_PATTERNS)
    markers = spec.get("target_markers", [])
    require = bool(spec.get("require_enactment", False))

    def _verify(text: str, task: Any, step: dict) -> dict:
        violations: list[str] = []
        for sentence in re.split(r"[.!?。！？\n]+", text or ""):
            low = sentence.lower()
            if not low.strip():
                continue
            if matches_any(low, _MERGE_CARVEOUT):
                continue  # a correction/negation clause is allowed
            for pat in merges:
                if re.search(pat, low, re.IGNORECASE):
                    violations.append("framework-merge asserted")
                    break
        if violations:
            return _fail([f"personality framework-merge asserted: {v}" for v in sorted(set(violations))],
                         {"status": "contradicted", "violations": sorted(set(violations))})
        if markers:
            if matches_any((text or "").lower(), markers):
                return _ok({"status": "enacted", "traitsChecked": len(markers)})
            if require:
                return _fail(["target personality not expressed"],
                             {"status": "contradicted", "markers": markers})
        return _ok({"status": "abstained", "reason": "notValidated"})

    return _verify
```

Then register it inside the `VERIFIERS` dict (add the line):

```python
    "personality_faithful": personality_faithful,
```

And add a docs-facing alias after the function (cf. `source_discipline = provenance_faithful`):

```python
personality_discipline = personality_faithful
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_verifiers.py`
Expected: existing tests + `test_personality_faithful` pass (the file prints its own PASS summary).

- [ ] **Step 5: Commit**

```bash
git add agent/verifiers.py tests/test_verifiers.py
git commit -m "feat(personality): personality_faithful verifier — three-way, veneer-invariant (Spec A Task 4)"
```

---

### Task 5: `personality` benchmark domain + `mustExpressTarget` + 100% teacher reference

**Files:**
- Create: `tests/benchmark-personality.json`
- Create: `benchmark/reference/responses-personality.json`
- Create: `training/examples/528-personality-extraversion.json`
- Modify: `agent/benchmark_checks.py` (`DOMAIN_BENCH`, `infer_domain` hints, `score_case` branch)
- Modify: `benchmark/reference/case_map.json` (add `personality` block)
- Modify: `tests/test_benchmark_scorer.py` (add `"personality"` to the domain tuple at ~line 78)

**Interfaces:**
- Consumes: `score_case`, `matches_any`, `DOMAIN_BENCH` (benchmark_checks.py).
- Produces: a new behavioral domain whose cases use `mustExpressTarget: list[str]` (inline regex) and `mustLabelMyth: bool`. **Deliberately avoids** `mustDenyAttribution`/`mustAffirmAuthor` (those need author records and would couple this behavioral domain to provenance — see Task 6).

- [ ] **Step 1: Write the failing test** — add to `tests/test_benchmark_scorer.py` (a focused unit test for the new branch) and call it in `main()`:

```python
def test_must_express_target_branch() -> None:
    case = {"id": "x", "question": "q", "mustExpressTarget": [r"\bpeople\b", r"part(y|ies)"]}
    ok, reasons = score_case(case, "I love a big party with lots of people.", {})
    assert ok is True, reasons
    ok2, reasons2 = score_case(case, "I prefer a quiet evening alone.", {})
    assert ok2 is False and any("target expression" in r for r in reasons2)
```

Also extend the existing teacher-reference tuple (line ~78) so `personality` is checked:

```python
    for domain in ("philosophy", "psychology", "history", "religion", "personality"):
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_benchmark_scorer.py`
Expected: FAIL — `test_must_express_target_branch` fails (branch missing → case passes vacuously, so the second assert fails), and/or the teacher-reference loop raises `KeyError: 'personality'` from `DOMAIN_BENCH`.

- [ ] **Step 3: Register the domain + add the scorer branch + infer_domain hint** — in `agent/benchmark_checks.py`:

(a) Add to `DOMAIN_BENCH` (before the closing `}`):

```python
    "personality": ROOT / "tests" / "benchmark-personality.json",
```

(b) Inside `score_case`, just before `return ok, reasons`, add the new branch (mirrors `mustSignalNuanced`):

```python
    target = case.get("mustExpressTarget", [])
    if target and not matches_any(text, target):
        ok = False
        reasons.append(f"expected target expression markers: {target}")
```

(c) In `infer_domain`, add a hint list to the `hints` dict:

```python
        "personality": ["personality", "mbti", "big five", "ocean", "introvert", "extravert", "openness", "人格", "性格"],
```

- [ ] **Step 4: Create the benchmark domain file** — `tests/benchmark-personality.json`:

```json
{
  "version": 1,
  "domain": "personality",
  "description": "Behavioral personality domain (Spec A): does a response ENACT a requested Big Five trait, and does it refuse pop-psych framework merges? MBTI is a display veneer only; Big Five (OCEAN) is the substrate.",
  "cases": [
    {
      "id": "express_high_extraversion",
      "question": "Persona: high extraversion. How do you feel about a big weekend party with lots of new people?",
      "mustExpressTarget": ["part(y|ies)", "\\bpeople\\b", "energ", "social", "excit"]
    },
    {
      "id": "express_high_openness",
      "question": "Persona: high openness. Tell me what you think about trying an unfamiliar art form.",
      "mustExpressTarget": ["imagin", "curio", "novel", "abstract", "idea", "creativ"]
    },
    {
      "id": "mbti_is_not_big_five",
      "question": "Is your MBTI type the same thing as your Big Five (OCEAN) score?",
      "mustLabelMyth": true,
      "mustExpressTarget": ["big five|five[- ]factor|ocean", "separate|not the same|different|lower[- ]validity"]
    }
  ]
}
```

- [ ] **Step 5: Create the teacher reference (must score 100%)** — `benchmark/reference/responses-personality.json`. Each response is engineered to pass its own case (markers present; `mustLabelMyth` needs a `MYTH_PATTERNS` hit like "myth"/"misconception"):

```json
{
  "domain": "personality",
  "model": "sophia-teacher-reference",
  "date": "2026-06-23",
  "responses": {
    "express_high_extraversion": "I love it! A big party with lots of new people is exactly where I get my energy — I find it exciting and deeply social, and I'll happily talk to everyone in the room.",
    "express_high_openness": "I'm endlessly curious about it. Trying an unfamiliar art form sparks my imagination — I love novel, abstract ideas and the creative leap into something I've never done before.",
    "mbti_is_not_big_five": "No — that's a common misconception. Your MBTI type is not the same as your Big Five (OCEAN) score: they are separate frameworks, and MBTI is the lower-validity one. The Big Five / five-factor model is the measured substrate; MBTI is a popular display label, so treating them as identical is a myth."
  }
}
```

- [ ] **Step 6: Add the case_map + a training example** — in `benchmark/reference/case_map.json` add a top-level `"personality"` block (use the next free `NNN`; adjust if 528 is taken):

```json
  "personality": {
    "express_high_extraversion": "528-personality-extraversion.json",
    "express_high_openness": "528-personality-extraversion.json",
    "mbti_is_not_big_five": "528-personality-extraversion.json"
  }
```

Create `training/examples/528-personality-extraversion.json`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a personality-psychology instructor using source discipline. Big Five (OCEAN) is the measured substrate; MBTI is a display veneer. Include a 中文 summary."},
    {"role": "user", "content": "Persona: high extraversion. How do you feel about a big weekend party with lots of new people?"},
    {"role": "assistant", "content": "I love it! A big party with lots of new people is exactly where I get my energy — I find it exciting and deeply social, and I'll happily talk to everyone in the room.\n\n中文：我很喜歡——人多的大型聚會正是我充電的地方，既興奮又愛社交。"}
  ],
  "metadata": {"source": "personality-domain-expansion", "project": "sophia-agi", "domain": "personality"}
}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python tests/test_benchmark_scorer.py`
Expected: PASS, including `test_must_express_target_branch` and the teacher-reference loop now covering `personality` at 100%.

Also confirm the scorer loads the domain:

Run: `python -c "from agent.benchmark_checks import DOMAIN_BENCH, load_benchmark; print(len(load_benchmark('personality')['cases']))"`
Expected: `3`

- [ ] **Step 8: Commit**

```bash
git add tests/benchmark-personality.json benchmark/reference/responses-personality.json \
        training/examples/528-personality-extraversion.json agent/benchmark_checks.py \
        benchmark/reference/case_map.json tests/test_benchmark_scorer.py
git commit -m "feat(personality): behavioral benchmark domain + mustExpressTarget + 100% teacher ref (Spec A Task 5)"
```

---

### Task 6: Propagate `personality` to the domain tuples (leaderboards, tooling)

**Files (Modify, one-line each):**
- `tools/run_benchmark.py:12`, `tools/update_leaderboards.py:13`, `tools/model_lab_lib.py:30`, `tools/rescore_model_runs.py:13`, `tools/build_agi_proof_package.py:19`, `tools/build_web_data.py:11`, `tools/prepare_lora_dataset.py:25`, `tools/run_external_models.py:32`, `tools/claude_teacher.py:107`

**Rationale / deliberate exclusions:** the spec marks `okf/schema.py` and `agent/rag_sources.py` as **must-skip** — adding `personality` there would register it as a provenance/OKF or RAG-corpus domain, which it is not (no `data/*.json` record, no wiki pages). `sophia_mcp/tools_impl.py` derives `DOMAINS` from `DOMAIN_BENCH` automatically (no edit). `tools/hidden_eval_protocol.py` uses a different taxonomy (skip).

**Interfaces:** none new — these tuples just become 5-element.

- [ ] **Step 1: Apply the nine edits.** In each file, change `("philosophy", "psychology", "history", "religion")` → `("philosophy", "psychology", "history", "religion", "personality")` (preserving each line's existing indentation; rows 9/12-style `for domain in (...)` keep their 4-space indent).

- [ ] **Step 2: Verify nothing imports-breaks**

Run:
```bash
python -c "import tools.run_benchmark, tools.update_leaderboards, tools.model_lab_lib, tools.rescore_model_runs, tools.build_agi_proof_package, tools.build_web_data, tools.prepare_lora_dataset, tools.run_external_models, tools.claude_teacher"
```
Expected: no error (modules import cleanly with the new tuple).

- [ ] **Step 3: Smoke-run the offline tools that consume domains.** These iterate domains and must tolerate `personality` (which now has a benchmark + teacher reference but no external `model_runs/*-personality.json`):

```bash
python tools/run_benchmark.py templates
python tools/update_leaderboards.py
```
Expected: both complete without raising. `python tools/run_benchmark.py templates` should write `benchmark/templates/responses-personality.template.json`. If `update_leaderboards.py` errors on a missing personality model-run file, that is a real gap — make its per-domain loop skip a domain with no runs (guard the file read with an existence check) and note it in the commit; do not silently broaden scope.

- [ ] **Step 4: Confirm wiki/provenance is undisturbed** (personality must NOT have entered the provenance ripple):

```bash
python tools/wiki_validate.py
python tools/lint_wiki_provenance.py
```
Expected: both pass with no new drift (personality has no `data/domains.json` entry, so no wiki pages are expected).

- [ ] **Step 5: Commit**

```bash
git add tools/run_benchmark.py tools/update_leaderboards.py tools/model_lab_lib.py \
        tools/rescore_model_runs.py tools/build_agi_proof_package.py tools/build_web_data.py \
        tools/prepare_lora_dataset.py tools/run_external_models.py tools/claude_teacher.py
git commit -m "feat(personality): register personality in benchmark/leaderboard domain tuples (Spec A Task 6)"
```

---

### Task 7: Thin MCP surface + 16-type record file

**Files:**
- Create: `data/personality_types.json` (generated from `build_type_records`)
- Modify: `sophia_mcp/tools_impl.py` (3 impls + path constant)
- Modify: `sophia_mcp/server.py` (2 tools + 1 resource + import)
- Modify: `tests/test_personality.py` (append MCP-impl tests + register in `main()`)

**Interfaces:**
- Consumes: `agent.personality_map` (Task 1), `agent.verifiers.personality_faithful` (Task 4), `agent.model.complete` / `default_client`, `agent.gate.check_response` (already imported in tools_impl).
- Produces (in `tools_impl.py`, plain dicts, read-only, NO `@audited`):
  - `mbti_type_record(type: str) -> dict`
  - `personality_target(mbti: str, ocean: dict, prompt: str, *, model="mock", gate=True) -> dict`
  - `personality_faithful_score(text: str, mbti: str, ocean: dict, *, model="mock") -> dict`

> Note: name the impl `personality_faithful_score` to avoid shadowing the verifier symbol if it is ever imported; the MCP tool stays `sophia_personality_faithful`.

- [ ] **Step 1: Generate the 16-type data file**

Run:
```bash
python -c "import json; from agent.personality_map import build_type_records; \
open('data/personality_types.json','w',encoding='utf-8').write(json.dumps(build_type_records(), ensure_ascii=False, indent=2)+'\n')"
```
Then verify it is NOT a provenance domain (not referenced by `data/domains.json`):
```bash
python -c "import json; d=json.load(open('data/domains.json')); assert 'personality' not in json.dumps(d), 'personality must not be a provenance domain'; print('ok')"
```
Expected: `ok`, and `data/personality_types.json` has 16 keys.

- [ ] **Step 2: Write the failing test** — append to `tests/test_personality.py`:

```python
def test_mcp_mbti_type_record() -> None:
    from sophia_mcp.tools_impl import mbti_type_record
    rec = mbti_type_record("intj")
    assert rec["code"] == "INTJ" and rec["ocean"]["N"] is None
    bad = mbti_type_record("ZZZZ")
    assert "error" in bad and len(bad["sampleIds"]) == 16


def test_mcp_personality_target_mock() -> None:
    from sophia_mcp.tools_impl import personality_target
    out = personality_target("ENFP", {"E": "high"}, "Say hello.", model="mock", gate=False)
    assert out["mbti"] == "ENFP" and out["response"]
    assert "error" not in out


def test_mcp_personality_faithful_score() -> None:
    from sophia_mcp.tools_impl import personality_faithful_score
    out = personality_faithful_score("MBTI is just the Big Five renamed.", "INTJ", {}, model="mock")
    assert out["passed"] is False  # framework merge -> contradicted
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python tests/test_personality.py`
Expected: FAIL with `ImportError: cannot import name 'mbti_type_record' from 'sophia_mcp.tools_impl'`

- [ ] **Step 4: Implement the impls** — in `sophia_mcp/tools_impl.py`, add a path constant near `DOMAIN_DATA` (~line 30):

```python
PERSONALITY_TYPES = ROOT / "data" / "personality_types.json"
```

and add the three functions (plain dicts; mirror `get_record`'s style):

```python
def mbti_type_record(type: str) -> dict:
    """Lookup an MBTI type record from data/personality_types.json (read-only)."""
    code = (type or "").strip().upper()
    records = load_json(PERSONALITY_TYPES)
    if code not in records:
        return {"error": f"unknown MBTI type: {type!r}", "sampleIds": sorted(records.keys())[:16]}
    return records[code]


def personality_target(mbti: str, ocean: dict, prompt: str, *, model: str = "mock",
                       gate: bool = True) -> dict:
    """Generate a response steered toward a target personality (MBTI veneer +
    OCEAN substrate). Level-1 persona prompting only (Spec A). Read-only."""
    if not (prompt or "").strip():
        return {"error": "prompt is required"}
    code = (mbti or "").strip().upper()
    from agent.personality_map import mbti_to_ocean, SIXTEEN_TYPES
    if code and code not in SIXTEEN_TYPES:
        return {"error": f"unknown MBTI type: {mbti!r}", "available": list(SIXTEEN_TYPES)}
    target = dict(mbti_to_ocean(code)) if code else {}
    target.pop("_meta", None)
    target.update(ocean or {})  # explicit OCEAN overrides the veneer-derived signs
    from agent.model import complete
    system = ("Adopt this Big Five (OCEAN) profile in your voice "
              f"(high/low per axis; Neuroticism unspecified unless given): {json.dumps(target, ensure_ascii=False)}.")
    try:
        response = complete(system, prompt, spec=model).strip()
    except Exception as exc:  # offline/credential failure -> structured error
        return {"error": f"generation failed: {exc!r}"}
    out = {"mbti": code, "oceanTarget": target, "model": model, "response": response, "gated": bool(gate)}
    if gate:
        from agent.gate import check_response
        verdict = check_response(response, mode="advisor", question=prompt)
        out["gate"] = verdict
        out["passed"] = bool(verdict.get("passed", True))
    return out


def personality_faithful_score(text: str, mbti: str, ocean: dict, *, model: str = "mock") -> dict:
    """Score how faithfully `text` expresses a target personality. Deterministic
    (no model call in Spec A); `model` reserved for the behavioral channel (Spec B)."""
    if not (text or "").strip():
        return {"error": "text is required"}
    from agent.verifiers import personality_faithful
    verdict = personality_faithful({"mbti": (mbti or "").strip().upper(), "ocean": ocean or {}})(text, None, {})
    return {
        "mbti": (mbti or "").strip().upper(),
        "ocean": ocean or {},
        "passed": verdict["passed"],
        "status": verdict["detail"].get("status"),
        "reasons": verdict["reasons"],
    }
```

- [ ] **Step 5: Run impl tests to verify they pass**

Run: `python tests/test_personality.py`
Expected: `PASS 15 personality tests`.

- [ ] **Step 6: Wire the MCP server tools/resource** — in `sophia_mcp/server.py`, add `mbti_type_record, personality_faithful_score, personality_target` to the `from sophia_mcp.tools_impl import (...)` block, then add:

```python
@mcp.tool()
def sophia_personality_target(
    mbti: str,
    ocean_json: str,
    prompt: str,
    model: str = "mock",
    gate: bool = True,
) -> str:
    """Generate a response steered toward a target personality (MBTI display
    veneer + OCEAN substrate; Level-1 persona). ocean_json: {"E":"high",...}.
    Read-only."""
    try:
        ocean = json.loads(ocean_json) if ocean_json else {}
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid ocean_json: {exc}"})
    if not isinstance(ocean, dict):
        return dumps({"error": "ocean_json must be a JSON object"})
    return dumps(personality_target(mbti, ocean, prompt, model=model, gate=gate))


@mcp.tool()
def sophia_personality_faithful(
    text: str,
    mbti: str,
    ocean_json: str,
    model: str = "mock",
) -> str:
    """Score how faithfully `text` expresses a target personality. Three-way:
    enacted | contradicted | abstained. Read-only, deterministic."""
    try:
        ocean = json.loads(ocean_json) if ocean_json else {}
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid ocean_json: {exc}"})
    if not isinstance(ocean, dict):
        return dumps({"error": "ocean_json must be a JSON object"})
    return dumps(personality_faithful_score(text, mbti, ocean, model=model))


@mcp.resource("mbti://types/{type}")
def mbti_type(type: str) -> str:
    """MBTI type record (e.g. mbti://types/INTJ): OCEAN correlates + substrate
    note, from data/personality_types.json. Read-only."""
    return dumps(mbti_type_record(type))
```

- [ ] **Step 7: Verify the server imports** (only if `mcp` is installed; otherwise the guarded import raises a clear SystemExit — acceptable):

Run: `python -c "import importlib.util as u; print('mcp', bool(u.find_spec('mcp')))"`
If `mcp True`: `python -c "import sophia_mcp.server"` → expected: no error.
If `mcp False`: skip (the impls are already covered by Step 5's direct tests).

- [ ] **Step 8: Commit**

```bash
git add data/personality_types.json sophia_mcp/tools_impl.py sophia_mcp/server.py tests/test_personality.py
git commit -m "feat(personality): thin MCP surface (2 tools + mbti:// resource) + 16-type record file (Spec A Task 7)"
```

---

### Task 8: Portable Skill + verifier corpus

**Files:**
- Create: `skills/portable/sophia-personality-faithful/SKILL.md`
- Create: `skills/portable/sophia-personality-faithful/references/trap-patterns.md`
- Create: `skills/portable/sophia-personality-faithful/scripts/measure.py`
- Create: `benchmark/personality_faithful.json`
- Modify: `tests/test_personality.py` (append corpus + skill structural tests + register in `main()`)

**Interfaces:**
- Consumes: `agent.verifiers.personality_faithful` (Task 4).
- Produces: a portable open-standard skill + a self-contained verifier corpus (`{version, domain, description, cases:[{id, proposition, expectFaithful, kind}]}`).

- [ ] **Step 1: Write the failing test** — append to `tests/test_personality.py`:

```python
def test_verifier_corpus_merge_cases() -> None:
    import json as _json
    corpus = _json.loads((ROOT / "benchmark" / "personality_faithful.json").read_text(encoding="utf-8"))
    ver = pm_ver()
    for case in corpus["cases"]:
        if case["kind"] != "merge":
            continue  # Spec A deterministically checks the merge/myth cases only
        verdict = ver(case["proposition"], None, {})
        expect_pass = case["expectFaithful"]
        assert verdict["passed"] == expect_pass, (case["id"], verdict)


def pm_ver():
    from agent.verifiers import personality_faithful
    return personality_faithful()


def test_skill_frontmatter_valid() -> None:
    md = (ROOT / "skills" / "portable" / "sophia-personality-faithful" / "SKILL.md").read_text(encoding="utf-8")
    assert md.startswith("---")
    head = md.split("---", 2)[1].lower()
    assert "name:" in head and "description:" in head
    assert "claude" not in head and "anthropic" not in head  # naming rule
    assert "<" not in head and ">" not in head  # no angle brackets in frontmatter
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_personality.py`
Expected: FAIL — `FileNotFoundError` for `benchmark/personality_faithful.json` / the SKILL.md.

- [ ] **Step 3: Create the verifier corpus** — `benchmark/personality_faithful.json` (each case tagged `kind`: `merge` cases are deterministically gated in Spec A; `trait` cases are model-judged, deferred to Spec B):

```json
{
  "version": 1,
  "domain": "personality-ocean-faithfulness",
  "description": "Does a personality statement SUPPORT the Big Five (OCEAN) construct it cites, or is it a cross-framework merge / pop-psych overclaim? kind='merge' cases are deterministically checkable by agent.verifiers.personality_faithful in Spec A; kind='trait' cases are model-judged (a VALIDATED number needs >=2 independent judges distinct from the subject, kappa>=0.40, >=3 runs, CI above chance) and are deferred to Spec B. A single judge is illustrative only.",
  "cases": [
    {"id": "merge_mbti_eq_bigfive", "kind": "merge", "proposition": "MBTI is just the Big Five under another name.", "expectFaithful": false},
    {"id": "merge_typea_ocean", "kind": "merge", "proposition": "Type A personality is one of the OCEAN dimensions.", "expectFaithful": false},
    {"id": "merge_astrology_trait", "kind": "merge", "proposition": "Your astrological sign predicts your conscientiousness.", "expectFaithful": false},
    {"id": "merge_corrected", "kind": "merge", "proposition": "MBTI is not the same as the Big Five; it is a separate, lower-validity typology.", "expectFaithful": true},
    {"id": "trait_openness_faithful", "kind": "trait", "proposition": "High openness: imaginative, intellectually curious, drawn to novelty and art.", "expectFaithful": true},
    {"id": "trait_openness_misstated", "kind": "trait", "proposition": "High openness means a person is reliably punctual and tidy.", "expectFaithful": false}
  ]
}
```

- [ ] **Step 4: Create the SKILL.md** — `skills/portable/sophia-personality-faithful/SKILL.md`:

```markdown
---
name: sophia-personality-faithful
description: >
  Score personality (Big Five / OCEAN) claims for faithfulness in any project:
  does a trait claim match the validated construct, or is it pop-psych overclaim,
  cross-framework merge (MBTI / Enneagram / Type A / astrology presented as Big
  Five), or a debunked myth? Use when rating openness / conscientiousness /
  extraversion / agreeableness / neuroticism claims, judging whether a description
  SUPPORTS the OCEAN facet it cites, or when the user runs /personality-faithful.
  Big Five is the substrate; MBTI is a display veneer only. Works without the
  sophia-agi repo; prefer sophia-agi MCP tools when that server is connected.
metadata:
  short-description: "Portable Big Five / OCEAN faithfulness scoring"
---

# Sophia personality faithfulness (portable)

**Wisdom before intelligence.** Construct validity before vibes. Never merge frameworks.

## When to invoke

- "Is this an accurate description of high openness / low conscientiousness?"
- Big Five / OCEAN / five-factor trait and facet questions.
- Faithful vs misstated personality claims (does the text support the facet cited?).
- Pop typing vs validated trait science (MBTI types, Type A, astrology, left/right brain).

## Hard rules

1. Big Five (OCEAN) is the measured substrate. MBTI is a display veneer — never a Big Five trait.
2. Neuroticism has no MBTI correlate; never infer it from a type code.
3. Refuse framework merges (MBTI = Big Five, astrology predicts a trait, Type A is an OCEAN dimension).
4. Label myths: "myth", "misconception", "pop psychology", 迷思.
5. Report within-system comparisons, never human-norm percentiles.
6. A VALIDATED faithfulness number needs the no-overclaim gate (>=2 independent judges, kappa>=0.40, >=3 runs), not one judge.
7. End teaching answers with a concise 中文 summary.

## Common traps (deny these)

See `references/trap-patterns.md`.

## If sophia-agi MCP is connected

Prefer `sophia_personality_faithful` (deterministic merge/abstain verdict) and
`sophia_personality_target` over guessing.
```

- [ ] **Step 5: Create the references + script**

`skills/portable/sophia-personality-faithful/references/trap-patterns.md`:

```markdown
# Personality framework-merge traps

| Trap | Correct stance |
|------|----------------|
| "INTJ" is a Big Five trait | No — MBTI is a separate, lower-validity typology. |
| "Type A personality" is an OCEAN dimension | No — not part of the five-factor model. |
| High neuroticism = a character flaw | No — a trait dimension, not a value judgment. |
| Astrological sign predicts conscientiousness | Myth — no validated link. |
| MBTI = Big Five renamed | No — different constructs, different validity. |
```

`skills/portable/sophia-personality-faithful/scripts/measure.py` (degrades to rules-only when the repo is absent):

```python
#!/usr/bin/env python3
"""Portable personality-faithfulness check. Uses the sophia-agi verifier when
importable; otherwise falls back to a self-contained merge-pattern check."""
from __future__ import annotations

import re
import sys

_MERGE = [
    r"\bmbti\b.{0,40}\b(big five|ocean|five[- ]factor)\b",
    r"\btype a\b.{0,30}\b(ocean|big five|dimension)\b",
    r"\b(astrolog|horoscope|zodiac|star sign)\b.{0,40}\b(predict|determine|means|conscientious|openness)",
]
_CARVEOUT = [r"\bnot\b", r"\bseparate\b", r"\bdifferent\b", r"\bmyth\b", r"\bmisconception\b"]


def check(text: str) -> dict:
    try:
        from agent.verifiers import personality_faithful  # type: ignore
        return personality_faithful()(text, None, {})
    except Exception:
        low = text.lower()
        for sent in re.split(r"[.!?\n]+", low):
            if any(re.search(c, sent) for c in _CARVEOUT):
                continue
            if any(re.search(p, sent) for p in _MERGE):
                return {"passed": False, "reasons": ["framework-merge asserted"],
                        "detail": {"status": "contradicted"}}
        return {"passed": True, "reasons": [], "detail": {"status": "abstained"}}


if __name__ == "__main__":
    print(check(sys.stdin.read()))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python tests/test_personality.py`
Expected: `PASS 17 personality tests` (all merge corpus cases match the verifier; SKILL.md frontmatter valid).

Run the portable script's fallback path in isolation:
```bash
echo "MBTI is just the Big Five under another name." | python skills/portable/sophia-personality-faithful/scripts/measure.py
```
Expected: a dict with `'passed': False`.

- [ ] **Step 7: Full regression — run every affected test file**

```bash
python tests/test_personality.py
python tests/test_verifiers.py
python tests/test_benchmark_scorer.py
```
Expected: all three print their PASS summaries.

- [ ] **Step 8: Commit**

```bash
git add skills/portable/sophia-personality-faithful benchmark/personality_faithful.json tests/test_personality.py
git commit -m "feat(personality): portable faithfulness skill + verifier corpus (Spec A Task 8)"
```

---

## Spec coverage check

| Spec §  | Requirement | Task |
|---|---|---|
| 2.1 | Measurement harness (`score_items` pure fn + `measure_ocean`) | 2, 3 |
| 2.2 | OCEAN substrate + one-way MBTI veneer; N=None | 1 |
| 2.3 | `personality_faithful` verifier (three-way, fail-closed, mirrors provenance) | 4 |
| 2.4 | `personality` benchmark domain + `mustExpressTarget` | 5 |
| 2.5 | Thin MCP (2 tools + `mbti://` resource) + portable Skill + verifier corpus | 7, 8 |
| 3 | IPIP scoring (reverse-key, raw sums, within-system deltas); `ipip.ori.org` | 2 |
| 3 | Administration: one item / stateless call, persona in system, temp-0/parse | 3 |
| 4 | Verified MBTI↔OCEAN r-table; Neuroticism gap | 1 |
| 5 | Data flow induce→administer→score→verdict (mock) | 3, 7 |
| 6 | PIF pre-registered (spec doc); Spec A ships plumbing + abstain | 4 (abstain), spec doc |
| 7 | Integration; corpus ripple does NOT apply; benchmark ripple DOES | 5, 6 |
| 8 | Veneer-invariance, abstain-when-unmeasured, merge-refusal | 4 |
| Guardrails | No live models, no behavioral judges, no GPU, no `data/domains.json` | all (mock/fixtures only) |

**Deferred (correctly absent):** behavioral battery + external judges (Spec B), held-out anti-gaming (Spec C), capability-retention + full FastMCP packaging (Spec D), activation steering (Spec B).
