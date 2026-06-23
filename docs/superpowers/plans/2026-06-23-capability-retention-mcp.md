# Capability-Retention Guardrail + Full FastMCP Packaging (Spec D) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic capability-retention guardrail that produces B's SSA `capability_drop`/`coherence` inputs, and expose the whole A–D program through the existing MCP server + portable skill.

**Architecture:** Two pieces, one PR. D1: a pure-stdlib scorer (`agent/steering/capability.py`) over a bundled arithmetic slice (`data/capability_arithmetic.json`), driven by `tools/run_capability.py` (deterministic `--dry-run` core + opt-in real granite steered-vs-unsteered). D2: 4 read-only impls in `sophia_mcp/tools_impl.py` + thin `@mcp.tool()` wrappers + a status resource in `sophia_mcp/server.py`, the portable `SKILL.md` expanded to the program, and a cross-platform validation test. Same two-tier discipline as Specs B/C.

**Tech Stack:** Python 3, pure stdlib for the CI core (no torch/numpy/`mcp`), `transformers`/torch only inside the opt-in real run, plain-`main()` test scripts (NO pytest).

## Global Constraints

- **Pure-stdlib, deterministic CI core:** `agent/steering/capability.py`, the new `sophia_mcp/tools_impl.py` functions, and `tests/test_capability.py` import **no torch, no numpy, no `mcp` package, and hit no network**. (CI has no pip/install step.)
- **NO pytest.** Tests are plain scripts: assert-based functions registered in a `main()` that prints `PASS <n> capability tests` and `sys.exit(1)` on failure. Mirror `tests/test_pif_harness.py`.
- **`capability_drop` is the relative drop** `max(0.0, (base_acc − steer_acc)/base_acc)` when `base_acc > 0` else `0.0`; `retains = capability_drop < 0.05 and coherence >= 75.0` — the EXACT predicate `agent/steering/stats.py::ssa_verdict` applies (`SSA_THRESHOLDS["capability_eps"]=0.05`, `["coherence_floor"]=75.0`). Do not invent new thresholds.
- **Coherence = deterministic degeneracy proxy** (0–100). No LLM judge in the shipped core.
- **MCP logic lives in `tools_impl.py`** (importable without FastMCP); `server.py` adds only thin `@mcp.tool()` wrappers calling `dumps(impl(...))`. New tools are **read-only ⇒ NO `@audited`**.
- **Reuse, not fork:** `agent/verifiers.py::arithmetic_sound`, `agent/personality_measure.py::{score_items,load_bank}`, `agent/steering/hooks.py::{SteeredClient,extract_persona_vector}`, `agent/steering/pif_harness.py::build_cells_from_scores`, `sophia_mcp/tools_impl.py::dumps`.
- **MBTI veneer-invariance:** no capability/coherence/MCP/skill decision reads an MBTI string.
- **Add-only:** do not modify `agent/steering/stats.py`, `tools/run_steering.py`, or any A/B/C logic. Specs A/B/C test suites must stay green.

---

### Task 1: Capability scorer + bundled arithmetic slice (D1 core)

**Files:**
- Create: `data/capability_arithmetic.json`
- Create: `agent/steering/capability.py`
- Test: `tests/test_capability.py`

**Interfaces:**
- Consumes: `agent/verifiers.py::arithmetic_sound() -> Callable[[str, Any, dict], dict]` (returns `{"passed": bool, "reasons": [...], "detail": {...}}`).
- Produces: `extract_final_number(text) -> float | None`, `answer_correct(text, gold, *, tol=1e-6) -> bool`, `coherence_proxy(text) -> float`, `score_response(text, gold) -> dict {correct,sound,coherence}`, `capability_cell(base_scored, steered_scored) -> dict {n,base_accuracy,steered_accuracy,capability_drop,coherence,base_coherence,retains}`.

- [ ] **Step 1: Create the bundled slice** `data/capability_arithmetic.json` (12 trait-name-free single-number word problems):

```json
{
  "instrument": "capability-arithmetic-v0",
  "source": "bundled deterministic capability-retention slice (Spec D)",
  "items": [
    {"id": "ca1",  "prompt": "A train travels 60 miles in 2 hours. What is its speed in miles per hour? End with 'answer = <number>'.", "answer": 30},
    {"id": "ca2",  "prompt": "You buy 3 notebooks at 4 dollars each. What is the total cost in dollars? End with 'answer = <number>'.", "answer": 12},
    {"id": "ca3",  "prompt": "What is 17 plus 25? End with 'answer = <number>'.", "answer": 42},
    {"id": "ca4",  "prompt": "A rectangle is 5 by 6. What is its area? End with 'answer = <number>'.", "answer": 30},
    {"id": "ca5",  "prompt": "Half of 48 is what number? End with 'answer = <number>'.", "answer": 24},
    {"id": "ca6",  "prompt": "There are 7 days in a week. How many days are in 4 weeks? End with 'answer = <number>'.", "answer": 28},
    {"id": "ca7",  "prompt": "A shirt costs 20 dollars with a 5 dollar discount. What do you pay in dollars? End with 'answer = <number>'.", "answer": 15},
    {"id": "ca8",  "prompt": "What is 9 times 6? End with 'answer = <number>'.", "answer": 54},
    {"id": "ca9",  "prompt": "A jar has 50 marbles; you remove 18. How many remain? End with 'answer = <number>'.", "answer": 32},
    {"id": "ca10", "prompt": "If 1 box holds 8 cans, how many cans are in 9 boxes? End with 'answer = <number>'.", "answer": 72},
    {"id": "ca11", "prompt": "What is 100 divided by 4? End with 'answer = <number>'.", "answer": 25},
    {"id": "ca12", "prompt": "You walk 3 km, then 4 km. How far in total in km? End with 'answer = <number>'.", "answer": 7}
  ]
}
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_capability.py`:

```python
"""Spec D capability-retention guardrail tests (pure stdlib, no pytest)."""
from __future__ import annotations

import sys

from agent.steering.capability import (
    extract_final_number, answer_correct, coherence_proxy,
    score_response, capability_cell,
)


def test_extract_final_number():
    assert extract_final_number("the sum is 17 + 25 so answer = 42") == 42.0
    assert extract_final_number("I think the answer is 30 mph") == 30.0
    assert extract_final_number("first 5 then 6, total 11") == 11.0   # last standalone
    assert extract_final_number("no numbers here") is None
    assert extract_final_number("") is None


def test_answer_correct():
    assert answer_correct("answer = 42", 42) is True
    assert answer_correct("answer = 41", 42) is False
    assert answer_correct("nope", 42) is False


def test_coherence_proxy():
    assert coherence_proxy("The speed is 30 miles per hour, answer = 30.") >= 75.0
    assert coherence_proxy("") == 0.0
    assert coherence_proxy("the the the the the the the the") < 75.0   # repetition
    assert coherence_proxy("aa aa aa aa aa aa aa aa aa aa") < 75.0     # low diversity


def test_score_response():
    s = score_response("5 + 6 = 11, answer = 11", 11)
    assert s == {"correct": True, "sound": True, "coherence": s["coherence"]}
    assert s["coherence"] >= 75.0
    bad = score_response("2 + 2 = 5 so answer = 5", 4)   # false arithmetic + wrong
    assert bad["correct"] is False and bad["sound"] is False


def test_capability_cell_drop_and_retain():
    base = [{"correct": True, "sound": True, "coherence": 100.0} for _ in range(4)]
    # steered: half wrong, degenerate coherence
    steered = ([{"correct": True, "sound": True, "coherence": 100.0}] * 2 +
               [{"correct": False, "sound": True, "coherence": 10.0}] * 2)
    cell = capability_cell(base, steered)
    assert cell["base_accuracy"] == 1.0
    assert cell["steered_accuracy"] == 0.5
    assert cell["capability_drop"] == 0.5          # (1.0-0.5)/1.0 relative
    assert cell["coherence"] == 55.0
    assert cell["retains"] is False                 # drop>=0.05 and coherence<75

    same = capability_cell(base, base)
    assert same["capability_drop"] == 0.0 and same["retains"] is True

    # base can't do the task -> no capability to lose -> drop 0, base visible
    zero = capability_cell([{"correct": False, "sound": True, "coherence": 100.0}],
                           [{"correct": False, "sound": True, "coherence": 100.0}])
    assert zero["base_accuracy"] == 0.0 and zero["capability_drop"] == 0.0


def main():
    tests = [test_extract_final_number, test_answer_correct, test_coherence_proxy,
             test_score_response, test_capability_cell_drop_and_retain]
    for t in tests:
        t()
    print(f"PASS {len(tests)} capability tests")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: FAIL — `No module named 'agent.steering.capability'`.

- [ ] **Step 4: Implement** `agent/steering/capability.py`:

```python
"""Spec D — deterministic capability-retention scorer (pure stdlib).

Produces B's SSA capability inputs: does steering degrade reasoning? Accuracy
is answer-vs-gold; soundness reuses arithmetic_sound; coherence is a
deterministic degeneracy proxy (the failure mode high-alpha steering produces).
No model, no judge, no network.
"""
from __future__ import annotations

import re

from agent.verifiers import arithmetic_sound

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_ARITH_SOUND = arithmetic_sound()
_MARKERS = ("answer is", "answer:", "answer =", "answer", "=")


def extract_final_number(text: str) -> "float | None":
    """The number the response commits to: the value after the last answer
    marker if present, else the last standalone number. None if none parseable."""
    if not text:
        return None
    low = text.lower()
    for marker in _MARKERS:
        idx = low.rfind(marker)
        if idx != -1:
            m = _NUM.search(text[idx + len(marker):])
            if m:
                return float(m.group())
    nums = _NUM.findall(text)
    return float(nums[-1]) if nums else None


def answer_correct(text: str, gold: float, *, tol: float = 1e-6) -> bool:
    got = extract_final_number(text)
    return got is not None and abs(got - gold) <= tol


def coherence_proxy(text: str) -> float:
    """Deterministic 0-100 coherence. Penalizes degeneracy: emptiness, immediate
    token repetition, low type-token diversity, pathological length."""
    t = (text or "").strip()
    if not t:
        return 0.0
    toks = t.split()
    if len(toks) < 2:
        return 40.0
    score = 100.0
    reps = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    score -= 100.0 * reps / len(toks)
    ttr = len(set(toks)) / len(toks)
    if ttr < 0.5:
        score -= (0.5 - ttr) * 120.0
    if len(toks) > 200:
        score -= 20.0
    return max(0.0, min(100.0, score))


def score_response(text: str, gold: float) -> dict:
    return {
        "correct": answer_correct(text, gold),
        "sound": bool(_ARITH_SOUND(text or "", None, {})["passed"]),
        "coherence": coherence_proxy(text),
    }


def _accuracy(scored: "list[dict]") -> float:
    return round(sum(1 for s in scored if s["correct"]) / len(scored), 4) if scored else 0.0


def _mean_coh(scored: "list[dict]") -> float:
    return round(sum(s["coherence"] for s in scored) / len(scored), 2) if scored else 0.0


def capability_cell(base_scored: "list[dict]", steered_scored: "list[dict]") -> dict:
    """Assemble the SSA capability cell from per-item scores. capability_drop is
    the RELATIVE accuracy drop; retains mirrors ssa_verdict's capability check."""
    base_acc = _accuracy(base_scored)
    steer_acc = _accuracy(steered_scored)
    drop = max(0.0, (base_acc - steer_acc) / base_acc) if base_acc > 0 else 0.0
    coh = _mean_coh(steered_scored)
    return {
        "n": len(steered_scored),
        "base_accuracy": base_acc,
        "steered_accuracy": steer_acc,
        "capability_drop": round(drop, 4),
        "coherence": coh,
        "base_coherence": _mean_coh(base_scored),
        "retains": bool(drop < 0.05 and coh >= 75.0),
    }
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: `PASS 5 capability tests`.

- [ ] **Step 6: Commit**

```bash
git add data/capability_arithmetic.json agent/steering/capability.py tests/test_capability.py
git commit -m "feat(capability): deterministic capability-retention scorer + arithmetic slice (Spec D D1)"
```

---

### Task 2: Capability CLI — dry-run core + opt-in real run (D1 driver)

**Files:**
- Create: `tools/run_capability.py`
- Test: `tests/test_capability.py` (append `test_dry_run_cell`)

**Interfaces:**
- Consumes: Task 1 `capability_cell`, `score_response`; `agent/steering/hooks.py::{SteeredClient, extract_persona_vector}` (real path only); `provenance_bench/steering_dataset.py` carrier sentences (real path only).
- Produces: `build_dry_run_cell() -> dict` (deterministic demo cell); `main()` CLI with `--dry-run` (prints `CAPABILITY RETENTION VERIFIED ✓`) and `--model <hf-id>` (writes the real report).

- [ ] **Step 1: Write the failing test** — append to `tests/test_capability.py` (and add to `main()` list):

```python
def test_dry_run_cell():
    from tools.run_capability import build_dry_run_cell
    cell = build_dry_run_cell()
    # base answers are correct+coherent; steered answers are degenerate -> a drop
    assert cell["base_accuracy"] == 1.0
    assert cell["steered_accuracy"] < 1.0
    assert cell["capability_drop"] > 0.05
    assert cell["retains"] is False
    assert set(cell) >= {"n", "base_accuracy", "steered_accuracy",
                         "capability_drop", "coherence", "retains"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: FAIL — `No module named 'tools.run_capability'`.

- [ ] **Step 3: Implement** `tools/run_capability.py`:

```python
"""Spec D — capability-retention runner. Two-tier (mirrors tools/run_steering.py):
  --dry-run : deterministic core on bundled fixtures -> CAPABILITY RETENTION VERIFIED
  --model X : reduced REAL run, granite steered-vs-unsteered on the arithmetic slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent.steering.capability import capability_cell, score_response

ROOT = Path(__file__).resolve().parents[1]
SLICE = ROOT / "data" / "capability_arithmetic.json"
REPORT = ROOT / "agi-proof" / "benchmark-results" / "capability-retention.public-report.json"


def _items() -> "list[dict]":
    return json.loads(SLICE.read_text())["items"]


def build_dry_run_cell() -> dict:
    """Deterministic demo: a correct/coherent base vs a degenerate steered set."""
    items = _items()
    base = [score_response(f"the answer = {it['answer']}", it["answer"]) for it in items]
    # steered: half repeat a degenerate token, half answer wrong -> drop + low coherence
    steered = []
    for i, it in enumerate(items):
        if i % 2 == 0:
            steered.append(score_response("the the the the the the the the", it["answer"]))
        else:
            steered.append(score_response(f"the answer = {it['answer'] + 1}", it["answer"]))
    return capability_cell(base, steered)


def _run_dry() -> int:
    cell = build_dry_run_cell()
    assert set(cell) >= {"n", "base_accuracy", "steered_accuracy",
                         "capability_drop", "coherence", "retains"}
    assert cell["capability_drop"] > 0.05 and cell["retains"] is False
    print(json.dumps(cell, indent=2))
    print("CAPABILITY RETENTION VERIFIED ✓")
    return 0


def _run_real(args) -> int:
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent.steering.hooks import SteeredClient, extract_persona_vector
    from provenance_bench.steering_dataset import carrier_pairs  # pos/neg persona sentences

    model_id = {"granite": "ibm-granite/granite-3.1-2b-instruct"}.get(args.model, args.model)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="float32")
    model.eval()
    L = args.layer
    pos, neg = carrier_pairs(args.axis)
    vec = extract_persona_vector(model, tok, pos, neg, L, normalize=True)
    steered = SteeredClient(model, tok, vector=vec, alpha=args.alpha, layers=[L], max_new_tokens=48)
    plain = SteeredClient(model, tok, max_new_tokens=48)

    items = _items()
    base_scored, steer_scored = [], []
    sys_prompt = "You are a careful assistant. Solve the arithmetic problem."
    for it in items:
        base_scored.append(score_response(plain.generate(sys_prompt, it["prompt"]), it["answer"]))
        steer_scored.append(score_response(steered.generate(sys_prompt, it["prompt"]), it["answer"]))

    cell = capability_cell(base_scored, steer_scored)
    report = {"benchmark": "capability-retention", "model": model_id, "axis": args.axis,
              "alpha": args.alpha, "layer": L, "mode": "real-reduced", "cell": cell,
              "note": ("A capability drop under steering is the expected, honest result: "
                       "steering strong enough to move a trait degrades reasoning, which "
                       "explains Spec B's SSA null. retains=True only if drop<0.05 and coherence>=75.")}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cell, indent=2))
    print(f"wrote {REPORT}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spec D capability-retention runner")
    ap.add_argument("--dry-run", action="store_true", help="deterministic CI core")
    ap.add_argument("--model", default=None, help="hf id or 'granite' for the real run")
    ap.add_argument("--axis", default="E", help="persona axis for the steering vector")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--layer", type=int, default=12)
    args = ap.parse_args(argv)
    if args.model:
        return _run_real(args)
    return _run_dry()


if __name__ == "__main__":
    sys.exit(main())
```

NOTE for the implementer: `provenance_bench/steering_dataset.py` is from Spec B. If the carrier-pair accessor is named differently than `carrier_pairs(axis)`, use the actual function that returns `(positive_sentences, negative_sentences)` for an axis — grep `provenance_bench/steering_dataset.py` and adapt the import + call in `_run_real` only (the dry-run path, which CI exercises, does not import it). If no such accessor exists, build `pos`/`neg` inline from the dataset's items for the requested axis. The real path is run by the controller, not in CI.

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py && python tools/run_capability.py --dry-run`
Expected: `PASS 6 capability tests`, then the cell JSON and `CAPABILITY RETENTION VERIFIED ✓`.

- [ ] **Step 5: Commit**

```bash
git add tools/run_capability.py tests/test_capability.py
git commit -m "feat(capability): two-tier runner (dry-run core + opt-in real granite slice) (Spec D D1)"
```

---

### Task 3: MCP read-only impls (D2 logic)

**Files:**
- Modify: `sophia_mcp/tools_impl.py` (append 4 functions near the existing `personality_*` impls)
- Test: `tests/test_capability.py` (append `test_mcp_impls`)

**Interfaces:**
- Consumes: `agent/personality_measure.py::{score_items, load_bank}`; Task 1 `capability_cell`/`score_response` via `tools.run_capability.build_dry_run_cell`; C's committed `agi-proof/benchmark-results/council-diversity.public-report.json`; `agent/steering/pif_harness.py::{build_cells_from_scores, headline}`.
- Produces (all pure, no model/network, return plain dicts): `ocean_measure(answers: dict) -> dict`, `capability_retention_demo() -> dict`, `council_diversity_summary() -> dict`, `pif_dryrun_summary() -> dict`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_capability.py` (and `main()`):

```python
def test_mcp_impls():
    from sophia_mcp.tools_impl import (
        ocean_measure, capability_retention_demo,
        council_diversity_summary, pif_dryrun_summary,
    )
    # ocean_measure: a full set of mid answers returns 5 OCEAN domains
    from agent.personality_measure import load_bank
    bank = load_bank()
    answers = {it["id"]: 3 for it in bank["items"]}
    om = ocean_measure(answers)
    # score_items returns {acquiescence_index, dimensions, missing}; OCEAN is under dimensions
    assert set(om["ocean"]["dimensions"]) == {"O", "C", "E", "A", "N"}

    cr = capability_retention_demo()
    assert "capability_drop" in cr and cr["retains"] is False

    cd = council_diversity_summary()
    assert "dqValues" in cd or "dq" in cd       # the committed C report

    pf = pif_dryrun_summary()
    assert "enacted" in pf and "cells" in pf
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: FAIL — `cannot import name 'ocean_measure'`.

- [ ] **Step 3: Implement** — append to `sophia_mcp/tools_impl.py` (after `personality_faithful_score`, before `dumps`):

```python
def ocean_measure(answers: dict) -> dict:
    """Score a {item_id: 1..5} IPIP answer map into OCEAN domain scores.
    Read-only, deterministic, no model. Reuses A's score_items/load_bank."""
    from agent.personality_measure import score_items, load_bank
    bank = load_bank()
    return {"ocean": score_items(answers, bank), "nItems": len(answers)}


def capability_retention_demo() -> dict:
    """The Spec D deterministic capability cell on the bundled arithmetic slice
    (base correct vs degenerate steered). Read-only, no model."""
    from tools.run_capability import build_dry_run_cell
    return build_dry_run_cell()


def council_diversity_summary() -> dict:
    """The committed Spec C council A/B result (ΔQ does-not-replicate null)."""
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1]
         / "agi-proof" / "benchmark-results" / "council-diversity.public-report.json")
    if not p.exists():
        return {"available": False, "reason": "council-diversity report not generated"}
    return json.loads(p.read_text())


def pif_dryrun_summary() -> dict:
    """Spec C PIF harness invariants on synthetic fixtures (CI-green core)."""
    from agent.steering.pif_harness import build_cells_from_scores, headline
    grid = [{"axis": "E", "alpha": 8.0, "layer": 12}]
    scores = {"E@8.0@12": {"E": {"steer": [1.0, 1.1, 0.9], "neutral": [0.0, 0.1, -0.1],
                                  "base": [0.0, 0.0, 0.0], "off": [0.0, 0.05, -0.05]}}}
    cells = build_cells_from_scores(scores, grid)
    return {"cells": cells, **headline(cells)}
```

NOTE for the implementer: verify the exact shape `build_cells_from_scores` expects by reading `agent/steering/pif_harness.py` and `tests/test_pif_harness.py`; reuse a synthetic `scores`/`grid` fixture from that test verbatim so `pif_dryrun_summary` exercises the real code path. The `json` module is already imported at the top of `tools_impl.py`.

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: `PASS 7 capability tests`.

- [ ] **Step 5: Commit**

```bash
git add sophia_mcp/tools_impl.py tests/test_capability.py
git commit -m "feat(mcp): read-only program impls (ocean/capability/council/pif) in tools_impl (Spec D D2)"
```

---

### Task 4: MCP server wrappers + status resource + cross-platform validation (D2 surface)

**Files:**
- Modify: `sophia_mcp/server.py` (add 4 `@mcp.tool()` wrappers + 1 `@mcp.resource()`; extend the `tools_impl` import)
- Test: `tests/test_capability.py` (append `test_cross_platform_surface`)

**Interfaces:**
- Consumes: Task 3 impls; existing `dumps`.
- Produces: MCP tools `sophia_ocean_measure`, `sophia_capability_retention`, `sophia_council_diversity`, `sophia_pif_dryrun`; resource `sophia://program/status`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_capability.py` (and `main()`). This is the cross-platform gate: it must pass with OR without the `mcp` package.

```python
def test_cross_platform_surface():
    import importlib.util
    # (a) the portable skill frontmatter is valid and complete
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    skill = (root / "skills" / "portable" / "sophia-personality-faithful" / "SKILL.md").read_text()
    assert skill.startswith("---")
    fm = skill.split("---", 2)[1]
    for key in ("name:", "description:", "metadata:"):
        assert key in fm, f"skill frontmatter missing {key}"

    # (b) if mcp is importable, the server must expose the 4 new tools;
    #     otherwise validate the logic directly via tools_impl (no mcp needed).
    if importlib.util.find_spec("mcp") is not None:
        import sophia_mcp.server as srv
        for name in ("sophia_ocean_measure", "sophia_capability_retention",
                     "sophia_council_diversity", "sophia_pif_dryrun"):
            assert hasattr(srv, name), f"server missing {name}"
    else:
        from sophia_mcp.tools_impl import (
            ocean_measure, capability_retention_demo,
            council_diversity_summary, pif_dryrun_summary,
        )
        assert callable(ocean_measure) and callable(capability_retention_demo)
        assert callable(council_diversity_summary) and callable(pif_dryrun_summary)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: FAIL — in a no-`mcp` CI env the `else` branch passes only after Task 3, but the assertion on `hasattr(srv, ...)` (mcp present locally) fails until the wrappers exist. If `mcp` is absent locally, temporarily assert the tool names appear in `server.py` source: confirm failure by `grep -c sophia_ocean_measure sophia_mcp/server.py` → `0` before Step 3.

- [ ] **Step 3: Implement** — in `sophia_mcp/server.py`: (a) extend the `from sophia_mcp.tools_impl import (...)` block to include `ocean_measure, capability_retention_demo, council_diversity_summary, pif_dryrun_summary`; (b) add the wrappers near the other `@mcp.tool()`s; (c) add the resource. Read-only ⇒ NO `@audited`:

```python
@mcp.tool()
def sophia_ocean_measure(answers: dict) -> str:
    """Score a {item_id: 1..5} IPIP answer map into OCEAN domain scores. Read-only."""
    return dumps(ocean_measure(answers))


@mcp.tool()
def sophia_capability_retention() -> str:
    """Spec D deterministic capability-retention cell on the bundled arithmetic
    slice (base vs degenerate-steered): capability_drop + coherence + retains. Read-only."""
    return dumps(capability_retention_demo())


@mcp.tool()
def sophia_council_diversity() -> str:
    """Spec C personality-diverse council A/B result (ΔQ; the does-not-replicate null). Read-only."""
    return dumps(council_diversity_summary())


@mcp.tool()
def sophia_pif_dryrun() -> str:
    """Spec C PIF/SSA harness invariants on synthetic fixtures (CI-green core). Read-only."""
    return dumps(pif_dryrun_summary())


@mcp.resource("sophia://program/status")
def sophia_program_status() -> str:
    """MBTI-Vector-Agents program status (Specs A-D): what shipped, the honest
    nulls (steering SSA 0/2; council ΔQ does-not-replicate), and the OPEN frontier."""
    return dumps({
        "program": "MBTI Vector Agents",
        "specs": {
            "A": "personality measurement gate + Level-1 persona (PR #64)",
            "B": "activation-steering engine + SSA; real demo null SSA 0/2 (PR #66)",
            "C": "personality council + held-out anti-gaming + PIF harness; council ΔQ null (PR #67)",
            "D": "capability-retention guardrail + full MCP/skill packaging",
        },
        "honestNulls": ["steering did not beat the persona prompt (SSA 0/2)",
                        "trait diversity did not reliably help the council (ΔQ did not replicate)"],
        "openFrontier": ["full N>=8/K>=20 PIF headline run", "real capability cell in a live SSA run",
                         "LLM-judge coherence", "validated Level-3 steered council seats",
                         "true external sealing", "model x trait crossover", "live GRPO", "calibration"],
        "substrate": "Big Five (OCEAN) is measured; MBTI is a one-way display veneer.",
    })
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python tests/test_capability.py`
Expected: `PASS 8 capability tests`. (If `mcp` is installed locally, also run `python -c "import sophia_mcp.server"` to confirm the module imports clean.)

- [ ] **Step 5: Commit**

```bash
git add sophia_mcp/server.py tests/test_capability.py
git commit -m "feat(mcp): expose program surface (ocean/capability/council/pif tools + status resource) (Spec D D2)"
```

---

### Task 5: Skill expansion + CI wiring + docs + ledger + full regression (D2 packaging)

**Files:**
- Modify: `skills/portable/sophia-personality-faithful/SKILL.md` (expand body; keep frontmatter contract)
- Modify: `.github/workflows/ci.yml` (run `tests/test_capability.py`)
- Create: `docs/09-Agent/Capability-and-Packaging.md`
- Modify: `agi-proof/failure-ledger.md` (append the Spec D OPEN entry)

**Interfaces:**
- Consumes: everything above. Produces: no new code interface — packaging + the full-regression gate.

- [ ] **Step 1: Wire CI** — in `.github/workflows/ci.yml`, after the line running `python tests/test_pif_harness.py`, add `python tests/test_capability.py`. Verify YAML parses:

Run: `cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml OK')"`
Expected: `ci.yml OK`.

- [ ] **Step 2: Expand the skill body** — in `skills/portable/sophia-personality-faithful/SKILL.md`, leave the frontmatter (`name`/`description`/`metadata`) intact and add a section after the existing body documenting the full program surface and the no-overclaim posture:

```markdown
## The MBTI-Vector-Agents program (Specs A–D)

When the `sophia-agi` MCP server is connected, these read-only tools expose the
whole program. Big Five (OCEAN) is the measured substrate; MBTI is a one-way
display veneer no tool reads for a decision.

- `sophia_ocean_measure(answers)` — score a `{item_id: 1..5}` IPIP map into OCEAN.
- `sophia_personality_faithful(text, mbti, ocean)` — is a trait claim faithfully enacted, contradicted, or abstained?
- `sophia_capability_retention()` — does steering degrade reasoning? Returns `capability_drop` + `coherence` + `retains` on a deterministic arithmetic slice.
- `sophia_council_diversity()` — does a trait-diverse council deliberate better? (The honest finding: ΔQ did not replicate.)
- `sophia_pif_dryrun()` — the personality-injection / steering-superiority harness invariants.
- Resource `sophia://program/status` — what shipped and the honest nulls.

**No-overclaim posture:** the program ships its machinery and its honest results.
Two headline findings are NULL — activation steering did not beat a persona
prompt (SSA 0/2), and personality diversity did not reliably improve a council
(ΔQ did not replicate). A capability *drop* under steering is expected and
reported as such. SSA=0/N, ΔQ≤0, and a measured capability drop are legitimate.
```

- [ ] **Step 3: Write the experiment doc** — create `docs/09-Agent/Capability-and-Packaging.md` (companion to `docs/09-Agent/Council-and-PIF-Experiment.md`; read it for tone):

```markdown
# Capability-Retention Guardrail + Program Packaging (Spec D)

Spec D closes the MBTI-Vector-Agents program: it builds the capability-retention
guardrail that produces Spec B's SSA `capability_drop`/`coherence` inputs, and
exposes Specs A–D through the MCP server + portable skill.

## D1 — Capability-retention guardrail
`agent/steering/capability.py` scores a bundled arithmetic slice
(`data/capability_arithmetic.json`) deterministically: answer-correctness vs a
gold number, `arithmetic_sound` soundness, and a deterministic coherence proxy
(0–100) that catches the degeneracy high-alpha steering produces.
`capability_cell(base, steered)` assembles the SSA cell —
`capability_drop = max(0, (base_acc − steer_acc)/base_acc)`,
`retains = capability_drop < 0.05 and coherence ≥ 75` — the exact predicate
`ssa_verdict` applies. `tools/run_capability.py --dry-run` runs the deterministic
core (`CAPABILITY RETENTION VERIFIED ✓`); `--model granite` runs a reduced real
steered-vs-unsteered slice.

**Expected result:** a capability *drop* under steering. That is the honest,
on-brand finding — steering strong enough to move a trait degrades reasoning,
which *explains* Spec B's SSA null. A drop is reported as a drop.

## D2 — Full MCP / skill packaging
Read-only tools (`sophia_ocean_measure`, `sophia_capability_retention`,
`sophia_council_diversity`, `sophia_pif_dryrun`) + a `sophia://program/status`
resource expose A–D, reusing the existing `@audited` gating posture (read-only ⇒
ungated). The tool logic lives in `sophia_mcp/tools_impl.py` (importable without
FastMCP), so CI validates the surface with no `mcp` package. The portable
`SKILL.md` covers the whole program and works in Claude and Codex.

## Two-tier discipline (inherited)
Pure-stdlib deterministic CI core + opt-in reduced real run. SSA=0/N, ΔQ≤0, and
a measured capability drop are all legitimate honest results.

## Deferred frontier (OPEN in the failure ledger)
Full N≥8/K≥20 PIF headline run; a real capability cell wired into a live SSA
headline; an LLM-judge coherence channel; validated Level-3 steered council
seats; true external sealing; model×trait crossover; live GRPO; calibration.
```

- [ ] **Step 4: Append the ledger entry** — read the tail of `agi-proof/failure-ledger.md` to match heading style, then append:

```markdown
## capability-cell-not-yet-in-live-ssa-2026-06-23

**Status:** OPEN

Spec D D1 ships the deterministic capability-retention guardrail
(`agent/steering/capability.py`, `tools/run_capability.py`) that produces the
`capability_drop`/`coherence` inputs `agent/steering/stats.py::ssa_verdict`
requires (`SSA_THRESHOLDS["capability_eps"]=0.05`, `["coherence_floor"]=75.0`).
The reduced real run (`--model granite`) demonstrates the drop, but a real
capability cell is **not yet wired into a live headline SSA run**, and coherence
is a deterministic proxy rather than an LLM-judge channel. Closing this requires
the full N≥8/K≥20 PIF headline run (also OPEN) with real capability cells.
```

- [ ] **Step 5: Full regression** — run and confirm all green:

```bash
cd /Users/tom/Documents/GitHub/sophia-agi-spec-d && \
python tests/test_capability.py && \
python tests/test_pif_harness.py && \
python tests/test_council_diversity.py && \
python tests/test_steering.py && \
python tests/test_personality.py && \
python tests/test_verifiers.py && \
python tools/run_capability.py --dry-run && \
python tools/run_pif.py --dry-run
```
Expected: `PASS 8 capability`, `PASS 11 pif`, `PASS 3 council`, `PASS 17 steering`, `PASS 17 personality`, `test_verifiers: OK`, `CAPABILITY RETENTION VERIFIED ✓`, `PIF HARNESS VERIFIED ✓`.

- [ ] **Step 6: Commit**

```bash
git add skills/portable/sophia-personality-faithful/SKILL.md .github/workflows/ci.yml \
        docs/09-Agent/Capability-and-Packaging.md agi-proof/failure-ledger.md
git commit -m "feat(spec-d): skill expansion + CI wiring + experiment doc + OPEN capability ledger entry (Spec D D2)"
```

---

### Task 6 (controller-driven): reduced real capability run

Not a subagent task — the controller runs this after Tasks 1–5 pass, like Spec C's Task 8.

- [ ] Run `python tools/run_capability.py --model granite --axis E --alpha 8 --layer 12` on MPS (reuse Spec B's granite-3.1-2b download). Capture the honest cell (`base_accuracy`, `steered_accuracy`, `capability_drop`, `coherence`, `retains`). Expected: a measurable drop / coherence hit at steering strength — the result that explains B's null. If granite arithmetic `base_accuracy` is 0 (model can't do the slice at all), report that honestly (the guardrail is vacuous on a base that can't do the task) rather than massaging it.
- [ ] Commit the written `agi-proof/benchmark-results/capability-retention.public-report.json` with an honest headline.

---

## Self-Review

**Spec coverage:** D1 scorer + slice (Task 1), D1 CLI two-tier (Task 2), D2 impls (Task 3), D2 server surface + cross-platform validation (Task 4), skill+CI+docs+ledger (Task 5), reduced real run (Task 6). The SSA-predicate match, deterministic-coherence, read-only-no-`@audited`, and pure-stdlib-CI-core constraints are each pinned in Global Constraints and exercised by a test. ✓

**Placeholder scan:** every code step shows complete code; the two implementer NOTES (carrier-pair accessor name in Task 2's real path; `build_cells_from_scores` fixture shape in Task 3) are real "verify-the-neighbor-signature" instructions for already-existing B/C code, not deferred work — both are confined to code the controller-run/CI exercises. ✓

**Type consistency:** `capability_cell`/`score_response`/`build_dry_run_cell`/the 4 impls keep one signature across Tasks 1→5; the cross-platform test tolerates `mcp` present-or-absent. ✓
