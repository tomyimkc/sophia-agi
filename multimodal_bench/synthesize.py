# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-checked synthesis of chart / table / document traps (workstream D).

The roadmap's data pillar: synthesise QA where the *answer is machine-verifiable*
(rendered charts/tables/documents with known values), and trust a synthetic row
only if the verifier (``multimodal_bench/verifiers.py``) can re-derive its label.
The generator is deterministic given a seed, so the committed
``data/visual_traps_synth.json`` is reproducible (a test regenerates it and
asserts byte-equality of the logical rows + ``gold_matches_check`` on every row).

Distractors (``trap_answer``) are *plausible* errors, not random: an adjacent
bar's value, the runner-up category, a neighbouring table cell, a transposed
document number — the kinds of mistakes a real VLM actually makes.
"""

from __future__ import annotations

import random

_CHART_TOPICS = [
    ("Quarterly revenue", ["Q1", "Q2", "Q3", "Q4"], "$M"),
    ("Site visits by day", ["Mon", "Tue", "Wed", "Thu"], "k"),
    ("Votes per option", ["A", "B", "C"], "votes"),
    ("Rainfall by month", ["Jan", "Feb", "Mar", "Apr"], "mm"),
]
_TABLE_TOPICS = [
    ("price", ["apple", "pear", "plum", "kiwi"]),
    ("stock", ["bolt", "nut", "washer", "screw"]),
]
_DOC_TEMPLATES = [
    {"Invoice No": "INV-{a}", "Total": "${b}.{c}", "Due": "Net {d}"},
    {"Order ID": "ORD-{a}", "Amount": "${b}.{c}", "Items": "{d}"},
]


def _transpose_digits(s: str) -> str:
    """Swap the last two digits of a numeric-ish string (a classic OCR slip)."""
    chars = list(s)
    pos = [i for i, ch in enumerate(chars) if ch.isdigit()]
    if len(pos) >= 2:
        i, j = pos[-1], pos[-2]
        chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)


def _chart_traps(rng: random.Random, n: int) -> list:
    out = []
    for k in range(n):
        title, labels, unit = _CHART_TOPICS[k % len(_CHART_TOPICS)]
        values = [rng.randrange(10, 90) for _ in labels]
        # ensure a unique max/min so chart_extreme is well-defined
        while len(set(values)) < len(values):
            values = [rng.randrange(10, 90) for _ in labels]
        bars = [{"label": labels[i], "value": values[i]} for i in range(len(labels))]
        scene = {"width": 512, "height": 512, "chart": {"kind": "bar", "title": title, "unit": unit, "bars": bars}, "objects": [], "texts": []}
        if k % 3 == 2:  # every third chart trap asks for the extreme
            which = "max" if k % 2 == 0 else "min"
            gold_label = max(bars, key=lambda b: b["value"])["label"] if which == "max" else min(bars, key=lambda b: b["value"])["label"]
            runner_up = sorted(bars, key=lambda b: b["value"], reverse=(which == "max"))[1]["label"]
            out.append({
                "id": f"synth-chart-extreme-{k}", "category": "chart_qa",
                "scene": scene, "question": f"In '{title}', which bar is the {'highest' if which == 'max' else 'lowest'}?",
                "answer_type": "text", "gold_answer": gold_label, "trap_answer": runner_up,
                "check": {"type": "chart_extreme", "which": which},
                "reason": f"The {which} bar is {gold_label}; naming the runner-up {runner_up} is a chart-reading error.",
            })
        else:  # ask for a specific bar's value
            qi = rng.randrange(len(labels))
            gold = values[qi]
            distractor = values[(qi + 1) % len(labels)]  # an adjacent bar's value
            out.append({
                "id": f"synth-chart-value-{k}", "category": "chart_qa",
                "scene": scene, "question": f"In '{title}', what is the value of bar {labels[qi]} ({unit})?",
                "answer_type": "count", "gold_answer": str(gold), "trap_answer": str(distractor),
                "check": {"type": "chart_value", "label": labels[qi], "expect": gold},
                "reason": f"Bar {labels[qi]} = {gold}; reading {distractor} is the adjacent bar's value (a misread).",
            })
    return out


def _table_traps(rng: random.Random, n: int) -> list:
    out = []
    for k in range(n):
        col, items = _TABLE_TOPICS[k % len(_TABLE_TOPICS)]
        chosen = items[: 3 + (k % 2)]
        values = {it: rng.randrange(2, 99) for it in chosen}
        rows = [[it, str(values[it])] for it in chosen]
        scene = {"width": 512, "height": 512, "table": {"columns": ["item", col], "rows": rows}, "objects": [], "texts": []}
        qi = rng.randrange(len(chosen))
        key = chosen[qi]
        gold = str(values[key])
        neighbour = chosen[(qi + 1) % len(chosen)]
        distractor = str(values[neighbour])  # the next row's value in the same column
        out.append({
            "id": f"synth-table-cell-{k}", "category": "table_qa",
            "scene": scene, "question": f"In the table, what is the {col} of '{key}'?",
            "answer_type": "text", "gold_answer": gold, "trap_answer": distractor,
            "check": {"type": "table_cell", "row": key, "col": col, "expect": gold},
            "reason": f"The {col} of {key} is {gold}; {distractor} is the adjacent row '{neighbour}' (a row-slip).",
        })
    return out


def _document_traps(rng: random.Random, n: int) -> list:
    out = []
    for k in range(n):
        tmpl = _DOC_TEMPLATES[k % len(_DOC_TEMPLATES)]
        a, b, c, d = rng.randrange(100, 999), rng.randrange(10, 99), rng.randrange(10, 99), rng.choice([15, 30, 45, 2, 3])
        fields = {name: pat.format(a=a, b=b, c=c, d=d) for name, pat in tmpl.items()}
        scene = {"width": 512, "height": 512, "document": {"fields": fields}, "objects": [], "texts": []}
        field_name = list(fields)[rng.randrange(len(fields))]
        gold = fields[field_name]
        distractor = _transpose_digits(gold)
        if distractor == gold:  # no digits to transpose -> use another field's value
            other = [v for nm, v in fields.items() if nm != field_name]
            distractor = other[0] if other else gold + "X"
        out.append({
            "id": f"synth-doc-field-{k}", "category": "document_qa",
            "scene": scene, "question": f"In this document, what is the '{field_name}'?",
            "answer_type": "text", "gold_answer": gold, "trap_answer": distractor,
            "check": {"type": "doc_field", "name": field_name, "expect": gold},
            "reason": f"The {field_name} reads {gold}; {distractor} transposes its digits (an OCR slip).",
        })
    return out


def build_synth_traps(*, seed: int = 0, n_chart: int = 6, n_table: int = 4, n_document: int = 4) -> list:
    """Deterministically generate the synthetic chart/table/document trap rows."""
    rng = random.Random(seed)
    traps = _chart_traps(rng, n_chart) + _table_traps(rng, n_table) + _document_traps(rng, n_document)
    return traps


def build_payload(*, seed: int = 0) -> dict:
    """The full JSON payload written to data/visual_traps_synth.json."""
    return {
        "_meta": {
            "description": "Verifier-checked SYNTHETIC chart/table/document traps, generated deterministically by multimodal_bench/synthesize.py (build_synth_traps). Every label is re-derivable by the scene verifier; a test regenerates this file and asserts equality + gold_matches_check on every row. Distractors are plausible misreads (adjacent bar/cell, runner-up, digit transposition), not random.",
            "generator": "multimodal_bench.synthesize.build_synth_traps",
            "seed": seed,
            "labelSource": "deterministic scene verifier (judge-free)",
            "schemaVersion": 1,
            "categories": ["chart_qa", "table_qa", "document_qa"],
        },
        "traps": build_synth_traps(seed=seed),
    }
