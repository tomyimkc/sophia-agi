#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""End-to-end preference-data pipeline runner (the orchestrator for items 1a/1b/1c).

Chains the three stages of the Verifier-Gated Preference Engine into one runnable,
reportable pipeline and runs the AUTHORITATIVE decontamination gate on the output.
This is a thin orchestrator — it composes tools that each have their own offline
self-test, never reimplementing them:

    Stage A (optional): GENERATE candidates            tools/gen_preference_candidates.py
                         (needs a model spec + keys; produces the `candidates` field)
    Stage B (one of):   LABEL candidates with the      tools/gen_verifier_dpo.py
                         machine verifiers              OR
                        INGEST foreign fuel re-scored   tools/ingest_foreign_tool_fuel.py
                         through our verifiers
    Stage C (always):   DECONTAMINATE the output via    provenance_bench.dataset_guard
                         check_contamination (exact)    + tools/assert_decontam.py (shingle)

The two labelling paths share one property that makes the pipeline honest: the label
provenance is always a **machine verifier**, never an LLM judge and never a foreign
label. The output pack matches ``training/tool_use/dpo_pairs.jsonl`` exactly, so it is
a drop-in growth path for the 200-row pack toward the 10k target (plan §3-4).

Why an orchestrator (and why now). The three pieces already exist and are CI-tested in
isolation; the missing artefact is the **reportable** path that proves the whole chain
runs, the skip-reason histogram is visible, and the output is decontaminated against
every eval surface before it is allowed into training. That is the artefact item 1 of
the handoff asks for: "REPORT BACK: rows minted, skip-reason histogram, decontam result."

Honest scope (pre-registered — see ``docs/06-Roadmap/Frontier-Positioning-Plan.md``):
  * The minted pack is a TRAINING INPUT, not a result. External transfer of an adapter
    trained on it is an OPEN gate (ledger: ``v4-adapter-externally-unvalidated``).
  * Decontamination here is the EXACT-prompt layer; the near-duplicate shingle layer is
    run by the separate authoritative ``tools/assert_decontam.py`` in CI. Both must pass
    before a pack is registered for training.

Usage::

    # Path 1 — generate (needs model) then label offline:
    python tools/run_preference_pipeline.py generate-label \\
        --tasks tasks.jsonl --out training/tool_use/dpo_pairs_v2.jsonl \\
        --n 4 --spec openai:gpt-4o-mini

    # Path 2 — ingest foreign fuel, re-scored offline:
    python tools/run_preference_pipeline.py ingest-foreign \\
        --in foreign.jsonl --out training/tool_use/dpo_pairs_foreign.jsonl --source ToolACE

    # Dry-run / wiring check (no model, no network) — uses the deterministic fixtures:
    python tools/run_preference_pipeline.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset_guard import check_contamination, eval_prompt_set  # noqa: E402
from tools.gen_verifier_dpo import run as label_run  # noqa: E402
from tools.gen_preference_candidates import (  # noqa: E402
    SELF_TEST_ROWS as GEN_FIXTURES,
    _fake_complete_factory,
    run as gen_run,
)
from tools.ingest_foreign_tool_fuel import (  # noqa: E402
    SELF_TEST_ROWS as FOREIGN_FIXTURES,
    run as foreign_run,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _decontam_report(rows: list[dict], *, root: Path = ROOT) -> dict:
    """Run the authoritative EXACT-prompt decontamination gate on minted pairs.

    Returns the check_contamination verdict plus the count of unique prompts. The
    near-duplicate shingle layer is left to ``tools/assert_decontam.py`` in CI; this
    function fails closed on any EXACT overlap with a held-out eval prompt.
    """
    verdict = check_contamination(rows, root=root)
    prompts = {r.get("prompt", "").strip() for r in rows if r.get("prompt")}
    verdict["uniquePrompts"] = len(prompts)
    return verdict


def _label(generated: list[dict]) -> "tuple[list[dict], dict]":
    """Stage B (machine-verifier labelling) over generated candidate-rows."""
    pairs, stats = label_run(generated)
    return pairs, stats


def pipeline_generate_label(
    *, tasks: list[dict], n: int, complete_fn, spec: "str | None", root: Path = ROOT,
) -> dict:
    """Path 1: generate candidates → label with machine verifiers → decontaminate."""
    generated, gstats = gen_run(tasks, n=n, complete_fn=complete_fn, spec=spec)
    pairs, lstats = _label(generated)
    decontam = _decontam_report(pairs, root=root)
    return {"path": "generate-label", "gen": gstats.as_dict(), "label": lstats,
            "decontam": decontam, "pairs": pairs}


def pipeline_ingest_foreign(
    *, rows: list[dict], source: str, root: Path = ROOT,
) -> dict:
    """Path 2: ingest foreign fuel re-scored by our verifiers → decontaminate."""
    pairs, fstats = foreign_run(rows, source=source)
    decontam = _decontam_report(pairs, root=root)
    return {"path": "ingest-foreign", "ingest": fstats, "decontam": decontam,
            "pairs": pairs}


def self_test() -> int:
    """Prove both pipeline paths run end-to-end offline and decontaminate cleanly.

    Uses the deterministic fixtures from gen_preference_candidates (scripted model)
    and ingest_foreign_tool_fuel (no model). Asserts: pairs are minted with
    machine-verified provenance, the skip-reason histogram is populated, and the
    output passes the authoritative exact-prompt decontamination gate."""
    ok = True
    msgs: list[str] = []

    # Path 1: generate → label. Scripted candidates that the gate separates.
    script = {
        "Socrates": [
            "No — Socrates wrote nothing himself; The Republic was written by Plato.",
            "Yes, Socrates wrote The Republic.",
            "Socrates is the author of The Republic.",
        ],
    }
    fake = _fake_complete_factory(script)
    r1 = pipeline_generate_label(tasks=GEN_FIXTURES, n=3, complete_fn=fake, spec=None)
    if r1["label"]["pairs"] < 1:
        ok = False
        msgs.append(f"path1 minted {r1['label']['pairs']} pairs (expected >=1)")
    if not r1["decontam"]["clean"]:
        ok = False
        msgs.append(f"path1 decontam NOT clean: {r1['decontam']['overlap']}")
    for p in r1["pairs"]:
        if p["metadata"].get("label_source") != "machine_verified":
            ok = False
            msgs.append("path1 pair lacks machine_verified provenance")

    # Path 2: ingest foreign → label. One mappable (mints), one unmappable (skipped).
    r2 = pipeline_ingest_foreign(rows=FOREIGN_FIXTURES, source="ToolACE-shape")
    if r2["ingest"]["pairs"] < 1:
        ok = False
        msgs.append(f"path2 minted {r2['ingest']['pairs']} pairs (expected >=1)")
    if r2["ingest"]["reasons"].get("all_candidates_skipped", 0) < 1:
        ok = False
        msgs.append("path2 did not skip the unmappable trace (partial-coverage check)")
    if not r2["decontam"]["clean"]:
        ok = False
        msgs.append(f"path2 decontam NOT clean: {r2['decontam']['overlap']}")

    print("Preference-pipeline self-test:", "PASS" if ok else "FAIL")
    print(f"  path1 generate-label : pairs={r1['label']['pairs']} "
          f"gen_reasons={r1['gen']['reasons']} decontam={r1['decontam']['clean']}")
    print(f"  path2 ingest-foreign : pairs={r2['ingest']['pairs']} "
          f"ingest_reasons={r2['ingest']['reasons']} decontam={r2['decontam']['clean']}")
    for m in msgs:
        print(f"  [XX] {m}")
    return 0 if ok else 1


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p_gl = sub.add_parser("generate-label", help="generate candidates then label offline")
    p_gl.add_argument("--tasks", type=Path, required=True, help="input tasks JSONL")
    p_gl.add_argument("--out", type=Path, required=True, help="output DPO pairs JSONL")
    p_gl.add_argument("--n", type=int, default=4)
    p_gl.add_argument("--spec", default=None, help="agent.model spec")
    p_gl.add_argument("--max-tokens", type=int, default=400)

    p_if = sub.add_parser("ingest-foreign", help="ingest foreign fuel re-scored by our verifiers")
    p_if.add_argument("--in", dest="in_path", type=Path, required=True)
    p_if.add_argument("--out", type=Path, required=True)
    p_if.add_argument("--source", default="foreign")

    ap.add_argument("--self-test", action="store_true",
                    help="run the deterministic offline wiring check (no model, no network)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    def _load(p: Path) -> list[dict]:
        rows = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return rows

    if args.cmd == "generate-label":
        from tools.gen_preference_candidates import _import_complete
        tasks = _load(args.tasks)
        report = pipeline_generate_label(
            tasks=tasks, n=args.n, complete_fn=_import_complete(), spec=args.spec)
        _write_jsonl(args.out, report["pairs"])
        print(json.dumps({"path": report["path"], "out": str(args.out),
                          "gen": report["gen"], "label": report["label"],
                          "decontam": {k: v for k, v in report["decontam"].items() if k != "overlap"}
                          }, ensure_ascii=False))
        return 0 if report["decontam"]["clean"] else 2  # exit 2 = decontam FAIL

    if args.cmd == "ingest-foreign":
        rows = _load(args.in_path)
        report = pipeline_ingest_foreign(rows=rows, source=args.source)
        _write_jsonl(args.out, report["pairs"])
        print(json.dumps({"path": report["path"], "out": str(args.out),
                          "ingest": report["ingest"],
                          "decontam": {k: v for k, v in report["decontam"].items() if k != "overlap"}
                          }, ensure_ascii=False))
        return 0 if report["decontam"]["clean"] else 2

    ap.error("specify a subcommand (generate-label | ingest-foreign) or --self-test")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
