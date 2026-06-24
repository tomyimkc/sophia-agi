#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine the OKF wiki's provenance graph into SFT + DPO training data (the flywheel).

Every page with a doNotAttributeTo edge becomes training signal for the exact
behavior Sophia sells: the lineage-correct answer is `chosen`, the merge is
`rejected`. As the librarian grows the wiki, the model's grasp of who-wrote-what
compounds — provenance maintenance IS the supervision. Benchmark-leak guarded so
held-out questions never leak into training.

    python tools/wiki_to_training.py            # write training/wiki_provenance_{sft,dpo}.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json  # noqa: E402
from agent.gate import check_response  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402
from okf import page as okf_page  # noqa: E402

WIKI_DIR = ROOT / "wiki"
SYSTEM = MODE_PROMPTS["advisor"]


def _benchmark_questions() -> set:
    questions: set = set()
    for path in DOMAIN_BENCH.values():
        if path.exists():
            for case in load_json(path).get("cases", []):
                q = str(case.get("question", "")).strip().lower()
                if q:
                    questions.add(q)
    return questions


def collect(wiki_dir: Path = WIKI_DIR, *, deleak: bool = True, gate_filter: bool = True) -> dict:
    holdout = _benchmark_questions() if deleak else set()
    sft: list = []
    dpo: list = []
    leaked = 0
    chosen_gate_violations = 0
    inconsistent_skipped = 0
    for page in okf_page.load_pages(wiki_dir):
        meta = page.meta
        author = meta.get("attributedAuthor")
        forbidden = [a for a in (meta.get("doNotAttributeTo") or [])]
        title = meta.get("canonicalTitleEn") or meta.get("id")
        if not (author and forbidden and title):
            continue
        question = f"Who wrote {title}?"
        if deleak and question.strip().lower() in holdout:
            leaked += 1
            continue

        # Anti-circularity / provenance-consistency guard: the `rejected` sample's
        # wrongness is established by the wiki provenance graph itself (the author it
        # asserts is in this page's doNotAttributeTo set), NOT by the gate — the
        # deterministic gate has no canonical attribution table for arbitrary titles,
        # so it cannot adjudicate this and must not be relied on to. Reject any page
        # whose metadata is self-contradictory (the forbidden lineage overlaps the
        # attributed author), which would make the preference pair incoherent.
        if author.strip().lower() in {f.strip().lower() for f in forbidden}:
            inconsistent_skipped += 1
            continue

        confidence = meta.get("authorConfidence") or "attributed"
        chosen = (
            f"{title} is attributed to {author} (confidence: {confidence}). "
            f"It must not be attributed to {', '.join(forbidden)} — those are distinct lineages. "
            f"中文摘要：{title} 歸於 {author}，勿混淆其他傳統。"
        )

        # Fail-closed on the TARGET we actually train toward: the `chosen` string must
        # carry no intrinsic gate violation (fabricated citation / false arithmetic).
        # Intrinsic-only (no question) — see tools/train_lora.py guard_filter for why.
        if gate_filter and check_response(chosen, mode="advisor")["violations"]:
            chosen_gate_violations += 1
            continue

        sft.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": chosen},
            ],
            "metadata": {"source": "wiki-provenance", "pageId": page.id},
        })
        dpo.append({
            "prompt": question,
            "chosen": chosen,
            "rejected": f"{forbidden[0].title()} wrote {title}.",
            "metadata": {"source": "wiki-provenance", "pageId": page.id,
                         "rejectedForbiddenBy": "doNotAttributeTo"},
        })
    return {"sft": sft, "dpo": dpo, "leakedSkipped": leaked, "leakageChecked": deleak,
            "gateFilter": gate_filter, "chosenGateViolations": chosen_gate_violations,
            "inconsistentSkipped": inconsistent_skipped}


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine OKF wiki provenance into SFT/DPO data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training")
    parser.add_argument("--no-deleak", action="store_true")
    parser.add_argument("--no-gate-filter", action="store_true",
                        help="Skip the intrinsic gate-check on chosen targets (not recommended)")
    args = parser.parse_args()
    data = collect(deleak=not args.no_deleak, gate_filter=not args.no_gate_filter)
    _write_jsonl(args.out_dir / "wiki_provenance_sft.jsonl", data["sft"])
    _write_jsonl(args.out_dir / "wiki_provenance_dpo.jsonl", data["dpo"])
    print(json.dumps({"sftRows": len(data["sft"]), "dpoPairs": len(data["dpo"]),
                      "leakedSkipped": data["leakedSkipped"],
                      "chosenGateViolations": data["chosenGateViolations"],
                      "inconsistentSkipped": data["inconsistentSkipped"],
                      "outDir": str(args.out_dir)},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
