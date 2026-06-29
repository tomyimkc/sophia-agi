#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated distillation: teacher model -> verified SFT data + rejected set.

Runs a teacher (any adapter provider — GLM-5.2, DeepSeek, Claude) over a set of
prompts, gates each answer through the epistemic gate + required-keyword checks,
keeps only verified-good answers for SFT, and captures rejected answers + the full
trajectory. This is how you grow a smaller local student that behaves like a
specialized frontier agent — without training on teacher hallucinations.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402
from agent.model import ModelClient, default_client  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402
from provenance_bench.dataset_guard import (  # noqa: E402
    build_eval_shingles, eval_prompt_set, nearest_eval, normalize)

# verification_provenance vocabulary — HOW a kept SFT row was verified. Tagged on every
# distilled row so the dataset builder can oversample the hard-won ones (the debug-and-
# recover signal) and the linter can require the tag (tools/lint_training_rows.py).
PROV_PASSED = "passed_first_try"        # main teacher passed the gate on the first attempt
PROV_PATCHED = "patched_after_failure"  # main teacher failed; a stronger teacher patched it (T1)
PROV_SELF_CONSISTENT = "self_consistent"  # recovered by verified self-consistency voting (T7)
PROV_VALUES = {PROV_PASSED, PROV_PATCHED, PROV_SELF_CONSISTENT}

DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {"id": "ddj", "prompt": "Did Confucius write the Dao De Jing? Identify the correct tradition and author.", "mustInclude": ["Laozi"], "mustAvoid": ["Confucius wrote"]},
    {"id": "rome", "prompt": "When did the Western Roman Empire fall, and why is one date an oversimplification?", "mustInclude": ["476"]},
    {"id": "brain", "prompt": "Is the '10% of the brain' claim supported? Label any myth.", "mustInclude": ["myth"]},
]


def _sft_row(item_prompt: str, answer: str, *, teacher: str, item_id: Any,
             provenance: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one SFT row, stamping the verification_provenance (T8) on the metadata."""
    meta = {"source": "distillation", "teacher": teacher, "id": item_id,
            "verification_provenance": provenance}
    if extra:
        meta.update(extra)
    return {
        "messages": [
            {"role": "system", "content": MODE_PROMPTS["advisor"]},
            {"role": "user", "content": item_prompt},
            {"role": "assistant", "content": answer},
        ],
        "metadata": meta,
    }


def distill_one(item: dict[str, Any], client: ModelClient) -> dict[str, Any]:
    system = item.get("system") or MODE_PROMPTS["advisor"]
    prompt = item["prompt"]
    result = client.generate(system, prompt)
    base = {"id": item.get("id"), "prompt": prompt, "costUsd": result.cost_usd, "model": result.model}
    if not result.ok or not result.text.strip():
        return {**base, "accepted": False, "reasons": [result.error or "empty"], "answer": result.text}
    gate = check_response(result.text, mode="advisor", question=prompt)
    lowered = result.text.lower()
    missing = [k for k in item.get("mustInclude", []) if k.lower() not in lowered]
    forbidden = [k for k in item.get("mustAvoid", []) if k.lower() in lowered]
    accepted = gate.get("passed", False) and not missing and not forbidden
    reasons = list(gate.get("warnings", [])) + list(gate.get("violations", []))
    reasons += [f"missing:{k}" for k in missing] + [f"forbidden:{k}" for k in forbidden]
    return {**base, "accepted": accepted, "answer": result.text, "reasons": reasons}


def _eval_contamination(prompt: str, evalset: set[str],
                        eval_sh: list[tuple[str, set]], *, jaccard_thr: float, k: int):
    """Return (is_contaminated, detail) for a teacher-trace prompt vs the held-out eval.

    A frontier teacher may have memorised public benchmarks, so a 'verified' trace whose
    prompt IS an eval prompt (exact) or paraphrases one (near-dup) would leak the eval into
    training and inflate any later uplift. Exact membership first (cheap), then the shingle
    near-dup tier. ``None`` evalset/eval_sh means decontam is disabled.
    """
    if normalize(prompt) in evalset:
        return True, {"kind": "exact"}
    j, matched = nearest_eval(prompt, eval_sh, k=k)
    if j >= jaccard_thr:
        return True, {"kind": "near-dup", "jaccard": round(j, 3), "evalPrompt": matched[:80]}
    return False, {}


def distill(prompts: list[dict[str, Any]], client: ModelClient, *, decontam: bool = True,
            eval_root: Path = ROOT, eval_prompts: set[str] | None = None,
            jaccard_thr: float = 0.9, shingle_k: int = 5) -> dict[str, Any]:
    """Run the teacher over ``prompts``, gate each answer, and (by default) DECONTAMINATE
    every kept trace against the held-out eval before it can become SFT data (T4).

    Buckets: ``sft`` (gate-passed AND eval-disjoint), ``rejectedRows`` (gate failed),
    ``decontaminatedRows`` (gate passed but the prompt collides with the held-out eval).
    Every SFT row is tagged ``verification_provenance: passed_first_try`` (T8).

    ``eval_prompts`` (a set of NORMALIZED prompts) may be injected to override the loaded
    held-out set — used by tests and by callers that already hold the eval surface.
    """
    evalset: set[str] = set()
    eval_sh: list[tuple[str, set]] = []
    if decontam:
        evalset = eval_prompts if eval_prompts is not None else eval_prompt_set(root=eval_root)
        eval_sh = build_eval_shingles(evalset, k=shingle_k)

    sft: list[dict] = []
    rejected: list[dict] = []
    decontaminated: list[dict] = []
    trajectory: list[dict] = []
    total_cost = 0.0
    for item in prompts:
        outcome = distill_one(item, client)
        total_cost += outcome.get("costUsd", 0.0) or 0.0
        traj = {k: v for k, v in outcome.items() if k != "answer"} | {"accepted": outcome["accepted"]}
        if outcome["accepted"]:
            leak, detail = (False, {})
            if decontam:
                leak, detail = _eval_contamination(outcome["prompt"], evalset, eval_sh,
                                                    jaccard_thr=jaccard_thr, k=shingle_k)
            if leak:
                traj["decontaminated"] = detail
                decontaminated.append({"id": outcome["id"], "prompt": outcome["prompt"],
                                       "reason": detail, "answer": outcome["answer"]})
            else:
                sft.append(_sft_row(outcome["prompt"], outcome["answer"],
                                    teacher=outcome["model"], item_id=outcome["id"],
                                    provenance=PROV_PASSED))
        else:
            rejected.append({"id": outcome["id"], "prompt": outcome["prompt"], "rejected": outcome["answer"], "reasons": outcome["reasons"]})
        trajectory.append(traj)
    n = len(prompts)
    return {
        "accepted": len(sft),
        "rejected": len(rejected),
        "decontaminated": len(decontaminated),
        "total": n,
        "acceptRate": round(len(sft) / n, 3) if prompts else 0.0,
        "totalCostUsd": round(total_cost, 6),
        # cost per VERIFIED-and-KEPT row — the real sample-efficiency number (T9).
        "costPerVerifiedRow": round(total_cost / len(sft), 6) if sft else None,
        "sft": sft,
        "rejectedRows": rejected,
        "decontaminatedRows": decontaminated,
        "trajectory": trajectory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier-gated distillation export")
    parser.add_argument("prompts", nargs="?", type=Path, default=None, help="prompts JSON [{id,prompt,mustInclude,mustAvoid}]")
    parser.add_argument("--provider", default=None, help="teacher provider (default from env; e.g. deepseek, glm:glm-5.2)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training")
    parser.add_argument("--no-decontam", action="store_true",
                        help="DANGER: skip the held-out-eval decontamination of teacher traces")
    parser.add_argument("--jaccard", type=float, default=0.9, help="near-dup decontam threshold")
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8")) if args.prompts else DEFAULT_PROMPTS
    client = default_client(args.provider)
    data = distill(prompts, client, decontam=not args.no_decontam, jaccard_thr=args.jaccard)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "distill_sft.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["sft"]) + ("\n" if data["sft"] else ""), encoding="utf-8")
    (args.out_dir / "distill_rejected.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["rejectedRows"]) + ("\n" if data["rejectedRows"] else ""), encoding="utf-8")
    (args.out_dir / "distill_decontaminated.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["decontaminatedRows"]) + ("\n" if data["decontaminatedRows"] else ""), encoding="utf-8")
    summary = {k: data[k] for k in ("accepted", "rejected", "decontaminated", "total",
                                    "acceptRate", "totalCostUsd", "costPerVerifiedRow")}
    summary["runAt"] = datetime.now().isoformat(timespec="seconds")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
