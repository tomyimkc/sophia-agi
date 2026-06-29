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

from agent import outcome_oracle as oracle  # noqa: E402
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
             provenance: str, source: str = "distillation",
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one SFT row, stamping the verification_provenance (T8) on the metadata.

    ``source`` distinguishes a teacher-distilled row ("distillation") from an on-policy
    RLVR harvest ("rlvr_harvest", T3) — both share this single row schema so the dataset
    builder and linters treat them uniformly."""
    meta = {"source": source, "teacher": teacher, "id": item_id,
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


def _spec_of(item: dict[str, Any]) -> dict[str, Any]:
    """Build the shared OracleSpec for an item — the SAME contract the RLVR reward uses,
    plus the fail-closed advisor gate that distillation additionally requires."""
    spec = {"epistemicGate": True}
    for src, dst in (("mustInclude", "mustInclude"), ("mustAvoid", "mustAvoid"),
                     ("expected", "expected"), ("regex", "regex"),
                     ("mathEquivalent", "mathEquivalent"), ("code", "code"),
                     ("citations", "citations")):
        if item.get(src) is not None:
            spec[dst] = item[src]
    return spec


def distill_one(item: dict[str, Any], client: ModelClient) -> dict[str, Any]:
    system = item.get("system") or MODE_PROMPTS["advisor"]
    prompt = item["prompt"]
    result = client.generate(system, prompt)
    base = {"id": item.get("id"), "prompt": prompt, "costUsd": result.cost_usd, "model": result.model}
    if not result.ok or not result.text.strip():
        return {**base, "accepted": False, "reasons": [result.error or "empty"], "answer": result.text}
    # Single outcome oracle (agent.outcome_oracle) — the verifier seam shared with RLVR.
    verdict = oracle.evaluate(_spec_of(item), result.text, question=prompt)
    return {**base, "accepted": verdict["passed"], "answer": result.text,
            "reasons": verdict["reasons"], "checks": verdict.get("checks", {})}


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


def _higher_temp(client: ModelClient, temperature: float) -> ModelClient:
    """A copy of ``client`` whose primary config samples at a higher temperature (for the
    self-consistency vote, T7) — diversity is what makes a verified majority meaningful."""
    import copy
    import dataclasses

    new = copy.copy(client)
    try:
        new.primary = dataclasses.replace(client.primary, temperature=temperature)
    except Exception:  # noqa: BLE001 - duck-typed/fake clients: leave primary as-is
        pass
    return new


def _self_consistent_answer(item: dict[str, Any], client: ModelClient, n: int,
                            temperature: float) -> tuple[str | None, float, dict]:
    """Sample ``n`` higher-temperature completions and keep one only if a VERIFIED majority
    (>= ceil(n/2)) pass the oracle (T7). Returns (answer_or_None, cost, detail)."""
    hot = _higher_temp(client, temperature)
    spec = _spec_of(item)
    passing: list[str] = []
    cost = 0.0
    for _ in range(max(1, n)):
        res = hot.generate(item.get("system") or MODE_PROMPTS["advisor"], item["prompt"])
        cost += res.cost_usd or 0.0
        if res.ok and res.text.strip() and oracle.evaluate(spec, res.text, question=item["prompt"])["passed"]:
            passing.append(res.text)
    need = (n // 2) + 1
    detail = {"n": n, "passed": len(passing), "needed": need}
    return (passing[0] if len(passing) >= need else None, cost, detail)


def distill(prompts: list[dict[str, Any]], client: ModelClient, *, decontam: bool = True,
            eval_root: Path = ROOT, eval_prompts: set[str] | None = None,
            jaccard_thr: float = 0.9, shingle_k: int = 5,
            patch_client: ModelClient | None = None,
            self_consistency_n: int = 0, patch_temperature: float = 0.9) -> dict[str, Any]:
    """Run the teacher over ``prompts``, gate each answer, DECONTAMINATE every kept trace
    against the held-out eval (T4), and optionally RECOVER the failed subset (T1/T7).

    Buckets: ``sft`` (kept: passed-first-try, patched, or self-consistent — all eval-disjoint),
    ``rejectedRows`` (gate failed AND not recovered), ``decontaminatedRows`` (verified but
    collides with the held-out eval), ``dpoPairs`` ({prompt,chosen,rejected} mined from a
    main-teacher failure that a stronger teacher patched — teaches debug-and-recover).

    Recovery tiers (applied to the failed subset, in this order):
      * ``patch_client`` (T1): a stronger auxiliary teacher re-answers; re-verified passes are
        tagged ``patched_after_failure`` and also yield a DPO pair vs the main failure.
      * ``self_consistency_n`` (T7): if no patch client, sample N higher-temp completions from
        the SAME teacher and keep one iff a verified majority pass — tagged ``self_consistent``.

    Every kept SFT row carries a ``verification_provenance`` tag (T8). ``eval_prompts`` (a set
    of NORMALIZED prompts) may be injected to override the loaded held-out set (tests/callers).
    """
    evalset: set[str] = set()
    eval_sh: list[tuple[str, set]] = []
    if decontam:
        evalset = eval_prompts if eval_prompts is not None else eval_prompt_set(root=eval_root)
        eval_sh = build_eval_shingles(evalset, k=shingle_k)

    def route_clean(prompt: str, answer: str, *, teacher: str, item_id: Any, provenance: str):
        """Decontaminate then bucket: ('sft', row) or ('decontam', info)."""
        if decontam:
            leak, detail = _eval_contamination(prompt, evalset, eval_sh, jaccard_thr=jaccard_thr, k=shingle_k)
            if leak:
                return "decontam", {"id": item_id, "prompt": prompt, "reason": detail, "answer": answer}
        return "sft", _sft_row(prompt, answer, teacher=teacher, item_id=item_id, provenance=provenance)

    sft: list[dict] = []
    rejected: list[dict] = []
    decontaminated: list[dict] = []
    dpo_pairs: list[dict] = []
    trajectory: list[dict] = []
    teacher_split: dict[str, int] = {}
    total_cost = 0.0

    # ---- main pass ----
    failures: list[tuple[dict, dict]] = []  # (item, main_outcome) for the recovery stage
    for item in prompts:
        outcome = distill_one(item, client)
        total_cost += outcome.get("costUsd", 0.0) or 0.0
        traj = {k: v for k, v in outcome.items() if k != "answer"} | {"accepted": outcome["accepted"]}
        if outcome["accepted"]:
            bucket, payload = route_clean(outcome["prompt"], outcome["answer"],
                                          teacher=outcome["model"], item_id=outcome["id"],
                                          provenance=PROV_PASSED)
            if bucket == "sft":
                sft.append(payload)
                teacher_split[outcome["model"]] = teacher_split.get(outcome["model"], 0) + 1
            else:
                traj["decontaminated"] = payload["reason"]
                decontaminated.append(payload)
        else:
            rejected.append({"id": outcome["id"], "prompt": outcome["prompt"],
                             "rejected": outcome["answer"], "reasons": outcome["reasons"]})
            failures.append((item, outcome))
        trajectory.append(traj)

    # ---- recovery stage (T1 patch tier, else T7 self-consistency) ----
    patched = self_consistent = 0
    for item, main_outcome in failures:
        recovered_answer: str | None = None
        provenance = ""
        teacher = ""
        if patch_client is not None:
            po = distill_one(item, patch_client)
            total_cost += po.get("costUsd", 0.0) or 0.0
            if po["accepted"]:
                recovered_answer, provenance, teacher = po["answer"], PROV_PATCHED, po["model"]
        elif self_consistency_n:
            ans, cost, _detail = _self_consistent_answer(item, client, self_consistency_n, patch_temperature)
            total_cost += cost
            if ans is not None:
                recovered_answer, provenance, teacher = ans, PROV_SELF_CONSISTENT, main_outcome["model"]
        if recovered_answer is None:
            continue
        bucket, payload = route_clean(item["prompt"], recovered_answer,
                                      teacher=teacher, item_id=item.get("id"), provenance=provenance)
        if bucket != "sft":
            decontaminated.append(payload)
            continue
        sft.append(payload)
        teacher_split[teacher] = teacher_split.get(teacher, 0) + 1
        if provenance == PROV_PATCHED:
            patched += 1
            # mine the recovery as a preference pair: stronger-teacher answer is chosen,
            # the main teacher's failed answer is rejected (debug-and-recover signal).
            if (main_outcome.get("answer") or "").strip() and main_outcome["answer"] != recovered_answer:
                dpo_pairs.append({"prompt": item["prompt"], "chosen": recovered_answer,
                                  "rejected": main_outcome["answer"],
                                  "metadata": {"source": "distill_patch", "teacher": teacher,
                                               "verification_provenance": PROV_PATCHED}})
        else:
            self_consistent += 1
        # this failure was recovered — drop it from the rejected bucket
        rejected = [r for r in rejected if r["id"] != item.get("id")]

    n = len(prompts)
    return {
        "accepted": len(sft),
        "passedFirstTry": len(sft) - patched - self_consistent,
        "patched": patched,
        "selfConsistent": self_consistent,
        "rejected": len(rejected),
        "decontaminated": len(decontaminated),
        "total": n,
        "acceptRate": round(len(sft) / n, 3) if prompts else 0.0,
        "totalCostUsd": round(total_cost, 6),
        # cost per VERIFIED-and-KEPT row — the real sample-efficiency number (T9).
        "costPerVerifiedRow": round(total_cost / len(sft), 6) if sft else None,
        "teacherSplit": teacher_split,  # rows kept per teacher (main vs patch) — T9 budgeting
        "sft": sft,
        "rejectedRows": rejected,
        "decontaminatedRows": decontaminated,
        "dpoPairs": dpo_pairs,
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
    parser.add_argument("--patch-provider", default=None,
                        help="T1: stronger auxiliary teacher for the failed subset (e.g. glm:glm-5.2, "
                             "claude); patched answers are SFT'd + mined as DPO pairs")
    parser.add_argument("--self-consistency-n", type=int, default=0,
                        help="T7: if no --patch-provider, sample N higher-temp completions from the "
                             "main teacher on failures and keep a verified majority")
    parser.add_argument("--patch-temperature", type=float, default=0.9,
                        help="sampling temperature for the self-consistency vote (T7)")
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8")) if args.prompts else DEFAULT_PROMPTS
    client = default_client(args.provider)
    patch_client = default_client(args.patch_provider) if args.patch_provider else None
    data = distill(prompts, client, decontam=not args.no_decontam, jaccard_thr=args.jaccard,
                   patch_client=patch_client, self_consistency_n=args.self_consistency_n,
                   patch_temperature=args.patch_temperature)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "distill_sft.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["sft"]) + ("\n" if data["sft"] else ""), encoding="utf-8")
    (args.out_dir / "distill_rejected.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["rejectedRows"]) + ("\n" if data["rejectedRows"] else ""), encoding="utf-8")
    (args.out_dir / "distill_decontaminated.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["decontaminatedRows"]) + ("\n" if data["decontaminatedRows"] else ""), encoding="utf-8")
    (args.out_dir / "distill_dpo.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data["dpoPairs"]) + ("\n" if data["dpoPairs"] else ""), encoding="utf-8")
    summary = {k: data[k] for k in ("accepted", "passedFirstTry", "patched", "selfConsistent",
                                    "rejected", "decontaminated", "total", "acceptRate",
                                    "totalCostUsd", "costPerVerifiedRow", "teacherSplit")}
    summary["dpoPairs"] = len(data["dpoPairs"])
    summary["runAt"] = datetime.now().isoformat(timespec="seconds")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
