#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Assemble the `training/local_sophia_v2/` dataset packs for a LOCAL Sophia wisdom
model (NOT AGI). Composes the existing builders' outputs into role-split packs, runs
the train/eval contamination guard (fail-closed), and writes an honest manifest that
records BOTH the present packs and the still-MISSING required inputs.

Run the upstream builders first:
    python tools/export_training_jsonl.py
    python tools/wiki_to_training.py
    python tools/mine_hard_negatives.py --out training/hard_negatives_dpo.jsonl
    python tools/build_moral_gate_sft.py
    python tools/prepare_lora_dataset.py

Then:
    python tools/build_local_sophia_dataset.py            # build + guard
    python tools/build_local_sophia_dataset.py --check    # guard only, no writes (CI)

This does NOT train anything (training needs your Mac/MLX or a cloud GPU).
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

DEFAULT_OUT = ROOT / "training" / "local_sophia_v2"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT = DEFAULT_OUT  # backward compat for tests
BASE_MODEL = DEFAULT_BASE_MODEL
MLX_MAX_TOKENS = 1024  # must match the trainer's --max-seq-length; rows are fit to this


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate malformed/partial JSONL lines; skip and keep reading
                pass
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _format_chat(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip() or "user"
        content = str(msg.get("content", "")).strip()
        if content:
            parts.append(f"<|{role}|>\n{content}")
    parts.append("<|end|>")
    return "\n".join(parts)


def _to_mlx_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        msgs = row.get("messages")
        if isinstance(msgs, list) and msgs:
            # MLX-LM ChatDataset supports --mask-prompt only when rows carry a
            # messages/chat feature. Do not flatten to text here.
            out.append({"messages": msgs, "metadata": row.get("metadata", {})})
    return out


HOLDOUT_SRC = "training/lora/holdout.jsonl"


def _sft_sources(out: Path) -> list[tuple[str, str, str]]:
    gi = out / "general_instruct.jsonl"
    gi_rel = str(gi.relative_to(ROOT)) if gi.exists() else "training/local_sophia_v2/general_instruct.jsonl"
    return [
        ("training/corpus.jsonl", "sft_source_discipline.jsonl", "sft"),
        ("training/wiki_provenance_sft.jsonl", "sft_wiki_provenance.jsonl", "sft"),
        ("training/council/traces.jsonl", "sft_council_traces.jsonl", "sft"),
        ("training/council/religion_repair_c4.jsonl", "sft_religion_repair_c4.jsonl", "sft"),
        ("training/moral_gate_sft.jsonl", "sft_moral_gate.jsonl", "sft"),
        (gi_rel, "general_instruct.jsonl", "sft"),
        # C4: human-reviewed, promoted gate-feedback misses (optional; absent → skipped).
        # Decontaminated like any source, so it cannot leak eval/holdout prompts.
        ("training/feedback/sft_from_feedback.jsonl", "sft_from_feedback.jsonl", "sft"),
        ("training/hk_advisor/sft_traces.jsonl", "sft_hk_advisor.jsonl", "sft"),
    ]


def _required_inputs(out: Path) -> dict:
    gi = out / "general_instruct.jsonl"
    gi_rel = str(gi.relative_to(ROOT)) if gi.exists() else "training/local_sophia_v2/general_instruct.jsonl"
    return {
        "general_instruction_retention": {
            "source": gi_rel,
            "message": "Bring a license-clean external instruct slice (~10% of mix) or the model becomes a narrow refusal machine.",
        },
        "moral_gate_sft": {
            "source": "training/moral_gate_sft.jsonl",
            "message": "Convert moral_corpus/ structured data into routing SFT examples (allow/revise/retrieve/clarify/escalate/abstain/block).",
        },
    }


# Legacy alias for imports/tests
SFT_SOURCES = _sft_sources(ROOT / "training" / "local_sophia_v2")
DPO_SOURCES = [
    ("training/hard_negatives_dpo.jsonl", "dpo_hard_negatives.jsonl", "dpo"),
    ("training/wiki_provenance_dpo.jsonl", "dpo_wiki_provenance.jsonl", "dpo"),
    ("training/hk_advisor/dpo_pairs.jsonl", "dpo_hk_advisor.jsonl", "dpo"),
    ("training/tool_use/dpo_pairs.jsonl", "dpo_tool_use_mcp.jsonl", "dpo"),
]

REQUIRED_INPUTS = _required_inputs(ROOT / "training" / "local_sophia_v2")


def _existing_baseline(out: Path) -> dict | None:
    manifest = out / "manifest.json"
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    baseline = data.get("baseline")
    return baseline if isinstance(baseline, dict) else None


def build(
    check_only: bool,
    *,
    out: Path = DEFAULT_OUT,
    base_model: str = DEFAULT_BASE_MODEL,
) -> int:
    from provenance_bench.dataset_guard import normalize, prompt_of
    from provenance_bench.holdout_seal import verify_manifest

    seal_path = ROOT / "agi-proof" / "sophia-7b-train-verify" / "heldout-seal.manifest.json"
    if seal_path.exists():
        seal = verify_manifest(seal_path, ROOT / HOLDOUT_SRC)
        if not seal["ok"]:
            print("::error:: holdout seal mismatch — run tools/seal_sophia_7b_holdout.py", file=sys.stderr)
            return 1

    out = out.resolve()
    holdout = _read_jsonl(ROOT / HOLDOUT_SRC)
    holdout_prompts = {normalize(prompt_of(r)) for r in holdout if prompt_of(r)}
    evalset = eval_prompt_set(root=ROOT)
    # forbidden = held-out eval prompts ∪ the local holdout prompts
    forbidden = set(evalset) | holdout_prompts

    packs: dict[str, dict] = {}
    all_train: list[dict] = []
    all_sft: list[dict] = []
    dropped_total = 0

    sft_sources = _sft_sources(out)
    required_inputs = _required_inputs(out)

    for rel, name, kind in sft_sources + DPO_SOURCES:
        rows = _read_jsonl(ROOT / rel)
        # DECONTAMINATE: drop any row whose prompt collides with eval/holdout.
        clean_rows, dropped = [], 0
        for r in rows:
            pr = prompt_of(r)
            if pr and normalize(pr) in forbidden:
                dropped += 1
            else:
                clean_rows.append(r)
        dropped_total += dropped
        packs[name] = {"source": rel, "kind": kind, "rows": len(clean_rows),
                       "present": bool(rows), "droppedForDecontamination": dropped}
        if clean_rows:
            all_train.extend(clean_rows)
            if kind == "sft":
                all_sft.extend(clean_rows)
            if not check_only:
                _write_jsonl(out / name, clean_rows)

    if holdout and not check_only:
        _write_jsonl(out / "holdout.jsonl", holdout)

    # MLX-LM consumes a data directory with train/valid JSONL. Train only on SFT-style
    # messages; DPO/preference rows are retained for future preference training, not MLX SFT.
    # Fit every row under the training max-seq-length so nothing is silently truncated
    # (the v2 run truncated overlong rows — see the failure ledger).
    from tools.split_long_training_rows import fit_rows

    mlx_train_fitted, mlx_train_fit = fit_rows(_to_mlx_rows(all_sft), max_tokens=MLX_MAX_TOKENS)
    mlx_valid_fitted, mlx_valid_fit = fit_rows(_to_mlx_rows(holdout), max_tokens=MLX_MAX_TOKENS)
    if not check_only:
        _write_jsonl(out / "mlx" / "train.jsonl", mlx_train_fitted)
        _write_jsonl(out / "mlx" / "valid.jsonl", mlx_valid_fitted)

    # --- fail-closed guard: after decontamination, train MUST be disjoint ---
    contam = check_contamination(all_train, evalset, root=ROOT)
    holdout_overlap = [prompt_of(r) for r in all_train
                       if prompt_of(r) and normalize(prompt_of(r)) in holdout_prompts]

    missing_required = {
        key: spec["message"]
        for key, spec in required_inputs.items()
        if not _read_jsonl(ROOT / spec["source"])
    }

    manifest = {
        "schema": "sophia.local_sophia_dataset.v2",
        "trainingGoal": "local verifier-gated wisdom model — NOT AGI",
        "baseModel": base_model,
        "packs": packs,
        "holdout": {"source": HOLDOUT_SRC, "rows": len(holdout)},
        "trainRowsTotal": len(all_train),
        "recommendedMix": {  # documented target ratios (see training doc)
            "sft_source_discipline": 0.30, "sft_council_traces": 0.20,
            "moral_gate_sft": 0.15, "tool_use_mcp": 0.15,
            "dpo_hard_negatives": 0.10, "general_instruction_retention": 0.10,
        },
        "missingRequiredInputs": missing_required,
        "excluded": ["benchmark/eval holdouts", "hidden-eval packs", "API keys",
                     "unverified self-generated answers"],
        "baseline": _existing_baseline(out),  # fill via tools/eval_ladder.py on your hardware BEFORE training
        "promotionRule": "promote only if provenance/citation improves at acceptable "
                         "false-positive cost (no useful-correctness regression).",
        "contamination": {
            "droppedForDecontamination": dropped_total,
            "vsEval": contam,
            "vsHoldoutOverlapCount": len(holdout_overlap),
            "clean": contam["clean"] and not holdout_overlap,
        },
        "mlx": {"trainRows": len(mlx_train_fitted), "validRows": len(mlx_valid_fitted),
                "path": str(out.relative_to(ROOT) / "mlx"), "maxTokens": MLX_MAX_TOKENS,
                "fit": {"train": mlx_train_fit, "valid": mlx_valid_fit}},
        "claimBoundary": "Trains behavioral discipline, not general intelligence. "
                         "External MCP/verifier gates enforce correctness at runtime.",
    }

    if not check_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in
                      ("trainRowsTotal", "missingRequiredInputs", "contamination", "mlx")}, indent=2))

    if not manifest["contamination"]["clean"]:
        print("::error:: CONTAMINATION — training prompts overlap eval/holdout. Fail-closed.")
        return 1
    print("contamination guard: CLEAN" + ("  (check-only)" if check_only else f"  → wrote {out}"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="guard only; no writes (CI)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output pack directory")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="target base model id")
    args = ap.parse_args(argv)
    return build(args.check, out=args.out, base_model=args.base_model)


if __name__ == "__main__":
    raise SystemExit(main())
