#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Certify a QAT-trained model served low-RAM (NVFP4) against its 16-bit self — the Boundary-3 number.

This is the GPU glue the DGX Spark runs after `tools/train_lora.py --qat`: it produces 16-bit (BF16 on
CUDA) vs served-quantized next-token distributions from the REAL model over a probe set and runs them
through the no-overclaim gate (`serving.lowram_eval.LowRamGate`). A pass is the first real "low-RAM,
capability-retained" evidence (`docs/11-Platform/Cheap-Compute-Boundary.md` Boundary 3).

What it does:
  1. Load the base model + the trained LoRA adapter (BF16) and **merge** the adapter → the served artifact.
  2. Build a probe set from a decontaminated, deployment-distribution file (default
     `training/lora/train.jsonl`). This is a *quantization-fidelity* probe — both arms see the SAME
     inputs and we measure their divergence, so it deliberately does NOT read the eval-sealed benchmark
     holdout (see `provenance_bench.holdout_seal`). Pass any decontaminated text via ``--eval-file``.
  3. Collect next-token softmax distributions over the SAME positions twice:
       - ``full``   : the BF16 merged model as-is.
       - ``lowram`` : every ``nn.Linear`` weight passed through the served quantization
                      (``training.qat`` STE fake-quant, default NVFP4 — the Blackwell-native grid).
  4. ``LowRamGate().evaluate(full, lowram, mem_ratio=...)`` → print the ``LowRamReport``
     (``passed``, ``mean_kl``, ``top1_agreement``, ``mem_ratio``). Exits non-zero on FAIL.

The quantizer and the gate are the merged, CI-tested repo pieces (`training/qat.py`, `moe/quant.py`,
`serving/lowram_eval.py`); only the model forward + weight-swap here are torch/GPU, so they are guarded
and not run in CI. ``--selftest`` exercises the non-torch logic (probe-set loader + gate on synthetic
distributions) so this file still carries an offline invariant.

Spark notes: BF16 + NVFP4 only — do NOT need bitsandbytes (no ``--4bit`` here). Honest scope: a Spark
pass is benchmark evidence, not the registered result (`REPLICATION.md`); the headline claim still needs
>=2 judge families / >=3 seeds / CIs (RESULTS.md), for which `config/inference.local.mac-judge.json`
wires the Spark+Mac judge farm.

Usage (on the Spark, after the QAT train):
    python tools/certify_lowram.py \
        --base-model allenai/OLMoE-1B-7B-0924-Instruct \
        --adapter training/lora/checkpoints/olmoe-qat-spark --scheme nvfp4
    python tools/certify_lowram.py --selftest      # CI-safe, no torch/GPU
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Default probe set: deployment-distribution training data (NOT the eval-sealed benchmark holdout —
# this is a full-vs-quant fidelity probe on identical inputs, so contamination does not apply).
DEFAULT_EVAL = "training/lora/train.jsonl"


# --------------------------------------------------------------------------- #
# Probe-set loading (pure-python, CI-tested)
# --------------------------------------------------------------------------- #
def _row_text(row: dict) -> str:
    """Reconstruct a chat row (or use the flat 'text' field) into one string."""
    if isinstance(row.get("messages"), list):
        parts = []
        for m in row["messages"]:
            content = str(m.get("content", "")).strip()
            if content:
                parts.append(f"<|{m.get('role', 'user')}|>\n{content}")
        return "\n".join(parts)
    return str(row.get("text", "")).strip()


def load_eval_texts(path: Path, *, max_rows: int, min_chars: int = 16) -> "list[str]":
    """Load up to ``max_rows`` non-trivial probe texts from a JSONL file. Skips short/blank rows.

    Streams the file line-by-line and short-circuits once ``max_rows`` is reached, so a large probe
    file is never fully read into memory (the Spark is low-RAM focused).
    """
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _row_text(row)
            if len(text) >= min_chars:
                texts.append(text)
            if len(texts) >= max_rows:
                break
    return texts


# --------------------------------------------------------------------------- #
# Torch glue (deployment path; guarded — not run in CI)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def quantized_linears(model, scheme: str):  # pragma: no cover - torch-only
    """Temporarily replace every nn.Linear weight with its served quantization, then restore.

    Uses the SAME fake-quant as QAT (``training.qat._torch_ste_quant``) so the certified
    low-bit weights match what training co-adapted to. Restores originals in ``finally``.
    """
    import torch
    from training.qat import _torch_ste_quant

    ste = _torch_ste_quant()
    saved: dict[str, "torch.Tensor"] = {}
    try:
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear) and m.weight is not None:
                    saved[name] = m.weight.data
                    m.weight.data = ste.apply(m.weight.data, scheme)
        yield
    finally:
        with torch.no_grad():
            for name, m in model.named_modules():
                if name in saved:
                    m.weight.data = saved[name]


def _tokenize_sequences(tokenizer, texts, *, max_seq_len):  # pragma: no cover - torch-only
    import torch

    seqs = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_seq_len).input_ids
        if ids.shape[1] >= 2:
            seqs.append(ids)
    return seqs


def _probs_for_sequences(model, seqs, *, max_positions_per_seq, max_total, device):  # pragma: no cover
    """Next-token softmax over fixed positions of cached sequences → (N, vocab) float32 numpy.

    Deterministic position selection (first ``max_positions_per_seq`` predictable positions of each
    sequence), so two calls (full / lowram) align row-for-row.
    """
    import numpy as np
    import torch

    out = []
    total = 0
    with torch.no_grad():
        for ids in seqs:
            if total >= max_total:
                break
            ids = ids.to(device)
            logits = model(ids).logits[0]                       # (seq, vocab)
            probs = torch.softmax(logits[:-1].float(), dim=-1)  # predict tok t+1 from pos t
            take = min(max_positions_per_seq, probs.shape[0], max_total - total)
            out.append(probs[:take].cpu().numpy().astype(np.float32))
            total += take
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype="float32")


def certify(*, base_model: str, adapter: str, scheme: str, eval_file: str, max_rows: int,
            max_positions_per_seq: int, max_total: int, max_seq_len: int) -> dict:  # pragma: no cover
    """Run the full FP16-vs-low-RAM certification on the real model. Returns the report dict."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from moe.quant import int8_memory_reduction, nvfp4_memory_reduction
    from serving.lowram_eval import LowRamGate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype,
                                                 trust_remote_code=True).to(device)
    if adapter and Path(adapter).exists():
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()        # fold LoRA into the base → the served artifact
    model.eval()

    texts = load_eval_texts(Path(eval_file), max_rows=max_rows)
    if not texts:
        return {"error": f"no eval texts in {eval_file}"}
    seqs = _tokenize_sequences(tok, texts, max_seq_len=max_seq_len)

    full = _probs_for_sequences(model, seqs, max_positions_per_seq=max_positions_per_seq,
                                max_total=max_total, device=device)
    with quantized_linears(model, scheme):
        lowram = _probs_for_sequences(model, seqs, max_positions_per_seq=max_positions_per_seq,
                                      max_total=max_total, device=device)

    mem_ratio = nvfp4_memory_reduction(16) if scheme == "nvfp4" else int8_memory_reduction(16)
    report = LowRamGate().evaluate(full, lowram, mem_ratio=mem_ratio)
    out = report.as_dict()
    out.update({"base_model": base_model, "adapter": adapter, "scheme": scheme,
                "device": device, "n_positions": int(full.shape[0])})
    return out


def main(argv: "list[str] | None" = None) -> int:  # pragma: no cover - torch-only
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--adapter", default="training/lora/checkpoints/olmoe-qat-spark")
    ap.add_argument("--scheme", choices=("int8", "nvfp4"), default="nvfp4")
    ap.add_argument("--eval-file", default=DEFAULT_EVAL,
                    help="decontaminated probe JSONL (deployment distribution); NOT the eval-sealed "
                         "benchmark set. Default: deployment-distribution training data.")
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--max-positions-per-seq", type=int, default=8)
    ap.add_argument("--max-total", type=int, default=256, help="cap on total eval positions")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    args = ap.parse_args(argv)

    report = certify(base_model=args.base_model, adapter=args.adapter, scheme=args.scheme,
                     eval_file=args.eval_file, max_rows=args.max_rows,
                     max_positions_per_seq=args.max_positions_per_seq, max_total=args.max_total,
                     max_seq_len=args.max_seq_len)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if "error" in report:
        return 1
    passed = bool(report.get("passed"))
    verdict = "PASS" if passed else "FAIL"
    print(f"\nLow-RAM certification [{args.scheme}] {verdict}: "
          f"mean_kl={report.get('mean_kl')} top1_agreement={report.get('top1_agreement')} "
          f"(gate: mean_kl<=0.05, top1>=0.97)")
    # Non-zero on a gate FAIL so scripts/CI can branch on the exit code (2 = FAIL, 1 = error).
    return 0 if passed else 2


# --------------------------------------------------------------------------- #
# Offline invariants (CI-safe; no torch/GPU)
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy as np
    except Exception:
        return False, {"checks": {"numpy_available": False}}
    import tempfile

    from serving.lowram_eval import LowRamGate

    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Eval-set loader: reads messages + text rows, skips short/blank, respects max_rows.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "probe.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": "x" * 40}, {"role": "assistant", "content": "y" * 40}]},
            {"text": "z" * 40},
            {"text": "tiny"},                      # too short -> skipped
            "",                                    # blank line -> skipped
        ]
        p.write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows), encoding="utf-8")
        texts = load_eval_texts(p, max_rows=10)
        checks["loader_reads_two_valid"] = len(texts) == 2
        checks["loader_skips_short"] = all(len(t) >= 16 for t in texts)
        texts_capped = load_eval_texts(p, max_rows=1)
        checks["loader_respects_max_rows"] = len(texts_capped) == 1

    # 2. The gate call (the certify path's verdict step) behaves on synthetic distributions:
    #    identical model passes, badly-degraded fails — mirrors serving.lowram_eval.
    rng = np.random.default_rng(0)
    z = rng.standard_normal((32, 40)) * 3
    e = np.exp(z - z.max(1, keepdims=True))
    full = e / e.sum(1, keepdims=True)
    gate = LowRamGate()
    checks["identical_passes"] = gate.evaluate(full, full.copy(), mem_ratio=3.56).passed
    z2 = z + rng.standard_normal((32, 40)) * 6
    e2 = np.exp(z2 - z2.max(1, keepdims=True))
    bad = e2 / e2.sum(1, keepdims=True)
    rep_bad = gate.evaluate(full, bad)
    checks["degraded_fails"] = not rep_bad.passed
    checks["report_has_fields"] = {"passed", "mean_kl", "top1_agreement", "mem_ratio"} <= set(rep_bad.as_dict())
    detail["bad_mean_kl"] = round(rep_bad.mean_kl, 4)

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        ok, detail = offline_invariants()
        print("Certify-lowram offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        raise SystemExit(0 if ok else 1)
    raise SystemExit(main())
