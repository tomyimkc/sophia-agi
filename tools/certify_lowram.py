#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Certify a QAT-trained LoRA artifact against its NVFP4 serving quantization.

*Why this exists.* ``tools/train_lora.py --qat --qat-scheme nvfp4`` produces a LoRA
adapter that was *co-adapted* to NVFP4 serving (``training/qat.py`` installs a fake-quant
forward during training). That training is the *mechanism*; it does not, on its own, license
the claim "this model serves at NVFP4 without losing quality". ``serving/lowram_eval.py``
(the :class:`LowRamGate`) is the *measurement*: it compares the full-precision next-token
distributions against the low-RAM (quantized) ones over a held-out calibration set and decides
pass/fail against an explicit, fail-closed contract. This script is the GPU glue that produces
those two distribution sets for a *real* model + adapter and runs the gate.

*The two paths it compares (the only honest comparison):*
  - **full**   : base model + LoRA delta, in bf16 — the artifact at full precision.
  - **nvfp4**  : the SAME merged model, with the served linear weights snapped to the exact
                 NVFP4 grid the model trained against (``training.qat`` / ``moe.quant``), and
                 everything else (embeddings, norms, router gate, lm_head) left in bf16.

*Two correctness rules this encodes — both were the failure mode of the first attempt:*
  1. **The full path must actually contain the LoRA.** A peft/transformers version skew can make
     ``PeftModel.from_pretrained`` raise (e.g. ``WeightConverter.__init__() got an unexpected
     keyword argument 'distributed_operation'``). Falling back to the *base model only* makes the
     comparison meaningless. We instead fall back to a **correct manual LoRA merge** read straight
     from ``adapter_model.safetensors`` + ``adapter_config.json``, so the full path is always
     base+LoRA.
  2. **Only the served set gets quantized.** NVFP4 serving quantizes the big attention/MLP/expert
     projection linears — NOT the embeddings, layernorms, MoE router gate, or ``lm_head``. Crushing
     ``lm_head`` and the embeddings to 4 bits is catastrophic and inflates output KL by an order of
     magnitude (the first run's ``mean_kl=0.41`` artifact). Quantizing exactly the QAT-co-adapted
     linears is both the standard recipe and the only fair test of what was trained.

The numerics (manual-merge math, NVFP4 round-trip, the gate, and the "quantizing lm_head is worse"
lesson) are verified GPU-free in :func:`offline_invariants` / ``--selftest`` — the repo's reference
discipline. The torch/transformers forward passes are the on-device piece, guarded behind imports.

Usage on the DGX Spark (inside .venv, NVFP4 is Blackwell-native there)::

    TRITON_INTERPRET=1 python tools/certify_lowram.py \
        --base-model allenai/OLMoE-1B-7B-0924-Instruct \
        --adapter training/lora/checkpoints/olmoe-qat-spark \
        --calib training/lora/holdout.jsonl \
        --scheme nvfp4 --dtype bf16 --attn sdpa \
        --n-eval 256 --out training/lora/checkpoints/olmoe-qat-spark/lowram_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The attn + MLP/expert projection linears NVFP4 serving quantizes and that QAT co-adapts
# (mirrors tools/train_lora.ATTN_MLP_MODULES). Embeddings / norms / router gate / lm_head stay bf16.
SERVED_LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj")

# Default no-overclaim contract — identical to serving.lowram_eval.LowRamGate defaults and the
# certify contract wired in tools/runpod_qat_lowram.py.
DEFAULT_CONTRACT = dict(max_mean_kl=0.05, min_top1_agreement=0.97,
                        protected_max_kl=0.10, protected_min_agreement=0.95)


# --------------------------------------------------------------------------- #
# Prompt formatting — reuse the EXACT chat format train_lora.py trained on, so the
# certify distributions are over the same surface the adapter saw.
# --------------------------------------------------------------------------- #
def _split_prompt_completion(messages: list[dict]) -> "tuple[str, str]":
    """Same reconstruction as tools/train_lora.split_prompt_completion (kept local to avoid
    importing the trainer's heavy module graph)."""
    asst_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not asst_idx:
        return "", ""
    last_asst = max(asst_idx)
    prefix_parts: list[str] = []
    for m in messages[:last_asst]:
        content = str(m.get("content", "")).strip()
        if content:
            prefix_parts.append(f"<|{m.get('role', 'user')}|>\n{content}")
    prompt = "\n".join(prefix_parts) + "\n<|assistant|>\n"
    completion = str(messages[last_asst].get("content", "")).strip() + "\n<|end|>"
    return prompt, completion


def _load_calib_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# LoRA loading — robust to peft/transformers version skew (the first-attempt bug).
# --------------------------------------------------------------------------- #
def _manual_merge_lora(model: Any, adapter_dir: Path) -> "tuple[int, dict]":
    """Merge a PEFT LoRA adapter into ``model``'s weights in place, read directly from
    ``adapter_model.safetensors`` + ``adapter_config.json``.

    For each target linear PEFT stores ``lora_A`` (r, in) and ``lora_B`` (out, r); the delta is
    ``scaling * (B @ A)`` with ``scaling = alpha / r`` (``alpha / sqrt(r)`` under rsLoRA), honoring
    per-module ``rank_pattern`` / ``alpha_pattern``. This reproduces exactly what
    ``PeftModel.merge_and_unload`` does — used only when the peft API path raises.

    Returns (modules_merged, info). Raises if the adapter files are missing (fail closed — never
    silently degrade to a base-only "full" path).
    """
    import torch
    from safetensors.torch import load_file

    cfg_path = adapter_dir / "adapter_config.json"
    wpath = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not wpath.exists():
        raise FileNotFoundError(
            f"adapter files missing under {adapter_dir} "
            f"(need adapter_config.json + adapter_model.safetensors)")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    r_default = int(cfg.get("r", 16))
    alpha_default = float(cfg.get("lora_alpha", 32))
    use_rslora = bool(cfg.get("use_rslora", False))
    rank_pattern = cfg.get("rank_pattern", {}) or {}
    alpha_pattern = cfg.get("alpha_pattern", {}) or {}

    sd = load_file(str(wpath))
    # Group A/B tensors by their target-module key (strip peft's wrapper prefixes/suffixes).
    targets: dict[str, dict] = {}
    for key, tensor in sd.items():
        if ".lora_A." in key:
            base, ab = key.split(".lora_A.", 1)[0], "A"
        elif ".lora_B." in key:
            base, ab = key.split(".lora_B.", 1)[0], "B"
        else:
            continue
        # base like "base_model.model.model.layers.0.self_attn.q_proj" -> module path under model
        mod = base
        for pfx in ("base_model.model.", "base_model."):
            if mod.startswith(pfx):
                mod = mod[len(pfx):]
                break
        targets.setdefault(mod, {})[ab] = tensor

    name_to_module = dict(model.named_modules())
    merged = 0
    skipped: list[str] = []
    for mod, ab in targets.items():
        if "A" not in ab or "B" not in ab:
            skipped.append(mod)
            continue
        module = name_to_module.get(mod)
        if module is None or not hasattr(module, "weight"):
            skipped.append(mod)
            continue
        # Per-module rank/alpha (rank_pattern/alpha_pattern keys are the module suffix).
        r_eff = r_default
        a_eff = alpha_default
        for pat, val in rank_pattern.items():
            if mod.endswith(pat):
                r_eff = int(val)
        for pat, val in alpha_pattern.items():
            if mod.endswith(pat):
                a_eff = float(val)
        scaling = a_eff / (r_eff ** 0.5) if use_rslora else a_eff / r_eff
        with torch.no_grad():
            A = ab["A"].to(device=module.weight.device, dtype=torch.float32)
            B = ab["B"].to(device=module.weight.device, dtype=torch.float32)
            delta = (B @ A) * scaling                       # (out, in)
            module.weight.add_(delta.to(module.weight.dtype))
        merged += 1
    info = {"merged": merged, "skipped": skipped, "scaling_rslora": use_rslora,
            "r": r_default, "alpha": alpha_default}
    return merged, info


def load_merged_model(base_model: str, adapter_dir: Path, *, dtype_str: str,
                      attn: str, device: str) -> "tuple[Any, Any, dict]":
    """Load base + LoRA as a single merged bf16 model. Tries the peft API first, falls back to a
    correct manual merge on version skew. Returns (model, tokenizer, load_info)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
    if attn and attn != "auto":
        load_kwargs["attn_implementation"] = attn
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    info: dict[str, Any] = {"adapter": str(adapter_dir)}
    merged_model = None
    try:
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        merged_model = peft_model.merge_and_unload()
        info["lora_load"] = "peft.merge_and_unload"
    except Exception as exc:  # noqa: BLE001 - version skew is the documented failure
        info["lora_load"] = f"manual_merge (peft path failed: {type(exc).__name__}: {exc})"
        n, mi = _manual_merge_lora(base, adapter_dir)
        info["manual_merge"] = mi
        if n == 0:
            raise RuntimeError(
                "manual LoRA merge applied 0 modules — refusing to certify a base-only model. "
                f"Check target-module name matching against {adapter_dir}.") from exc
        merged_model = base

    if device != "cuda":
        merged_model = merged_model.to(device)
    merged_model.eval()
    merged_model.config.use_cache = False
    return merged_model, tok, info


# --------------------------------------------------------------------------- #
# NVFP4 quantization of the served linears (in place) — the low-RAM deployment.
# --------------------------------------------------------------------------- #
def quantize_served_linears(model: Any, *, scheme: str,
                            suffixes: "tuple[str, ...]" = SERVED_LINEAR_SUFFIXES) -> dict:
    """Snap every served linear's weight to its serving grid in place; leave the rest bf16.

    Uses the exact fake-quant the model trained against: ``training.qat`` torch NVFP4
    (``_torch_nvfp4``) / per-channel INT8. Returns a report of params quantized vs kept so the
    memory ratio is computed honestly (not a flat headline number)."""
    import torch
    from training.qat import _torch_nvfp4

    def fq(w: "torch.Tensor") -> "torch.Tensor":
        if scheme == "nvfp4":
            return _torch_nvfp4(w)
        # int8 per-channel symmetric (matches training.qat._torch_ste_quant int8 branch)
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        return torch.clamp(torch.round(w / scale), -127, 127) * scale

    q_params = 0
    kept_params = 0
    q_modules = 0
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        is_linear = type(module).__name__ == "Linear" and module.weight.dim() == 2
        served = any(name.endswith(s) for s in suffixes)
        n = module.weight.numel()
        if is_linear and served:
            with torch.no_grad():
                module.weight.copy_(fq(module.weight.float()).to(module.weight.dtype))
            q_params += n
            q_modules += 1
        else:
            kept_params += n
    return {"quantized_modules": q_modules, "quantized_params": q_params,
            "kept_params": kept_params}


def effective_mem_ratio(q_params: int, kept_params: int, *, scheme: str,
                        from_bits: int = 16) -> "tuple[float, float]":
    """(per-tensor ratio, whole-model effective ratio) for the quantized fraction.

    Per-tensor NVFP4 = 4.5 bits → ``from_bits/4.5`` (≈3.56 vs bf16). Whole-model effective ratio
    accounts for the bf16-kept embeddings/norms/lm_head, so it is the honest deployment number."""
    from moe.quant import nvfp4_memory_reduction

    # NVFP4 = 4.5 effective bits (4 + 8/block); INT8 = 8 bits.
    per_tensor = nvfp4_memory_reduction(from_bits=from_bits) if scheme == "nvfp4" else from_bits / 8.0
    q_eff_bits = from_bits / per_tensor
    total = q_params + kept_params
    if total == 0:
        return per_tensor, 1.0
    avg_bits = (q_params * q_eff_bits + kept_params * from_bits) / total
    return per_tensor, from_bits / avg_bits


# --------------------------------------------------------------------------- #
# Forward pass — next-token distributions over the held-out completion positions.
# --------------------------------------------------------------------------- #
def collect_next_token_probs(model: Any, tok: Any, rows: list[dict], *,
                             n_eval: int, max_seq_len: int, device: str,
                             protected_ids: "set[str] | None" = None):
    """Run ``model`` over the calibration rows and return (probs (N,V), protected_mask, ids).

    Scores the assistant/completion token positions (the next-token predictions the gate
    compares), capped at ``n_eval`` positions total. Softmax in fp32 → numpy float64."""
    import numpy as np
    import torch

    probs: list = []
    mask: list[bool] = []
    used_ids: list[str] = []
    protected_ids = protected_ids or set()
    model_device = next(model.parameters()).device
    for row in rows:
        if len(probs) >= n_eval:
            break
        msgs = row.get("messages") or []
        prompt, completion = _split_prompt_completion(msgs)
        if not completion:
            continue
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt + completion, add_special_tokens=False)["input_ids"]
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
        if len(full_ids) <= len(prompt_ids) + 1:
            continue
        ids = torch.tensor([full_ids], dtype=torch.long, device=model_device)
        with torch.no_grad():
            logits = model(ids).logits[0]                 # (T, V)
        # Position t predicts token t+1; score the completion positions [len(prompt)-1 .. T-2].
        start = max(len(prompt_ids) - 1, 0)
        is_prot = str(row.get("id", "")) in protected_ids
        for t in range(start, len(full_ids) - 1):
            if len(probs) >= n_eval:
                break
            p = torch.softmax(logits[t].float(), dim=-1).to("cpu").numpy().astype(np.float64)
            probs.append(p)
            mask.append(is_prot)
            used_ids.append(str(row.get("id", "")))
    if not probs:
        return np.zeros((0, 0)), [], []
    return np.stack(probs), mask, used_ids


# --------------------------------------------------------------------------- #
# Main — orchestrate full vs nvfp4 and run the gate.
# --------------------------------------------------------------------------- #
def run_certify(args: argparse.Namespace) -> dict:
    import numpy as np

    from serving.lowram_eval import LowRamGate

    adapter_dir = Path(args.adapter)
    calib = Path(args.calib)
    rows = _load_calib_rows(calib)
    protected_ids = set(args.protected_ids.split(",")) if args.protected_ids else None

    model, tok, load_info = load_merged_model(
        args.base_model, adapter_dir, dtype_str=args.dtype, attn=args.attn, device=args.device)
    print(f"[load] {load_info.get('lora_load')}", flush=True)

    # 1) full bf16 (base + LoRA) next-token distributions.
    full_probs, prot_mask, ids = collect_next_token_probs(
        model, tok, rows, n_eval=args.n_eval, max_seq_len=args.max_seq_len,
        device=args.device, protected_ids=protected_ids)
    print(f"[full] collected {full_probs.shape[0]} next-token positions", flush=True)

    # 2) quantize the served linears in place → low-RAM (NVFP4) distributions over the SAME rows.
    qinfo = quantize_served_linears(model, scheme=args.scheme)
    per_tensor_ratio, eff_ratio = effective_mem_ratio(
        qinfo["quantized_params"], qinfo["kept_params"], scheme=args.scheme)
    print(f"[nvfp4] quantized {qinfo['quantized_modules']} linears "
          f"({qinfo['quantized_params']/1e6:.1f}M params); per-tensor {per_tensor_ratio:.2f}x, "
          f"whole-model {eff_ratio:.2f}x vs {args.dtype}", flush=True)
    low_probs, _, _ = collect_next_token_probs(
        model, tok, rows, n_eval=args.n_eval, max_seq_len=args.max_seq_len,
        device=args.device, protected_ids=protected_ids)

    # 3) run the no-overclaim gate.
    gate = LowRamGate(**DEFAULT_CONTRACT)
    report = gate.evaluate(full_probs, low_probs,
                           protected_mask=prot_mask if protected_ids else None,
                           mem_ratio=eff_ratio)
    out = report.as_dict()
    out["device"] = args.device
    out["scheme"] = args.scheme
    out["base_model"] = args.base_model
    out["adapter"] = str(adapter_dir)
    out["lora_load"] = load_info.get("lora_load")
    out["per_tensor_mem_ratio"] = round(per_tensor_ratio, 4)
    out["quantized_modules"] = qinfo["quantized_modules"]
    out["quantized_params"] = qinfo["quantized_params"]
    out["kept_params"] = qinfo["kept_params"]
    out["n_calib_rows"] = len(rows)
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base-model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    ap.add_argument("--adapter", default="training/lora/checkpoints/olmoe-qat-spark",
                    help="PEFT adapter dir (adapter_model.safetensors + adapter_config.json)")
    ap.add_argument("--calib", default="training/lora/holdout.jsonl",
                    help="held-out calibration rows (jsonl with chat 'messages')")
    ap.add_argument("--scheme", choices=("int8", "nvfp4"), default="nvfp4")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--attn", choices=("auto", "sdpa", "eager", "flash_attention_2"), default="sdpa")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--n-eval", type=int, default=256, help="max next-token positions to score")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--protected-ids", default="",
                    help="comma-separated calib row ids forming the must-not-regress slice")
    ap.add_argument("--out", type=Path, default=None, help="write the LowRamReport JSON here")
    ap.add_argument("--selftest", action="store_true",
                    help="run GPU-free offline invariants and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        ok, detail = offline_invariants()
        print("certify_lowram offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        print(f"  detail: {json.dumps({k: v for k, v in detail.items() if k != 'checks'})}")
        return 0 if ok else 1

    out = run_certify(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n=== LowRamReport ===")
    print(json.dumps(out, indent=2))
    print(f"\nVERDICT: {'PASS' if out['passed'] else 'FAIL'} "
          f"(mean_kl={out['mean_kl']}, top1={out['top1_agreement']}, "
          f"mem_ratio={out['mem_ratio']}x, n={out['n_eval']})")
    return 0 if out["passed"] else 2


# --------------------------------------------------------------------------- #
# Offline invariants — GPU-free, prove the merge/quant/gate logic on synthetic tensors.
# --------------------------------------------------------------------------- #
def offline_invariants() -> "tuple[bool, dict]":
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return False, {"checks": {"numpy_available": False}}

    from serving.lowram_eval import LowRamGate
    from training.qat import fake_quant

    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    # ---- 1. Manual LoRA merge math reproduces the reference delta = scaling * (B @ A). --------
    out_f, in_f, r = 32, 24, 8
    alpha = 16.0
    W = rng.standard_normal((out_f, in_f))
    A = rng.standard_normal((r, in_f))      # peft lora_A: (r, in)
    B = rng.standard_normal((out_f, r))     # peft lora_B: (out, r)
    scaling = alpha / r
    delta = scaling * (B @ A)
    W_merged = W + delta
    # A hidden vector through merged weight == through base + the lora low-rank path.
    x = rng.standard_normal((4, in_f))
    lhs = x @ W_merged.T
    rhs = x @ W.T + scaling * ((x @ A.T) @ B.T)
    checks["manual_merge_matches_lowrank_path"] = bool(np.allclose(lhs, rhs, atol=1e-9))
    detail["merge_resid"] = float(np.max(np.abs(lhs - rhs)))

    # rsLoRA scaling = alpha / sqrt(r), distinct from alpha / r.
    checks["rslora_scaling_differs"] = abs(alpha / (r ** 0.5) - alpha / r) > 1e-6

    # ---- 2. Build a tiny 2-layer "model": quantizing only the served linear keeps KL low; -----
    #         additionally quantizing the lm_head (the first-attempt bug) makes KL much worse.
    V, H, N = 40, 32, 64
    h = rng.standard_normal((N, H)) * 0.7
    W_served = rng.standard_normal((H, H)) * 0.3      # an inner projection (served, quantize it)
    W_head = rng.standard_normal((V, H)) * 0.3        # lm_head (DO NOT quantize in the good path)

    def logits_of(Ws, Wh):
        return (h @ Ws.T) @ Wh.T

    full_logits = logits_of(W_served, W_head)
    full = softmax(full_logits)

    good = softmax(logits_of(fake_quant(W_served, "nvfp4"), W_head))           # served-only
    bad = softmax(logits_of(fake_quant(W_served, "nvfp4"),
                            fake_quant(W_head, "nvfp4")))                      # + lm_head (bug)
    gate = LowRamGate()
    rep_good = gate.evaluate(full, good, mem_ratio=3.56)
    rep_bad = gate.evaluate(full, bad, mem_ratio=3.56)
    checks["served_only_kl_below_lm_head_kl"] = rep_good.mean_kl < rep_bad.mean_kl
    detail["served_only_mean_kl"] = round(rep_good.mean_kl, 5)
    detail["plus_lm_head_mean_kl"] = round(rep_bad.mean_kl, 5)

    # ---- 3. Identical model passes (~0 KL, 100% agreement) — sanity floor on the gate. --------
    rep_id = gate.evaluate(full, full.copy(), mem_ratio=3.56)
    checks["identical_passes"] = rep_id.passed and rep_id.mean_kl < 1e-9

    # ---- 4. effective_mem_ratio: whole-model ratio sits between 1x and the per-tensor ratio ---
    #         (bf16-kept params pull the headline 3.56x down — the honest deployment number).
    per_t, eff = effective_mem_ratio(900_000_000, 100_000_000, scheme="nvfp4")
    checks["per_tensor_nvfp4_is_3p56"] = abs(per_t - 3.5556) < 0.01
    checks["eff_ratio_between_1_and_per_tensor"] = 1.0 < eff < per_t
    detail["eff_ratio_90pct_quantized"] = round(eff, 4)

    # ---- 5. Served-suffix matching only fires on the attn/MLP projections. --------------------
    served_names = ["model.layers.0.self_attn.q_proj", "model.layers.3.mlp.experts.7.down_proj"]
    kept_names = ["lm_head", "model.embed_tokens", "model.layers.0.input_layernorm",
                  "model.layers.0.mlp.gate"]  # MoE router gate stays bf16
    checks["served_match_projections"] = all(
        any(n.endswith(s) for s in SERVED_LINEAR_SUFFIXES) for n in served_names)
    checks["served_skip_head_norm_router"] = not any(
        any(n.endswith(s) for s in SERVED_LINEAR_SUFFIXES) for n in kept_names)

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    raise SystemExit(main())
