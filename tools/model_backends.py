# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The single model seam for the wisdom-internalization harness.

Both ``tools/gen_distill_traces.py`` (teacher farm) and ``tools/run_wisdom_ablation.py``
(proof matrix) consume a ``generate(system, user) -> ModelResult`` callable. This module
is the one place that builds that callable, for three backends:

  * ``mock`` — deterministic, offline; ``mock-base`` fabricates attributions, ``mock-student``
    abstains. Lets the whole pipeline run in CI with no weights.
  * ``mlx``  — Apple Silicon (Mac Studio M3 Ultra). Lazy-imports ``mlx_lm``; refuses off-Darwin.
  * ``hf``   — CUDA/CPU (DGX Spark). Lazy-imports ``transformers`` (+ optional 4-bit / PEFT adapter).

Heavy deps are imported INSIDE the factory so importing this module is free everywhere
(CI on Linux never pulls mlx; the Mac never pulls bitsandbytes). The returned callable
reuses ``agent.model.ModelResult`` so it drops straight into ``provenance_bench.runner``.

Decode params (temperature/top_p/max_tokens/seed) are recorded by callers into the data
passport, so they live here as one resolved dict.
"""
from __future__ import annotations

import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model import ModelResult

GenerateFn = Callable[[str, str], ModelResult]
_DID = re.compile(r"did (.+?) write (.+?)\?")


@dataclass(frozen=True)
class Decode:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 1337

    def as_dict(self) -> dict:
        return {"temperature": self.temperature, "top_p": self.top_p,
                "max_tokens": self.max_tokens, "seed": self.seed}


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# mock — offline, deterministic; the CI substrate
# --------------------------------------------------------------------------- #
def _make_mock(model: str) -> GenerateFn:
    student = "student" in model

    def generate(system: str, user: str) -> ModelResult:
        m = _DID.match(user.lower())
        if m:
            if student:  # internalized: abstains on ungrounded attribution, gate OFF
                text = "I can't confirm that attribution without provenance."
            else:        # naive base: asserts the (false) attribution by name
                text = f"Yes, {m.group(1).title()} wrote {m.group(2).title()}."
        else:
            text = "(answer)"
        return ModelResult(text=text, ok=True, provider="mock", model=model,
                           prompt_tokens=max(1, len(user) // 4),
                           completion_tokens=max(1, len(text) // 4), finish_reason="stop")
    return generate


# --------------------------------------------------------------------------- #
# mlx — Apple Silicon (Mac Studio). Lazy import; Darwin-only.
# --------------------------------------------------------------------------- #
def _make_mlx(model: str, adapter: str | None, dec: Decode) -> GenerateFn:
    if platform.system() != "Darwin":
        raise SystemExit("backend=mlx is Apple-Silicon only (fails closed off-Darwin). "
                         "Use --backend hf on the DGX Spark.")
    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError as e:  # pragma: no cover - env-specific
        raise SystemExit('mlx-lm not installed. On the Mac: pip install "mlx-lm>=0.20"') from e

    mdl, tok = mlx_load(model, adapter_path=adapter)

    def generate(system: str, user: str) -> ModelResult:
        prompt = tok.apply_chat_template(_messages(system, user), add_generation_prompt=True,
                                         tokenize=False)
        try:  # newer mlx_lm uses a sampler object for temperature
            from mlx_lm.sample_utils import make_sampler
            text = mlx_generate(mdl, tok, prompt=prompt, max_tokens=dec.max_tokens,
                                sampler=make_sampler(temp=dec.temperature, top_p=dec.top_p),
                                verbose=False)
        except Exception:  # older mlx_lm signature
            text = mlx_generate(mdl, tok, prompt=prompt, max_tokens=dec.max_tokens,
                                temp=dec.temperature, verbose=False)
        return ModelResult(text=(text or "").strip(), ok=True, provider="mlx", model=model,
                           completion_tokens=max(1, len(text or "") // 4), finish_reason="stop")
    return generate


# --------------------------------------------------------------------------- #
# hf — CUDA/CPU (DGX Spark). Lazy import; optional 4-bit + PEFT adapter.
# --------------------------------------------------------------------------- #
def _make_hf(model: str, adapter: str | None, dec: Decode, *, load_4bit: bool) -> GenerateFn:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("transformers/torch not installed. "
                         "On the Spark: pip install -r requirements-lora.txt") from e

    kwargs: dict = {"torch_dtype": "auto", "device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    if adapter:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, adapter)
    mdl.eval()
    torch.manual_seed(dec.seed)
    do_sample = dec.temperature > 0.0

    def generate(system: str, user: str) -> ModelResult:
        ids = tok.apply_chat_template(_messages(system, user), add_generation_prompt=True,
                                      return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=dec.max_tokens, do_sample=do_sample,
                               temperature=dec.temperature or None, top_p=dec.top_p,
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        return ModelResult(text=text.strip(), ok=True, provider="hf", model=model,
                           prompt_tokens=int(ids.shape[1]),
                           completion_tokens=int(out.shape[1] - ids.shape[1]),
                           finish_reason="stop")
    return generate


# --------------------------------------------------------------------------- #
# public factory
# --------------------------------------------------------------------------- #
def make_generate(backend: str, model: str, *, adapter: str | None = None,
                  decode: Decode | None = None, load_4bit: bool = False) -> GenerateFn:
    """Return a ``generate(system, user) -> ModelResult`` for the chosen backend.

    backend: ``mock`` | ``mlx`` | ``hf``. ``model`` is a HF repo id or local path;
    ``adapter`` is an optional LoRA adapter dir (the trained student)."""
    dec = decode or Decode()
    if backend == "mock":
        return _make_mock(model)
    if backend == "mlx":
        return _make_mlx(model, adapter, dec)
    if backend == "hf":
        return _make_hf(model, adapter, dec, load_4bit=load_4bit)
    raise SystemExit(f"unknown backend {backend!r} (choose: mock | mlx | hf)")
