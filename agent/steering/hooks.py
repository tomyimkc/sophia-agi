"""Residual-stream steering hooks + SteeredClient (the ONLY torch module).

torch is imported lazily inside functions so importing agent.steering never
requires torch. Steering math (the vector) stays a plain list[float] until the
hook boundary, where it becomes an fp32 tensor cast to the hidden-state dtype.
"""
from __future__ import annotations

import contextlib
import os

from agent.steering.vectors import Vector

# MPS safety: opt into CPU fallback for any unimplemented op before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def make_steering_hook(vec_f32, alpha: float):
    """register_forward_hook callback: add alpha*v to the residual-stream output.
    Handles transformers 4.x (tuple output) and 5.x (bare tensor). Casts v to the
    hidden state's dtype/device inside the hook (fp32 master → fp16 add on MPS)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            v = vec_f32.to(device=hs.device, dtype=hs.dtype)
            return (hs + alpha * v,) + tuple(output[1:])
        hs = output
        v = vec_f32.to(device=hs.device, dtype=hs.dtype)
        return hs + alpha * v
    return hook


@contextlib.contextmanager
def attach_hooks(model, vector: Vector, alpha: float, layers: "list[int]"):
    """Register a steering hook on each model.model.layers[L]; remove all in finally."""
    import torch
    vec_f32 = torch.tensor(list(vector), dtype=torch.float32)
    handles = []
    try:
        for L in layers:
            layer = model.model.layers[L]
            handles.append(layer.register_forward_hook(make_steering_hook(vec_f32, alpha)))
        yield
    finally:
        for h in handles:
            h.remove()


def capture_residual(model, layer_idx: int, run) -> Vector:
    """Register a capturing hook on model.model.layers[layer_idx], call run() (which
    triggers a forward), and return the captured residual stream mean-pooled over all
    positions as a plain list[float]. The pure-stdlib diff_of_means/normalize then turn
    these into the steering direction (those are CI-tested in Task 1)."""
    import torch  # noqa: F401
    captured: dict = {}

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        # mean over every dim except the last (hidden) → a [hidden] vector
        captured["v"] = hs.detach().float().mean(dim=tuple(range(hs.dim() - 1))).cpu().tolist()

    h = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        run()
    finally:
        h.remove()
    return captured["v"]


def extract_persona_vector(model, tokenizer, pos_prompts, neg_prompts, layer: int,
                           *, normalize: bool = True) -> Vector:
    """CAA difference-of-means axis vector (real path): mean residual on positive
    trait prompts minus mean on negatives, at `layer`, then normalize."""
    from agent.steering.vectors import diff_of_means, normalize as _normalize

    def _vecs(prompts):
        out = []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
            out.append(capture_residual(model, layer, lambda ids=ids: model(ids)))
        return out

    raw = diff_of_means(_vecs(pos_prompts), _vecs(neg_prompts))
    return _normalize(raw) if normalize else raw


class _Result:
    """Duck-types agent.model.ModelResult for measure_ocean (.ok / .text)."""
    def __init__(self, text: str, ok: bool = True):
        self.text = text
        self.ok = ok


class SteeredClient:
    """In-process HF model with an active steering vector. Duck-types
    measure_ocean's client: generate(system, user) -> object with .ok/.text.
    When vector/alpha are None it is the unsteered baseline client."""

    def __init__(self, model, tokenizer, *, vector: "Vector | None" = None,
                 alpha: float = 0.0, layers: "list[int] | None" = None,
                 max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.vector = vector
        self.alpha = alpha
        self.layers = layers or []
        self.max_new_tokens = max_new_tokens

    def _run(self, system: str, user: str) -> str:
        import torch
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    def generate(self, system: str, user: str):
        try:
            if self.vector is not None and self.layers:
                with attach_hooks(self.model, self.vector, self.alpha, self.layers):
                    return _Result(self._run(system, user).strip())
            return _Result(self._run(system, user).strip())
        except Exception as exc:  # never crash the measurement loop
            return _Result("", ok=False)
