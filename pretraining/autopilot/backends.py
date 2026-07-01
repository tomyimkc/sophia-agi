# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Execution backends for the autonomous experiment runner.

Two backends, by design:

* ``LocalBackend`` — runs a REAL nano-LM training experiment on CPU (pure Python, free,
  deterministic). This is what the autonomous loop actually iterates on: every score is a
  genuinely measured held-out loss, never a simulated number.

* ``RunPodEscalation`` — a deliberately *gated* adapter that turns a chosen config into a
  real ``tools/runpod_train.py`` command. It DEFAULTS TO DRY-RUN and never spends GPU money
  on its own: launching requires an explicit ``launch=True`` AND a cost ceiling AND an API
  key in the environment. It exists so a config the loop liked cheaply can be escalated to
  real GPU *with a human cost decision*, not autonomously. Honest caveat: the nano loop
  searches a toy model; the RunPod pipeline trains a real Qwen LoRA — escalation transfers
  the *methodology and the decision to spend*, not literal hyperparameters.
"""
from __future__ import annotations

import math
from typing import Any

from pretraining.nano import (
    NanoLM, eval_loss, make_source, mixed_corpus, sample_stream,
    source_entropy, to_examples, train,
)

# Mirror of tools/runpod_train.py's cheapest-first GPU fallback + rough public $/hr ranges
# (indicative only; the real price is whatever RunPod quotes at launch).
GPU_FALLBACK = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA L40S"]
EST_USD_PER_HR = {"NVIDIA GeForce RTX 4090": (0.34, 0.74), "NVIDIA RTX A5000": (0.26, 0.46),
                  "NVIDIA A40": (0.39, 0.79), "NVIDIA L40S": (0.79, 1.19)}


class LocalBackend:
    """Run a real nano experiment described by ``config`` and return measured metrics."""

    def run(self, config: "dict[str, Any]") -> "dict[str, Any]":
        vocab = config.get("vocab", 8)
        order = config.get("order", 2)
        context = config.get("context", 2)
        hidden = config.get("hidden", 16)
        D = config.get("D", 1600)
        epochs = config.get("epochs", 12)
        lr = config.get("lr", 0.03)
        seed = config.get("seed", 0)
        optimizer = config.get("optimizer", "adam")

        src = make_source(vocab=vocab, order=order, seed=config.get("source_seed", 1), peak=3.0)
        E = source_entropy(src)
        held_target = config.get("target", "A")

        if "mix" in config:
            # data-mixing experiment: two sources, fixed budget D, target distribution
            srcB = make_source(vocab=vocab, order=order, seed=config.get("source_seed_b", 2),
                               peak=3.0)
            wA = config["mix"]
            ex = mixed_corpus([src, srcB], [wA, 1 - wA], D, context=context, seed=10 + seed)
            if held_target == "B":
                held = to_examples(sample_stream(srcB, 1000, seed=778), context)
                E = source_entropy(srcB)
            elif held_target == "blend":
                held = (to_examples(sample_stream(src, 500, seed=777), context)
                        + to_examples(sample_stream(srcB, 500, seed=778), context))
            else:
                held = to_examples(sample_stream(src, 1000, seed=777), context)
        else:
            ex = to_examples(sample_stream(src, D + context, seed=2 + seed), context)
            held = to_examples(sample_stream(src, 1000, seed=99), context)

        m = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=seed)
        hist = train(m, ex, epochs=epochs, optimizer=optimizer, lr=lr, seed=seed)
        held_loss = eval_loss(m, held)
        diverged = bool(hist["diverged"]) or math.isnan(held_loss)  # NaN
        params = m.num_params()
        return {
            "held_loss": round(held_loss, 5) if not diverged else float("inf"),
            "train_loss": round(hist["final_train_loss"], 5),
            "diverged": diverged,
            "params": params,
            "compute_proxy": params * D,            # ~ params × tokens
            "floor_E": round(E, 5),
            "excess": round(held_loss - E, 5) if not diverged else float("inf"),
        }


class RunPodEscalation:
    """Turn a chosen config into a real (but gated) RunPod training command.

    Never launches on its own. ``plan()`` returns the exact dry-run command + a cost
    estimate. ``plan(launch=True, cost_ceiling_usd=X)`` is the ONLY way to request a real
    run, and even then it just emits the ``--yes`` command + asserts the guard — it does
    not shell out here, because spending GPU money must be a human action, not a side effect
    of a loop.
    """

    def plan(self, config: "dict[str, Any]", *, branch: str, epochs: int = 1,
             launch: bool = False, cost_ceiling_usd: float | None = None,
             est_hours: float = 1.0) -> "dict[str, Any]":
        gpu = GPU_FALLBACK[0]
        lo, hi = EST_USD_PER_HR[gpu]
        est_cost = (round(lo * est_hours, 2), round(hi * est_hours, 2))
        base = (f"python tools/runpod_train.py --branch {branch} "
                f"--epochs {epochs} --seed {config.get('seed', 0)}")
        guard_ok = launch and cost_ceiling_usd is not None and est_cost[1] <= cost_ceiling_usd
        return {
            "launched": False,                       # this adapter NEVER launches inline
            "mode": "launch_requested" if launch else "dry_run",
            "dry_run_command": base + " --dry-run",
            "launch_command": base + " --yes  # requires RUNPOD_API_KEY in env",
            "gpu_fallback": GPU_FALLBACK,
            "est_cost_usd": {"gpu": gpu, "hours": est_hours, "range": est_cost},
            "cost_ceiling_usd": cost_ceiling_usd,
            "guard": ("PASS — within ceiling; run the launch_command yourself to spend"
                      if guard_ok else
                      "BLOCKED — set launch=True, a cost_ceiling_usd >= est high end, and "
                      "RUNPOD_API_KEY; cost is a human decision, never autonomous"),
            "honesty_note": ("nano search ≠ Qwen LoRA hyperparameters; escalation transfers "
                             "the methodology and the spend decision, not literal config."),
        }


__all__ = ["LocalBackend", "RunPodEscalation", "GPU_FALLBACK", "EST_USD_PER_HR"]
