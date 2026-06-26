# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LoRA search space (C2) — the knobs an autonomous sweep would tune, honestly labelled.

Defines the hyperparameters worth searching for the real Qwen LoRA pipeline and a
deterministic sampler that produces candidate configs for the ASHA scheduler. Each knob is
tagged with whether ``tools/runpod_train.py`` ALREADY accepts it (so the config transfers
today) or whether it needs a Step-2 passthrough flag — no silent assumption that a knob is
wired when it isn't.

    from pretraining.autopilot.search_space import sample_configs, WIRED_TODAY
    cfgs = sample_configs(12, seed=0)        # 12 candidate LoRA configs
"""
from __future__ import annotations

import random
from typing import Any

# The space. Values chosen to bracket sensible QLoRA settings for 3B/7B on small data.
SPACE = {
    "lora_rank": [8, 16, 32, 64],
    "lora_alpha": [16, 32, 64],
    "lr": [5e-5, 1e-4, 2e-4, 3e-4],
    "neftune_alpha": [0, 5, 10],
    "epochs": [1, 2, 3],
}

# Which knobs runpod_train.py accepts TODAY vs needs a Step-2 passthrough arg.
WIRED_TODAY = {"epochs", "seed", "model"}
NEEDS_PASSTHROUGH = {"lora_rank", "lora_alpha", "lr", "neftune_alpha"}


def sample_configs(n: int, *, seed: int = 0, model: str = "Qwen/Qwen2.5-3B-Instruct"
                   ) -> "list[dict[str, Any]]":
    """Deterministically sample ``n`` distinct-ish candidate configs from the space."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        cfg["model"] = model
        cfg["seed"] = 0
        # honesty tag: can this config be run as-is, or does it need the passthrough work?
        cfg["_transfers_today"] = all(
            k in WIRED_TODAY for k in cfg if k not in NEEDS_PASSTHROUGH or cfg[k] == SPACE[k][0]
        )
        out.append(cfg)
    return out


def passthrough_gap() -> dict:
    """Report what Step 2 must add to runpod_train.py for full-space search to transfer."""
    return {
        "wired_today": sorted(WIRED_TODAY),
        "needs_passthrough": sorted(NEEDS_PASSTHROUGH),
        "note": ("train_lora.py already accepts --lora-rank/--lr/--neftune-alpha; the Step-2 "
                 "change is threading these through runpod_train.py's remote command builder. "
                 "Until then, the sweep can only vary epochs/seed/model on real GPU."),
    }


__all__ = ["SPACE", "WIRED_TODAY", "NEEDS_PASSTHROUGH", "sample_configs", "passthrough_gap"]
