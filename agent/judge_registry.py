# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judge-family registry and openness classification (Leiden value 5: autonomy).

The no-overclaim gate requires >=2 INDEPENDENT judge families. Today the panel is served
by proprietary inference (e.g. ``openrouter:...``). The Leiden Declaration favours
non-proprietary, publicly governed tooling, so this module makes two distinctions explicit
and machine-checkable for any panel of judge ids:

  * ``open_weights``         — the judged MODEL has openly released weights (llama, mistral,
                               qwen, gemma, deepseek, ...). Several current judges already are.
  * ``self_hostable``        — the INFERENCE runs without a proprietary API (a local /
                               self-hosted OpenAI-compatible endpoint). This is the real gap:
                               an open-weights model behind a proprietary API is not yet a
                               non-proprietary validation path.

A judge id is ``"<provider>:<model>"`` (e.g. ``"openrouter:meta-llama/llama-3.3-70b-instruct"``
or ``"local:qwen2.5-32b-instruct"``). Pure stdlib, deterministic, offline.
"""
from __future__ import annotations

# Providers that are proprietary hosted-inference APIs.
PROPRIETARY_INFERENCE_PROVIDERS = frozenset({
    "openrouter", "llmhub", "openai", "anthropic", "google", "gemini", "together",
    "fireworks", "groq", "deepinfra",
})
# Providers that denote a local / self-hosted OpenAI-compatible endpoint (no proprietary API).
SELF_HOSTABLE_PROVIDERS = frozenset({
    "local", "local-endpoint", "vllm", "ollama", "tgi", "llamacpp", "lmstudio", "sglang",
})
# Model-name fragments whose weights are openly released (license openness varies; see notes).
OPEN_WEIGHTS_MODEL_FAMILIES = frozenset({
    "llama", "mistral", "mixtral", "qwen", "gemma", "deepseek", "phi", "yi", "falcon",
    "olmo", "smollm", "nemotron", "command-r", "dbrx", "stablelm",
})


def parse_judge_id(judge_id: str) -> "tuple[str, str]":
    """Split ``provider:model``; provider is lowercased. No colon -> provider 'unknown'."""
    s = str(judge_id or "").strip()
    if ":" in s:
        provider, model = s.split(":", 1)
        return provider.strip().lower(), model.strip()
    return "unknown", s


def _is_open_weights(model: str) -> bool:
    m = model.lower()
    return any(fam in m for fam in OPEN_WEIGHTS_MODEL_FAMILIES)


def classify_judge(judge_id: str) -> "dict":
    """Classify a single judge id into openness booleans."""
    provider, model = parse_judge_id(judge_id)
    self_hostable = provider in SELF_HOSTABLE_PROVIDERS
    proprietary_inference = provider in PROPRIETARY_INFERENCE_PROVIDERS
    open_weights = _is_open_weights(model)
    return {
        "id": judge_id,
        "provider": provider,
        "model": model,
        "open_weights": open_weights,
        # a local endpoint is self-hostable; if it serves an open-weights model it is also
        # a fully non-proprietary path.
        "self_hostable": self_hostable,
        "proprietary_inference": proprietary_inference,
        "non_proprietary_path": self_hostable and open_weights,
    }


def classify_panel(judge_ids: "list[str]") -> "dict":
    """Aggregate openness over a panel of judge ids (the >=2-family validation set)."""
    judges = [classify_judge(j) for j in judge_ids]
    n = len(judges)
    n_open_weights = sum(1 for j in judges if j["open_weights"])
    n_self_hostable = sum(1 for j in judges if j["self_hostable"])
    n_non_proprietary = sum(1 for j in judges if j["non_proprietary_path"])
    return {
        "judges": judges,
        "n_total": n,
        "n_open_weights": n_open_weights,
        "n_self_hostable": n_self_hostable,
        "n_non_proprietary_path": n_non_proprietary,
        "has_open_weights_judge": n_open_weights > 0,
        "has_self_hostable_judge": n_self_hostable > 0,
        # Leiden value 5 is fully served only when at least one corroborating family runs
        # on a non-proprietary path (open weights + self-hosted inference).
        "has_non_proprietary_path": n_non_proprietary > 0,
    }


__all__ = [
    "PROPRIETARY_INFERENCE_PROVIDERS", "SELF_HOSTABLE_PROVIDERS",
    "OPEN_WEIGHTS_MODEL_FAMILIES", "parse_judge_id", "classify_judge", "classify_panel",
]
