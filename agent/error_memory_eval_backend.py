# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evaluation backends for error-memory RAG measurement.

Default CI path uses DeterministicOracleBackend (disjoint held-out labels).
LocalLLMBackend is a stub for future live-model validation — not wired in CI.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Answer a held-out case without consulting the failure store labels."""

    name: str

    def answer(self, case: dict, *, seed: int = 0) -> str:
        """Return the model's candidate answer for ``case['question']``."""


class DeterministicOracleBackend:
    """Disjoint-oracle backend: baseline wrong/correct from held-out labels."""

    name = "deterministic-oracle"

    def answer(self, case: dict, *, seed: int = 0) -> str:
        del seed
        if case.get("modelWasRight") or not case.get("wouldRepeat"):
            return str(case["correctAnswer"])
        return str(case["wrongAnswer"])


class LocalLLMBackend:
    """Stub for live local-model eval — requires explicit wiring + credentials."""

    name = "local-llm-stub"

    def __init__(self, *, model_id: str = "unconfigured") -> None:
        self.model_id = model_id

    def answer(self, case: dict, *, seed: int = 0) -> str:
        del case, seed
        raise NotImplementedError(
            "LocalLLMBackend is not configured. Set model endpoint/credentials "
            "and implement generation before live-model error-memory eval."
        )


def default_backend() -> ModelBackend:
    return DeterministicOracleBackend()
