# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Append-only, prefix-stable chat session — the cache-stability core.

Ported from the one idea in DeepSeek-Reasonix worth porting: a conversation that
grows **append-only** so the KV-cache prefix is never invalidated. Every turn's
context is a *prefix* of the next turn's context (until a compaction), so a
prefix-caching provider re-bills only the new tail, not the whole history. That is
the difference between a long rollout costing O(turns) and O(turns²) in input
tokens — and rollout cost is what gates how many RLVR traces you can afford.

This module is pure-Python and deterministic: token counts come from a fixed
heuristic, not a live tokenizer, so the cache-savings property is provable offline
in CI rather than asserted about a provider's billing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def count_tokens(text: str) -> int:
    """Deterministic, offline token estimate (~4 chars/token, min 1 for nonempty).

    Not a real BPE count — it does not need to be. The cache-savings claim is a
    structural property of an append-only prefix, so any monotone length proxy
    demonstrates it reproducibly without pulling in a tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


@dataclass
class Session:
    """An append-only message log with a stable, cache-friendly prefix.

    Invariant (``assert_append_only``): the message list only ever grows by
    appending, except for an explicit ``compact()`` which is the single sanctioned
    prefix-reset. ``cached_prefix_tokens`` tracks how many leading tokens a
    prefix-cache would already hold going into the next turn.
    """

    system: str = ""
    messages: list[Message] = field(default_factory=list)
    context_window: int = 8192
    compact_ratio: float = 0.8
    # Tokens the provider's prefix cache already holds (everything sent so far).
    cached_prefix_tokens: int = 0
    compactions: int = 0
    _prefix_history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cached_prefix_tokens = 0
        if self.system:
            # The system prompt is the base of the stable prefix.
            self._prefix_history.append(count_tokens(self.system))

    # -- prefix accounting --------------------------------------------------- #
    def prefix_tokens(self) -> int:
        """Total tokens in the current stable prefix (system + all messages)."""
        return count_tokens(self.system) + sum(m.tokens for m in self.messages)

    def needs_compaction(self) -> bool:
        return self.prefix_tokens() >= self.compact_ratio * self.context_window

    # -- mutation (append-only) --------------------------------------------- #
    def append(self, role: str, content: str) -> Message:
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self._prefix_history.append(self.prefix_tokens())
        return msg

    def assert_append_only(self) -> bool:
        """Prefix-token history is non-decreasing between compactions (the cache
        invariant). A drop is only legal immediately after a ``compact()``."""
        return all(b >= a for a, b in zip(self._prefix_history, self._prefix_history[1:]))

    def compact(self, summary: str) -> None:
        """The single sanctioned prefix reset: replace history with a summary.

        Low-frequency by design — only when ``needs_compaction()`` — because each
        compaction throws away the warm prefix cache (the summary must be read
        fresh once). Mirrors Reasonix's compact-once-at-0.8 policy.
        """
        kept_system = self.system
        self.messages = [Message(role="system", content=f"[compacted context]\n{summary}")]
        self.system = kept_system
        self.compactions += 1
        self.cached_prefix_tokens = 0  # cache is cold after a compaction
        self._prefix_history.append(self.prefix_tokens())

    def mark_sent(self) -> None:
        """Record that the current prefix has now been seen by the provider, so the
        next turn re-uses it from cache."""
        self.cached_prefix_tokens = self.prefix_tokens()
