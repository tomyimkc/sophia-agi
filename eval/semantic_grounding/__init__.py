# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Semantic-grounding & compositional-grammar benchmark (Phase 0).

A sealed, deterministic, offline benchmark that asks whether grounding word
meaning in explicit OKF definitions (D1) and composing meaning by a small
concept grammar (D2) behaves better than ungrounded distributional guessing.

See docs/06-Roadmap/Semantic-Grounding-And-Compositional-Grammar-Program.md.

Status: candidateOnly. canClaimAGI=false. This is a MEASUREMENT HARNESS, not a
capability claim. The scorer is fail-closed: an axis whose inputs are absent
reports ``None`` (N/A), never a guessed pass.
"""
