# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reasoning operations over the OKF consequence graph.

This subpackage holds the iterative / branching consequence machinery that sits
on top of the non-destructive ``okf.revision`` primitives:

- ``ko_detector`` — GO ko-rule on an iterative revision sequence: detect when a
  multi-step revise/reassert loop returns to a prior belief state and force the
  gate to escalate (a ko is irreducible without new information).

- ``revise_loop`` — the iterative consumer that drives ``okf.revise`` round by
  round over a retraction schedule and runs ``ko_detector`` after each round. The
  ko-detector's contract only becomes live once a caller iterates; this module is
  that caller.

The single-shot ``simulate_cascade`` (one retraction -> one cascade) lives in
``agent.consequence_gate`` as the 8th conscience path; the operations here are
the multi-step extensions.
"""
from reasoning.consequence.ko_detector import KO_MAX_ROUNDS, KOAlert, detect_ko, is_ko
from reasoning.consequence.revise_loop import ReviseLoopState, run_revise_loop

__all__ = ["KO_MAX_ROUNDS", "KOAlert", "detect_ko", "is_ko", "ReviseLoopState", "run_revise_loop"]
