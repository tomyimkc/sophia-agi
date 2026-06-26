# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Frontier dangerous-capability evaluation harness (Sophia-AGI).

POSTURE (non-negotiable): this package *measures and refuses* dangerous
capability — it never produces uplift. Every artifact is defensive, category-level,
and contains no operational hazardous detail. Findings are fail-closed and follow
responsible disclosure. candidateOnly=true; not Level-3 evidence; no AGI claim.

This is the MEASUREMENT half of frontier safety. The repo already has the POLICY
half — `agent/ssil_capability_ceiling.py` (G8, RSP-style HALT) and
`agent/ssil_sandbagging.py` (G8E, elicitation gate). This harness upgrades the
scattered refusal-style probes to the standard `agent/uplift.py` already uses:
paired control-vs-treatment, held-out test split, bootstrap CIs, and a
gold-calibrated scorer whose own error rate is published, not assumed zero.

Roadmap: docs/06-Roadmap/frontier-readiness/06-frontier-evals.md (M0 = Milestone A core).
"""

__all__ = ["harness", "banks", "scoring", "rubric", "report"]
