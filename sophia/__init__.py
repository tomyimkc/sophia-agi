# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Public package surface for Sophia AGI.

Sophia is packaged as verifier-gated AGI-candidate training/proof machinery,
not as a claim that AGI has been achieved.
"""

from sophia.trainer import (
    CommandSpec,
    ExperimentConfig,
    build_experiment_plan,
    load_experiment_config,
)

__version__ = "0.9.0"

__all__ = [
    "__version__",
    "CommandSpec",
    "ExperimentConfig",
    "build_experiment_plan",
    "load_experiment_config",
]
