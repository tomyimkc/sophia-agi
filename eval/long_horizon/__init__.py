# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-horizon execution eval — deterministic, offline, per-step verifiable.

Sophia already has two long-horizon *engine* pieces:

  * ``agent/long_horizon.py`` — the durable task-tree execution engine.
  * ``agent/horizon.py`` — a METR-style effective-horizon *curve* (oracle-judged
    chained arithmetic).

and one *autonomy logger* (``tools/run_long_horizon.py`` →
``agi-proof/long-horizon-runs/``) that records interventions/tool-calls and judges
a whole run with a coarse autonomy classification or a single objective gate.

What was MISSING — and what this package adds — is a measurement harness that scores
multi-step task execution by a *sequence of dependent, deterministic per-step
checkpoints* (success only if every dependent sub-goal is met; verified by pure
functions, NOT an LLM judge), and summarises it with the three honest long-horizon
constructs:

  * **completion rate** — fraction of tasks where every checkpoint passed,
  * **step-level success** — fraction of all checkpoints passed,
  * **horizon length** — longest fully-correct prefix (one slip ends the horizon),

each with a bootstrap / anytime-valid CI from ``tools/eval_stats.py``.

Everything here is deterministic and offline: tasks carry pure-function verifiers and
the harness drives an agent through a narrow :class:`~eval.long_horizon.tasks.Agent`
interface, so a deterministic mock agent makes the whole pipeline testable with no
model and no network. A real model run is OUT OF SCOPE here; the committed artifact is
PENDING/not-run (see ``tools/run_long_horizon_eval.py``).
"""

from eval.long_horizon.tasks import (
    Agent,
    Checkpoint,
    LongHorizonTask,
    Step,
    StepResult,
    example_tasks,
)
from eval.long_horizon.harness import (
    HarnessResult,
    TaskRun,
    horizon_length,
    run_task,
    run_tasks,
)

__all__ = [
    "Agent",
    "Checkpoint",
    "LongHorizonTask",
    "Step",
    "StepResult",
    "example_tasks",
    "HarnessResult",
    "TaskRun",
    "horizon_length",
    "run_task",
    "run_tasks",
]
