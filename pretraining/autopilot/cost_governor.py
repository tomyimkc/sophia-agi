# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cost governor — a hard USD ceiling for autonomous GPU search.

The one rule that makes autonomous GPU experimentation safe: **never spend past a number
the human set.** This tracks projected and actual spend against a ceiling and answers a
single question before every trial — *can we afford the next run?* — fail-closed.

  * ``estimate_trial()`` — projected $ for one trial = price/hr × est hours/trial.
  * ``can_afford(n)``    — would launching ``n`` more trials stay within the ceiling,
    given what's already been spent? (Uses the *projected* per-trial cost, not hope.)
  * ``record(hours, price_per_hr)`` — book ACTUAL spend after a real run completes.
  * ``remaining()`` / ``spent()`` — live accounting.

Defaults are anchored to a REAL observed price (``$0.69/hr``, from a prior RunPod run logged
in agi-proof/benchmark-results/runpod-train/). All pure stdlib, deterministic, no I/O.
"""
from __future__ import annotations

from typing import Any

# Observed in agi-proof/benchmark-results/runpod-train/sft-3seed-*.log: costPerHr=0.69 on a
# 24–48 GB pod. Per-trial hours is a planning estimate until the calibration run measures it.
OBSERVED_PRICE_PER_HR = 0.69
DEFAULT_EST_HOURS_PER_TRIAL = 0.75   # 3B QLoRA on ~800 rows + eval ladder + pod overhead


class BudgetExceeded(RuntimeError):
    """Raised when an action would push projected/actual spend past the ceiling."""


class CostGovernor:
    def __init__(self, ceiling_usd: float, *, price_per_hr: float = OBSERVED_PRICE_PER_HR,
                 est_hours_per_trial: float = DEFAULT_EST_HOURS_PER_TRIAL,
                 overhead_frac: float = 0.30) -> None:
        if ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be > 0")
        self.ceiling = float(ceiling_usd)
        self.price_per_hr = float(price_per_hr)
        self.est_hours = float(est_hours_per_trial)
        self.overhead_frac = float(overhead_frac)   # image pulls / evictions / SSH retries
        self._spent = 0.0
        self._trials = 0

    # -- projections ----------------------------------------------------------
    def estimate_trial(self) -> float:
        """Projected $ for one trial, incl. a buffer for pod overhead/retries."""
        return round(self.price_per_hr * self.est_hours * (1 + self.overhead_frac), 4)

    def projected_total(self, n_trials: int) -> float:
        return round(self._spent + n_trials * self.estimate_trial(), 4)

    def can_afford(self, n_trials: int = 1) -> bool:
        """Fail-closed: only True if launching ``n_trials`` more stays within the ceiling."""
        return self.projected_total(n_trials) <= self.ceiling

    def max_affordable_trials(self) -> int:
        per = self.estimate_trial()
        if per <= 0:
            return 0
        return max(0, int((self.ceiling - self._spent) // per))

    # -- guards ---------------------------------------------------------------
    def guard(self, n_trials: int = 1) -> None:
        """Raise BudgetExceeded unless ``n_trials`` more are affordable."""
        if not self.can_afford(n_trials):
            raise BudgetExceeded(
                f"projected ${self.projected_total(n_trials)} exceeds ceiling ${self.ceiling} "
                f"(spent ${self._spent}, est ${self.estimate_trial()}/trial)")

    # -- actuals --------------------------------------------------------------
    def record(self, hours: float, *, price_per_hr: float | None = None) -> dict:
        """Book ACTUAL spend for a completed run. Returns the updated ledger snapshot."""
        price = self.price_per_hr if price_per_hr is None else float(price_per_hr)
        cost = round(hours * price, 4)
        self._spent = round(self._spent + cost, 4)
        self._trials += 1
        # refine the per-trial estimate from observed reality (calibration in action)
        self.est_hours = hours if self._trials == 1 else (self.est_hours + hours) / 2
        if price_per_hr is not None:
            self.price_per_hr = price
        over = self._spent > self.ceiling
        return {"trial_cost_usd": cost, "spent_usd": self._spent, "trials": self._trials,
                "over_ceiling": over, "refined_est_hours": round(self.est_hours, 4)}

    def spent(self) -> float:
        return self._spent

    def remaining(self) -> float:
        return round(max(0.0, self.ceiling - self._spent), 4)

    def snapshot(self) -> dict:
        return {
            "ceiling_usd": self.ceiling,
            "price_per_hr": self.price_per_hr,
            "est_hours_per_trial": self.est_hours,
            "est_cost_per_trial": self.estimate_trial(),
            "spent_usd": self._spent,
            "remaining_usd": self.remaining(),
            "trials_run": self._trials,
            "max_affordable_trials": self.max_affordable_trials(),
        }


__all__ = ["CostGovernor", "BudgetExceeded", "OBSERVED_PRICE_PER_HR",
           "DEFAULT_EST_HOURS_PER_TRIAL"]
