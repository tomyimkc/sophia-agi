# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""C2 — ship-steering-or-ABSTAIN, mirroring the selfextend false-accept contract.
Ship a vector only if all three falsifiable invariants hold on the HELD-OUT split;
else ABSTAIN + record. fail-closed. An unvalidated steering vector is worse than none."""
from __future__ import annotations

GAP = 0.20            # mirrors verified_reward.reward_is_hackable gap
OFF_TARGET_MAX = 0.10  # mirrors flywheel heldoutFalseAcceptRate cap


def ship_steering(*, fit_shift: float, held_shift: float, heldout_off_target_rate: float,
                  target_moved_on_heldout: bool, axis: str, ledger=None) -> dict:
    invariants = {
        "not_gamed": (fit_shift - held_shift) <= GAP,
        "off_target_bounded": heldout_off_target_rate <= OFF_TARGET_MAX,
        "target_moved": bool(target_moved_on_heldout),
    }
    ship = all(invariants.values())
    reason = None
    if not ship:
        # reported reason: first failing invariant in this fixed order
        if not invariants["target_moved"]:
            reason = "target_not_moved"
        elif not invariants["not_gamed"]:
            reason = "steering_gamed"
        else:
            reason = "steering_off_target"
        if ledger is not None:
            ledger.record(domain=axis, reason=reason)
    return {"ship": ship, "invariants": invariants, "reason": reason}
