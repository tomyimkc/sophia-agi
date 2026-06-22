"""selfextend — the self-extending verification flywheel (the path-to-AGI engine).

Connects Sophia's static pieces into a loop that grows its OWN competence:

    abstain -> localize the gap (abstention ledger) -> synthesize a verifier ->
    validate it on held-out data -> promote only if it clears the bar (else stay
    abstained) -> coverage & competence rise -> repeat.

Plus the components that loop needs and Sophia did not yet have: calibrated
uncertainty everywhere (ECE/Brier), a competence self-model, a causal world model
(do-operator, beyond provenance), cross-domain transfer of the one loop,
environment-as-verifier (verify by executing), and the verified-reward signal a
live RLVR run would optimize (with an anti-gaming held-out check).

Everything here is deterministic, dependency-free, and offline-testable: the
machinery and its falsifiable metrics, not a capability claim. Live RLVR (GPU) and
live grounding (network) consume these interfaces but are out of scope to *run* here.
"""

from selfextend.abstention_ledger import AbstentionLedger
from selfextend.verifier_synthesis import Rule, propose_and_validate, synthesize_verifier
from selfextend.flywheel import run_flywheel
from selfextend.calibration_metrics import brier_score, expected_calibration_error
from selfextend.competence_map import CompetenceMap
from selfextend.causal_graph import CausalGraph
from selfextend.transfer import run_transfer
from selfextend.env_verifier import verify_by_execution
from selfextend.verified_reward import reward_is_hackable, verified_reward
from selfextend.long_horizon import run_long_horizon
from selfextend.loop import close_loop

__all__ = [
    "AbstentionLedger", "Rule", "synthesize_verifier", "propose_and_validate",
    "run_flywheel", "expected_calibration_error", "brier_score", "CompetenceMap",
    "CausalGraph", "run_transfer", "verify_by_execution", "verified_reward",
    "reward_is_hackable", "run_long_horizon", "close_loop",
]
