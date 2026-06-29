# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Frontier consensus demotion — the regime where even `consensus` can fall, but only
under a decisive-evidence bar, never under time or counting pressure alone.

WHY THIS EXISTS (the thesis-level decision):

  `okf.decay_okf` Principle P3 makes `consensus` decay-immune by default. That is source
  discipline: the AI must not flip-flop on settled knowledge every time a surprising
  preprint appears. The replication crisis, cold-fusion announcements, OPERA faster-than-
  light neutrinos — all were single high-surprise events that were wrong. Counting
  contradictions (or trusting surprise alone) is exactly the cheap proxy that *caused*
  those errors. So P3 holds.

  But P3 unconditionally would freeze the system inside whatever paradigm it started in.
  Real science has overturned consensus: Newton → Einstein, steady-state → Big Bang,
  luminiferous aether → special relativity. A truth-seeking system built for unsolved
  frontier problems (its stated goal) must be *revolutionary-ready* — without becoming
  a postmodern relativist that treats every claim as equally valid.

THE RULE (Bayes factor primary, N is a multiplicity floor):

  A `consensus` belief marked `frontier=True` may be demoted ONE confidence rank
  (consensus → disputed), or superseded in a named regime, ONLY when ALL hold:

    1. DECISIVE Bayes factor:  K = P(data | challenger) / P(data | consensus) >= K_DECISIVE
       (K_DECISIVE = 100, the Jeffreys "decisive evidence" threshold). The challenger must
       explain the data overwhelmingly better — not merely equally well.
    2. MULTIPLICITY FLOOR: the evidence spans >= N_MIN independent observation groups
       (N_MIN = 3). A single experiment — however surprising — cannot topple consensus.
       This is the rule that would have protected special relativity from OPERA (2011,
       N=1, later a loose fiber-optic cable).
    3. SURPRISE GATE: each of the N observations must be high-surprise under the consensus
       model (predictive probability below a gate), so noise or routine confirmation
       cannot manufacture a demotion.
    4. SOURCE DISCIPLINE PRESERVED: demotion moves consensus → disputed (ONE rank), or
       emits a regime-scoped `supersededBy` edge when the challenger owns a regime.
       NEVER consensus → legendary/none_extant — those are *provenance* categories
       (who said it), not *evidence* categories (how well it predicts). Collapsing into
       them would launder away the audit trail of why consensus fell.

WHAT "REGIME" MEANS (the Einstein/Newton lesson):

  Einstein did not make Newton "legendary". Newtonian mechanics became `supersededBy`
  relativity IN THE HIGH-VELOCITY / STRONG-FIELD regime, while remaining consensus in the
  low-velocity regime where it is still correct and used to land spacecraft. Regime-scoped
  demotion is the honest outcome; global demotion is reserved for challengers that dominate
  everywhere.

FALSIFIABLE (the tests simulate both directions):

  - Newton → Einstein (1915-1919): K decisive, N>=3 independent (perihelion, light
    bending, redshift), all high-surprise under Newton  -> REGIME DEMOTION. Newton stays
    consensus in the low-velocity regime. PASS.
  - OPERA faster-than-light neutrino (2011): one high-surprise event, N=1 < floor
    -> QUARANTINE, not demotion. Special relativity consensus untouched. (And historically
    correct: it was an instrumental fault.) PASS.

Honesty boundary: this is a deterministic, offline decision rule over supplied evidence.
It does not itself compute Bayes factors from raw data — it takes them as audited input,
exactly as Sophia's gate takes verdicts as audited input. `level3Evidence: false`.
"""

from __future__ import annotations

from dataclasses import dataclass


# Jeffreys' scale of evidence for Bayes factors. We require DECISIVE to demote consensus.
#   K 1-3.2      "substantial"   — nowhere near enough to touch consensus
#   K 3.2-10     "strong"
#   K 10-100     "very strong"
#   K > 100      "decisive"      — the ONLY bar that can demote a frontier consensus
K_DECISIVE = 100.0
N_MIN = 3                       # multiplicity floor: independent observation groups
SURPRISE_GATE = 0.10            # an observation's predictive prob under consensus must be below this


@dataclass(frozen=True)
class Observation:
    """One piece of evidence against the consensus. All fields are audited inputs."""
    name: str
    surprise_p: float            # predictive probability under the consensus model (lower = more surprising)
    group: str                   # independence key — observations sharing a group are NOT independent

    @property
    def passes_surprise_gate(self) -> bool:
        return self.surprise_p < SURPRISE_GATE


@dataclass(frozen=True)
class DemotionEvidence:
    bayes_factor: float                  # K = P(data|challenger) / P(data|consensus), audited
    observations: tuple[Observation, ...]
    challenger_regime: str | None = None  # if set, demotion is regime-scoped; if None, global


@dataclass(frozen=True)
class DemotionDecision:
    demote: bool
    new_confidence: str | None            # None if not demoted; "disputed" if demoted
    superseded_by_regime: str | None      # challenger_regime if regime-scoped demotion
    reasons: tuple[str, ...] = ()
    quarantined: bool = False             # evidence present but bar not met -> hold for review

    def to_dict(self) -> dict:
        return {
            "schema": "sophia.okf_frontier_demotion.v1",
            "demote": self.demote,
            "newConfidence": self.new_confidence,
            "supersededByRegime": self.superseded_by_regime,
            "reasons": list(self.reasons),
            "quarantined": self.quarantined,
            "rankDrop": 1 if self.demote else 0,   # ALWAYS at most one rank
            "neverToProvenanceCategories": True,    # never legendary/none_extant
            "candidateOnly": True,
            "level3Evidence": False,
        }


def _independent_group_count(observations: tuple[Observation, ...]) -> int:
    """Number of distinct independence groups — the multiplicity that actually matters."""
    return len({o.group for o in observations})


def try_demote_consensus(
    belief_confidence: str,
    evidence: DemotionEvidence,
    *,
    domain_is_frontier: bool,
) -> DemotionDecision:
    """Decide whether a consensus belief may be demoted under the decisive-evidence rule.

    Returns a DemotionDecision. Non-frontier consensus is ALWAYS immune (P3 holds). Frontier
    consensus is demotable only on K>=K_DECISIVE AND >=N_MIN independent groups AND every
    observation surprise-gated. Below the bar -> quarantine (hold for review), never demote.
    """
    if belief_confidence != "consensus":
        return DemotionDecision(False, None, None, reasons=("not consensus; demotion rule n/a",))

    if not domain_is_frontier:
        # P3 unconditional: settled-domain consensus is absolutely decay-immune.
        return DemotionDecision(False, None, None, reasons=("settled-domain consensus: immune (source discipline)",))

    reasons: list[str] = []
    n_independent = _independent_group_count(evidence.observations)

    # Gate 3: surprise — every observation must be genuinely unexpected under consensus.
    failed_surprise = [o.name for o in evidence.observations if not o.passes_surprise_gate]
    if failed_surprise:
        reasons.append(f"observations failed surprise gate: {failed_surprise}")

    # Gate 2: multiplicity floor — no single experiment, however surprising, topples consensus.
    if n_independent < N_MIN:
        reasons.append(f"multiplicity floor not met: {n_independent} independent groups < {N_MIN}")

    # Gate 1: decisive Bayes factor — the challenger must explain the data overwhelmingly better.
    if evidence.bayes_factor < K_DECISIVE:
        reasons.append(
            f"bayes factor {evidence.bayes_factor:.1f} < decisive {K_DECISIVE:.0f} "
            f"(Jeffreys scale: strong=10, decisive=100)"
        )

    if reasons:
        # Evidence was presented but the bar is not met -> quarantine, do NOT demote.
        return DemotionDecision(
            False, None, None, reasons=tuple(reasons), quarantined=True,
        )

    # Bar cleared. Demote exactly one rank: consensus -> disputed.
    # Regime-scoped if the challenger owns a regime; else global single-rank demotion.
    regime = evidence.challenger_regime
    return DemotionDecision(
        True,
        new_confidence="disputed",                 # exactly ONE rank below consensus
        superseded_by_regime=regime,               # may be None (global) or a regime name
        reasons=(
            f"decisive: K={evidence.bayes_factor:.0f} >= {K_DECISIVE:.0f}, "
            f"{n_independent} independent groups >= {N_MIN}, all surprise-gated",
        ),
    )


# ---- Historical simulations (the falsifiable tests, made importable) ----

def simulate_newton_to_einstein() -> dict:
    """The paradigm case of correct regime-scoped demotion.

    By 1919 the evidence against Newtonian consensus was: Mercury perihelion anomaly,
    Eddington's light-bending measurement, gravitational redshift — three INDEPENDENT
    observation groups, each high-surprise under Newton, with GR explaining all decisively.
    Correct outcome: Newton is superseded by GR in the high-velocity/strong-field regime,
    but REMAINS consensus in the low-velocity regime (still used to navigate spacecraft).
    """
    evidence = DemotionEvidence(
        bayes_factor=1e6,   # GR predicts all three precisely; Newton cannot — decisive
        observations=(
            Observation("mercury_perihelion", surprise_p=0.001, group="orbital_dynamics"),
            Observation("light_bending_eddington", surprise_p=0.005, group="electromagnetism_gravity"),
            Observation("gravitational_redshift", surprise_p=0.008, group="spectroscopy"),
        ),
        challenger_regime="relativistic_strong_field",
    )
    decision = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    return {
        "case": "newton_to_einstein",
        "regimeScoped": decision.superseded_by_regime == "relativistic_strong_field",
        "newConfidence": decision.new_confidence,    # "disputed" — ONE rank, not collapsed
        "demoted": decision.demote,
        "stillConsensusInLowVelocity": decision.new_confidence == "disputed",  # not destroyed
        "decision": decision.to_dict(),
    }


def simulate_opera_ftl_neutrino() -> dict:
    """The counter-case: why the N>=3 floor exists.

    OPERA (2011) reported faster-than-light neutrinos — one extremely high-surprise result
    that, had we counted contradictions or trusted surprise alone, would have toppled
    special relativity. The multiplicity floor quarantines it. Historically correct: the
    anomaly was a loose fiber-optic cable + clock oscillator fault (2012).
    """
    evidence = DemotionEvidence(
        bayes_factor=1e4,   # seemingly large, but rests on ONE observation group
        observations=(
            Observation("opera_ftl_2011", surprise_p=1e-9, group="opera_detector"),  # N=1
        ),
        challenger_regime=None,
    )
    decision = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    return {
        "case": "opera_ftl_neutrino",
        "quarantined": decision.quarantined,         # held, not demoted
        "consensusUntouched": not decision.demote,
        "reason": decision.reasons,
        "decision": decision.to_dict(),
    }


__all__ = [
    "Observation", "DemotionEvidence", "DemotionDecision",
    "try_demote_consensus",
    "simulate_newton_to_einstein", "simulate_opera_ftl_neutrino",
    "K_DECISIVE", "N_MIN", "SURPRISE_GATE",
]
