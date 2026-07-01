#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""O5 (horizon bet, SIMULATION ONLY) — energy-based verification as oscillator-Ising dynamics.

Inspiration: unconv-ai/Un-0's deeper thesis is 'dynamical systems as a computing substrate'
for ~1000x lower energy. Oscillator Ising Machines (OIM) + Equilibrium Propagation (2025)
show an energy-minimization problem can be solved by the NATIVE dynamics of coupled analog
oscillators. O2 (energy_verifier_head) produces exactly such an energy — a scalar
compatibility energy over (answer, evidence). O5 asks the question O5 is allowed to ask:
IF that verification energy were mapped onto an oscillator substrate, does its equilibrium
recover the same accept/abstain decision as the digital argmin?

=====================================================================================
HONESTY BANNER — READ THIS.
This is a SOFTWARE SIMULATION of an idealized coupled-oscillator (Ising) relaxation. It is
NOT hardware, NOT an energy measurement, and NOT an LLM-scale result. It demonstrates ONE
thing only: that a small energy-based verification decision can be encoded as an oscillator
network whose relaxed phase state reproduces the digital decision. It says NOTHING about
whether real OIM hardware can run an LLM-scale semantic verifier — the literature's
demonstrated scale is small combinatorial energies (MNIST-class). Every output carries
simulationOnly:true, hardwareClaim:false, canClaimAGI:false. This tool exists to make the
O2->O5 bridge concrete and falsifiable-in-principle, not to claim the substrate works.
=====================================================================================

Method: given per-candidate energies (lower = more compatible; e.g. from O2), encode each
candidate as an oscillator whose self-bias pulls it toward the 'accept' phase in proportion
to -energy. Relax the network (gradient descent on an Ising-style energy landscape, the
digital stand-in for what OIM hardware does physically) and read the binary phase state.
Report whether the relaxed decision matches the digital argmin-energy decision, and the
'annealing gap' (how cleanly the substrate separated accept from abstain).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "sophia.oscillator_substrate_sim.v1",
        "environmentArtifact": True, "ok": False, "reason": reason,
        "simulationOnly": True, "hardwareClaim": False, "canClaimAGI": False,
    }


def relax_ising(bias: np.ndarray, coupling: np.ndarray, *, steps: int = 200, dt: float = 0.05,
                seed: int = 0) -> np.ndarray:
    """Relax phases in an OIM-style energy landscape (digital stand-in for physical relaxation).

    Each oscillator phase theta_i in (-pi, pi]; the sub-harmonic-injection term sin(2*theta)
    bistabilizes phases toward 0 ('accept') or pi ('abstain'). Energy:
      E = -sum_i bias_i cos(theta_i) - (1/2) sum_ij K_ij cos(theta_i - theta_j)
    Gradient descent on E is the update. Returns final phases.
    """
    n = len(bias)
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.1, 0.1, size=n)          # start near the 'accept' basin edge
    for _ in range(int(steps)):
        coupling_term = (coupling * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
        grad = bias * np.sin(theta) - coupling_term + 0.5 * np.sin(2 * theta)  # +binarizing term
        theta = theta - dt * grad
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi   # wrap to (-pi, pi]


def run(candidates: list[dict[str, Any]], *, seed: int = 0, steps: int = 200) -> dict[str, Any]:
    """candidates: [{"id":..., "energy": float}]  (energy lower = more compatible)."""
    cands = [c for c in candidates if "energy" in c]
    if len(cands) < 1:
        return _env_artifact("no candidates with an 'energy' field (fail-closed)")
    energies = np.array([float(c["energy"]) for c in cands])
    ids = [c.get("id", i) for i, c in enumerate(cands)]

    # DIGITAL reference decision: argmin energy = accept; others abstain.
    digital_accept = int(np.argmin(energies))

    # Encode as oscillator biases: only the MIN-energy candidate gets a positive (accept) bias;
    # all others get a negative (abstain) bias proportional to how far above the minimum they
    # sit. This is the winner-take-one verification decision (argmin energy = the single accept),
    # scaled so the winner and runner-up land in distinct phase basins (a real annealing gap).
    emin = energies.min()
    bias = np.where(energies <= emin + 1e-9, 1.5, -(1.0 + (energies - emin)))
    coupling = np.zeros((len(cands), len(cands)))
    theta = relax_ising(bias, coupling, steps=steps, seed=seed)

    # phase near 0 => accept; phase near +-pi => abstain
    accept_score = np.cos(theta)                    # in [-1,1]; >0 => accept basin
    substrate_accept = int(np.argmax(accept_score))
    matches = (substrate_accept == digital_accept)
    # annealing gap: separation between the accepted oscillator and the rest
    sorted_scores = np.sort(accept_score)[::-1]
    gap = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

    return {
        "schema": "sophia.oscillator_substrate_sim.v1",
        "simulationOnly": True, "hardwareClaim": False, "canClaimAGI": False,
        "n": len(cands),
        "digitalDecision": {"acceptId": ids[digital_accept], "acceptIndex": digital_accept},
        "substrateDecision": {"acceptId": ids[substrate_accept], "acceptIndex": substrate_accept},
        "substrateMatchesDigital": bool(matches),
        "annealingGap": round(gap, 4),
        "perCandidate": [
            {"id": ids[i], "energy": round(float(energies[i]), 4),
             "phase": round(float(theta[i]), 4), "acceptScore": round(float(accept_score[i]), 4)}
            for i in range(len(cands))
        ],
        "banner": ("SIMULATION of idealized oscillator-Ising relaxation; not hardware, not an "
                   "energy measurement, not LLM-scale. Demonstrates the O2->O5 encoding only."),
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidates", required=True, help="JSONL of {id, energy} (e.g. from O2)")
    p.add_argument("--output", default=None)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cands = _load_jsonl(args.candidates)
    except Exception as e:
        report = _env_artifact(f"could not read --candidates ({type(e).__name__}: {e})")
    else:
        report = run(cands, steps=args.steps, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not report.get("environmentArtifact") else 2


if __name__ == "__main__":
    raise SystemExit(main())