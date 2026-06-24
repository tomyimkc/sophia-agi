# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""C1 — personality-diverse council A/B (Spec C). Pure stdlib + the council engine.

Seats carry an OCEAN persona via a prefix wrapper passed through deliberate()'s
seat_clients seam (verified: each seat is called via _gen -> client.generate
(system,user) -> result with .ok/.text). The deterministic score_case judge
cannot collude with the seats. A NULL ΔQ is the expected, honest result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case, load_traditions
from agent.council_deliberate import deliberate
from provenance_bench.aggregate import _ci


def ocean_persona_prompt(name: str, traits: dict) -> str:
    def band(x, hi, lo):
        return hi if x >= 0.5 else lo
    o = band(traits.get("O", 0.5), "imaginative, open to unconventional framings",
             "conventional, prefers established framings")
    c = band(traits.get("C", 0.5), "methodical, detail-checking, risk-averse",
             "exploratory, big-picture, tolerant of loose ends")
    e = band(traits.get("E", 0.5), "assertive and decisive in stating a view",
             "reserved, hedged, careful to qualify")
    a = band(traits.get("A", 0.5), "cooperative, seeks common ground",
             "skeptical, willing to dissent and challenge")
    n = band(traits.get("N", 0.5), "highly alert to downside risks and failure modes",
             "calm, unbothered by tail risks")
    return (f"PERSONA ({name}): Adopt this cognitive style throughout. You are {o}; {c}; "
            f"{e}; {a}; {n}. Let this style shape WHICH considerations you surface and how "
            f"you weigh them — but never fabricate facts or citations to fit the persona.")


@dataclass
class PersonaClient:
    base: object
    persona_name: str
    traits: dict
    spec: str = field(default="", init=False)
    model: str = field(default="", init=False)

    def __post_init__(self):
        base_spec = getattr(self.base, "spec", "") or getattr(self.base, "model", "")
        self.spec = f"{base_spec}|persona:{self.persona_name}" if base_spec else f"persona:{self.persona_name}"
        self.model = self.spec

    def generate(self, system: str, user: str):
        merged = f"{ocean_persona_prompt(self.persona_name, self.traits)}\n\n{system}"
        return self.base.generate(merged, user)


def arm_passrate(domain: str, *, client, seat_clients=None, traditions=None) -> dict:
    traditions = traditions if traditions is not None else load_traditions()
    cases = load_json(DOMAIN_BENCH[domain]).get("cases", [])
    passed = 0
    per_case = []
    for case in cases:
        d = deliberate(case["question"], client=client, seat_clients=seat_clients)
        ok, _ = score_case(case, d.synthesis, traditions)
        passed += int(ok)
        per_case.append(int(ok))
    fams = sorted({getattr(c, "spec", "") for c in (seat_clients or [])}) or ["<homogeneous>"]
    return {"passrate": passed / len(cases) if cases else 0.0, "n": len(cases),
            "per_case": per_case, "seat_families": fams}


def council_diversity(domain: str, *, client, profiles: "list[tuple[str, dict]]") -> dict:
    """Three matched arms on the same gold cases:
    single (bare client), homogeneous-persona (N copies of one profile),
    diverse-persona (N distinct profiles). ΔQ = diverse − homogeneous (paired-bootstrap CI)."""
    traditions = load_traditions()
    diverse_clients = [PersonaClient(client, name, t) for name, t in profiles]
    homo_name, homo_t = profiles[0]
    homo_clients = [PersonaClient(client, homo_name, homo_t) for _ in profiles]

    single = arm_passrate(domain, client=client, seat_clients=None, traditions=traditions)
    homogeneous = arm_passrate(domain, client=client, seat_clients=homo_clients, traditions=traditions)
    diverse = arm_passrate(domain, client=client, seat_clients=diverse_clients, traditions=traditions)

    # paired ΔQ across cases: per-case (diverse_ok − homo_ok), bootstrap its mean CI
    diffs = [d - h for d, h in zip(diverse["per_case"], homogeneous["per_case"])]
    import random as _r, statistics as _s
    rng = _r.Random(0)
    boot = [_s.fmean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(2000)] if diffs else [0.0]
    dq_ci = _ci(boot)
    return {"domain": domain, "single": single, "homogeneous": homogeneous, "diverse": diverse,
            "dq": round(diverse["passrate"] - homogeneous["passrate"], 4), "dq_ci": dq_ci,
            "profiles": [n for n, _ in profiles]}
