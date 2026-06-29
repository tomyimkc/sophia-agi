#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate oracle-verified chemistry + biology curriculum at graded difficulty tiers.

Training oracle only (NOT the wisdom gate). Every row's gold answer is produced by a
deterministic oracle (``agent.chem_verifier`` / ``agent.bio_verifier``) and then
re-verified through the same oracle before it is kept — the chem/bio analogue of the
sympy/exec gating in ``tools/generate_math_code_curriculum.py``.

Outputs:
  * ``training/sophia-chem-bio-curriculum/`` — verified SFT rows (tier0-2), trained.
  * ``eval/chem_bio_capability/heldout_v1.jsonl`` — tier3 held-out eval items, NEVER
    trained. Because it lives under ``eval/**``, the dataset contamination guard
    (``provenance_bench.dataset_guard``) automatically treats it as a held-out surface,
    so any train/eval prompt overlap fails closed.

    python tools/generate_chem_bio_curriculum.py
    python tools/generate_chem_bio_curriculum.py --check   # validate, no writes

``trainingOracleOnly: true`` — passing these oracles is NOT benchmark proof.
``canClaimAGI`` stays False.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import bio_verifier as bv  # noqa: E402
from agent import chem_verifier as cv  # noqa: E402
from provenance_bench.dataset_guard import check_contamination, eval_prompt_set, normalize  # noqa: E402

OUT_DIR = ROOT / "training" / "sophia-chem-bio-curriculum"
EVAL_DIR = ROOT / "eval" / "chem_bio_capability"
EVAL_HELDOUT = EVAL_DIR / "heldout_v1.jsonl"
CURRICULUM_SEED = 20260629
SOURCE = "sophia-chem-bio-curriculum"
TAIL = " End your reply with a line 'Answer: <value>'."

_ORACLE = {
    "molar_mass": "chem_verifier", "atom_count": "chem_verifier", "balance": "chem_verifier",
    "chem_value": "chem_verifier", "gc": "bio_verifier", "revcomp": "bio_verifier",
    "transcribe": "bio_verifier", "translate": "bio_verifier", "bio_value": "bio_verifier",
    "ratio": "bio_verifier", "abstention": "abstention-design",
}

# Common compounds: (name, formula)
_COMPOUNDS = [
    ("water", "H2O"), ("carbon dioxide", "CO2"), ("table salt", "NaCl"),
    ("calcium carbonate", "CaCO3"), ("glucose", "C6H12O6"), ("ammonia", "NH3"),
    ("sulfuric acid", "H2SO4"), ("methane", "CH4"), ("ethanol", "C2H6O"),
    ("magnesium oxide", "MgO"), ("sodium hydroxide", "NaOH"), ("nitric acid", "HNO3"),
    ("calcium hydroxide", "Ca(OH)2"), ("aluminium oxide", "Al2O3"),
]
_REACTIONS = [
    (["H2", "O2"], ["H2O"]), (["CH4", "O2"], ["CO2", "H2O"]),
    (["N2", "H2"], ["NH3"]), (["C3H8", "O2"], ["CO2", "H2O"]),
    (["Fe", "O2"], ["Fe2O3"]), (["Al", "O2"], ["Al2O3"]),
    (["Na", "Cl2"], ["NaCl"]), (["KClO3"], ["KCl", "O2"]),
    (["C2H6", "O2"], ["CO2", "H2O"]), (["H2", "Cl2"], ["HCl"]),
]


# --------------------------------------------------------------------------- #
# Abstention oracle (deterministic): a calibrated refusal carries hedge/refusal
# vocabulary and asserts no fabricated specific value.
# --------------------------------------------------------------------------- #
_ABSTAIN_MARKERS = re.compile(
    r"can('|no)?t verify|cannot verify|don'?t know|do not know|insufficient (information|data)|"
    r"no reliable source|unable to|won'?t guess|will not guess|not (something i can|able to) (confirm|verify)",
    re.I,
)


def verify_abstention(response: str) -> dict[str, Any]:
    ans = cv.extract_answer(response)
    has_marker = bool(_ABSTAIN_MARKERS.search(ans))
    has_number = bool(re.search(r"\d", ans))
    if has_marker and not has_number:
        return {"verdict": "accepted", "reasons": [], "detail": {"abstention": True}}
    return {"verdict": "rejected",
            "reasons": ["not a calibrated abstention (missing hedge marker or fabricated value)"],
            "detail": {"hasMarker": has_marker, "hasNumber": has_number}}


# --------------------------------------------------------------------------- #
# Row + response helpers
# --------------------------------------------------------------------------- #
def _response(reasoning: str, gold_str: str) -> str:
    return f"Let me work through this step by step.\n\n{reasoning}\n\nAnswer: {gold_str}"


def verify_problem(prob: dict) -> dict[str, Any]:
    resp = _response(prob["reasoning"], prob["gold_str"])
    k, pr = prob["kind"], prob["params"]
    if k == "molar_mass":
        return cv.verify_molar_mass(resp, pr["formula"])
    if k == "atom_count":
        return cv.verify_atom_count(resp, pr["formula"], pr["element"])
    if k == "balance":
        return cv.verify_balanced_coeffs(resp, pr["reactants"], pr["products"])
    if k == "chem_value":
        return cv.verify_value(resp, pr["gold"], rtol=pr.get("rtol", 0.02))
    if k == "gc":
        return bv.verify_gc_content(resp, pr["seq"])
    if k == "revcomp":
        return bv.verify_reverse_complement(resp, pr["seq"])
    if k == "transcribe":
        return bv.verify_transcription(resp, pr["seq"])
    if k == "translate":
        return bv.verify_translation(resp, pr["seq"], to_stop=pr.get("to_stop", False))
    if k == "bio_value":
        return bv.verify_value(resp, pr["gold"], rtol=pr.get("rtol", 0.02))
    if k == "ratio":
        return bv.verify_ratio(resp, tuple(pr["gold"]))
    if k == "abstention":
        return verify_abstention(resp)
    return {"verdict": "abstain", "reasons": [f"unknown kind {k}"], "detail": {}}


def _build_row(prob: dict, tier: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prob["prompt"]},
            {"role": "assistant", "content": _response(prob["reasoning"], prob["gold_str"])},
        ],
        "metadata": {
            "source": SOURCE, "project": "sophia-agi", "domain": prob["domain"],
            "tier": tier, "family": prob["family"], "id": prob["id"],
            "gold": prob["gold_str"], "verifierOracle": _ORACLE[prob["kind"]],
            "verifierVerdict": "accepted", "trainingOracleOnly": True, "candidateOnly": True,
        },
    }


def _rand_dna(rng: random.Random, n: int, *, start_atg: bool = False) -> str:
    s = "".join(rng.choice("ACGT") for _ in range(n))
    return ("ATG" + s[3:]) if start_atg and n >= 3 else s


# --------------------------------------------------------------------------- #
# Curriculum families (gold computed by the oracle, then re-verified)
# --------------------------------------------------------------------------- #
def gen_chem(rng: random.Random) -> dict[str, list[dict]]:
    t0, t1, t2 = [], [], []
    # tier0 — molar mass
    for i, (name, f) in enumerate(_COMPOUNDS[:10]):
        mm = cv.molar_mass(f)
        t0.append({"domain": "chemistry", "family": "molar_mass", "id": f"cb-chem-mm-{i}",
                   "prompt": f"What is the molar mass of {name} ({f}) in g/mol (one decimal)?{TAIL}",
                   "reasoning": f"Sum the standard atomic weights across {f}.",
                   "gold_str": f"{mm:.1f} g/mol", "kind": "molar_mass", "params": {"formula": f}})
    # tier0 — atom count
    ac = [("Ca(OH)2", "O"), ("Al2(SO4)3", "O"), ("C6H12O6", "C"), ("H2SO4", "H"),
          ("CaCO3", "O"), ("NH3", "H"), ("Al2O3", "Al"), ("C2H6O", "C")]
    for i, (f, el) in enumerate(ac):
        c = cv.parse_formula(f)[el]
        t0.append({"domain": "chemistry", "family": "atom_count", "id": f"cb-chem-ac-{i}",
                   "prompt": f"How many {el} atoms are in one formula unit of {f}?{TAIL}",
                   "reasoning": f"Expand {f}, applying subscripts through parentheses.",
                   "gold_str": str(c), "kind": "atom_count", "params": {"formula": f, "element": el}})
    # tier1 — balance
    for i, (r, p) in enumerate(_REACTIONS):
        coeffs = cv.balance_equation(r, p)
        eq = " + ".join(r) + " -> " + " + ".join(p)
        t1.append({"domain": "chemistry", "family": "balance", "id": f"cb-chem-bal-{i}",
                   "prompt": f"Balance: {eq}. Give the coefficients (reactants then products) "
                             f"as a comma-separated list.{TAIL}",
                   "reasoning": "Conserve each element; take the smallest positive integer set.",
                   "gold_str": ", ".join(str(c) for c in coeffs),
                   "kind": "balance", "params": {"reactants": r, "products": p}})
    # tier1 — moles from grams / grams from moles
    mg = [("water", "H2O", 36.0), ("carbon dioxide", "CO2", 88.0), ("glucose", "C6H12O6", 90.0),
          ("ammonia", "NH3", 34.0), ("table salt", "NaCl", 58.5), ("methane", "CH4", 32.0),
          ("sodium hydroxide", "NaOH", 80.0), ("calcium carbonate", "CaCO3", 50.0)]
    for i, (name, f, mass) in enumerate(mg):
        n = mass / cv.molar_mass(f)
        t1.append({"domain": "chemistry", "family": "moles_from_grams", "id": f"cb-chem-mol-{i}",
                   "prompt": f"How many moles are in {mass} g of {name} ({f})? Give mol to 3 decimals.{TAIL}",
                   "reasoning": f"n = m / M, with M the molar mass of {f}.",
                   "gold_str": f"{n:.3f} mol", "kind": "chem_value", "params": {"gold": n, "rtol": 0.01}})
    nm = [("water", "H2O", 2.0), ("carbon dioxide", "CO2", 0.5), ("ammonia", "NH3", 3.0),
          ("methane", "CH4", 1.5), ("glucose", "C6H12O6", 0.25), ("table salt", "NaCl", 4.0)]
    for i, (name, f, mol) in enumerate(nm):
        m = mol * cv.molar_mass(f)
        t1.append({"domain": "chemistry", "family": "grams_from_moles", "id": f"cb-chem-grm-{i}",
                   "prompt": f"What is the mass of {mol} mol of {name} ({f})? Give grams to 2 decimals.{TAIL}",
                   "reasoning": "m = n * M.", "gold_str": f"{m:.2f} g",
                   "kind": "chem_value", "params": {"gold": m, "rtol": 0.01}})
    # tier2 — percent by mass of an element
    pm = [("H2O", "O"), ("CO2", "C"), ("CaCO3", "Ca"), ("C6H12O6", "C"), ("NH3", "N"),
          ("H2SO4", "S"), ("NaCl", "Na"), ("CH4", "H")]
    for i, (f, el) in enumerate(pm):
        c = cv.parse_formula(f)[el]
        pct = 100.0 * c * cv.ATOMIC_WEIGHTS[el] / cv.molar_mass(f)
        t2.append({"domain": "chemistry", "family": "percent_by_mass", "id": f"cb-chem-pct-{i}",
                   "prompt": f"What is the percent by mass of {el} in {f}? Give a percentage to 2 decimals.{TAIL}",
                   "reasoning": f"%(E) = n(E)*A(E) / M({f}) * 100.",
                   "gold_str": f"{pct:.2f}%", "kind": "chem_value", "params": {"gold": pct, "rtol": 0.005}})
    # tier2 — mass-to-mass stoichiometry (combustion of methane)
    m2m = [("CH4", "CO2", ["CH4", "O2"], ["CO2", "H2O"], 16.0),
           ("CH4", "CO2", ["CH4", "O2"], ["CO2", "H2O"], 32.0),
           ("H2", "H2O", ["H2", "O2"], ["H2O"], 4.0),
           ("H2", "H2O", ["H2", "O2"], ["H2O"], 10.0),
           ("N2", "NH3", ["N2", "H2"], ["NH3"], 28.0),
           ("C3H8", "CO2", ["C3H8", "O2"], ["CO2", "H2O"], 44.0)]
    for i, (sf, pf, r, p, mass) in enumerate(m2m):
        coeffs = cv.balance_equation(r, p)
        ci = coeffs[r.index(sf)]
        cj = coeffs[len(r) + p.index(pf)]
        prod_mass = (mass / cv.molar_mass(sf)) * (cj / ci) * cv.molar_mass(pf)
        eq = " + ".join(r) + " -> " + " + ".join(p)
        t2.append({"domain": "chemistry", "family": "mass_to_mass", "id": f"cb-chem-m2m-{i}",
                   "prompt": f"For {eq}, how many grams of {pf} form from {mass} g of {sf} "
                             f"(excess of the other reactant)? Give grams to 2 decimals.{TAIL}",
                   "reasoning": "moles of reactant -> mole ratio from the balanced equation -> mass of product.",
                   "gold_str": f"{prod_mass:.2f} g", "kind": "chem_value",
                   "params": {"gold": prod_mass, "rtol": 0.01}})
    return {"tier0": t0, "tier1": t1, "tier2": t2}


def gen_bio(rng: random.Random) -> dict[str, list[dict]]:
    t0, t1, t2 = [], [], []
    # tier0 — GC content / reverse complement / transcription
    for i in range(8):
        seq = _rand_dna(rng, rng.choice([8, 10, 12]))
        gc = bv.gc_content(seq)
        t0.append({"domain": "biology", "family": "gc_content", "id": f"cb-bio-gc-{i}",
                   "prompt": f"What is the GC content of the DNA sequence 5'-{seq}-3'? "
                             f"Give a percentage to one decimal.{TAIL}",
                   "reasoning": "GC% = (G+C)/length * 100.", "gold_str": f"{gc:.1f}%",
                   "kind": "gc", "params": {"seq": seq}})
    for i in range(8):
        seq = _rand_dna(rng, rng.choice([6, 8, 10]))
        t0.append({"domain": "biology", "family": "reverse_complement", "id": f"cb-bio-rc-{i}",
                   "prompt": f"Give the reverse complement of the DNA strand 5'-{seq}-3'.{TAIL}",
                   "reasoning": "Complement each base (A-T, G-C), then reverse.",
                   "gold_str": bv.reverse_complement(seq), "kind": "revcomp", "params": {"seq": seq}})
    for i in range(6):
        seq = _rand_dna(rng, rng.choice([6, 9]))
        t0.append({"domain": "biology", "family": "transcription", "id": f"cb-bio-tx-{i}",
                   "prompt": f"Transcribe the DNA sense strand 5'-{seq}-3' into mRNA.{TAIL}",
                   "reasoning": "Transcription copies the sense strand, replacing T with U.",
                   "gold_str": bv.transcribe(seq), "kind": "transcribe", "params": {"seq": seq}})
    # tier1 — translation
    for i in range(10):
        seq = "ATG" + "".join(rng.choice("ACGT") for _ in range(rng.choice([6, 9, 12])))
        prot = bv.translate(seq)
        t1.append({"domain": "biology", "family": "translation", "id": f"cb-bio-tr-{i}",
                   "prompt": f"Translate the DNA coding sequence 5'-{seq}-3' into a one-letter "
                             f"amino-acid string (use '*' for a stop codon).{TAIL}",
                   "reasoning": "Read codons 5'->3' from the start, mapping each via the standard genetic code.",
                   "gold_str": prot, "kind": "translate", "params": {"seq": seq}})
    # tier1 — Hardy-Weinberg
    for i, p in enumerate([0.6, 0.7, 0.8, 0.5, 0.4, 0.3, 0.9, 0.2]):
        hw = bv.hardy_weinberg(p)
        t1.append({"domain": "biology", "family": "hardy_weinberg", "id": f"cb-bio-hw-{i}",
                   "prompt": f"In Hardy-Weinberg equilibrium with dominant-allele frequency p={p}, "
                             f"what fraction of the population is heterozygous (Aa)? Give 3 decimals.{TAIL}",
                   "reasoning": "Heterozygote frequency = 2pq with q = 1 - p.",
                   "gold_str": f"{hw['Aa']:.3f}", "kind": "bio_value",
                   "params": {"gold": hw["Aa"], "rtol": 0.01}})
    # tier2 — Punnett phenotype/genotype ratios
    crosses = [("Aa", "Aa", (3, 1), "phenotype dominant:recessive"),
               ("Aa", "aa", (1, 1), "phenotype dominant:recessive"),
               ("Bb", "Bb", (1, 2, 1), "genotype BB:Bb:bb"),
               ("Aa", "Aa", (1, 2, 1), "genotype AA:Aa:aa")]
    for i, (g1, g2, ratio, label) in enumerate(crosses):
        t2.append({"domain": "biology", "family": "punnett_ratio", "id": f"cb-bio-pn-{i}",
                   "prompt": f"For the monohybrid cross {g1} x {g2}, give the {label} ratio "
                             f"as colon-separated integers.{TAIL}",
                   "reasoning": "Enumerate the 4 offspring boxes and reduce to lowest integer ratio.",
                   "gold_str": ":".join(str(x) for x in ratio), "kind": "ratio", "params": {"gold": list(ratio)}})
    # tier2 — expected offspring counts
    exp = [("Aa", "Aa", "recessive", 0.25, 200), ("Aa", "aa", "recessive", 0.5, 80),
           ("Aa", "Aa", "dominant", 0.75, 160), ("Bb", "Bb", "recessive", 0.25, 400)]
    for i, (g1, g2, pheno, frac, n) in enumerate(exp):
        gold = frac * n
        t2.append({"domain": "biology", "family": "expected_offspring", "id": f"cb-bio-exp-{i}",
                   "prompt": f"From the cross {g1} x {g2} with {n} offspring, how many are expected to show "
                             f"the {pheno} phenotype? Give a whole number.{TAIL}",
                   "reasoning": f"Expected count = P({pheno}) * N.",
                   "gold_str": str(int(gold)), "kind": "bio_value", "params": {"gold": gold, "rtol": 0.001}})
    return {"tier0": t0, "tier1": t1, "tier2": t2}


def gen_abstention(rng: random.Random) -> list[dict]:
    items = [
        ("chemistry", "What is the exact boiling point of the compound 'fictionium fluoride'?"),
        ("chemistry", "Give the precise molar mass of the undisclosed proprietary polymer 'PX-9'."),
        ("biology", "What is the exact 3D folded structure of an arbitrary protein from its sequence alone?"),
        ("biology", "State the precise lifespan in days of the fictional organism 'Glubworm'."),
        ("chemistry", "What is the standard reduction potential of the made-up element 'Zorganium'?"),
        ("biology", "Give the exact number of genes in the genome of an unnamed, unsequenced species."),
        ("chemistry", "What is the heat of formation of a compound whose structure I have not given you?"),
        ("biology", "What allele frequency does an unspecified population have for an unnamed gene?"),
        ("chemistry", "What is the pKa of 'mysteryacid' with no structure provided?"),
        ("biology", "Translate a DNA sequence I have not provided into a protein."),
        ("chemistry", "How many isomers does an unspecified molecular formula have?"),
        ("biology", "What is the precise melting temperature of an unstated primer sequence?"),
    ]
    out = []
    for i, (dom, q) in enumerate(items):
        out.append({"domain": dom, "family": "abstention", "id": f"cb-abst-{i}",
                    "prompt": q + TAIL,
                    "reasoning": "The prompt lacks the structure/data any reliable method would require.",
                    "gold_str": "I can't verify this from the information given, and I won't guess.",
                    "kind": "abstention", "params": {}})
    return out


def generate_problems() -> dict[str, dict[str, list[dict]]]:
    rng = random.Random(CURRICULUM_SEED)
    chem, bio = gen_chem(rng), gen_bio(rng)
    abst = gen_abstention(rng)
    tiers: dict[str, dict[str, list[dict]]] = {}
    for t in ("tier0", "tier1", "tier2"):
        tiers[t] = {"chemistry": chem[t], "biology": bio[t]}
    tiers["tier1"]["abstention"] = abst  # abstention rows live in tier1
    return tiers


# --------------------------------------------------------------------------- #
# Held-out tier3 eval (NEVER trained) — distinct families + values
# --------------------------------------------------------------------------- #
def generate_heldout() -> list[dict]:
    out: list[dict] = []
    # chem: percent yield, molarity, multi-step mass-to-mass (distinct from train)
    yld = [(50.0, 64.0), (12.0, 20.0), (8.5, 10.0)]
    for i, (actual, theo) in enumerate(yld):
        out.append({"id": f"cb-ho-yield-{i}", "domain": "chemistry", "kind": "chem_value",
                    "question": f"A reaction produced {actual} g of product against a theoretical {theo} g. "
                                f"What is the percent yield (2 decimals)?",
                    "goldAnswer": f"{100.0*actual/theo:.2f}%"})
    mol = [(0.5, 2.0), (1.5, 0.5), (0.2, 0.25)]
    for i, (n, v) in enumerate(mol):
        out.append({"id": f"cb-ho-molarity-{i}", "domain": "chemistry", "kind": "chem_value",
                    "question": f"What is the molarity of a solution with {n} mol solute in {v} L (3 decimals)?",
                    "goldAnswer": f"{n/v:.3f} M"})
    dilute = [("NaOH", 40.0, 0.25), ("KCl", 74.55, 0.10)]
    for i, (f, _m, mol_amt) in enumerate(dilute):
        out.append({"id": f"cb-ho-mass-{i}", "domain": "chemistry", "kind": "chem_value",
                    "question": f"What mass (g, 2 decimals) of {f} is needed for {mol_amt} mol?",
                    "goldAnswer": f"{mol_amt*cv.molar_mass(f):.2f} g"})
    # bio: ORF translation to stop, dihybrid recessive fraction, carrier frequency
    orfs = ["ATGGCATTTGGATAA", "ATGAAACCCGGGTGA", "ATGTGGTGCTGA"]
    for i, seq in enumerate(orfs):
        out.append({"id": f"cb-ho-orf-{i}", "domain": "biology", "kind": "translate",
                    "question": f"Translate 5'-{seq}-3' to protein, stopping at the first stop codon.",
                    "goldAnswer": bv.translate(seq, to_stop=True)})
    for i, p in enumerate([0.5, 0.6, 0.7]):
        hw = bv.hardy_weinberg(p)
        out.append({"id": f"cb-ho-carrier-{i}", "domain": "biology", "kind": "bio_value",
                    "question": f"In HW equilibrium with p={p}, what fraction is homozygous recessive (3 decimals)?",
                    "goldAnswer": f"{hw['aa']:.3f}"})
    for i, n in enumerate([320, 64]):
        out.append({"id": f"cb-ho-dihybrid-{i}", "domain": "biology", "kind": "bio_value",
                    "question": f"A dihybrid cross AaBb x AaBb yields {n} offspring. How many show both "
                                f"recessive traits (9:3:3:1; whole number)?",
                    "goldAnswer": str(int(n / 16))})
    # Eval prompts MUST carry the same TAIL ("Answer: <value>") instruction the training
    # prompts use, so the held-out reply format matches what the model was taught and the
    # grader can isolate the final answer. Re-baseline v1: this was previously omitted.
    for it in out:
        it["question"] = it["question"] + TAIL
        it.update({"candidateOnly": True, "trainingOracleOnly": True,
                   "labelSource": "oracle-generated committed fixture (NOT externally authored)",
                   "externalSource": False})
    return out


# --------------------------------------------------------------------------- #
# Verify + assemble
# --------------------------------------------------------------------------- #
def verify_and_build_rows(tiers: dict[str, dict[str, list[dict]]]) -> tuple[list[dict], dict[str, Any]]:
    stats: dict[str, Any] = {"tiers": {}, "totals": {"generated": 0, "kept": 0, "dropped": 0}}
    rows: list[dict] = []
    evalset = eval_prompt_set(root=ROOT)
    for tier, buckets in tiers.items():
        tstat: dict[str, Any] = {"kept": 0, "dropped": 0, "generated": 0, "byDomain": {}}
        for domain, probs in buckets.items():
            gen = kept = dropped = 0
            verdicts: dict[str, int] = {}
            for prob in probs:
                gen += 1
                stats["totals"]["generated"] += 1
                if normalize(prob["prompt"]) in evalset:
                    dropped += 1
                    verdicts["decontam"] = verdicts.get("decontam", 0) + 1
                    continue
                v = verify_problem(prob)["verdict"]
                verdicts[v] = verdicts.get(v, 0) + 1
                if v != "accepted":
                    dropped += 1
                    continue
                rows.append(_build_row(prob, tier))
                kept += 1
            tstat["byDomain"][domain] = {"generated": gen, "kept": kept, "dropped": dropped,
                                         "verdicts": verdicts}
            tstat["generated"] += gen
            tstat["kept"] += kept
            tstat["dropped"] += dropped
        stats["tiers"][tier] = tstat
        stats["totals"]["kept"] += tstat["kept"]
        stats["totals"]["dropped"] += tstat["dropped"]
    return rows, stats


def build_manifest(rows: list[dict], stats: dict[str, Any], contam: dict, heldout: list[dict]) -> dict:
    chem = [r for r in rows if r["metadata"]["domain"] == "chemistry"]
    bio = [r for r in rows if r["metadata"]["domain"] == "biology"]
    abst = [r for r in rows if r["metadata"]["domain"] not in ("chemistry", "biology")
            or r["metadata"]["family"] == "abstention"]
    return {
        "schema": "sophia.chem_bio_curriculum.v1",
        "experimentId": SOURCE,
        "baseModel": "Qwen/Qwen2.5-7B-Instruct",
        "seed": CURRICULUM_SEED,
        "trainingOracleOnly": True,
        "canClaimAGI": False,
        "counts": {
            "total": len(rows),
            "chemistry": len(chem),
            "biology": len(bio),
            "abstention": len(abst),
            "byTier": {t: s["kept"] for t, s in stats["tiers"].items()},
            "heldoutEval": len(heldout),
        },
        "verification": stats,
        "contamination": contam,
        "oracles": {
            "chemistry": "agent.chem_verifier (formula parse, molar mass, exact rational equation balancing)",
            "biology": "agent.bio_verifier (genetic-code translation, reverse-complement, GC%, Hardy-Weinberg, Punnett)",
            "abstention": "deterministic calibrated-refusal check (hedge marker present, no fabricated value)",
            "citableAsBenchmarkEvidence": False,
        },
        "tierLadder": {
            "tier0": "facts & parsing — molar mass, atom counts, GC%, reverse-complement, transcription",
            "tier1": "single-step quantitative + abstention — balancing, mole<->gram, translation, Hardy-Weinberg",
            "tier2": "multi-step — percent-by-mass, mass-to-mass stoichiometry, Punnett ratios, expected counts",
            "heldout": "tier3 (eval/chem_bio_capability/heldout_v1.jsonl): percent yield, molarity, ORF-to-stop, "
                       "dihybrid, carrier frequency — NEVER trained",
        },
        "outputs": {
            "sft_chem.jsonl": len(chem),
            "sft_bio.jsonl": len(bio),
            "sft_all.jsonl": len(rows),
            "eval/chem_bio_capability/heldout_v1.jsonl": len(heldout),
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(*, check_only: bool = False) -> tuple[int, dict[str, Any]]:
    tiers = generate_problems()
    rows, stats = verify_and_build_rows(tiers)
    heldout = generate_heldout()

    # In-memory disjointness (belt-and-suspenders, independent of files on disk).
    train_prompts = {normalize(r["messages"][0]["content"]) for r in rows}
    ho_prompts = {normalize(h["question"]) for h in heldout}
    self_overlap = sorted(train_prompts & ho_prompts)

    if not check_only:
        _write_jsonl(EVAL_HELDOUT, heldout)  # write held-out first so the guard sees it

    contam = check_contamination(rows, root=ROOT)
    result = {"stats": stats, "contamination": contam, "rowCount": len(rows),
              "heldoutCount": len(heldout), "selfOverlap": self_overlap}

    if self_overlap or not contam["clean"]:
        print(f"CONTAMINATION: selfOverlap={len(self_overlap)} guard={contam['overlapCount']}",
              file=sys.stderr)
        return 1, result

    if check_only:
        print(json.dumps(result, indent=2))
        return (0 if stats["totals"]["kept"] > 0 else 1), result

    chem = [r for r in rows if r["metadata"]["domain"] == "chemistry"]
    bio = [r for r in rows if r["metadata"]["domain"] == "biology"]
    _write_jsonl(OUT_DIR / "sft_chem.jsonl", chem)
    _write_jsonl(OUT_DIR / "sft_bio.jsonl", bio)
    _write_jsonl(OUT_DIR / "sft_all.jsonl", rows)
    manifest = build_manifest(rows, stats, contam, heldout)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "contamination": contam}, indent=2))
    print(f"wrote {OUT_DIR} ({len(rows)} verified rows) + {EVAL_HELDOUT} ({len(heldout)} held-out)")
    return 0, result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    code, _ = run(check_only=ap.parse_args(argv).check)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
