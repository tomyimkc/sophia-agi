# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle (molecular) biology verification: codon translation, reverse-complement,
GC content, transcription, Hardy–Weinberg and monohybrid Punnett ratios →
accepted / rejected / abstain.

The biology analogue of ``agent.math_verifier``. Every check here is a deterministic,
closed-form function (the standard genetic-code table; complement/transcription string
maps; closed-form genotype arithmetic), so it is **pure stdlib** and always runs in CI —
no GPU, no optional backend. ``biopython`` is only needed for long-sequence alignment
(not used by the base curriculum); the curriculum's oracles are self-contained here.

Like the math verifier, an answer that cannot be parsed/extracted yields ``abstain``
(fail-closed), never a fabricated verdict.
"""
from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from math import isclose
from typing import Any, Literal

Verdict = Literal["accepted", "rejected", "abstain"]

# Standard genetic code (DNA sense-strand codons → 1-letter amino acid; '*' = stop).
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


# --------------------------------------------------------------------------- #
# Pure-Python oracles
# --------------------------------------------------------------------------- #
def _clean_dna(seq: str) -> str | None:
    s = re.sub(r"\s+", "", (seq or "").upper())
    return s if s and all(c in _COMPLEMENT for c in s) else None


def reverse_complement(seq: str) -> str | None:
    s = _clean_dna(seq)
    return "".join(_COMPLEMENT[c] for c in reversed(s)) if s else None


def transcribe(seq: str) -> str | None:
    """DNA sense strand → mRNA (T→U)."""
    s = _clean_dna(seq)
    return s.replace("T", "U") if s else None


def gc_content(seq: str) -> float | None:
    """GC content as a percentage (0–100)."""
    s = _clean_dna(seq)
    if not s:
        return None
    return 100.0 * (s.count("G") + s.count("C")) / len(s)


def translate(seq: str, *, to_stop: bool = False) -> str | None:
    """Translate a DNA sense strand to a 1-letter protein. Trailing partial codon is
    ignored. ``to_stop`` truncates at the first stop codon (exclusive)."""
    s = _clean_dna(seq)
    if not s:
        return None
    out = []
    for i in range(0, len(s) - 2, 3):
        aa = CODON_TABLE[s[i:i + 3]]
        if aa == "*" and to_stop:
            break
        out.append(aa)
    return "".join(out)


def hardy_weinberg(p: float) -> dict[str, float] | None:
    """Genotype frequencies {AA, Aa, aa} for dominant-allele frequency ``p`` (q=1-p)."""
    if not (0.0 <= p <= 1.0):
        return None
    q = 1.0 - p
    return {"AA": p * p, "Aa": 2 * p * q, "aa": q * q}


def punnett_monohybrid(g1: str, g2: str) -> dict[str, Any] | None:
    """Genotype + phenotype ratios for a monohybrid cross of two single-gene genotypes
    (e.g. ``Aa`` × ``Aa``). Dominance = uppercase allele. Returns counts out of 4."""
    def _norm(g: str) -> tuple[str, str] | None:
        g = (g or "").strip()
        if len(g) != 2 or g[0].upper() != g[1].upper() or not g[0].isalpha():
            return None
        return (g[0], g[1])

    a, b = _norm(g1), _norm(g2)
    if a is None or b is None or a[0].upper() != b[0].upper():
        return None
    letter = a[0].upper()
    geno: Counter = Counter()
    pheno: Counter = Counter()
    for x in a:
        for y in b:
            key = "".join(sorted([x, y], key=lambda c: (c.lower(), c.islower())))
            geno[key] += 1
            pheno["dominant" if (x.isupper() or y.isupper()) else "recessive"] += 1
    return {"letter": letter, "genotype": dict(geno), "phenotype": dict(pheno)}


# --------------------------------------------------------------------------- #
# Answer extraction + verification entrypoints
# --------------------------------------------------------------------------- #
def extract_answer(text: str) -> str:
    s = str(text or "")
    idx = s.rfind("Answer:")
    return s[idx + len("Answer:"):].strip() if idx >= 0 else s.strip()


def _seq_token(text: str, alphabet: str, *, min_len: int = 2) -> str | None:
    """Longest run over ``alphabet`` (case-insensitive) in the extracted answer, at
    least ``min_len`` long. The length floor avoids matching a stray letter inside an
    English word (e.g. the 'a' in "no idea") as if it were a sequence."""
    runs = [r for r in re.findall(rf"[{alphabet}]+", extract_answer(text).upper()) if len(r) >= min_len]
    return max(runs, key=len) if runs else None


def _last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", extract_answer(text))
    return float(nums[-1]) if nums else None


def verify_reverse_complement(answer: str, seq: str) -> dict[str, Any]:
    gold = reverse_complement(seq)
    if gold is None:
        return {"verdict": "abstain", "reasons": [f"invalid DNA: {seq!r}"], "detail": {"seq": seq}}
    got = _seq_token(answer, "ACGT")
    if not got:
        return {"verdict": "abstain", "reasons": ["no DNA answer found"], "detail": {"gold": gold}}
    ok = got == gold
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got} != {gold}"], "detail": {"gold": gold, "got": got}}


def verify_transcription(answer: str, seq: str) -> dict[str, Any]:
    gold = transcribe(seq)
    if gold is None:
        return {"verdict": "abstain", "reasons": [f"invalid DNA: {seq!r}"], "detail": {"seq": seq}}
    got = _seq_token(answer, "ACGU")
    if not got:
        return {"verdict": "abstain", "reasons": ["no mRNA answer found"], "detail": {"gold": gold}}
    ok = got == gold
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got} != {gold}"], "detail": {"gold": gold, "got": got}}


def verify_translation(answer: str, seq: str, *, to_stop: bool = False) -> dict[str, Any]:
    gold = translate(seq, to_stop=to_stop)
    if gold is None:
        return {"verdict": "abstain", "reasons": [f"invalid DNA: {seq!r}"], "detail": {"seq": seq}}
    got = _seq_token(answer, "ACDEFGHIKLMNPQRSTVWY*", min_len=1)
    if not got:
        return {"verdict": "abstain", "reasons": ["no protein answer found"], "detail": {"gold": gold}}
    ok = got.rstrip("*") == gold.rstrip("*")
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got} != {gold}"], "detail": {"gold": gold, "got": got}}


def verify_gc_content(answer: str, seq: str, *, abs_tol: float = 0.5) -> dict[str, Any]:
    gold = gc_content(seq)
    if gold is None:
        return {"verdict": "abstain", "reasons": [f"invalid DNA: {seq!r}"], "detail": {"seq": seq}}
    got = _last_number(answer)
    if got is None:
        return {"verdict": "abstain", "reasons": ["no numeric answer found"], "detail": {"gold": gold}}
    ok = isclose(got, gold, abs_tol=abs_tol)
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got}% != {round(gold, 2)}%"],
            "detail": {"gold": gold, "got": got}}


def verify_value(answer: str, gold: float, *, rtol: float = 0.01, abs_tol: float = 1e-4) -> dict[str, Any]:
    """Generic numeric check (genotype frequency, expected count, …)."""
    got = _last_number(answer)
    if got is None:
        return {"verdict": "abstain", "reasons": ["no numeric answer found"], "detail": {"gold": gold}}
    ok = isclose(got, gold, rel_tol=rtol, abs_tol=abs_tol)
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got} != {gold}"], "detail": {"gold": gold, "got": got}}


def verify_ratio(answer: str, gold: tuple[int, ...]) -> dict[str, Any]:
    """Verify an integer ratio (e.g. a 3:1 or 1:2:1 Punnett ratio), order-sensitive,
    accepting any positive scalar multiple."""
    raw = extract_answer(answer)
    got = tuple(int(round(float(x))) for x in re.findall(r"\d+", raw))[:len(gold)]
    if len(got) != len(gold) or any(x <= 0 for x in got):
        return {"verdict": "abstain", "reasons": [f"expected {len(gold)} ratio terms"],
                "detail": {"gold": list(gold), "got": list(got)}}
    ok = len({Fraction(g, x) for g, x in zip(got, gold)}) == 1
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"ratio {got} != {gold}"],
            "detail": {"gold": list(gold), "got": list(got)}}
