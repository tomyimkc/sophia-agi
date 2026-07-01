# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Biology reference verifier — deterministic, dependency-free sequence checks.

Reference-grade, NOT a production bioinformatics oracle: it catches the cheap, machine-checkable
errors a biology council seat must never ship — sequences over an **invalid alphabet** (a "DNA"
string containing letters outside ACGT), a claimed reverse-complement that does not actually
complement, and a "coding" sequence whose length is not a multiple of 3. It is the standalone gate
that lets a biology seat clear the trust boundary. A real Biopython backend can replace it; until
then it is candidate-only and fails *closed*.
"""

from __future__ import annotations

import re

DNA = set("ACGT")
RNA = set("ACGU")
AMINO = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino-acid one-letter codes
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def classify_sequence(seq: str) -> "str | None":
    """Return 'dna' | 'rna' | 'protein' | None for an uppercase sequence token."""
    s = seq.upper()
    if not s or not s.isalpha():
        return None
    chars = set(s)
    if chars <= DNA:
        return "dna"
    if chars <= RNA:
        return "rna"
    if chars <= AMINO:
        return "protein"
    return None


def reverse_complement(dna: str) -> "str | None":
    dna = dna.upper()
    if set(dna) - DNA:
        return None
    return "".join(_COMPLEMENT[b] for b in reversed(dna))


def biology_sound():
    """Verifier-style callable ``v(text, record, ctx) -> {passed, reasons, detail}``.

    Flags: (1) a token explicitly labelled DNA/RNA/protein sequence that is over an invalid
    alphabet; (2) a stated reverse-complement that does not actually complement; (3) a sequence
    labelled 'coding'/'ORF' whose length is not a multiple of 3. No checkable biology -> passes."""

    def _v(text, _record=None, _ctx=None) -> dict:
        text = (text or "")[:8000]  # bound untrusted input
        reasons: list[str] = []
        checked = 0

        # (1) explicitly labelled sequences: the sequence token is the next UPPERCASE run
        # (so filler words like "sequence" are skipped). "DNA sequence ACGTX" -> ACGTX.
        for label, seq in re.findall(r"\b(DNA|RNA|protein)\b[^A-Z]{0,20}([A-Z]{4,})", text):
            checked += 1
            kind = classify_sequence(seq)
            want = label.lower()
            if kind is None or (want in ("dna", "rna") and kind != want):
                reasons.append(f"[biology] invalid {label.upper()} sequence over its alphabet: {seq}")

        # (2) reverse-complement claims: "reverse complement of ACGT is ACGT"
        for a, b in re.findall(r"reverse[- ]complement of ([ACGTacgt]{2,})\s+is\s+([ACGTacgt]{2,})", text, re.I):
            checked += 1
            if reverse_complement(a) != b.upper():
                reasons.append(f"[biology] wrong reverse-complement: rc({a.upper()}) != {b.upper()}")

        # (3) coding/ORF length must be a multiple of 3 (next uppercase ACGTU run after filler)
        for _kw, seq in re.findall(r"\b(coding|ORF|reading frame)\b[^A-Z]{0,20}([ACGTU]{3,})", text):
            checked += 1
            if len(seq) % 3 != 0:
                reasons.append(f"[biology] coding sequence length {len(seq)} not a multiple of 3: {seq.upper()}")

        return {"passed": not reasons, "reasons": reasons, "detail": {"checked": checked}}

    return _v


if __name__ == "__main__":
    v = biology_sound()
    assert v("The DNA sequence ACGTACGT encodes...")["passed"], "valid DNA should pass"
    assert not v("The DNA sequence ACGTX is read...")["passed"], "invalid DNA letter should fail"
    assert not v("The reverse complement of AAGG is AAGG.")["passed"], "wrong rc should fail"
    assert v("The reverse complement of ACGT is ACGT.")["passed"], "ACGT is its own rc -> pass"
    assert reverse_complement("ACGT") == "ACGT" and reverse_complement("AAGG") == "CCTT"
    assert not v("The coding sequence ACGTA is translated.")["passed"], "len%3!=0 should fail"
    assert v("Mitochondria are the powerhouse of the cell.")["passed"], "no biology -> pass"
    print("biology_verifier self-check: PASS")
