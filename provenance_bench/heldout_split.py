"""C2 — held-out family disjointness + sealing (pure stdlib). Construct-disjoint,
not just string-disjoint: hold out whole clusters, assert no shared content 3-gram,
and that the fit module never imports the held-out paths."""
from __future__ import annotations

import hashlib, json, re
from pathlib import Path

from agent.personality_measure import load_bank
from agent.personality_behavioral import load_battery

ROOT = Path(__file__).resolve().parents[1]
SEEN_ITEMS = ROOT / "data" / "personality_items.json"
HELDOUT_ITEMS = ROOT / "data" / "personality_items_heldout.json"
SEEN_BATTERY = ROOT / "data" / "behavioral_battery.json"
HELDOUT_BATTERY = ROOT / "data" / "behavioral_battery_heldout.json"
_STOP = {"the", "a", "an", "to", "of", "and", "i", "you", "your", "is", "are", "with", "for", "in", "on", "do"}


def _sealed(strings: "list[str]") -> str:
    return hashlib.sha256(json.dumps(sorted(strings)).encode()).hexdigest()[:16]


def _tokens(text: str) -> "list[str]":
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP]


def _ngrams(texts: "list[str]", n=3) -> set:
    out = set()
    for t in texts:
        toks = _tokens(t)
        out |= {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}
    return out


def held_out_disjoint(*, fit_module="tools/run_steering.py") -> dict:
    seen_items = [it["text"] for it in load_bank(SEEN_ITEMS)["items"]]
    ho_items = [it["text"] for it in load_bank(HELDOUT_ITEMS)["items"]]
    seen_ids = {it["id"] for it in load_bank(SEEN_ITEMS)["items"]}
    ho_ids = {it["id"] for it in load_bank(HELDOUT_ITEMS)["items"]}
    seen_b = [p for v in load_battery(SEEN_BATTERY)["prompts"].values() for p in v]
    ho_b = [p for v in load_battery(HELDOUT_BATTERY)["prompts"].values() for p in v]
    overlaps = sorted(" ".join(g) for g in (_ngrams(seen_items + seen_b) & _ngrams(ho_items + ho_b)))
    # nearest-neighbour token-Jaccard between any seen and any held-out text
    def jac(a, b):
        sa, sb = set(_tokens(a)), set(_tokens(b))
        return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    nn = max((jac(s, h) for s in seen_items + seen_b for h in ho_items + ho_b), default=0.0)
    fit_src = (ROOT / fit_module).read_text(encoding="utf-8")
    fit_reads = ("personality_items_heldout" in fit_src) or ("behavioral_battery_heldout" in fit_src)
    return {"ipip_intersection": sorted(seen_ids & ho_ids), "ngram_overlaps": overlaps,
            "seen_sealed": _sealed(seen_items), "heldout_sealed": _sealed(ho_items),
            "fit_reads_heldout": fit_reads, "nearest_neighbour_sim": round(nn, 4)}
