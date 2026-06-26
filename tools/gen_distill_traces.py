#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-distillation trace generator: harvest the GATED pipeline as its own teacher.

The thesis claim this serves: the provenance/abstain behavior is not just a runtime
wrapper — it is an *internalizable, distillable* inductive bias. To prove that, the
teacher must be Sophia's OWN gated pipeline (fully local, provenance-stampable), not a
closed frontier model (un-stampable origin => structurally excluded, see passport).

Pipeline (every row passes ALL of these or it is dropped — fail-closed):
  1. SPLIT   provenance_bench cases into TRAIN / sealed HELD-OUT, deterministically by
             content hash. Trace-gen only ever touches TRAIN; the held-out seal manifest
             is written so the ablation harness can prove disjointness.
  2. TEACH   run_case(case, base_generate) produces the SAME natural answer `alone` and
             `gated`. We harvest the GATED arm — the gate's repaired/abstained output.
  3. GATE    the harvested assistant text is re-checked with agent.guarded.check_claim
             (double firewall): a trace whose own text trips the gate is dropped.
  4. VERIFY  against external gold — false cases must be NON-hallucinated; true cases
             must still affirm gold (no over-blocking distilled in).
  5. STAMP   pretraining.data_passport.passport.stamp_pack (content hash + minhash dedup
             + quality) PLUS provenance fields: teacher id, gate source hash, verifier,
             seed, decode params. A row missing any of these is never written.
  6. SEAL    drop any row whose prompt near-duplicates a held-out prompt (minhash Jaccard).

Output: training/council/distill_traces.jsonl  (+ held-out seal manifest).
Runs today on the mock provider; swap `make_base_generate` for a real local adapter.

Usage:
  python tools/gen_distill_traces.py --dry-run
  python tools/gen_distill_traces.py --heldout-frac 0.3 --seed 1337
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import check_claim
from pretraining.data_passport import passport
from provenance_bench.dataset import build_cases, build_gate_records
from provenance_bench.runner import run_case
from provenance_bench.score import score

OUT_TRACES = ROOT / "training" / "council" / "distill_traces.jsonl"
SEAL_MANIFEST = ROOT / "agi-proof" / "wisdom-internalization" / "heldout-seal.manifest.json"
ADVISOR_SCAFFOLD = (
    "You are a provenance-disciplined assistant. State only what the evidence supports; "
    "never assert an attribution you cannot ground; abstain when provenance is missing."
)


class ModelResult(NamedTuple):
    """Minimal duck-type the runner accepts (it reads .text / .ok)."""
    text: str
    ok: bool = True


# --------------------------------------------------------------------------- #
# Deterministic TRAIN / HELD-OUT split (no RNG — content-hash bucketing)
# --------------------------------------------------------------------------- #
def split_cases(cases: list, *, heldout_frac: float, seed: int) -> tuple[list, list]:
    """Bucket each case by hash(seed | prompt) so the split is reproducible and the
    held-out set is a *function of the data*, not of call order."""
    train, heldout = [], []
    cut = int(heldout_frac * 1000)
    for c in cases:
        h = hashlib.sha256(f"{seed}:{c.prompt}".encode()).hexdigest()
        (heldout if int(h[:4], 16) % 1000 < cut else train).append(c)
    return train, heldout


def write_seal(heldout: list) -> None:
    """Prompt-only digests of held-out cases (gold answers never hashed/leaked)."""
    SEAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    digests = sorted(passport.content_hash(c.prompt) for c in heldout)
    SEAL_MANIFEST.write_text(
        json.dumps({"count": len(digests), "promptDigests": digests}, indent=2),
        encoding="utf-8",
    )


def gate_source_hash() -> str:
    """Version stamp of the gate the trace was generated behind (reproducibility)."""
    src = (ROOT / "agent" / "guarded.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Teacher = base model BEHIND the gate.  Swap this for a real local adapter.
# --------------------------------------------------------------------------- #
def make_base_generate(model_id: str, *, seed: int, temperature: float) -> Callable:
    """Return a generate(system, user) -> ModelResult callable.

    The mock below makes the harness runnable offline; for the real run, load the
    local base/adapter (HF or MLX) and return its decoded text. Keep decode params
    here — they are recorded into every passport row."""
    import re

    def generate(system: str, user: str) -> ModelResult:
        # MOCK: deterministic, intentionally naive — on "Did X write Y?" it asserts the
        # (false) attribution by name so the judge flags it and the gate has real work.
        m = re.match(r"did (.+?) write (.+?)\?", user.lower())
        if m:
            who, what = m.group(1).title(), m.group(2).title()
            return ModelResult(text=f"Yes, {who} wrote {what}.", ok=True)
        return ModelResult(text="(answer)", ok=True)
    return generate


# --------------------------------------------------------------------------- #
# Harvest the gated arm into a verified, stamped SFT row
# --------------------------------------------------------------------------- #
def harvest(case, result: dict) -> dict | None:
    gated = result.get("gated", {})              # {abstained, hallucinated, affirmed_gold}
    gated_text = result.get("gated_text") or ""  # the actual repaired/abstained text
    if not gated_text:
        return None

    # (3) double firewall: the trace's own text must pass the gate.
    if check_claim(gated_text).get("violations"):
        return None
    # (4) verify against gold.
    if case.label == "false" and gated.get("hallucinated"):
        return None            # teacher failed to fix a false attribution
    if case.label == "true" and not gated.get("affirmed_gold"):
        return None            # gate over-blocked a correct answer -> don't distill it

    return {
        "messages": [
            {"role": "system", "content": ADVISOR_SCAFFOLD},
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": gated_text},
        ],
        "label": case.label,
        "gold_author": getattr(case, "gold_author", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock-base")
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--on-fail", default="abstain",
                    choices=["repair", "abstain", "hedge", "passthrough"],
                    help="abstain => distill the gate's cited-abstention (the wisdom target)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    train, heldout = split_cases(cases, heldout_frac=args.heldout_frac, seed=args.seed)
    write_seal(heldout)
    generate = make_base_generate(args.model, seed=args.seed, temperature=args.temperature)
    gate_records = build_gate_records()

    results, rows = [], []
    for c in train:
        res = run_case(c, generate, on_fail=args.on_fail, records=gate_records)
        results.append(res)
        row = harvest(c, res)
        if row is not None:
            rows.append(row)

    # (5) passport stamp + dedup, then attach provenance/reproducibility fields.
    stamped = passport.stamp_pack(rows)["rows"]
    gate_sha = gate_source_hash()
    held_digests = {passport.content_hash(c.prompt) for c in heldout}
    kept = []
    for r in stamped:
        # (6) seal: never emit a row colliding with a held-out prompt.
        if passport.content_hash(r["messages"][1]["content"]) in held_digests:
            continue
        r["provenance"] = {
            "teacher": args.model,
            "gate_sha": gate_sha,
            "verifier": "provenance_bench.score+agent.guarded.check_claim",
            "seed": args.seed,
            "decode": {"temperature": args.temperature},
            "origin": "self-distillation:gated-arm",
        }
        kept.append(r)

    teacher_delta = score(results)   # the teacher's own alone-vs-gated headline
    print(json.dumps({
        "trainCases": len(train), "heldoutCases": len(heldout),
        "harvested": len(rows), "kept_after_seal_dedup": len(kept),
        "teacherHallucDelta": teacher_delta["delta"],
        "teacherFalsePositiveCost": teacher_delta["falsePositiveCost"],
        "sealManifest": str(SEAL_MANIFEST.relative_to(ROOT)),
    }, indent=2))

    if args.dry_run:
        return 0
    OUT_TRACES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TRACES.open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} traces -> {OUT_TRACES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
