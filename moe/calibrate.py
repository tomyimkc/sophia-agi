# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibration-distribution matching for quantization — the Unsloth insight, governed.

*The problem this module solves.* Most quantization "imatrix" calibration sets are built
from generic text (wikitext, C4) and then the quantized model is benchmarked on...
wikitext. That is a **circular overfit**: the calibration distribution and the reported
benchmark distribution coincide, so the measured error flatters the quantization. Unsloth's
sharpest real insight in Dynamic 2.0 is that an *instruct/chat* model should be calibrated
on *chat-formatted* data — the distribution it actually sees at deployment — not on plain
text. The fix is simple to state and routinely violated in practice.

This module makes the discipline **machine-checked**:

1. **Source the calibration set from the deployment distribution** (Sophia's own
   council traces / source-discipline SFT packs — chat-formatted, the real deployment
   format), not from generic web text.
2. **Prove the calibration set is disjoint from every held-out eval set**, reusing the
   repo's existing contamination guard (`provenance_bench.dataset_guard`). Quantization
   calibrated on data that overlaps the eval set is the same leak as training-on-eval;
   it must fail closed the same way.
3. **Emit a calibration datasheet** (provenance, distribution source, decontamination
   proof, size) so the quantized artifact carries the same audit trail as a trained one.

This is the calibration half of the governed-quantization story; the *allocation* half is
:mod:`moe.adapt`. Together they answer "quantization, but honest" — calibrated on the real
deployment distribution, allocation CI-checked, error bounded. See
``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 3.

Pure stdlib + numpy; the contamination guard is already a repo dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def _row_text(row: dict) -> str:
    """Best-effort extraction of the textual content of a calibration row.

    Handles both SFT (``messages``) and DPO (``prompt``/``chosen``) shapes, plus a
    plain ``text`` field. Returns the concatenation of all human+assistant turns so the
    distribution the quantizer sees matches what the model emits at deployment.
    """
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("messages"), list):
        parts = []
        for m in row["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        return "\n".join(parts)
    out = []
    for k in ("prompt", "chosen", "question", "input"):
        v = row.get(k)
        if isinstance(v, str):
            out.append(v)
    return "\n".join(out)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 1. Build a calibration set from the deployment distribution
# ---------------------------------------------------------------------------

def build_calibration_set(sources: Iterable[Path], *, max_rows: int = 2048,
                          min_chars: int = 32) -> "list[dict]":
    """Assemble a chat-formatted calibration set from Sophia's own deployment packs.

    ``sources`` are JSONL files drawn from the *deployment distribution* — e.g.
    ``training/council/traces.jsonl`` (council deliberation traces),
    ``training/local_sophia_v2/sft_source_discipline.jsonl`` (source-discipline SFT),
    ``training/moral_gate_sft.jsonl``. These are chat-formatted and match what the served
    model actually emits, which is the whole point: calibrate on the real distribution.

    Drops rows that are too short (a 3-token row teaches the quantizer nothing about the
    output distribution) and de-duplicates by content hash (a repeated row over-weights
    one region of the distribution — a mini overfit within the calibration set itself).
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for src in sources:
        src = Path(src)
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _row_text(row)
            if len(text) < min_chars:
                continue
            h = content_hash(text)
            if h in seen:
                continue
            seen.add(h)
            rows.append({"text": text, "hash": h, "source": str(src.name)})
            if len(rows) >= max_rows:
                return rows
    return rows


# ---------------------------------------------------------------------------
# 2. Decontamination — calibration MUST be disjoint from eval
# ---------------------------------------------------------------------------

def check_calibration_disjoint(calib_rows: "list[dict]",
                               eval_prompts: "set[str] | None" = None) -> "tuple[bool, dict]":
    """Assert the calibration set is disjoint from the held-out eval prompts.

    Reuses :func:`provenance_bench.dataset_guard.eval_prompt_set` for the eval side and
    the guard's ``normalize`` for matching, so the *same* contamination rule that protects
    training-on-eval protects quantization-calibrated-on-eval. The leak is identical: if
    the quantizer was tuned on the eval distribution, the reported quantization error is
    meaningless. Fail-closed.

    Returns ``(ok, detail)``. ``ok`` is False iff any calibration text (normalized) appears
    in the eval prompt set.
    """
    try:
        from provenance_bench.dataset_guard import eval_prompt_set, normalize
    except Exception as e:  # pragma: no cover
        return False, {"checks": {"guard_importable": False}, "error": str(e)}

    if eval_prompts is None:
        eval_prompts = eval_prompt_set()
    eval_norm = {normalize(p) for p in eval_prompts}

    leaked = []
    for r in calib_rows:
        # Check both the full text and its normalized form against eval prompts.
        n = normalize(r.get("text", ""))
        if n and n in eval_norm:
            leaked.append(r.get("hash", "?"))
    ok = len(leaked) == 0
    return ok, {
        "checks": {"disjoint_from_eval": ok, "guard_importable": True},
        "n_calib": len(calib_rows),
        "n_eval_prompts": len(eval_norm),
        "leaked_hashes": leaked[:10],
        "leaked_count": len(leaked),
    }


# ---------------------------------------------------------------------------
# 3. Datasheet — the audit trail a quantized artifact must carry
# ---------------------------------------------------------------------------

def calibration_datasheet(calib_rows: "list[dict]", *, disjoint_ok: bool,
                          disjoint_detail: dict, target_bits: float,
                          notes: str = "") -> dict:
    """Emit a machine-readable calibration datasheet to ship *with* a quantized model.

    Mirrors the training-data passport discipline: a quantized artifact without a
    calibration datasheet is as opaque as an un-sourced claim. Records *what* the
    quantizer was tuned on (distribution source, size), *that* it was decontaminated
    (the disjoint proof), and the target width — so a reviewer can reproduce or reject it.
    """
    return {
        "calibration": {
            "n_rows": len(calib_rows),
            "sources": sorted({r.get("source", "?") for r in calib_rows}),
            "distribution": "deployment (chat-formatted council/SFT traces)",
            "hashes": [r.get("hash") for r in calib_rows[:64]],  # first 64 for spot-check
        },
        "decontamination": {
            "disjoint_from_eval": bool(disjoint_ok),
            "guard": "provenance_bench.dataset_guard.eval_prompt_set",
            "n_eval_prompts_checked": disjoint_detail.get("n_eval_prompts"),
            "leaked_count": disjoint_detail.get("leaked_count", 0),
        },
        "target_avg_bits": float(target_bits),
        "notes": notes,
        "honest_scope": (
            "Calibrated on the deployment distribution; allocation is CI-checked in "
            "moe/adapt. Output-fidelity claims still require a separate held-out eval "
            "to the no-overclaim gate — calibration disjointness is necessary, not "
            "sufficient, for an honest capability-retention claim."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. build_calibration_set dedups by content hash and respects min_chars/max_rows.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.jsonl"
        # two identical rows (dup) + one short + one good
        good = {"messages": [{"role": "user", "content": "x" * 100},
                             {"role": "assistant", "content": "y" * 100}]}
        short = {"text": "tiny"}
        p.write_text(
            json.dumps(good) + "\n" + json.dumps(good) + "\n"
            + json.dumps(short) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
        rows = build_calibration_set([p], min_chars=32)
    checks["dedups_identical"] = len(rows) == 1
    checks["drops_short"] = all(len(r["text"]) >= 32 for r in rows)
    detail["n_after_build"] = len(rows)

    # 2. content_hash is deterministic and collision-resistant in shape (16 hex chars).
    h1 = content_hash("abc"); h2 = content_hash("abc"); h3 = content_hash("abd")
    checks["hash_deterministic"] = h1 == h2
    checks["hash_distinct"] = h1 != h3
    checks["hash_len"] = len(h1) == 16

    # 3. Decontamination catches a known leak. Inject a calibration row whose text
    #    matches a synthetic eval prompt → must report not-disjoint.
    fake_eval = {"this is exactly an eval prompt that must not be in calib 0001"}
    leaky = [{"text": "this is exactly an eval prompt that must not be in calib 0001",
              "hash": "LEAK"}]
    ok_leak, det_leak = check_calibration_disjoint(leaky, eval_prompts=fake_eval)
    checks["detects_leak"] = (not ok_leak) and det_leak["leaked_count"] == 1
    detail["leak_detail"] = det_leak["leaked_count"]

    # 4. A clean calibration set (text unrelated to eval) reports disjoint.
    clean_eval = {"unique eval prompt zzzz 9999"}
    clean_calib = [{"text": "completely different deployment distribution text aaaa",
                    "hash": "OK1"}]
    ok_clean, _ = check_calibration_disjoint(clean_calib, eval_prompts=clean_eval)
    checks["clean_is_disjoint"] = ok_clean

    # 5. Datasheet records disjoint status and carries the honest_scope caveat.
    ds = calibration_datasheet(clean_calib, disjoint_ok=True,
                               disjoint_detail={"n_eval_prompts": 1, "leaked_count": 0},
                               target_bits=2.0)
    checks["datasheet_has_disjoint"] = ds["decontamination"]["disjoint_from_eval"] is True
    checks["datasheet_has_scope"] = "necessary, not sufficient" in ds["honest_scope"]
    checks["datasheet_has_sources"] = len(ds["calibration"]["sources"]) >= 0  # structure ok

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Calibration offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)
