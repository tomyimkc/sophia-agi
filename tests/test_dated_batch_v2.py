# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CI regression for the realtime C1 dated-fact batch v2 + the pack quality gate.

Keeps the honest-growth discipline standing: (a) tools/validate_dated_pack passes its own offline
invariants; (b) batch v2 is well-formed, provenance-carrying, and temporally clean; (c) the seed +
batch together have no id collisions; (d) the pack is STILL underpowered (N<393) — so no run may be
promoted to a GO. No capability claim; canClaimAGI stays false. Runs under pytest OR as a script.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEED = ROOT / "eval" / "fact_check" / "phase1_dated_seed_v1.jsonl"
BATCH = ROOT / "eval" / "fact_check" / "phase1_dated_batch_v2.jsonl"
BATCHES = sorted((ROOT / "eval" / "fact_check").glob("phase1_dated_batch_v*.jsonl"))


def test_validator_offline_invariants():
    import validate_dated_pack
    ok, d = validate_dated_pack.offline_invariants()
    assert ok, d["checks"]


def test_all_batches_are_valid_and_sourced():
    import validate_dated_pack
    # every batch (v2, v3, …) must pass with provenance required
    ok, d = validate_dated_pack.validate(BATCHES, require_source=True)
    assert ok, d["errors"]
    assert d["total"] >= 30
    # honest label mix across the batches (real true/false/unknowable spread)
    assert d["mix"]["true"] >= 10 and d["mix"]["false"] >= 8 and d["mix"]["unknowable"] >= 4, d["mix"]


def test_seed_plus_batch_no_id_collision_and_still_underpowered():
    import validate_dated_pack
    # seed v1 predates the `source` field -> validate the union without requiring it
    ok, d = validate_dated_pack.validate([SEED, *BATCHES], require_source=False)
    assert ok, d["errors"]
    assert d["unique_ids"] == d["total"], "id collision between seed and batch"
    # the pack must STILL be underpowered — no GO is permissible below N=393
    assert d["underpowered"] and d["total"] < 393, d


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  [ok] {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  [XX] {fn.__name__}: {e}")
    print(f"dated-batch-v2 regression: {'PASS' if not failed else 'FAIL'} ({len(fns) - failed}/{len(fns)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
