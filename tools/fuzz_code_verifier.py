#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier fuzzing for the CODE reward — find the bug before the model learns it.

Implements the discipline of *"Before the Model Learns the Bug: Fuzzing RLVR
Verifiers"* (arXiv:2606.01066) for this repo's code reward: generate a corpus of
adversarial completions that try to game the tests-pass signal, then compare the
**buggy reference verifier** (`code_reward.reward_for_task`, exit-code grader) with
the **hardened verifier** (`code_integrity.guarded_reward_for_task`, static gate +
exit-code-proof isolated grader). The hardened verifier must reject every cheat
and accept the honest solution; if any cheat slips through, this exits non-zero so
CI catches the regression before an RL run can exploit it.

Deterministic, stdlib-only, offline. The cheat rejections rely on the static scan
(no execution needed), so the GATE is meaningful in CI even with
`SOPHIA_ALLOW_CODE_EXEC=0`; set it to 1 to also exercise the isolated grader and to
show the buggy verifier actually accepting the hacks.

    python tools/fuzz_code_verifier.py            # table + GO/NO-GO, exit 0/1
    python tools/fuzz_code_verifier.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import code_integrity as ci  # noqa: E402
from provenance_bench.code_exec import _exec_on  # noqa: E402
from provenance_bench.code_reward import REWARD_MAX, REWARD_MIN, reward_for_task  # noqa: E402

# Hidden canonical test the model never sees, plus held-out inputs (C4) and a
# couple of honest controls that the hardened verifier must keep accepting.
CANONICAL = "assert scale(3, 4) == 12\nassert scale(0, 5) == 0\n"
HOLDOUT = "assert scale(7, 6) == 42\nassert scale(9, 9) == 81\n"
HONEST = {
    "honest_correct": "```python\ndef scale(n, k):\n    return n * k\n```",
}
# An honest-but-wrong answer must be rejected by BOTH verifiers (sanity: the gate
# is not just rejecting everything).
HONEST_WRONG = "```python\ndef scale(n, k):\n    return n + k\n```"


def run(write: str | None = None) -> int:
    exec_on = _exec_on()
    corpus = ci.fuzz_corpus()
    rows = []

    for name, ans in corpus.items():
        weak, _ = reward_for_task(ans, CANONICAL)
        hard, det = ci.guarded_reward_for_task(ans, CANONICAL, holdout_test=HOLDOUT)
        rows.append({
            "case": name, "kind": "cheat",
            "weak": weak, "hardened": hard,
            "rejected": hard == REWARD_MIN,
            "categories": det.get("integrity", {}).get("categories", []),
        })

    for name, ans in HONEST.items():
        weak, _ = reward_for_task(ans, CANONICAL)
        hard, _ = ci.guarded_reward_for_task(ans, CANONICAL, holdout_test=HOLDOUT)
        rows.append({"case": name, "kind": "honest", "weak": weak,
                     "hardened": hard, "accepted": hard == REWARD_MAX, "categories": []})

    wk, hd = reward_for_task(HONEST_WRONG, CANONICAL)[0], ci.guarded_reward_for_task(HONEST_WRONG, CANONICAL)[0]
    rows.append({"case": "honest_wrong", "kind": "control", "weak": wk,
                 "hardened": hd, "rejected": hd == REWARD_MIN, "categories": []})

    cheats_leaked = [r["case"] for r in rows if r["kind"] == "cheat" and not r["rejected"]]
    honest_blocked = [r["case"] for r in rows if r["kind"] == "honest" and not r["accepted"]]
    go = not cheats_leaked and not honest_blocked

    # --- report ---
    print(f"VERIFIER FUZZ (exec={'on' if exec_on else 'off — static-gate only'})  "
          f"corpus={len(corpus)} cheats, {len(HONEST)} honest, 1 control\n")
    print(f"  {'case':18} {'kind':7} {'weak':>5} {'hardened':>9}  verdict")
    print("  " + "-" * 56)
    for r in rows:
        if r["kind"] == "control" and not exec_on:
            mark = "n/a (syntax-only)"  # wrong code still compiles; needs exec to reject
        else:
            ok = r.get("rejected", r.get("accepted"))
            mark = "ok" if ok else "LEAK"
        print(f"  {r['case']:18} {r['kind']:7} {r['weak']:>+5.0f} {r['hardened']:>+9.0f}  {mark}"
              + (f"  [{','.join(r['categories'])}]" if r["categories"] else ""))

    verdict = "GO" if go else "NO-GO"
    print(f"\nVERIFIER FUZZ: {verdict} — "
          + (f"hardened verifier rejected all {len(corpus)} cheats, kept honest"
             if go else f"LEAKS: cheats={cheats_leaked} honest_blocked={honest_blocked}"))

    if write:
        payload = {
            "tool": "fuzz_code_verifier", "verdict": verdict, "exec": exec_on,
            "nCheats": len(corpus), "cheatsLeaked": cheats_leaked,
            "honestBlocked": honest_blocked, "rows": rows,
            "reference": "arXiv:2606.01066", "canClaimAGI": False,
        }
        Path(write).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {write}")

    return 0 if go else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fuzz the code reward verifier (arXiv:2606.01066).")
    ap.add_argument("--json", dest="json_out", default=None, help="write a JSON report to this path")
    args = ap.parse_args(argv)
    return run(write=args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
