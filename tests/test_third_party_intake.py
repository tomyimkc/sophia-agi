# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the third-party verifiable-domain intake pipeline (tools/third_party_intake.py).

Covers the load-bearing guarantees:
  * a decontam-FAILING item is REFUSED and never enters the admitted set (fail-closed gate);
  * validity gate rejects non-machine-checkable / structurally-invalid items;
  * admittedCount and loopClosedCount are STRICTLY separate (loopClosed <= admitted, and
    admitting never bumps loopClosed);
  * loopClosedCount == 0 on the shipped synthetic sample manifest (honest N=0);
  * the intake baseline is not self-poisoned by a manifest living under eval/.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import third_party_intake as tpi  # noqa: E402

SAMPLE = ROOT / "eval" / "third_party_intake" / "sample_manifest.jsonl"

# A small controlled baseline: exactly one eval prompt (normalized) to check against.
_CONTAM_PROMPT = "prove that the sum of two even integers is even."
_BASELINE = {tpi.normalize(_CONTAM_PROMPT)}


def _item(item_id, prompt, kind="sympy", gold="ok", test="assert True", proof=True):
    scoring = {"kind": kind}
    if kind == "sympy":
        scoring["gold"] = gold
    elif kind == "exec":
        scoring["test"] = test
    it = {"itemId": item_id, "domain": "math", "prompt": prompt, "scoring": scoring}
    if proof:
        it["decontamProof"] = {"method": "author-warrant", "warrant": "independent"}
    return it


def test_decontam_failing_item_is_rejected():
    """An item whose prompt matches the eval baseline is REFUSED (never admitted)."""
    items = [
        _item("clean-1", "compute the 7th triangular number.", gold="28"),
        _item("contam-1", _CONTAM_PROMPT, gold="qed"),
    ]
    r = tpi.run_intake(items, eval_norm=set(_BASELINE))
    assert "clean-1" in r["admitted"], r
    assert "contam-1" not in r["admitted"], "contaminated item must NOT be admitted"
    rej_ids = {x["itemId"]: x for x in r["rejected"]}
    assert "contam-1" in rej_ids, "contaminated item must be in rejected list"
    assert rej_ids["contam-1"]["stage"] == "decontam"
    assert r["counters"]["admittedCount"] == 1
    assert r["counters"]["rejectedCount"] == 1


def test_near_duplicate_is_rejected():
    """A near-verbatim copy (high shingle Jaccard) is refused, not just exact matches.

    A long eval prompt with a single appended token stays well above the 0.9 Jaccard bar
    (this is what a near-verbatim lift of an eval item looks like); the decontam gate must
    refuse it. (Contrast: adding a word to a short sentence drops Jaccard below threshold and
    is legitimately admitted — the near-dup layer targets verbatim spans, not paraphrase.)
    """
    base = ("compute the determinant of the three by three matrix whose rows are one two "
            "three four five six and seven eight ten and then report the resulting integer "
            "value clearly")
    near = base + " please"
    items = [_item("near-1", near, kind="exec", test="assert True")]
    r = tpi.run_intake(items, eval_norm={tpi.normalize(base)}, jaccard=0.9, shingle=5)
    assert "near-1" not in r["admitted"], "near-duplicate must be refused"
    assert any(x["itemId"] == "near-1" and x["stage"] == "decontam" for x in r["rejected"])


def test_validity_gate_rejects_non_machine_checkable():
    """LLM-judged / missing-oracle / missing-proof items are refused at validity."""
    items = [
        {"itemId": "judge-1", "domain": "code", "prompt": "explain quicksort.",
         "scoring": {"kind": "llm-judge"}, "decontamProof": {"method": "x"}},
        {"itemId": "noproof-1", "domain": "math", "prompt": "add 2 and 2.",
         "scoring": {"kind": "sympy", "gold": "4"}},  # no decontamProof
        {"itemId": "empty-oracle", "domain": "math", "prompt": "add 3 and 3.",
         "scoring": {"kind": "sympy", "gold": ""}, "decontamProof": {"method": "x"}},
    ]
    r = tpi.run_intake(items, eval_norm=set())  # empty baseline -> only validity can reject
    assert r["admitted"] == [], "no item should be admitted"
    stages = {x["itemId"]: x["stage"] for x in r["rejected"]}
    assert stages["judge-1"] == "validity"
    assert stages["noproof-1"] == "validity"
    assert stages["empty-oracle"] == "validity"


def test_counters_are_strictly_separate():
    """admittedCount can be > 0 while loopClosedCount stays 0; invariant holds."""
    items = [
        _item("a", "compute 12 factorial.", gold="479001600"),
        _item("b", "compute 5 choose 2.", gold="10"),
    ]
    r = tpi.run_intake(items, eval_norm=set())
    c = r["counters"]
    assert c["admittedCount"] == 2, c
    assert c["loopClosedCount"] == 0, "admitting must NEVER close the loop"
    assert c["loopClosedCount"] <= c["admittedCount"], "invariant loopClosed<=admitted"
    # go/claim flags stay honest
    assert r["go"] is False
    assert r["canClaimAGI"] is False
    assert r["status"] == "preregistration_only"


def test_sample_manifest_loop_closed_is_zero():
    """On the shipped synthetic manifest: some admitted, loopClosedCount stays 0 (honest N=0)."""
    assert SAMPLE.exists(), f"missing sample manifest: {SAMPLE}"
    items = tpi._read_manifest(SAMPLE)
    baseline = tpi.eval_baseline(exclude=SAMPLE)
    r = tpi.run_intake(items, eval_norm=baseline)
    c = r["counters"]
    assert c["loopClosedCount"] == 0, "loopClosedCount MUST be 0 with no verifier+gain data"
    # the deliberately-contaminated + the llm-judge items must be rejected; the two clean
    # machine-checkable items must be admitted.
    assert c["admittedCount"] == 2, r
    rej = {x["itemId"]: x["stage"] for x in r["rejected"]}
    assert rej.get("SYNTH-EXT-math-CONTAM") == "decontam"
    assert rej.get("SYNTH-EXT-noscoring-001") == "validity"


def test_self_ingestion_guard():
    """A clean manifest item under eval/ is NOT rejected for matching itself."""
    items = tpi._read_manifest(SAMPLE)
    baseline = tpi.eval_baseline(exclude=SAMPLE)
    r = tpi.run_intake(items, eval_norm=baseline)
    # SYNTH-EXT-math-001 is unique to the manifest -> must NOT be a self-match reject.
    assert "SYNTH-EXT-math-001" in r["admitted"], (
        "clean manifest item spuriously rejected — self-ingestion guard failed")


def test_cli_exit_code_and_receipt():
    """The CLI returns exit 0 and a JSON receipt; missing manifest returns exit 2."""
    import json
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "third_party_intake.py"),
         "--manifest", "eval/third_party_intake/sample_manifest.jsonl"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["counters"]["loopClosedCount"] == 0
    assert receipt["go"] is False
    # missing manifest -> exit 2
    proc2 = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "third_party_intake.py"),
         "--manifest", "eval/third_party_intake/does_not_exist.jsonl"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert proc2.returncode == 2, proc2.stdout


if __name__ == "__main__":
    test_decontam_failing_item_is_rejected()
    test_near_duplicate_is_rejected()
    test_validity_gate_rejects_non_machine_checkable()
    test_counters_are_strictly_separate()
    test_sample_manifest_loop_closed_is_zero()
    test_self_ingestion_guard()
    test_cli_exit_code_and_receipt()
    print("ALL TESTS PASSED")
