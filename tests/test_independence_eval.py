# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the independence-eval machinery (pre-registration P2).

Covers: seal_eval_pack manifest hash stability + tamper detection; the runner's aggregation math
on a tiny synthetic mock-backend fixture; and that the emitted PENDING artifact is labeled
not-run / NO-GO. All deterministic, stdlib only — no model calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_independence_eval as rie  # noqa: E402
from tools import seal_eval_pack as sep  # noqa: E402


# --------------------------------------------------------------------------- #
# seal_eval_pack: hash stability + tamper detection
# --------------------------------------------------------------------------- #
def _write_pack(path: Path, items: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(it) for it in items) + "\n", encoding="utf-8")


def test_manifest_hash_is_stable_under_formatting(tmp_path: Path) -> None:
    items = [{"id": "a", "prompt": "who wrote X?"}, {"id": "b", "prompt": "source of Y?"}]
    p1 = tmp_path / "pack1.jsonl"
    p2 = tmp_path / "pack2.jsonl"
    _write_pack(p1, items)
    # same content, different key order + whitespace + extra blank line
    p2.write_text(
        '  {"prompt": "who wrote X?", "id": "a"}  \n\n{"prompt":"source of Y?","id":"b"}\n',
        encoding="utf-8",
    )
    h1, c1 = sep.manifest_hash(p1)
    h2, c2 = sep.manifest_hash(p2)
    assert h1 == h2, "canonicalization must make the hash invariant to formatting/key order"
    assert c1 == c2 == 2


def test_manifest_hash_changes_on_content_change(tmp_path: Path) -> None:
    p = tmp_path / "pack.jsonl"
    _write_pack(p, [{"id": "a", "prompt": "who wrote X?"}])
    h_before, _ = sep.manifest_hash(p)
    _write_pack(p, [{"id": "a", "prompt": "who wrote X? (tampered)"}])
    h_after, _ = sep.manifest_hash(p)
    assert h_before != h_after


def test_seal_then_verify_roundtrip_and_tamper_detect(tmp_path: Path) -> None:
    p = tmp_path / "pack.jsonl"
    _write_pack(p, [{"id": "a", "prompt": "p1"}, {"id": "b", "prompt": "p2"}])
    manifest = sep.seal(p, author="ext-reviewer:jane", sealed_at="2026-06-29T00:00:00Z")
    sep.manifest_path(p).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    # sealed-at is exactly what was passed (never auto-dated)
    assert manifest["sealedAt"] == "2026-06-29T00:00:00Z"
    assert manifest["author"] == "ext-reviewer:jane"

    ok = sep.verify(p)
    assert ok["ok"] is True, ok

    # tamper the pack -> verify must FAIL (sealed/unspent enforcement)
    _write_pack(p, [{"id": "a", "prompt": "p1"}, {"id": "b", "prompt": "p2-CHANGED"}])
    bad = sep.verify(p)
    assert bad["ok"] is False
    assert bad["expectedHash"] != bad["actualHash"]


def test_verify_fails_without_manifest(tmp_path: Path) -> None:
    p = tmp_path / "unsealed.jsonl"
    _write_pack(p, [{"id": "a", "prompt": "p1"}])
    res = sep.verify(p)
    assert res["ok"] is False
    assert "never sealed" in res["reason"]


# --------------------------------------------------------------------------- #
# runner aggregation math on a tiny synthetic fixture
# --------------------------------------------------------------------------- #
def test_aggregate_math_on_tiny_fixture() -> None:
    # 3 families, gated all-pass / raw all-fail -> delta exactly 1.0, CI excludes zero, consistent
    per_family = {
        f"fam{k}": {"raw": [0] * 40, "gated": [1] * 40} for k in range(3)
    }
    judge_verdicts = {
        "ollama:qwen2.5:7b-instruct": ["gated"] * 36 + ["raw"] * 4,
        "openai:Llama-3.3-70B-4bit": ["gated"] * 34 + ["raw"] * 6,
    }
    agg = rie.aggregate(per_family, judge_verdicts, primary_threshold=0.105)
    for fam in per_family:
        fr = agg["perFamily"][fam]
        assert fr["delta"] == 1.0
        assert fr["ciExcludesZero"] is True
        assert fr["improves"] is True
    assert agg["consistentAcrossFamilies"] is True
    prim = agg["adapterPromptVsBasePrompt"][rie.PRIMARY_METRIC]
    assert prim["improves"] is True
    assert prim["delta"] == 1.0
    # judges mostly pick gated -> winrate > 0.5
    for jf, w in agg["judgeWinrate"].items():
        assert w["adapter_winrate"] > 0.5
    # agreement statistics are computed
    pair = next(iter(agg["judgeAgreement"].values()))
    assert pair["cohen_kappa"] is not None
    assert pair["gwet_ac1"] is not None


def test_aggregate_single_family_is_not_headline() -> None:
    # a CI-clean win on ONE family must NOT set the headline 'improves' (needs >=3 families)
    per_family = {"fam0": {"raw": [0] * 40, "gated": [1] * 40}}
    agg = rie.aggregate(per_family, {"jA": ["gated"] * 40, "jB": ["gated"] * 40})
    assert agg["consistentAcrossFamilies"] is False
    assert agg["adapterPromptVsBasePrompt"][rie.PRIMARY_METRIC]["improves"] is False


def test_aggregate_no_signal_ci_includes_zero() -> None:
    # gated == raw -> delta 0, CI must include zero, not a win
    per_family = {f"fam{k}": {"raw": [1, 0] * 20, "gated": [1, 0] * 20} for k in range(3)}
    agg = rie.aggregate(per_family, {"jA": ["gated", "raw"] * 20, "jB": ["gated", "raw"] * 20})
    assert agg["adapterPromptVsBasePrompt"][rie.PRIMARY_METRIC]["improves"] is False


def test_mock_run_is_deterministic() -> None:
    a = rie.run_mock(n_cases=60, seeds=2)
    b = rie.run_mock(n_cases=60, seeds=2)
    assert a == b, "mock run must be deterministic (stable hashing, no RNG state)"


# --------------------------------------------------------------------------- #
# the PENDING artifact must be labeled not-run / NO-GO
# --------------------------------------------------------------------------- #
def test_pending_artifact_is_not_run_and_no_go() -> None:
    arts = rie.pending_artifacts()
    ev, jg = arts["eval"], arts["judge"]
    assert ev["status"] == "not_run"
    assert ev["verdict"] != "GO"
    assert ev["go"] is False
    assert ev["canClaimAGI"] is False
    # no primary metric is CI-clean -> claim_gate cannot read a result
    assert ev["adapterPromptVsBasePrompt"] == {}
    assert jg["status"] == "not_run"
    assert jg["verdict"] != "GO"


def test_emit_pending_writes_no_go_files(tmp_path: Path) -> None:
    arts = rie.pending_artifacts()
    ep = tmp_path / f"{rie.PREFIX}-eval.json"
    jp = tmp_path / f"{rie.PREFIX}-judge.json"
    ep.write_text(json.dumps(arts["eval"]) + "\n", encoding="utf-8")
    jp.write_text(json.dumps(arts["judge"]) + "\n", encoding="utf-8")
    ev = json.loads(ep.read_text(encoding="utf-8"))
    assert ev["status"] == "not_run" and ev["verdict"] != "GO"


def test_claim_gate_no_go_on_pending(tmp_path: Path) -> None:
    # feed the PENDING artifacts to the real claim_gate.gate() and assert NO-GO (no result claimed)
    from tools import claim_gate

    arts = rie.pending_artifacts()
    wm = tmp_path / "wm"
    wm.mkdir()
    (wm / f"{rie.PREFIX}-eval.json").write_text(json.dumps(arts["eval"]), encoding="utf-8")
    (wm / f"{rie.PREFIX}-judge.json").write_text(json.dumps(arts["judge"]), encoding="utf-8")
    spec = json.loads(rie.SPEC.read_text(encoding="utf-8"))
    orig = claim_gate.WM
    try:
        claim_gate.WM = wm
        result = claim_gate.gate(rie.PREFIX, spec)
    finally:
        claim_gate.WM = orig
    assert result["verdict"] == "NO-GO"
    assert result["ok"] is False
