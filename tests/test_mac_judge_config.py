# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The two-box (Spark vLLM + Mac MLX) judge farm must satisfy the real >=2-family gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.aggregate import _AGGREGATOR_PROVIDERS, _distinct_families  # noqa: E402

CONFIG = ROOT / "config" / "inference.local.mac-judge.json"


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _family(spec: str) -> str:
    """Replicate the gate's per-judge family derivation (provenance_bench.aggregate)."""
    prov, _, model = spec.partition(":")
    prov = prov.strip().lower()
    model = model.strip().split("@", 1)[0]
    if prov in _AGGREGATOR_PROVIDERS and "/" in model:
        return model.split("/", 1)[0].lower()
    return prov


def test_config_is_valid_two_box() -> None:
    cfg = _load()
    assert cfg["schema"] == "sophia.judge_farm.local.v1"
    assert set(cfg["boxes"]) == {"spark", "mac"}
    assert cfg["boxes"]["spark"]["engine"] == "vllm"
    assert cfg["boxes"]["mac"]["engine"] == "mlx"


def test_judges_clear_the_real_two_family_gate() -> None:
    """The judges must count as >=2 families under the SAME function the gate uses."""
    cfg = _load()
    judges = cfg["judge_farm"]["judges"]
    assert len(judges) >= 2
    assert _distinct_families(judges) >= 2, f"only {_distinct_families(judges)} family(ies): {judges}"
    # The gate keys vLLM by vendor ('qwen') and mlx by engine ('mlx').
    fams = {_family(j) for j in judges}
    assert fams == {"qwen", "mlx"}
    assert fams == set(cfg["judge_farm"]["expected_families"])


def test_same_vendor_judges_collapse_to_one_family() -> None:
    """The REAL pitfall: two SAME-vendor judges = 1 family (different vendors = 2)."""
    assert _distinct_families(["vllm:Qwen/Qwen2.5-7B-Instruct", "vllm:Qwen/Qwen2.5-14B-Instruct"]) == 1
    assert _distinct_families(["vllm:Qwen/Qwen2.5-7B-Instruct", "vllm:meta-llama/Llama-3.3-8B"]) == 2


def test_boxes_are_independent_graders() -> None:
    """The value-add of the Mac box: a different engine/runtime than the Spark (less-correlated)."""
    cfg = _load()
    engines = {b["engine"] for b in cfg["boxes"].values()}
    assert engines == {"vllm", "mlx"}                         # CUDA/vLLM vs Metal/MLX
    lineages = {b["model_lineage"] for b in cfg["boxes"].values()}
    assert len(lineages) == 2                                 # distinct model lineages too


def test_recommended_flag_matches_judges() -> None:
    farm = _load()["judge_farm"]
    flag = farm["recommended_flag"]
    assert flag.startswith("--judges ")
    assert set(flag[len("--judges "):].split(",")) == set(farm["judges"])


def test_config_loader_emits_judges_and_fails_gracefully(tmp_path, capsys) -> None:
    """tools/run_local_judge_eval.py --config: valid → judges; bad input → structured error, exit 1."""
    sys.path.insert(0, str(ROOT / "tools"))
    import run_local_judge_eval as rlje
    # valid config → exit 0, emits the recommended flag
    assert rlje.main(["--config", str(CONFIG)]) == 0
    assert "recommended_judge_flag" in capsys.readouterr().out
    # missing file → exit 1, structured error (no traceback)
    assert rlje.main(["--config", str(tmp_path / "nope.json")]) == 1
    assert "error" in json.loads(capsys.readouterr().out)
    # valid JSON but missing keys → exit 1, structured error
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "x"}', encoding="utf-8")
    assert rlje.main(["--config", str(bad)]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_judges_are_distinct_from_subject() -> None:
    """judge != subject: the OLMoE/Sophia-V1 subject lineage must not be a judge lineage."""
    cfg = _load()
    judge_lineages = {b["model_lineage"] for b in cfg["boxes"].values()}
    assert "olmoe" not in judge_lineages and "allenai" not in judge_lineages
    assert set(cfg["judge_farm"]["subject_lineages_to_avoid"]) == judge_lineages
