# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The THREE-family judge farm (m3-sft kappa-gap closer) must satisfy the real >=3-family gate.

Sibling of test_mac_judge_config.py (the 2-family reference). This config is dispatch-ready
for the m3-sft-2family-judge-not-validated-2026-06-29 run; the run itself is farm-gated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.aggregate import _AGGREGATOR_PROVIDERS, _distinct_families  # noqa: E402

CONFIG = ROOT / "config" / "inference.local.mac-judge-3family.json"


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


def test_config_is_valid_three_box() -> None:
    cfg = _load()
    assert cfg["schema"] == "sophia.judge_farm.local.v1"
    assert set(cfg["boxes"]) == {"spark", "mac", "deepseek"}
    assert cfg["boxes"]["spark"]["engine"] == "vllm"
    assert cfg["boxes"]["mac"]["engine"] == "mlx"
    assert cfg["boxes"]["deepseek"]["engine"] == "openrouter"


def test_judges_clear_the_real_three_family_gate() -> None:
    """The judges must count as >=3 families under the SAME function the gate uses."""
    cfg = _load()
    judges = cfg["judge_farm"]["judges"]
    assert len(judges) >= 3
    assert _distinct_families(judges) >= 3, f"only {_distinct_families(judges)} family(ies): {judges}"
    fams = {_family(j) for j in judges}
    assert fams == {"qwen", "mlx-community", "deepseek"}
    assert fams == set(cfg["judge_farm"]["expected_families"])


def test_third_family_is_genuinely_distinct() -> None:
    """The 3rd family (deepseek) must not collapse into qwen or the llama/mlx family."""
    cfg = _load()
    fams = [_family(j) for j in cfg["judge_farm"]["judges"]]
    assert len(fams) == len(set(fams)) == 3  # no two judges share a family


def test_judges_are_distinct_from_subject() -> None:
    """judge != subject: the gemma-3-4b-it subject lineage must not be a judge family."""
    cfg = _load()
    fams = {_family(j) for j in cfg["judge_farm"]["judges"]}
    assert "gemma" not in fams  # the M3-pilot subject is gemma-3-4b-it
    # subject_lineages_to_avoid documents the model lineages these judges rule out as subjects.
    assert set(cfg["judge_farm"]["subject_lineages_to_avoid"]) == {"qwen", "meta-llama", "deepseek"}


def test_recommended_flag_matches_judges() -> None:
    farm = _load()["judge_farm"]
    flag = farm["recommended_flag"]
    assert flag.startswith("--judges ")
    assert set(flag[len("--judges "):].split(",")) == set(farm["judges"])


def test_gate_preregisters_kappa_floor_and_forced_choice() -> None:
    """The gate block must pin the kappa>=0.40 floor, >=3 seeds, and forced-choice primary."""
    gate = _load()["gate"]
    assert gate["min_judge_families"] == 3
    assert gate["kappa_floor"] == 0.40
    assert gate["min_seeds"] == 3
    assert gate["ci_excludes_zero"] is True
    assert gate["primary_protocol"] == "forced-choice"
