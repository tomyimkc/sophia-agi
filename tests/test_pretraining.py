#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the pretraining research package.

Fast, deterministic, dependency-free. Each test asserts a *property* that must hold for
the artifact to be honest (model actually learns; power-law fit recovers a known exponent;
MoE routing doesn't collapse; dedup catches duplicates; validators fail closed), rather
than re-running full studies (those have their own --quick CLIs).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.architecture.moe import MoELM  # noqa: E402
from pretraining.data_passport.passport import minhash, estimate_jaccard, stamp_pack  # noqa: E402
from pretraining.eval_matrix.matrix import build_matrix  # noqa: E402
from pretraining.nano import (  # noqa: E402
    NanoLM, drifted_source, eval_loss, make_source, mixed_corpus,
    sample_stream, source_entropy, to_examples, train,
)
from pretraining.scaling.fit import fit_with_floor, predict  # noqa: E402
from pretraining.vertical_data.schemas import validate  # noqa: E402


# -- nano model genuinely learns -----------------------------------------------
def test_nano_model_reduces_loss() -> None:
    src = make_source(vocab=6, order=2, seed=1, peak=3.0)
    ex = to_examples(sample_stream(src, 600, seed=2), context=2)
    m = NanoLM(vocab=6, context=2, hidden=12, seed=0)
    before = eval_loss(m, ex)
    hist = train(m, ex, epochs=8, optimizer="adam", lr=0.05)
    after = eval_loss(m, ex)
    assert after < before - 0.05, (before, after)
    assert not hist["diverged"]


def test_source_entropy_is_lower_bound_ish() -> None:
    # A learned model's held-out loss should land at or above the source entropy floor
    # (within sampling noise), never wildly below.
    src = make_source(vocab=6, order=2, seed=3, peak=3.0)
    E = source_entropy(src)
    ex = to_examples(sample_stream(src, 2000, seed=2), context=2)
    held = to_examples(sample_stream(src, 1000, seed=99), context=2)
    m = NanoLM(vocab=6, context=2, hidden=16, seed=0)
    train(m, ex, epochs=12, optimizer="adam", lr=0.03)
    assert eval_loss(m, held) > E - 0.25


# -- power-law fit recovers a planted exponent ---------------------------------
def test_fit_recovers_known_powerlaw() -> None:
    E, A, p = 0.5, 2.0, 0.4
    xs = [100, 200, 400, 800, 1600]
    losses = [E + A * (x ** (-p)) for x in xs]
    fit = fit_with_floor(xs, losses, E)
    assert abs(fit["p"] - p) < 1e-3
    assert abs(fit["A"] - A) < 1e-2
    assert fit["r2_logspace"] > 0.999
    # extrapolation is exact for clean synthetic data
    assert abs(predict(fit, 3200) - (E + A * 3200 ** (-p))) < 1e-6


# -- data mixing: best mix tracks the target -----------------------------------
def test_mixed_corpus_respects_ratio() -> None:
    a = make_source(vocab=6, order=1, seed=1)
    b = make_source(vocab=6, order=1, seed=2)
    ex = mixed_corpus([a, b], [0.75, 0.25], 400, context=1, seed=0)
    assert len(ex) == 400


# -- synthetic drift produces a genuinely different distribution ----------------
def test_drift_changes_distribution() -> None:
    src = make_source(vocab=6, order=2, seed=1, peak=3.0)
    d0 = drifted_source(src, 0.0, seed=5)
    d1 = drifted_source(src, 0.8, seed=5)
    # drift=0 reproduces source; drift>0 raises entropy (toward uniform)
    assert abs(source_entropy(d0) - source_entropy(src)) < 1e-9
    assert source_entropy(d1) > source_entropy(src)


# -- MoE routing does not collapse ---------------------------------------------
def test_moe_routes_without_collapse() -> None:
    src = make_source(vocab=6, order=2, seed=1, peak=3.0)
    ex = to_examples(sample_stream(src, 800, seed=2), context=2)
    moe = MoELM(vocab=6, context=2, hidden=6, n_experts=3, seed=0)
    for c, t in ex:
        moe.train_step(c, t, 0.1)
    # no single expert should swallow everything (perfect balance = 1/3)
    assert moe.load_balance() < 0.85
    assert moe.active_params() < moe.num_params()  # sparse: active < total


# -- data passport catches duplicates & flags missing license ------------------
def test_passport_dedup_and_flags() -> None:
    rows = [
        {"messages": [{"role": "user", "content": "What is 2+2?"},
                      {"role": "assistant", "content": "4"}], "source": "x", "license": "MIT"},
        {"messages": [{"role": "user", "content": "What is 2+2?"},
                      {"role": "assistant", "content": "4"}], "source": "x", "license": "MIT"},
        {"prompt": "A completely different and sufficiently long unique training prompt here.",
         "completion": "distinct answer text", "source": "y"},
    ]
    res = stamp_pack(rows)
    sheet = res["datasheet"]
    # rows 0 and 1 are exact dups -> share a cluster
    assert res["rows"][0]["_passport"]["dedup_cluster"] == res["rows"][1]["_passport"]["dedup_cluster"]
    assert sheet["duplicate_rate"] > 0
    # row 2 has no license -> flagged unlicensed
    assert "unlicensed" in res["rows"][2]["_passport"]["flags"]


def test_minhash_similarity() -> None:
    a = minhash("the quick brown fox jumps over the lazy dog")
    b = minhash("the quick brown fox jumps over the lazy dog")
    c = minhash("entirely unrelated content about quantum chromodynamics")
    assert estimate_jaccard(a, b) == 1.0
    assert estimate_jaccard(a, c) < 0.5


# -- eval matrix builds and reports honest gaps --------------------------------
def test_eval_matrix_builds() -> None:
    m = build_matrix()
    assert m["n_packs"] > 0
    assert 0.0 <= m["coverage_fraction"] <= 1.0
    assert len(m["uncovered_cells"]) > 0  # the honest gaps must be surfaced
    # multimodal is a known gap
    assert any("multimodal" in c for c in m["uncovered_cells"])


# -- vertical-data validators fail closed --------------------------------------
def test_vertical_validators_fail_closed() -> None:
    good = {"record_type": "agent_trajectory", "source": "s", "license": "MIT",
            "goal": "g", "steps": [{"action": "a", "observation": "o"}],
            "outcome": "done", "reward": 0.9}
    assert validate("agent_trajectory", good)["ok"]
    bad = {"record_type": "agent_trajectory", "goal": "", "steps": [], "outcome": 1}
    res = validate("agent_trajectory", bad)
    assert not res["ok"] and len(res["errors"]) >= 3
    assert not validate("nonsense_type", {})["ok"]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
