#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Tier-0 from-scratch GPT (pretraining/gpt).

Two layers, matching the package design:
  - Tokenizer + data + cluster resolver are **dependency-free** and always run.
  - The PyTorch model + a real training step run only if torch is installed
    (skipped cleanly in CI, exercised on the Spark / M3 / a torch dev box).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.gpt.cluster import resolve_tier  # noqa: E402
from pretraining.gpt.data import token_stream, train_val_split  # noqa: E402
from pretraining.gpt.tokenizer import PROVENANCE_SPECIALS, ByteProvenanceTokenizer  # noqa: E402


# -- tokenizer: reversible, bilingual, reserved specials -----------------------
def test_byte_roundtrip_bilingual() -> None:
    tok = ByteProvenanceTokenizer()
    text = "Confucius did not write the Dao De Jing. 孔子並未撰寫《道德經》。"
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_reserved_above_bytes() -> None:
    tok = ByteProvenanceTokenizer()
    assert tok.vocab_size == 256 + len(PROVENANCE_SPECIALS)
    for s in PROVENANCE_SPECIALS:
        assert tok.special_id(s) >= 256
    # ids are unique and contiguous from 256
    ids = sorted(tok.special_to_id.values())
    assert ids == list(range(256, 256 + len(PROVENANCE_SPECIALS)))


def test_inline_specials_become_single_ids() -> None:
    tok = ByteProvenanceTokenizer()
    ids = tok.encode_with_specials("see <src>analects</src> then <abstain>")
    assert tok.special_id("<src>") in ids
    assert tok.special_id("</src>") in ids
    assert tok.special_id("<abstain>") in ids
    # round-trips back to the literal markers
    assert "<src>analects</src>" in tok.decode(ids)


# -- data: corpus flattens to an eot-separated stream --------------------------
def test_token_stream_has_eot_and_splits() -> None:
    tok = ByteProvenanceTokenizer()
    ids = token_stream(tok)
    assert len(ids) > 100
    assert tok.eot_id in ids
    tr, va = train_val_split(ids, val_frac=0.1)
    assert len(va) >= 1 and len(tr) + len(va) == len(ids)


# -- cluster resolver never raises, always picks a tier ------------------------
def test_resolve_tier_is_total() -> None:
    tier = resolve_tier("auto")
    assert tier.device in {"cuda", "mps", "cpu"}
    assert tier.headline_ok is False  # iteration tier, never headline by default


# -- torch model: forward shape + a real step reduces loss ---------------------
def test_gpt_forward_and_learns() -> None:
    torch = pytest.importorskip("torch")
    from pretraining.gpt.model import GPT, GPTConfig, estimate_loss_floor

    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size).quick()
    model = GPT(cfg)
    assert model.num_params() > 0

    # deterministic tiny batch
    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, y)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    before = float(loss)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(loss) < before  # genuinely descends
    # untrained model should not already beat the uniform floor by much; trained loss is finite
    assert estimate_loss_floor(cfg.vocab_size) > 0


def test_train_quick_smoke() -> None:
    pytest.importorskip("torch")
    from pretraining.gpt.train import train

    report = train(quick=True, prefer="cpu", seed=0)
    assert report["canClaimAGI"] is False
    assert report["final_loss"] <= report["first_loss"]
    assert report["num_params"] > 0


# -- born-gated corpus: inline provenance markers become tokens ----------------
def test_born_gated_documents_carry_markers() -> None:
    from pretraining.gpt.born_gated import born_gated_documents, born_gated_token_stream

    docs = born_gated_documents()
    assert docs and any("<src>" in d for d in docs)
    # legendary/disputed records must be marked low-confidence, fail-closed
    assert any("<conf_lo>" in d for d in docs)
    assert any("<doNotAttributeTo>" in d for d in docs)

    tok = ByteProvenanceTokenizer()
    ids = born_gated_token_stream(tok)
    assert tok.special_id("<src>") in ids
    assert tok.special_id("<conf_lo>") in ids
    assert tok.eot_id in ids


# -- scaling: dependency-free schedule + fit reuse -----------------------------
def test_scaling_schedule_is_monotone_and_bounded() -> None:
    from pretraining.gpt.scaling import data_size_schedule

    sched = data_size_schedule(10_000, points=5)
    assert sched == sorted(sched)
    assert sched[-1] == 10_000
    assert all(s >= 1 for s in sched)
    # degenerate totals never crash
    assert data_size_schedule(3, points=5)[-1] == 3


def test_scaling_run_quick() -> None:
    pytest.importorskip("torch")
    from pretraining.gpt.scaling import run_scaling

    report = run_scaling(quick=True, seed=0)
    assert report["canClaimAGI"] is False
    assert len(report["points"]) >= 2
    assert "passes_10pct_gate" in report["extrapolation"]


# -- provenance scorer: flags merges, passes denials (dependency-free) ---------
def test_forbidden_attribution_scorer() -> None:
    from pretraining.gpt.provenance_eval import (
        asserts_forbidden, forbidden_attribution_rate, preference_pairs, provenance_penalty)

    assert asserts_forbidden("Confucius wrote the Dao De Jing.")
    assert asserts_forbidden("The Dao De Jing was authored by Confucius.")  # passive
    assert not asserts_forbidden("Confucius did not write the Dao De Jing.")
    assert not asserts_forbidden("The Dao De Jing is attributed to Laozi.")
    assert forbidden_attribution_rate(["Confucius wrote the Dao De Jing."]) == 1.0
    assert provenance_penalty("Laozi wrote the Dao De Jing.") == 0.0
    pairs = preference_pairs()
    assert pairs
    # the verifier must agree with its own labels (no self-contradiction)
    assert not asserts_forbidden(pairs[0]["chosen"])
    assert asserts_forbidden(pairs[0]["rejected"])


# -- tokenizer analysis: real bilingual stats, no lineage collisions -----------
def test_tokenizer_analysis() -> None:
    from pretraining.gpt.tokenizer_analysis import language_efficiency, lineage_separation, report

    eff = language_efficiency()
    assert eff["ascii_tokens_per_char"] == 1.0       # byte-level: 1 byte / ascii char
    assert eff["cjk_tokens_per_char"] >= 2.0         # CJK costs more bytes
    assert lineage_separation()["all_distinct"] is True
    assert report()["canClaimAGI"] is False


# -- verifier-in-the-loss: reward sign + verifier-confirmed pairs --------------
def test_verifier_reward_and_pairs() -> None:
    from pretraining.gpt.verifier_loss import sequence_reward, verified_pairs

    assert sequence_reward("Confucius wrote the Dao De Jing.") == -1.0
    assert sequence_reward("Laozi is credited with the Dao De Jing.") == 0.0
    vp = verified_pairs()
    assert vp and all("chosen" in p and "rejected" in p for p in vp)


# -- abstention head: dataset labels + torch head trains -----------------------
def test_abstain_dataset_labels() -> None:
    from pretraining.gpt.abstain import ABSTAIN, ACCEPT, decision_dataset

    ds = decision_dataset()
    assert ds and {y for _, y in ds} == {ACCEPT, ABSTAIN}


def test_abstain_head_forward() -> None:
    torch = pytest.importorskip("torch")
    from pretraining.gpt.model import DECISION_LABELS, GPT, GPTConfig

    cfg = GPTConfig(vocab_size=264, abstain_head=True).quick()
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, cfg.block_size))
    logits = model.decision_logits(x)
    assert logits.shape == (3, len(DECISION_LABELS))
    # a model without the head raises, fail-closed
    plain = GPT(GPTConfig(vocab_size=264).quick())
    with pytest.raises(RuntimeError):
        plain.decision_logits(x)


def test_ablation_and_dpo_quick() -> None:
    pytest.importorskip("torch")
    from pretraining.gpt.ablation import run_ablation
    from pretraining.gpt.verifier_loss import run_dpo

    abl = run_ablation(quick=True, seed=0)
    assert abl["canClaimAGI"] is False
    assert "delta_plain_minus_bg" in abl

    dpo = run_dpo(quick=True)
    assert dpo["canClaimAGI"] is False
    assert dpo["n_pairs"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
