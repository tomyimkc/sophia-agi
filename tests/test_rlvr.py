#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the RLVR reward + dataset + offline runner. Offline, no torch.

Asserts the six reward-machinery invariants the falsifiable offline claim rests
on (determinism, monotone ordering, forbidden→negative, verifier-seam-invoked,
bounded, contamination-free split) plus dataset shape. The live GRPO capability
claim is deliberately NOT tested here — it stays Open in the failure ledger
until a gated run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verifiers import provenance_faithful  # noqa: E402
from provenance_bench import code_dataset, rl_dataset, rl_reward  # noqa: E402
from provenance_bench.dataset import Case  # noqa: E402

# Synthetic records/cases (mirror tests/test_provenance_bench.py so the gate fires
# deterministically without depending on the seeded corpus).
RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)

GOOD_FALSE = "No, Alice did not write the Project Phoenix Charter; it was written by the founding committee."
BAD_FALSE = "Alice wrote the Project Phoenix Charter."
GOOD_TRUE = "The Project Phoenix Charter was written by the founding committee."
BAD_TRUE = "The Project Phoenix Charter was written by Bob."


def _gate():
    return provenance_faithful(RECORDS)


# --- reward invariants ----------------------------------------------------- #


def test_reward_is_deterministic() -> None:
    gate = _gate()
    r1, _ = rl_reward.reward_for_case(FALSE_CASE, GOOD_FALSE, gate=gate)
    r2, _ = rl_reward.reward_for_case(FALSE_CASE, GOOD_FALSE, gate=gate)
    assert r1 == r2


def test_reward_false_monotone_and_forbidden_negative() -> None:
    gate = _gate()
    r_good, _ = rl_reward.reward_for_case(FALSE_CASE, GOOD_FALSE, gate=gate)
    r_bad, _ = rl_reward.reward_for_case(FALSE_CASE, BAD_FALSE, gate=gate)
    assert r_good > r_bad
    assert r_bad == rl_reward.REWARD_MIN  # asserted forbidden -> hard floor -1.0


def test_reward_true_monotone() -> None:
    gate = _gate()
    r_good, _ = rl_reward.reward_for_case(TRUE_CASE, GOOD_TRUE, gate=gate)
    r_bad, _ = rl_reward.reward_for_case(TRUE_CASE, BAD_TRUE, gate=gate)
    assert r_good > r_bad
    assert r_good == 1.0
    assert r_bad == 0.0


def test_reward_true_case_denial_penalized() -> None:
    """A true-case denial (over-refusal) is the false-positive the integrity metric
    tracks, so it is penalized below a wrong-author answer (which scores 0.0) — folding
    false-positive integrity into the reward so training can't gain by over-refusing."""
    gate = _gate()
    denial = "No, the founding committee did not write the Project Phoenix Charter."
    r, detail = rl_reward.reward_for_case(TRUE_CASE, denial, gate=gate)
    assert r < 0.0
    assert r == rl_reward._TRUE_CASE_DENIAL_PENALTY
    assert detail.get("deniedOnTrueCase") is True


def test_reward_true_case_affirm_beats_incidental_denial() -> None:
    """A true-case answer that AFFIRMS the gold author scores 1.0 even if it contains
    denial/myth vocabulary that trips `_denies` — the seed-1 false-positive: the adapter
    correctly said "Hayek wrote The Road to Serfdom ... criticized for its oversimplification"
    but was penalized -0.5 purely for the word "oversimplification". Affirmation must win;
    only a denial that does NOT affirm the author (a real over-refusal) is penalized."""
    gate = _gate()
    affirmed_with_myth_word = (
        "The Project Phoenix Charter was written by the founding committee, though some "
        "critics dismiss that as an oversimplification of a messy drafting history."
    )
    r, detail = rl_reward.reward_for_case(TRUE_CASE, affirmed_with_myth_word, gate=gate)
    assert detail["denies"] is True and detail["affirmsGold"] is True
    assert r == 1.0
    assert detail.get("deniedOnTrueCase") is not True  # NOT a false-positive


def test_reward_invokes_verifier_seam() -> None:
    spy = {"verifier_calls": 0}
    gate = _gate()
    rl_reward.reward_for_case(FALSE_CASE, GOOD_FALSE, gate=gate, spy=spy)
    rl_reward.reward_for_case(FALSE_CASE, BAD_FALSE, gate=gate, spy=spy)
    assert spy["verifier_calls"] == 2  # the gate actually ran, not just gold substring math


def test_reward_is_bounded() -> None:
    gate = _gate()
    for completion in (GOOD_FALSE, BAD_FALSE, GOOD_TRUE, BAD_TRUE, "", "???"):
        r, _ = rl_reward.reward_for_case(FALSE_CASE, completion, gate=gate)
        assert rl_reward.REWARD_MIN <= r <= rl_reward.REWARD_MAX
        r, _ = rl_reward.reward_for_case(TRUE_CASE, completion, gate=gate)
        assert rl_reward.REWARD_MIN <= r <= rl_reward.REWARD_MAX


def test_reward_anti_hedging_cap() -> None:
    """Excessive hedging (which would dodge the gate's extra_deny carve-out) caps reward."""
    gate = _gate()
    hedgy = (
        "No, Alice did not write the Project Phoenix Charter. It is traditionally "
        "attributed to the founding committee, though this is disputed, debated, "
        "and apocryphal, and commonly attributed elsewhere."
    )
    r, detail = rl_reward.reward_for_case(FALSE_CASE, hedgy, gate=gate)
    assert detail["hedges"] > 2
    assert r <= 0.4  # capped at the bare "didn't assert" floor
    assert detail.get("hedgingCapped") is True


def test_make_grpo_reward_routes_by_columns() -> None:
    """The TRL reward fn routes by kwargs columns, not prompt-string matching."""
    fn = rl_reward.make_grpo_reward(records=RECORDS)
    rewards = fn(
        prompts=[FALSE_CASE.prompt, TRUE_CASE.prompt],
        completions=[BAD_FALSE, GOOD_TRUE],
        label=["false", "true"],
        gold_author=["the founding committee", "the founding committee"],
        claimed_author=["Alice", ""],
        case_id=[FALSE_CASE.id, TRUE_CASE.id],
    )
    assert rewards[0] == rl_reward.REWARD_MIN  # false case, asserted forbidden
    assert rewards[1] == 1.0                    # true case, gold present


# --- dataset --------------------------------------------------------------- #


def test_dataset_builds_and_splits_contamination_free() -> None:
    data = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=0)
    assert data["train_rows"] and data["eval_rows"]
    assert data["entity_intersection"] == []
    for row in data["train_rows"][:1] + data["eval_rows"][:1]:
        assert {"prompt", "label", "gold_author", "claimed_author", "case_id"} <= set(row)


def test_gate_records_scoped_to_partition() -> None:
    """Eval gate records derive only from eval cases (no train-derived rules).

    Note: a work TITLE may appear on both sides with different authors (e.g.
    "Republic" as a Socrates-false case and a Plato-true case) — the
    contamination guard is on (work, author) entity pairs, checked separately.
    """
    data = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=0)
    eval_works = {c.work.lower() for c in data["eval_cases"] if c.label == "false"}
    # every eval gate record references a work that is actually an eval false case
    for rec in data["eval_gate_records"].values():
        assert rec["canonicalTitleEn"].lower() in eval_works
    # and the train gate records reference only train false-case works
    train_works = {c.work.lower() for c in data["train_cases"] if c.label == "false"}
    for rec in data["train_gate_records"].values():
        assert rec["canonicalTitleEn"].lower() in train_works


def test_sealed_hash_is_stable() -> None:
    a = rl_dataset.build_rl_dataset(seed=0)
    b = rl_dataset.build_rl_dataset(seed=0)
    assert a["train_sealed"] == b["train_sealed"]
    c = rl_dataset.build_rl_dataset(seed=7)
    assert a["train_sealed"] != c["train_sealed"] or a["train_cases"] != c["train_cases"]


# --- offline runner -------------------------------------------------------- #


def test_run_rlvr_mock_passes_invariants() -> None:
    from tools import run_rlvr

    ok, detail = run_rlvr._offline_invariants()
    assert ok, detail
    assert all(detail["checks"].values())
    assert detail["trainCases"] > 0 and detail["evalCases"] > 0


def test_ingest_refuses_committed_archive() -> None:
    """The gate must never re-read the durable rlvr-replication/ archive: doing so
    decoupled the live gate from the actual training (a fresh run re-promoting stale
    numbers — the stale-promote bug). The guard refuses such a path structurally."""
    from tools import ingest_rlvr_eval

    archive_path = (
        ROOT / "agi-proof" / "benchmark-results" / "rlvr-replication" / "seed0.adapter-eval.json"
    )
    try:
        ingest_rlvr_eval.ingest(archive_path)
    except SystemExit as e:
        assert ingest_rlvr_eval.ARCHIVE_DIR_NAME in str(e)
    else:
        raise AssertionError("ingest must refuse a file under the committed archive dir")


class _ChatTok:
    """Minimal stub of a chat tokenizer (no transformers/model download)."""

    chat_template = "{{ stub }}"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        assert messages == [{"role": "user", "content": messages[0]["content"]}]
        return f"<|user|>{messages[0]['content']}<|assistant|>"


class _BaseTok:
    """A base/completion tokenizer with no chat template."""

    chat_template = None


def test_code_chat_wrap_templates_for_chat_model() -> None:
    # The code task must wrap raw prompts in the chat template so an instruct model
    # emits an extractable fenced block (the rlvr-code-no-chat-template fix).
    wrapped = code_dataset.chat_wrap(_ChatTok(), "Write a function f(x).")
    assert wrapped == "<|user|>Write a function f(x).<|assistant|>"
    assert "Write a function f(x)." in wrapped


def test_code_chat_wrap_noop_without_template() -> None:
    # A base/completion model has no chat template — chat_wrap must pass through.
    assert code_dataset.chat_wrap(_BaseTok(), "Write a function f(x).") == "Write a function f(x)."


def main() -> int:
    test_reward_is_deterministic()
    test_reward_false_monotone_and_forbidden_negative()
    test_reward_true_monotone()
    test_reward_true_case_denial_penalized()
    test_reward_true_case_affirm_beats_incidental_denial()
    test_reward_invokes_verifier_seam()
    test_reward_is_bounded()
    test_reward_anti_hedging_cap()
    test_make_grpo_reward_routes_by_columns()
    test_dataset_builds_and_splits_contamination_free()
    test_gate_records_scoped_to_partition()
    test_sealed_hash_is_stable()
    test_run_rlvr_mock_passes_invariants()
    test_ingest_refuses_committed_archive()
    test_code_chat_wrap_templates_for_chat_model()
    test_code_chat_wrap_noop_without_template()
    print("test_rlvr: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
