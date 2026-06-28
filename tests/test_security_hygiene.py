# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-2 model/prompt hygiene layer: secret patterns, canaries,
corpus scrubbing, and the prompt-hygiene CLI gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import canary, corpus_scrub, secret_patterns
from tools import check_prompt_hygiene


# ── secret_patterns ───────────────────────────────────────────────────────────
def test_find_secrets_catches_provider_keys():
    text = "key=sk-ABCDEFGHIJKLMNOPQRSTUVWX and hf_ABCDEFGHIJKLMNOPQRSTUV plus AKIA0123456789ABCDEF"
    kinds = {f["kind"] for f in secret_patterns.find_secrets(text)}
    assert "openai_key" in kinds
    assert "hf_token" in kinds
    assert "aws_access_key" in kinds


def test_find_internal_and_pii():
    assert secret_patterns.find_internal("connect to 10.0.0.5 now")
    assert secret_patterns.find_internal("/home/alice/.env")
    assert secret_patterns.find_pii("mail me at a.b@example.com")
    assert secret_patterns.find_pii("ssn 123-45-6789")


def test_clean_prose_has_no_false_positives():
    text = "The quick brown fox discusses prompt injection and system prompt leakage."
    assert not secret_patterns.find_secrets(text)
    assert not secret_patterns.find_pii(text)


def test_redact_replaces_with_typed_token():
    out = secret_patterns.redact("token sk-ABCDEFGHIJKLMNOPQRSTUVWX here")
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in out
    assert "[REDACTED:openai_key]" in out


# ── canary ────────────────────────────────────────────────────────────────────
def test_canary_is_deterministic_and_seed_scoped():
    a = canary.make_canary("system_prompt", seed="s3cret-seed")
    b = canary.make_canary("system_prompt", seed="s3cret-seed")
    c = canary.make_canary("train_mix", seed="s3cret-seed")
    d = canary.make_canary("system_prompt", seed="other-seed")
    assert a == b           # deterministic
    assert a != c != d      # label- and seed-scoped
    assert a.startswith(canary.CANARY_PREFIX)


def test_canary_requires_seed():
    import pytest
    with pytest.raises(RuntimeError):
        canary.make_canary("x", seed="")


def test_scan_and_match_known_canary():
    tok = canary.make_canary("system_prompt", seed="seed")
    blob = f"...leaked instructions {tok} trailing..."
    assert canary.scan_for_canaries(blob) == [tok]
    assert canary.contains_known_canary(blob, [tok]) == [tok]
    assert canary.contains_known_canary("nothing here", [tok]) == []


# ── corpus_scrub ──────────────────────────────────────────────────────────────
def test_scrub_record_redacts_and_reports():
    rec = {"text": "reach me a@b.com with key sk-ABCDEFGHIJKLMNOPQRSTUVWX", "label": "x"}
    out, rep = corpus_scrub.scrub_record(rec)
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in out["text"]
    assert "a@b.com" not in out["text"]
    assert rep["secrets"] == 1 and rep["pii"] == 1
    assert out["label"] == "x"  # non-text keys untouched


def test_scrub_corpus_drops_canary_records():
    tok = canary.make_canary("train_mix", seed="seed")
    records = [
        {"text": "clean philosophy example"},
        {"text": f"this carries a canary {tok}"},
        {"text": "email me at x@y.org"},
    ]
    clean, summary = corpus_scrub.scrub_corpus(records)
    assert summary["out"] == 2                 # canary record dropped
    assert summary["dropped_canary_records"] == 1
    assert summary["leaked_canaries"] == [tok]
    assert summary["ok"] is False              # a leak is a hard fail
    assert all(tok not in r["text"] for r in clean)


# ── prompt hygiene CLI gate ───────────────────────────────────────────────────
def test_prompt_hygiene_flags_a_planted_secret(tmp_path):
    bad = tmp_path / "evil_prompt.py"
    bad.write_text('SYSTEM = "you are sophia. internal key sk-ABCDEFGHIJKLMNOPQRSTUVWX"\n')
    # run() skips untracked tmp files, so scan directly to exercise detection:
    findings = check_prompt_hygiene.scan_file(bad)
    assert any(f["kind"].startswith("secret:") for f in findings)
    # the finding must NOT echo the raw secret (clear-text-logging safe):
    assert all("ABCDEFGHIJKLMNOPQRSTUVWX" not in str(f) for f in findings)


def test_prompt_hygiene_passes_on_clean_string(tmp_path):
    good = tmp_path / "ok_prompt.py"
    good.write_text('SYSTEM = "You are Sophia, a verifier-gated epistemic assistant."\n')
    assert check_prompt_hygiene.scan_file(good) == []


def test_repo_public_prompts_are_clean():
    """The real repo's public prompt surfaces must pass the gate."""
    report = check_prompt_hygiene.run(check_prompt_hygiene.DEFAULT_GLOBS)
    assert report["clean"], f"prompt hygiene findings: {report['files_with_findings']}"
