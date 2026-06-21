#!/usr/bin/env python3
"""Tests for the LoRA leakage guard (#7).

A confidential/PII/secret example must never reach the training export; ordinary
historical prose must NOT be flagged (0 false positives on the real corpus); and
the canary harness measures regurgitation correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.training_safety import (  # noqa: E402
    canary_extraction_rate,
    filter_examples,
    is_safe_to_train,
    make_canary,
    unsafe_reasons,
)


def _ex(content, meta=None):
    return {"messages": [{"role": "assistant", "content": content}], "metadata": meta or {}}


def test_flags_confidential_pii_and_secrets() -> None:
    assert unsafe_reasons(_ex("hi", {"classification": "confidential"}))
    assert unsafe_reasons(_ex("hi", {"doNotTrain": True}))
    assert unsafe_reasons(_ex("email me at alice@example.com"))
    assert unsafe_reasons(_ex("ssn 123-45-6789"))
    assert unsafe_reasons(_ex("api_key=sk-abc123def456"))


def test_ordinary_prose_is_safe() -> None:
    # historical/philosophical prose with names + years must NOT be flagged
    assert is_safe_to_train(_ex("Laozi is traditionally associated with the Dao De Jing (c. 400 BCE)."))
    assert is_safe_to_train(_ex("Kant wrote the Critique of Pure Reason in 1781."))


def test_real_corpus_zero_false_positives() -> None:
    import glob
    import json

    exs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(str(ROOT / "training" / "examples" / "*.json")))]
    assert exs, "no training examples found"
    r = filter_examples(exs)
    assert r["nDropped"] == 0, f"unexpected drops: {r['reasonsHistogram']}"


def test_filter_drops_unsafe_keeps_safe() -> None:
    exs = [_ex("clean text about philosophy"), _ex("secret", {"classification": "secret"}), _ex("api_key=XYZ")]
    r = filter_examples(exs)
    assert r["nSafe"] == 1 and r["nDropped"] == 2


def test_scans_metadata_and_alt_fields() -> None:
    # PII hidden in metadata free-text is caught (not just messages)
    assert unsafe_reasons({"messages": [{"content": "hi"}], "metadata": {"source": "x SSN 123-45-6789"}})
    # alternate row schemas (DPO chosen/rejected, prompt/completion, instruction/io) are scanned
    assert unsafe_reasons({"prompt": "q", "chosen": "api_key=sk-live-ABCDEF", "rejected": "r"})
    assert unsafe_reasons({"instruction": "do", "input": "email me at bob@example.com", "output": "ok"})


def test_alt_classification_keys_and_string_flags() -> None:
    assert unsafe_reasons({"messages": [{"content": "q"}], "metadata": {"sensitivity": "confidential"}})
    assert unsafe_reasons({"messages": [{"content": "q"}], "metadata": {"doNotTrain": "true"}})   # stringified bool
    assert unsafe_reasons({"messages": [{"content": "q"}], "metadata": {"pii": True}})
    # a benign classification value is NOT flagged
    assert not unsafe_reasons({"messages": [{"content": "q"}], "metadata": {"classification": "public"}})


def test_canary_harness() -> None:
    c = make_canary("seed1")
    assert c.startswith("CANARY-") and c == make_canary("seed1")    # deterministic
    assert canary_extraction_rate(["nothing here"], [c]) == 0.0      # not regurgitated
    assert canary_extraction_rate([f"... {c} ..."], [c]) == 1.0      # regurgitated


def test_secret_value_match() -> None:
    secret = make_canary("planted")
    assert unsafe_reasons(_ex(f"the value is {secret}"), secrets=[secret])


def main() -> int:
    test_flags_confidential_pii_and_secrets()
    test_ordinary_prose_is_safe()
    test_real_corpus_zero_false_positives()
    test_scans_metadata_and_alt_fields()
    test_alt_classification_keys_and_string_flags()
    test_filter_drops_unsafe_keeps_safe()
    test_canary_harness()
    test_secret_value_match()
    print("test_training_safety: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
