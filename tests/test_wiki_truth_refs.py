# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for open-world truth-ref retrieval (agent.wiki_truth_refs).

No network, no keys, no torch: ``fetch_fn`` is mocked. Locks the fail-closed contract:
a parseable summary yields reference strings; a missing/404/empty/garbage body yields
``[]`` so the caller (the source-contamination bench) abstains instead of treating
"no reference" as "verified".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.wiki_truth_refs import WIKI_SUMMARY_API, fetch_truth_refs  # noqa: E402


def _summary_body(extract: str, description: str = "") -> str:
    payload = {"type": "standard", "title": "X", "extract": extract}
    if description:
        payload["description"] = description
    return json.dumps(payload)


def test_returns_parsed_summary_refs() -> None:
    """A parseable summary with multiple sentences yields up to n distinct refs."""
    body = _summary_body(
        "The Voynich manuscript is an illustrated codex. Its author and language remain unknown."
    )
    refs = fetch_truth_refs("Voynich manuscript", n=2, fetch_fn=lambda url, **k: body)
    assert len(refs) == 2, refs
    assert all(isinstance(r, str) and r.strip() for r in refs)
    assert "unknown" in refs[1].lower()


def test_url_uses_rest_summary_endpoint_and_underscores() -> None:
    """The fetched URL is the REST summary endpoint with the title slugified."""
    seen = {}

    def fake_fetch(url, **k):
        seen["url"] = url
        return _summary_body("Great Wall of China is a series of fortifications.")

    fetch_truth_refs("Great Wall of China", fetch_fn=fake_fetch)
    assert seen["url"] == WIKI_SUMMARY_API + "Great_Wall_of_China"


def test_short_summary_falls_back_to_single_ref() -> None:
    """A one-sentence summary still yields one usable ref (not nothing)."""
    body = _summary_body("Jupiter is the largest planet in the Solar System.")
    refs = fetch_truth_refs("Jupiter", n=2, fetch_fn=lambda url, **k: body)
    assert refs == ["Jupiter is the largest planet in the Solar System."]


def test_description_used_when_no_extract() -> None:
    """When extract is empty, the description is used as the ref text."""
    body = json.dumps({"type": "standard", "extract": "", "description": "Roman general"})
    refs = fetch_truth_refs("Someone", fetch_fn=lambda url, **k: body)
    assert refs == ["Roman general"]


def test_empty_fetch_returns_empty_list_fail_closed() -> None:
    """A None body (fetch failure) fails closed -> []."""
    assert fetch_truth_refs("Voynich manuscript", fetch_fn=lambda url, **k: None) == []


def test_not_found_returns_empty_list_fail_closed() -> None:
    """A REST not_found / missing page fails closed -> []."""
    nf = json.dumps({"type": "https://mediawiki.org/wiki/HyperSwitch/errors/not_found",
                     "title": "Not found."})
    assert fetch_truth_refs("Nonexistent_Page_Zzz", fetch_fn=lambda url, **k: nf) == []
    missing = json.dumps({"missing": True, "title": "Gone"})
    assert fetch_truth_refs("Gone", fetch_fn=lambda url, **k: missing) == []


def test_garbage_body_returns_empty_list_fail_closed() -> None:
    """Unparseable JSON fails closed -> [] (never raises)."""
    assert fetch_truth_refs("X", fetch_fn=lambda url, **k: "<<not json>>") == []


def test_blank_entity_returns_empty_without_fetch() -> None:
    """A blank entity short-circuits to [] and never calls fetch_fn."""
    called = {"n": 0}

    def fake_fetch(url, **k):
        called["n"] += 1
        return _summary_body("x")

    assert fetch_truth_refs("   ", fetch_fn=fake_fetch) == []
    assert called["n"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
