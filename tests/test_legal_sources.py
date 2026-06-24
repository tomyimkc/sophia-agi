#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/legal_sources/* — offline, with injected fake fetchers.

No test touches the real network: every source/resolver is driven by a fake
``fetch`` callable, so HKLII / e-Legislation behavior is simulated deterministically.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifiers as v  # noqa: E402
from agent.legal_sources.cache import ResolutionCache  # noqa: E402
from agent.legal_sources.elegislation import ELegislationSource  # noqa: E402
from agent.legal_sources.hklii import HKLIISource  # noqa: E402
from agent.legal_sources.registry import LegalResolver  # noqa: E402


def make_fetch(responses: dict, *, default=(404, "")):
    """fetch(url, timeout) -> canned (status, body) by substring match on url."""

    def _fetch(url, timeout=20):
        for needle, resp in responses.items():
            if needle in url:
                return resp
        return default

    return _fetch


def _tmp(tmp_path=None):
    """A temp dir under pytest (fixture) or as a plain script (mkdtemp)."""
    return Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())


# --------------------------- source-level behavior --------------------------- #

def test_elegislation_verified_and_not_found() -> None:
    src = ELegislationSource(base="https://example.test/cap")
    ok = make_fetch({"cap614": (200, "<title>Legislation Publication Ordinance</title>")})
    r = src.resolve("Cap. 614", fetch=ok)
    assert r.verified and r.status == "verified" and "example.test" in r.url

    missing = make_fetch({}, default=(404, ""))
    r2 = src.resolve("Cap. 999", fetch=missing)
    assert not r2.verified and r2.status == "not_found"


def test_elegislation_fetch_error_fails_closed() -> None:
    def boom(url, timeout=20):
        raise OSError("network down")

    r = ELegislationSource().resolve("Cap. 614", fetch=boom)
    assert r.verified is False and r.status == "error"


def test_hklii_verified_and_not_found() -> None:
    src = HKLIISource(base="https://example.test")
    body = "<a>Yu Hon Tong Thomas v Centaline [2025] HKCFI 808</a>"
    r = src.resolve("[2025] HKCFI 808", fetch=make_fetch({"search": (200, body)}))
    assert r.verified and r.court == "HKCFI" and r.year == "2025"

    r2 = src.resolve("[2024] HKCFI 9999", fetch=make_fetch({"search": (200, "no results")}))
    assert not r2.verified and r2.status == "not_found"


def test_hklii_requires_full_citation_not_just_year() -> None:
    # a stray year on the page must NOT count as a match
    src = HKLIISource(base="https://example.test")
    r = src.resolve("[2025] HKCFI 808", fetch=make_fetch({"search": (200, "decided in 2025, HKCFI generally")}))
    assert r.verified is False


# ------------------------------ cache behavior ------------------------------ #

def test_cache_roundtrip_and_miss(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    cache = ResolutionCache(path=tmp_path / "c.json")
    assert cache.get("[2025] HKCFI 808") is None  # fail-closed miss
    src = HKLIISource(base="https://example.test")
    res = src.resolve("[2025] HKCFI 808", fetch=make_fetch({"search": (200, "[2025] HKCFI 808")}))
    cache.put(res)
    cache.save()
    again = ResolutionCache(path=tmp_path / "c.json")
    hit = again.get("[2025] HKCFI 808")
    assert hit is not None and hit.verified


# ----------------------------- registry routing ----------------------------- #

def test_registry_cache_mode_is_offline(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    r = LegalResolver(mode="cache", cache=ResolutionCache(path=tmp_path / "c.json"),
                      fetch=make_fetch({"search": (200, "[2025] HKCFI 808")}))
    out = r.resolve("[2025] HKCFI 808")
    assert out.verified is False and out.status == "offline"  # cache miss, no network


def test_registry_live_mode_resolves_and_caches(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    cache = ResolutionCache(path=tmp_path / "c.json")
    fetch = make_fetch({"search": (200, "[2025] HKCFI 808"), "cap614": (200, "<title>x</title>")})
    r = LegalResolver(mode="live", cache=cache, fetch=fetch)
    assert r.resolve("[2025] HKCFI 808").verified is True
    assert r.resolve("Cap. 614").verified is True
    # second call for the case is served from cache (persisted)
    assert ResolutionCache(path=tmp_path / "c.json").get("[2025] HKCFI 808").verified


def test_registry_live_fabricated_fails(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    r = LegalResolver(mode="live", cache=ResolutionCache(path=tmp_path / "c.json"),
                      fetch=make_fetch({"search": (200, "no results")}))
    assert r.resolve("[2024] HKCFI 9999").verified is False


# -------------------------- verifier integration ---------------------------- #

def test_verifier_with_live_resolver_accepts_real_outside_register(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    # citation NOT in the static register, but the live source verifies it
    fetch = make_fetch({"search": (200, "Chan v X [2099] HKCFI 1")})
    resolver = LegalResolver(mode="live", cache=ResolutionCache(path=tmp_path / "c.json"), fetch=fetch).resolve
    ver = v.legal_citation_exists(set(), resolver=resolver)  # empty static register
    assert ver("See Chan v X [2099] HKCFI 1.", None, {})["passed"] is True


def test_verifier_resolver_fail_closed_on_error(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    def boom(citation):
        raise RuntimeError("resolver broken")

    ver = v.legal_citation_exists(set(), resolver=boom)
    assert ver("See [2099] HKCFI 1.", None, {})["passed"] is False  # broken resolver never passes


# ----------------------- federated sources (UK / US) ------------------------ #

def test_tna_verified_and_not_found() -> None:
    from agent.legal_sources.tna import TNASource

    src = TNASource(base="https://example.test")
    assert src.can_resolve("[2025] EWHC 1383 (Admin)") is True
    assert src.can_resolve("[2025] HKCFI 808") is False  # HK routes elsewhere
    body = "Ayinde v Haringey [2025] EWHC 1383 (Admin)"
    r = src.resolve("[2025] EWHC 1383 (Admin)", fetch=make_fetch({"search": (200, body)}))
    assert r.verified and r.court == "EWHC"
    r2 = src.resolve("[2023] EWHC 8888", fetch=make_fetch({"search": (200, "no results")}))
    assert not r2.verified and r2.status == "not_found"


def test_courtlistener_verified_and_not_found() -> None:
    from agent.legal_sources.courtlistener import CourtListenerSource

    src = CourtListenerSource(base="https://example.test")
    assert src.can_resolve("925 F.3d 1339") is True
    assert src.can_resolve("[2025] HKCFI 808") is False
    body = '{"count":1,"results":[{"citation":["925 F.3d 1339"]}]}'
    r = src.resolve("925 F.3d 1339", fetch=make_fetch({"courtlistener": (200, body), "search": (200, body)}))
    assert r.verified
    r2 = src.resolve("999 U.S. 123", fetch=make_fetch({"search": (200, '{"count":0,"results":[]}')}))
    assert not r2.verified and r2.status == "not_found"


def test_registry_routes_each_jurisdiction(tmp_path=None) -> None:
    tmp_path = _tmp(tmp_path)
    fetch = make_fetch({
        "elegislation": (200, "<title>ok</title>"),
        "hklii": (200, "[2025] HKCFI 808"),
        "nationalarchives": (200, "[2025] EWHC 1383"),
        "courtlistener": (200, '{"results":[{"citation":["925 F.3d 1339"]}]}'),
    })
    r = LegalResolver(mode="live", cache=ResolutionCache(path=tmp_path / "c.json"), fetch=fetch)
    assert r.resolve("Cap. 614").provider == "elegislation"
    assert r.resolve("[2025] HKCFI 808").provider == "hklii"
    assert r.resolve("[2025] EWHC 1383 (Admin)").provider == "tna"
    assert r.resolve("925 F.3d 1339").provider == "courtlistener"


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_legal_sources: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
