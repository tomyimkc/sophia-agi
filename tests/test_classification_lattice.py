#!/usr/bin/env python3
"""Tests for the classification lattice (#3): BLP + Biba + audited declassification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.security import (  # noqa: E402
    AuditLog,
    Conf,
    DeclassError,
    DeclassRule,
    Integ,
    Label,
    can_flow,
    combine,
    declassify,
)


def test_blp_and_biba_flow_rules() -> None:
    assert not can_flow(Label(Conf.SECRET), Label(Conf.PUBLIC)).allowed        # no write down
    assert can_flow(Label(Conf.PUBLIC), Label(Conf.SECRET)).allowed            # write up (conf) ok
    assert not can_flow(Label(integ=Integ.UNTRUSTED), Label(integ=Integ.AUTHORITATIVE)).allowed  # no write up
    assert can_flow(Label(integ=Integ.AUTHORITATIVE), Label(integ=Integ.UNTRUSTED)).allowed      # write down (integ) ok


def test_need_to_know() -> None:
    assert not can_flow(Label(compartments={"hr"}), Label(compartments=set())).allowed
    assert can_flow(Label(compartments={"hr"}), Label(compartments={"hr", "legal"})).allowed


def test_combine_creep_and_min_integrity() -> None:
    c = combine(Label(Conf.PUBLIC), Label(Conf.SECRET), Label(Conf.INTERNAL))
    assert c.conf == Conf.SECRET                                               # max -> creep
    i = combine(Label(integ=Integ.AUTHORITATIVE), Label(integ=Integ.UNTRUSTED))
    assert i.integ == Integ.UNTRUSTED                                          # min -> least trusted
    assert combine(Label(compartments={"a"}), Label(compartments={"b"})).compartments == frozenset({"a", "b"})


def test_declassify_success_and_audit() -> None:
    log = AuditLog()
    rule = DeclassRule("redacted", Conf.SECRET, Conf.CONFIDENTIAL, predicate=lambda v: "[R]" in str(v))
    new = declassify(Label(Conf.SECRET), "x [R]", rule, approver=lambda *a: True, audit=log)
    assert new is not None and new.conf == Conf.CONFIDENTIAL
    assert log.records("declassify") and log.verify()


def test_declassify_fail_closed() -> None:
    log = AuditLog()
    rule = DeclassRule("redacted", Conf.SECRET, Conf.CONFIDENTIAL, predicate=lambda v: "[R]" in str(v))
    assert declassify(Label(Conf.SECRET), "x [R]", rule, approver=None, audit=log) is None        # no approver
    assert declassify(Label(Conf.SECRET), "raw", rule, approver=lambda *a: True, audit=log) is None  # predicate fails
    assert declassify(Label(Conf.SECRET), "x [R]", rule, approver=lambda *a: "yes", audit=log) is None  # non-True
    # wrong-level rule does not downgrade a more-secret label silently
    assert len(log.records("declassify_denied")) == 3


def test_declassify_is_bounded() -> None:
    try:
        DeclassRule("bad", Conf.PUBLIC, Conf.SECRET, predicate=lambda v: True)   # would RAISE conf
        assert False, "should reject a conf-raising rule"
    except DeclassError:
        pass


def test_audit_tamper_evident() -> None:
    log = AuditLog()
    log.append("declassify", {"a": 1})
    log.append("declassify_denied", {"b": 2})
    assert log.verify() is True
    log.entries[0].detail["a"] = 999
    assert log.verify() is False


def test_demo_invariants_hold() -> None:
    from tools.run_classification_lattice import run_demo

    res = run_demo()
    assert res["ok"] is True, [k for k, v in res["invariants"].items() if not v]


def main() -> int:
    test_blp_and_biba_flow_rules()
    test_need_to_know()
    test_combine_creep_and_min_integrity()
    test_declassify_success_and_audit()
    test_declassify_fail_closed()
    test_declassify_is_bounded()
    test_audit_tamper_evident()
    test_demo_invariants_hold()
    print("test_classification_lattice: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
