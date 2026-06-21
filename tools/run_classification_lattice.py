#!/usr/bin/env python3
"""Classification lattice (#3): Bell-LaPadula + Biba + bounded, audited declassification.

Falsifiable invariants (deterministic, offline): no-write-down (BLP), no-write-up
(Biba), need-to-know, label creep from combination, declassification relieving creep
under predicate+approval, fail-closed refusal, bounded rules, and a tamper-evident
audit chain. Exits non-zero if any invariant fails.

    python tools/run_classification_lattice.py [--json]
"""

from __future__ import annotations

import argparse
import json
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


def run_demo() -> dict:
    log = AuditLog()

    blp = (not can_flow(Label(Conf.SECRET), Label(Conf.PUBLIC)).allowed
           and can_flow(Label(Conf.PUBLIC), Label(Conf.SECRET)).allowed)
    biba = (not can_flow(Label(integ=Integ.UNTRUSTED), Label(integ=Integ.AUTHORITATIVE)).allowed
            and can_flow(Label(integ=Integ.AUTHORITATIVE), Label(integ=Integ.UNTRUSTED)).allowed)
    ntk = (not can_flow(Label(compartments={"hr"}), Label(compartments=set())).allowed
           and can_flow(Label(compartments={"hr"}), Label(compartments={"hr", "legal"})).allowed)
    creep = (combine(Label(Conf.PUBLIC), Label(Conf.SECRET)).conf == Conf.SECRET
             and combine(Label(integ=Integ.AUTHORITATIVE), Label(integ=Integ.UNTRUSTED)).integ == Integ.UNTRUSTED)
    # Biba catches what BLP/confidentiality alone allows: PUBLIC->PUBLIC write is fine
    # for confidentiality, but untrusted->authoritative must be blocked.
    biba_adds = not can_flow(Label(Conf.PUBLIC, Integ.UNTRUSTED), Label(Conf.PUBLIC, Integ.AUTHORITATIVE)).allowed

    # Declassification relieves creep: a crept-to-SECRET item can reach a CONFIDENTIAL
    # sink only after a redaction-gated, approved, audited downgrade.
    rule = DeclassRule("redacted", Conf.SECRET, Conf.CONFIDENTIAL,
                       predicate=lambda v: "[REDACTED]" in str(v), justification="pii redacted")
    sink = Label(Conf.CONFIDENTIAL)
    crept = combine(Label(Conf.PUBLIC), Label(Conf.SECRET))      # -> SECRET
    before = can_flow(crept, sink).allowed                       # blocked (write down)
    new = declassify(crept, "content [REDACTED]", rule, approver=lambda *a: True, audit=log)
    after = new is not None and can_flow(new, sink).allowed
    denied_no_approver = declassify(crept, "content [REDACTED]", rule, approver=None, audit=log) is None
    denied_predicate = declassify(crept, "raw unredacted secret", rule, approver=lambda *a: True, audit=log) is None

    try:
        DeclassRule("bad", Conf.PUBLIC, Conf.SECRET, predicate=lambda v: True)
        bounded = False
    except DeclassError:
        bounded = True

    audit_intact = log.verify()
    # in-place edit + reorder are caught by the chain alone
    t1 = AuditLog(); t1.append("declassify", {"x": 1}); t1.append("declassify", {"y": 2})
    t1.entries[0].detail["x"] = 999
    edit_caught = not t1.verify()
    t2 = AuditLog(); t2.append("a", {"i": 0}); t2.append("b", {"i": 1})
    t2.entries[0], t2.entries[1] = t2.entries[1], t2.entries[0]
    reorder_caught = not t2.verify()
    # tail truncation (drop the incriminating last record): MISSED without an anchor,
    # CAUGHT with the persisted (count, head) anchor.
    t3 = AuditLog(); t3.append("a", {}); t3.append("declassify_denied", {})
    a_count, a_head = t3.count, t3.head()
    del t3.entries[-1]
    trunc_missed_unanchored = t3.verify()
    trunc_caught_anchored = not t3.verify(expected_count=a_count, expected_head=a_head)
    # forged append / rebuild: caught with the anchor
    t4 = AuditLog(); t4.append("a", {})
    f_count, f_head = t4.count, t4.head()
    t4.append("forged", {"z": 1})
    forged_caught_anchored = not t4.verify(expected_count=f_count, expected_head=f_head)

    invariants = {
        "blp_no_write_down": blp,
        "biba_no_write_up": biba,
        "need_to_know_enforced": ntk,
        "combine_causes_label_creep": creep,
        "biba_adds_protection_blp_lacks": biba_adds,
        "declassification_relieves_creep": (before is False) and (after is True),
        "declassification_fail_closed": denied_no_approver and denied_predicate,
        "declassification_is_bounded": bounded,
        "audit_chain_intact": audit_intact,
        "audit_catches_edit_and_reorder": edit_caught and reorder_caught,
        "audit_anchor_catches_truncation_and_forge": trunc_caught_anchored and forged_caught_anchored,
        "audit_unanchored_misses_tail_truncation": trunc_missed_unanchored,   # honest, documented limitation
        "all_declassifications_audited": len(log.records("declassify")) == 1 and len(log.records("declassify_denied")) == 2,
    }
    return {"invariants": invariants, "ok": all(invariants.values()), "auditEntries": len(log.entries)}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = run_demo()
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1
    print("Classification lattice — Bell-LaPadula + Biba + audited declassification (#3)")
    print("=" * 76)
    print(f"\naudit entries: {r['auditEntries']}\n")
    for k, v in r["invariants"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("\n" + ("ALL INVARIANTS HOLD" if r["ok"] else "INVARIANT FAILURE"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
