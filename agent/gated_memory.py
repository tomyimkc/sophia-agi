# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated persistent memory — a durable store where a write survives only if it
clears the machine gate.

Inspiration (honest framing). The "ECC" agent-memory pattern persists *everything* an agent
emits into a SQLite database via lifecycle hooks, so a sibling session can read it back. That
makes memory durable but also durable-ly wrong: a hallucination written once is recalled
forever. Sophia fuses that durability with its existing verifier (``agent.gate.check_response``):
a ``remember`` call persists into the readable ``accepted`` table **only if** the gate returns
no violations; otherwise the text lands in a separate ``quarantine`` table that ``recall`` never
reads. The trust boundary Sophia already enforces per-answer is here made durable across
sessions and processes.

Limits (not a truth oracle). The gate is a *filter*, not an oracle of truth: it catches the
machine-checkable failures it knows about (false equalities, forbidden attributions, invalid
citations, ...). A false statement with no detectable violation can still be stored. This bounds
persisted hallucination to "passed the gate", it does not eliminate it. ``canClaimAGI`` stays
false. Standard-library only; deterministic given an injected verifier; offline; no network.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS accepted (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        source TEXT,
        text TEXT NOT NULL,
        question TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS quarantine (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        source TEXT,
        text TEXT NOT NULL,
        question TEXT,
        reasons TEXT
    )""",
)


def _real_verifier(mode: str):
    """Return a verifier closure backed by the real gate. Imported lazily so the store can be
    used (with an injected verifier) without pulling in the full gate stack."""

    def _verify(text: str, question: "str | None") -> "tuple[bool, list[str]]":
        from agent.gate import check_response

        result = check_response(
            text,
            mode=mode,
            question=question or text,
            route_claims=True,
        )
        violations = list(result.get("violations") or [])
        return (len(violations) == 0, violations)

    return _verify


class GatedMemory:
    """Durable agent memory gated by a machine verifier.

    A custom ``verifier(text, question) -> (clean: bool, reasons: list[str])`` may be injected
    (tests use a deterministic stub so no model is needed). ``verifier=None`` uses the real gate.
    """

    def __init__(self, db_path: str = ":memory:", *, verifier=None, mode: str = "advisor") -> None:
        self.db_path = db_path
        self.mode = mode
        self._verify = verifier if verifier is not None else _real_verifier(mode)
        # check_same_thread=False keeps it usable from a worker thread; access stays single-writer.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    # -- write side (the trust boundary, made durable) -------------------------
    def remember(self, text: str, *, question: "str | None" = None, source: "str | None" = None) -> dict:
        """Run the verifier; persist into ``accepted`` iff clean, else ``quarantine``.

        Returns ``{"stored": True, "verdict": "accepted"}`` on a clean write, or
        ``{"stored": False, "verdict": "held", "reasons": [...]}`` when the gate flags it."""
        clean, reasons = self._verify(text, question)
        ts = time.time()
        if clean:
            self._conn.execute(
                "INSERT INTO accepted (ts, source, text, question) VALUES (?, ?, ?, ?)",
                (ts, source, text, question),
            )
            self._conn.commit()
            return {"stored": True, "verdict": "accepted"}
        reason_list = list(reasons or [])
        self._conn.execute(
            "INSERT INTO quarantine (ts, source, text, question, reasons) VALUES (?, ?, ?, ?, ?)",
            (ts, source, text, question, "\n".join(reason_list)),
        )
        self._conn.commit()
        return {"stored": False, "verdict": "held", "reasons": reason_list}

    # -- read side (what a sibling session sees) -------------------------------
    def recall(self, query: "str | None" = None, limit: int = 100) -> "list[dict]":
        """Return ONLY rows from ``accepted`` (never quarantine), optionally LIKE-filtered on text."""
        if query:
            cur = self._conn.execute(
                "SELECT id, ts, source, text, question FROM accepted "
                "WHERE text LIKE ? ORDER BY id LIMIT ?",
                (f"%{query}%", limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT id, ts, source, text, question FROM accepted ORDER BY id LIMIT ?",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]

    # -- audit surface (held rows; NEVER readable context) ---------------------
    def quarantined(self) -> "list[dict]":
        """Audit-only view of held rows. Reasons are returned as a list. Never used as context."""
        cur = self._conn.execute(
            "SELECT id, ts, source, text, question, reasons FROM quarantine ORDER BY id"
        )
        rows = []
        for r in cur.fetchall():
            d = dict(r)
            d["reasons"] = d["reasons"].split("\n") if d["reasons"] else []
            rows.append(d)
        return rows

    def audit(self) -> dict:
        """Reconciliation totals: ``{"accepted": n, "held": m}``."""
        n = self._conn.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
        m = self._conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
        return {"accepted": n, "held": m}

    def close(self) -> None:
        self._conn.close()


# -- deterministic stub verifiers (for invariants/tests; no model needed) ------
def _stub_clean(text: str, question: "str | None") -> "tuple[bool, list[str]]":
    return (True, [])


def _stub_flag(text: str, question: "str | None") -> "tuple[bool, list[str]]":
    return (False, ["stub: flagged for test"])


def offline_invariants() -> "tuple[bool, dict]":
    """Falsifiable invariants, proven with an INJECTED stub verifier (deterministic, no model),
    plus ONE check that exercises the REAL gate."""
    import tempfile

    checks: dict[str, bool] = {}

    # 1) Clean claim is stored and recalled (stub-clean verifier).
    m = GatedMemory(":memory:", verifier=_stub_clean)
    r = m.remember("Water boils at 100 C at sea level.", question="At what temp does water boil?")
    checks["clean_stored"] = r["stored"] and r["verdict"] == "accepted"
    rows = m.recall()
    checks["clean_recalled"] = len(rows) == 1 and "Water boils" in rows[0]["text"]

    # 2) Flagged claim is quarantined, NOT recalled, reasons retained (stub-flag verifier).
    mf = GatedMemory(":memory:", verifier=_stub_flag)
    rf = mf.remember("A false claim.", question="q?")
    checks["flag_held"] = (not rf["stored"]) and rf["verdict"] == "held"
    checks["flag_reasons_returned"] = rf.get("reasons") == ["stub: flagged for test"]
    checks["flag_not_recalled"] = mf.recall() == []
    held = mf.quarantined()
    checks["flag_reasons_retained"] = len(held) == 1 and held[0]["reasons"] == ["stub: flagged for test"]

    # 3) recall never returns a held row (mixed store).
    mm = GatedMemory(":memory:", verifier=lambda t, q: (("BAD" not in t), ["mixed"] if "BAD" in t else []))
    mm.remember("good one", question="q")
    mm.remember("BAD one", question="q")
    recalled = mm.recall()
    checks["recall_excludes_held"] = all("BAD" not in row["text"] for row in recalled) and len(recalled) == 1

    # 4) Audit totals reconcile.
    a = mm.audit()
    checks["audit_reconciles"] = a == {"accepted": 1, "held": 1}

    # 5) Cross-session persistence: a brand-new instance on the SAME db file sees the accept.
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "gated.db")
        first = GatedMemory(path, verifier=_stub_clean)
        first.remember("Durable fact across sessions.", question="q", source="sess-1")
        first.close()
        second = GatedMemory(path, verifier=_stub_flag)  # different verifier, same data
        sib = second.recall()
        checks["cross_session_persistence"] = (
            len(sib) == 1 and "Durable fact" in sib[0]["text"] and sib[0]["source"] == "sess-1"
        )
        second.close()

    # 6) REAL gate (no stub): a forbidden attribution is held; the corrected answer is accepted.
    try:
        rg = GatedMemory(":memory:")  # verifier=None -> real gate
        bad = rg.remember(
            "Confucius wrote the Dao De Jing.",
            question="Did Confucius write the Dao De Jing?",
        )
        good = rg.remember(
            "No, Confucius did not write the Dao De Jing; it is a Daoist text attributed to "
            "Laozi. This is a common Confucian misconception.",
            question="Did Confucius write the Dao De Jing?",
        )
        checks["real_gate_holds_bad"] = (not bad["stored"]) and bad["verdict"] == "held"
        checks["real_gate_accepts_good"] = good["stored"] and good["verdict"] == "accepted"
        # The held hallucination must never be recallable as context.
        checks["real_gate_bad_not_recalled"] = all(
            "Confucius wrote" not in row["text"] for row in rg.recall()
        )
    except Exception as exc:  # noqa: BLE001 - surface as a failed check, not a crash
        checks["real_gate_holds_bad"] = False
        checks["real_gate_accepts_good"] = False
        checks["real_gate_bad_not_recalled"] = False
        checks["real_gate_error"] = False
        return (False, {"checks": checks, "error": repr(exc)})

    ok = all(checks.values())
    return (ok, {"checks": checks})


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    ok, detail = offline_invariants()
    for name, passed in detail["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if "error" in detail:
        print(f"error: {detail['error']}")
    print(f"{'PASS' if ok else 'FAIL'} gated_memory offline_invariants")
    sys.exit(0 if ok else 1)
