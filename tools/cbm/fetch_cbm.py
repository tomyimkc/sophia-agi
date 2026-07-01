#!/usr/bin/env python3
"""Phase-1 supply-chain pin + verify for the codebase-memory-mcp binary.

The MCP binary is a THIRD-PARTY tool (DeusData/codebase-memory-mcp) that reads code
snippets live off disk, so before it is ever wired into `.mcp.json` (through
``tools/cbm/index_guard.py``, the locked-tree preflight) its exact bytes must be
pinned and verified. This module is the pin/verify half; ``index_guard.py`` is the
locked-tree half. Together: run ONLY a byte-pinned binary, and ONLY on a locked tree.

``cbm.pin.json`` records ``{repo, ref, binary_rel, sha256}``. An EMPTY ``sha256`` means
NOT yet pinned — :func:`verify` refuses, so indexing stays disabled by default.

  # once, after you clone+build the pinned ref yourself and inspect it:
  python tools/cbm/fetch_cbm.py --init  path/to/codebase-memory-mcp
  # before every run (also done by the .mcp.json wrapper — see the Phase-1 plan):
  python tools/cbm/fetch_cbm.py --verify path/to/codebase-memory-mcp   # exit 1 on mismatch

Fetch/build is deliberately NOT automated here: building a third-party binary from
source is environment-specific, and a human should do it once against the pinned
``ref`` and inspect it before ``--init``. That keeps the trust boundary explicit.
canClaimAGI:false.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN_PATH = ROOT / "cbm.pin.json"


def load_pin(path: Path = PIN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(binary: Path) -> str:
    h = hashlib.sha256()
    with open(binary, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(binary: Path, pin: dict) -> "tuple[bool, str]":
    """Return (ok, message). ok iff the pin is initialized AND sha256(binary) matches."""
    expected = (pin.get("sha256") or "").strip().lower()
    if not expected:
        return False, ("pin NOT initialized (cbm.pin.json sha256 is empty) — indexing stays "
                       "disabled; run --init after fetching+building+inspecting the pinned ref.")
    if not binary.exists():
        return False, f"binary not found: {binary}"
    actual = sha256_file(binary)
    if actual != expected:
        return False, f"sha256 MISMATCH — expected {expected[:12]}…, got {actual[:12]}… (refusing)."
    return True, f"verified sha256 {actual[:12]}… against pin (ref {pin.get('ref', '?')})."


def main(argv: "list[str]") -> int:
    ap = argparse.ArgumentParser(description="pin + verify the codebase-memory-mcp binary")
    ap.add_argument("--verify", metavar="BINARY",
                    help="recompute sha256(BINARY) and check it against cbm.pin.json (exit 1 on mismatch)")
    ap.add_argument("--init", metavar="BINARY",
                    help="record sha256(BINARY) into the pin (first-time pin; do this only after auditing the ref)")
    ap.add_argument("--print", action="store_true", help="print the current pin and exit")
    ap.add_argument("--pin", default=str(PIN_PATH), help="path to the pin file (default: repo-root cbm.pin.json)")
    args = ap.parse_args(argv)

    pin_path = Path(args.pin)
    pin = load_pin(pin_path)

    if args.print:
        print(json.dumps(pin, indent=2, sort_keys=True))
        return 0
    if args.init:
        b = Path(args.init)
        if not b.exists():
            sys.stderr.write(f"[cbm-pin] binary not found: {b}\n")
            return 2
        pin["sha256"] = sha256_file(b)
        pin_path.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[cbm-pin] pinned sha256 {pin['sha256'][:12]}… for ref {pin.get('ref', '?')}")
        return 0
    if args.verify:
        ok, msg = verify(Path(args.verify), pin)
        sys.stderr.write(f"[cbm-pin] {msg}\n")
        return 0 if ok else 1

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
