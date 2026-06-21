#!/usr/bin/env python3
"""Refresh the HK legal-authority snapshot from primary sources (strategy A).

Runs the live resolver over a list of citations and (a) warms the resolution
cache and (b) updates ``data/legal_authorities.json`` with the verified ones,
each stamped with its source URL and ``retrievedAt``. After a refresh, the
existing ``legal_citation_exists`` verifier works offline against a real,
provenance-stamped snapshot.

    # re-verify every citation already in the snapshot
    SOPHIA_LEGAL_SOURCE=live python tools/refresh_legal_authorities.py --existing

    # add / verify specific citations
    SOPHIA_LEGAL_SOURCE=live python tools/refresh_legal_authorities.py \
        --citations "[2025] HKCFI 808" "Cap. 614"

    # offline dry-run (cache-only): shows what is/ isn't already verified
    python tools/refresh_legal_authorities.py --existing

Be polite to HKLII / e-Legislation: this hits the network once per uncached
citation. Confirm each source's ToS / robots.txt before bulk runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_citations import normalize_citation  # noqa: E402
from agent.legal_sources import make_resolver, resolver_mode  # noqa: E402

REGISTER = ROOT / "data" / "legal_authorities.json"


def _load_register() -> dict:
    return json.loads(REGISTER.read_text(encoding="utf-8"))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh HK legal-authority snapshot from primary sources.")
    ap.add_argument("--citations", nargs="*", default=[], help="citations to verify/add")
    ap.add_argument("--existing", action="store_true", help="re-verify citations already in the snapshot")
    ap.add_argument("--write", action="store_true", help="write verified results back into the snapshot")
    args = ap.parse_args(argv)

    register = _load_register()
    targets = list(args.citations)
    if args.existing or not targets:
        targets += [a.get("citation", "") for a in register.get("authorities", []) if a.get("citation")]
    targets = sorted({normalize_citation(c) for c in targets if c})

    mode = resolver_mode()
    resolver = make_resolver()
    if resolver is None:
        print("SOPHIA_LEGAL_SOURCE=off — nothing to do (static register only).")
        return 0
    print(f"Resolver mode: {mode}  |  citations: {len(targets)}\n")

    by_cite = {normalize_citation(a["citation"]): a for a in register.get("authorities", []) if a.get("citation")}
    verified_count = 0
    for cite in targets:
        res = resolver(cite)
        flag = "✓" if res.verified else "✗"
        print(f" {flag} {cite:<24} {res.status:<10} {res.provider}"
              + (f"  {res.url}" if res.url else ""))
        if res.verified:
            verified_count += 1
            if args.write:
                entry = by_cite.setdefault(cite, {"citation": cite})
                entry.update({"verified": True, "url": res.url, "retrievedAt": res.retrievedAt})
                if res.title and not entry.get("name"):
                    entry["name"] = res.title

    print(f"\nVerified {verified_count}/{len(targets)}.")
    if args.write:
        register["authorities"] = list(by_cite.values())
        REGISTER.write_text(json.dumps(register, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {REGISTER.relative_to(ROOT)}.")
    elif verified_count and not args.write:
        print("(dry run — pass --write to persist into the snapshot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
