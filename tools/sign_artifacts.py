#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Model-artifact integrity (P5) — checksums, optional signature, and a minimal
SBOM for a release.

Supply-chain defense (OWASP LLM03/04): publish a ``SHA256SUMS`` alongside every
released adapter/weight so downstream users can verify they got the bytes you
shipped, optionally sign that manifest, and emit a CycloneDX-style SBOM of the
Python dependency set.

    python tools/sign_artifacts.py checksum training/mlx_adapters/sophia-v3 --out SHA256SUMS
    python tools/sign_artifacts.py verify SHA256SUMS
    python tools/sign_artifacts.py sign SHA256SUMS            # uses minisign/cosign if present
    python tools/sign_artifacts.py sbom --out sbom.cdx.json

Dependency-free. Signing shells out to ``minisign`` or ``cosign`` if installed;
otherwise it explains how to install one (it never fakes a signature).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(targets: "list[str]") -> "list[Path]":
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*")) if q.is_file()]
        elif p.is_file():
            out.append(p)
    return out


def compute_checksums(targets: "list[str]") -> "dict[str, str]":
    return {str(p): sha256_file(p) for p in iter_files(targets)}


def write_sums(checksums: "dict[str, str]", out: Path) -> None:
    # Standard `sha256sum` format so the GNU tool can also verify it.
    lines = [f"{digest}  {path}" for path, digest in sorted(checksums.items())]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_sums(sums_path: Path) -> dict:
    bad: list[str] = []
    missing: list[str] = []
    ok = 0
    # Resolve manifest entries relative to the SHA256SUMS file's directory, not the
    # caller's CWD, so `verify` works from anywhere (matching `sha256sum -c`).
    base = sums_path.resolve().parent
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, path = line.partition("  ")
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        if not p.is_file():
            missing.append(path)
        elif sha256_file(p) != digest:
            bad.append(path)
        else:
            ok += 1
    return {"ok": not bad and not missing, "verified": ok, "mismatched": bad, "missing": missing}


def sign(sums_path: Path) -> dict:
    if shutil.which("minisign"):
        subprocess.run(["minisign", "-Sm", str(sums_path)], check=True)
        return {"tool": "minisign", "signature": f"{sums_path}.minisig"}
    if shutil.which("cosign"):
        sig = sums_path.with_suffix(sums_path.suffix + ".sig")
        subprocess.run(["cosign", "sign-blob", "--yes", "--output-signature", str(sig), str(sums_path)], check=True)
        return {"tool": "cosign", "signature": str(sig)}
    return {"tool": None, "error": "install minisign (`brew install minisign`) or cosign to sign; "
                                   "checksums are still published unsigned"}


def _parse_requirement(line: str) -> "dict | None":
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    for sep in ("==", ">=", "<=", "~=", ">", "<"):
        if sep in line:
            name, ver = line.split(sep, 1)
            return {"name": name.strip(), "version": ver.strip()}
    return {"name": line, "version": None}


def build_sbom() -> dict:
    components: dict[str, dict] = {}
    for req in sorted(ROOT.glob("requirements*.txt")):
        for line in req.read_text(encoding="utf-8").splitlines():
            comp = _parse_requirement(line)
            if comp:
                components.setdefault(comp["name"].lower(), {
                    "type": "library", "name": comp["name"], "version": comp["version"],
                    "purl": f"pkg:pypi/{comp['name'].lower()}" + (f"@{comp['version']}" if comp["version"] else ""),
                })
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.5",
        "metadata": {"component": {"type": "application", "name": "sophia-agi"}},
        "components": list(components.values()),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("checksum", help="hash artifacts into a SHA256SUMS file")
    c.add_argument("targets", nargs="+", help="files or directories to hash")
    c.add_argument("--out", default="SHA256SUMS")

    v = sub.add_parser("verify", help="verify a SHA256SUMS file")
    v.add_argument("sums", help="path to SHA256SUMS")

    s = sub.add_parser("sign", help="sign a SHA256SUMS file (minisign/cosign)")
    s.add_argument("sums", help="path to SHA256SUMS")

    b = sub.add_parser("sbom", help="emit a CycloneDX SBOM of Python deps")
    b.add_argument("--out", default="sbom.cdx.json")

    args = ap.parse_args(argv)

    if args.cmd == "checksum":
        sums = compute_checksums(args.targets)
        if not sums:
            print("no files found to hash", file=sys.stderr)
            return 2
        write_sums(sums, Path(args.out))
        print(f"wrote {len(sums)} checksums to {args.out}")
        return 0
    if args.cmd == "verify":
        rep = verify_sums(Path(args.sums))
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    if args.cmd == "sign":
        rep = sign(Path(args.sums))
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("signature") else 1
    if args.cmd == "sbom":
        sbom = build_sbom()
        Path(args.out).write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
        print(f"wrote SBOM with {len(sbom['components'])} components to {args.out}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
