#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Witness signing — ed25519 signatures over evidence artifacts.

The ledger rule is already "no claim without artifact + sha256"; this upgrades the
highest-value artifacts (``published-results.json``, promotion reports, adapter
manifests) from *hash-committed* to *signature-witnessed*: a third party can verify
the artifact against a published public key without trusting the repo history.

    python tools/witness.py keygen  --key <private.key> [--pub <pubkey.hex>]
    python tools/witness.py sign    --key <private.key> <artifact> [...]
    python tools/witness.py verify  [--pub <pubkey.hex>] <artifact> [...]

Design:
  * The signature covers the artifact's **sha256 digest** (hex, utf-8), so the
    witness composes with every existing "artifact + sha256" ledger row.
  * Each artifact gets a sidecar ``<artifact>.witness.json`` carrying the digest,
    the public key, and the signature — self-contained, diffable, committable.
  * ``verify`` is fail-closed: a missing sidecar, digest mismatch, bad signature,
    or (when ``--pub`` is given) an unexpected signer all fail with a named reason.
  * The private key must live OUTSIDE the repo (same handling as the git-crypt
    key). ``keygen``/``sign`` refuse a key path inside the working tree.

Honest bound: a witness proves *who signed which bytes*, nothing about whether the
numbers inside are valid — that remains the claim gate's job. Requires the
``cryptography`` package (already a transitive dev dependency); every command
fails closed with a clear message when it is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "sophia.witness.v1"


def _crypto():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        return Ed25519PrivateKey, Ed25519PublicKey
    except ImportError:
        raise SystemExit(
            "witness: the 'cryptography' package is required (pip install cryptography); "
            "refusing to sign/verify without real ed25519")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_outside_repo(key_path: Path) -> None:
    try:
        key_path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return  # outside the repo — good
    raise SystemExit(f"witness: refusing key path inside the repo ({key_path}); "
                     f"private keys never enter the working tree")


def keygen(key_path: Path, pub_path: "Path | None") -> int:
    Ed25519PrivateKey, _ = _crypto()
    from cryptography.hazmat.primitives import serialization

    _assert_outside_repo(key_path)
    if key_path.exists():
        raise SystemExit(f"witness: {key_path} exists; refusing to overwrite a key")
    private = Ed25519PrivateKey.generate()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption()))
    key_path.chmod(0o600)
    pub_hex = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    if pub_path is not None:
        pub_path.parent.mkdir(parents=True, exist_ok=True)
        pub_path.write_text(pub_hex + "\n", encoding="utf-8")
    print(f"witness: key -> {key_path} (keep OUT of the repo); pubkey {pub_hex}"
          + (f" -> {pub_path}" if pub_path else ""))
    return 0


def sign(key_path: Path, artifacts: "list[Path]") -> int:
    Ed25519PrivateKey, _ = _crypto()
    from cryptography.hazmat.primitives import serialization

    _assert_outside_repo(key_path)
    private = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    pub_hex = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    for artifact in artifacts:
        if not artifact.is_file():
            raise SystemExit(f"witness: no such artifact {artifact}")
        digest = _sha256(artifact)
        sig = private.sign(digest.encode("utf-8")).hex()
        sidecar = artifact.with_name(artifact.name + ".witness.json")
        sidecar.write_text(json.dumps({
            "schema": SCHEMA,
            "artifact": artifact.name,
            "artifactSha256": digest,
            "publicKey": pub_hex,
            "signature": sig,
            "claimCeiling": "witness proves signer+bytes only; validity is the claim gate's job",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"witness: signed {artifact} -> {sidecar.name}")
    return 0


def verify(artifacts: "list[Path]", *, pub_hex: "str | None" = None) -> int:
    _, Ed25519PublicKey = _crypto()
    from cryptography.exceptions import InvalidSignature

    failures = 0
    for artifact in artifacts:
        sidecar = artifact.with_name(artifact.name + ".witness.json")
        label = str(artifact)
        if not artifact.is_file():
            print(f"FAIL {label}: artifact missing"); failures += 1; continue
        if not sidecar.is_file():
            print(f"FAIL {label}: no witness sidecar ({sidecar.name})"); failures += 1; continue
        try:
            w = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"FAIL {label}: unreadable sidecar"); failures += 1; continue
        if w.get("schema") != SCHEMA:
            print(f"FAIL {label}: unknown witness schema {w.get('schema')!r}"); failures += 1; continue
        digest = _sha256(artifact)
        if digest != w.get("artifactSha256"):
            print(f"FAIL {label}: sha256 mismatch (artifact changed after signing)")
            failures += 1; continue
        signer = w.get("publicKey", "")
        if pub_hex is not None and signer != pub_hex:
            print(f"FAIL {label}: signed by unexpected key {signer[:16]}…"); failures += 1; continue
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(signer)).verify(
                bytes.fromhex(w.get("signature", "")), digest.encode("utf-8"))
        except (InvalidSignature, ValueError):
            print(f"FAIL {label}: signature invalid"); failures += 1; continue
        print(f"OK   {label}: witnessed by {signer[:16]}… sha256 {digest[:12]}…")
    return 1 if failures else 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen", help="generate an ed25519 keypair (key stays outside the repo)")
    kg.add_argument("--key", type=Path, required=True)
    kg.add_argument("--pub", type=Path, default=None,
                    help="where to write the hex public key (committable)")
    sg = sub.add_parser("sign", help="sign artifacts, writing .witness.json sidecars")
    sg.add_argument("--key", type=Path, required=True)
    sg.add_argument("artifacts", nargs="+", type=Path)
    vf = sub.add_parser("verify", help="verify artifacts against their sidecars")
    vf.add_argument("--pub", default=None,
                    help="hex public key (or @file) the witness MUST be signed by")
    vf.add_argument("artifacts", nargs="+", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "keygen":
        return keygen(args.key, args.pub)
    if args.cmd == "sign":
        return sign(args.key, args.artifacts)
    pub = args.pub
    if pub and pub.startswith("@"):
        pub = Path(pub[1:]).read_text(encoding="utf-8").strip()
    return verify(args.artifacts, pub_hex=pub)


if __name__ == "__main__":
    raise SystemExit(main())
