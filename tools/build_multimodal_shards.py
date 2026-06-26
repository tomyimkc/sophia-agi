#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build WebDataset shards from image-text pairs: dedup -> score -> shard.

    python tools/build_multimodal_shards.py pairs.jsonl --out shards/part-000.tar

Input is a JSONL of image-text samples (``{id, caption, image_path|image_bytes|phash,
provenance}``). Runs perceptual-hash dedup (PIL used to hash real images when present;
otherwise a ``phash``/``image_matrix`` field is used), scores each pair (caption + provenance),
optionally drops rejects, and writes a WebDataset tar with a provenance/quality JSON per
sample. Offline; image decoding optional.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.multimodal import process, sample as smod, shards  # noqa: E402


def _load(path: str) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        # Support base64 image bytes in JSONL, and on-disk image_path.
        if isinstance(s.get("image_b64"), str):
            try:
                s["image_bytes"] = base64.b64decode(s.pop("image_b64"), validate=True)
            except Exception:  # malformed base64 -> skip this sample, keep going
                print(f"[skip] sample {s.get('id')!r}: bad base64 image", file=sys.stderr)
                continue
        elif s.get("image_path") and Path(s["image_path"]).is_file():
            s["image_bytes"] = Path(s["image_path"]).read_bytes()
        out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="JSONL of image-text samples")
    ap.add_argument("--out", required=True, help="output WebDataset tar")
    ap.add_argument("--max-distance", type=int, default=5, help="phash Hamming dup threshold")
    ap.add_argument("--keep-only", action="store_true")
    args = ap.parse_args(argv)

    samples = _load(args.input)
    bad = 0
    valid = []
    for i, s in enumerate(samples):
        problems = smod.validate(s)
        if problems:
            bad += 1
            print(f"[skip] sample {i}: {problems[0]}", file=sys.stderr)
            continue
        valid.append(s)

    result = process.dedup_samples(valid, max_distance=args.max_distance)
    kept = result["kept"]
    print(f"Dedup: kept {result['stats']['kept']}/{result['stats']['input']} "
          f"(removed {result['stats']['removed']}, ratio {result['stats']['dedupRatio']:.3f})")

    for s in kept:
        s["quality"] = process.score_sample(s)
    if args.keep_only:
        kept = [s for s in kept if s["quality"]["keep"]]
        print(f"Quality filter: {len(kept)} kept")

    n = shards.write_webdataset(kept, args.out)
    print(f"Wrote {n} samples -> {args.out}")
    if bad:
        print(f"{bad} invalid sample(s) skipped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
