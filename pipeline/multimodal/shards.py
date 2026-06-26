# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""WebDataset shard writer for image-text samples (multimodal stretch).

WebDataset is the standard multimodal shard format: a plain tar where each sample is a group
of files sharing a key — ``<key>.jpg`` (image), ``<key>.txt`` (caption), ``<key>.json``
(metadata incl. provenance + dedup + quality). Training loaders stream these tars directly.
This writes real tars with the stdlib ``tarfile`` — no dependency — so it is fully testable
offline, and the shards are loadable by any WebDataset reader.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def write_webdataset(samples, path: str | Path, *, image_ext: str = "jpg") -> int:
    """Write ``samples`` to a WebDataset tar at ``path``. Returns the sample count.

    Each sample contributes ``<key>.<ext>`` (from ``image_bytes`` if present), ``<key>.txt``
    (caption), and ``<key>.json`` (everything except raw image bytes — provenance, dedup,
    quality). The key is the sample ``id``.
    """
    samples = list(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for s in samples:
            key = str(s.get("id"))
            img = s.get("image_bytes")
            if isinstance(img, bytes):
                _add_bytes(tar, f"{key}.{image_ext}", img)
            _add_bytes(tar, f"{key}.txt", (s.get("caption") or "").encode("utf-8"))
            meta = {k: v for k, v in s.items() if k != "image_bytes" and k != "image_matrix"}
            _add_bytes(tar, f"{key}.json", json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return len(samples)


def read_webdataset(path: str | Path) -> list[dict]:
    """Read a WebDataset tar back into sample dicts (caption + metadata; image bytes restored)."""
    path = Path(path)
    by_key: dict[str, dict] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            # Split on the LAST dot so sample ids containing dots survive the round-trip.
            key, _, ext = member.name.rpartition(".")
            entry = by_key.setdefault(key, {"id": key})
            data = tar.extractfile(member).read()
            if ext == "txt":
                entry["caption"] = data.decode("utf-8")
            elif ext == "json":
                entry.update(json.loads(data.decode("utf-8")))
            else:
                entry["image_bytes"] = data
    return list(by_key.values())


__all__ = ["write_webdataset", "read_webdataset"]
