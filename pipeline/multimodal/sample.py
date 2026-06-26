# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Image-text sample contract (multimodal stretch).

An image-text pair flowing through the multimodal pipeline. Like ``pipeline.document``, this
is a lightweight stdlib validator (no jsonschema) returning a problem list (empty == valid).
A sample references its image by ``image_path`` or carries ``image_bytes``/``phash`` directly,
plus a ``caption`` and a ``provenance`` block for lineage (the same provenance discipline as
the text pipeline, extended to pairs).
"""

from __future__ import annotations


def validate(sample: dict) -> list[str]:
    """Return a list of problems with ``sample`` (empty == valid)."""
    problems: list[str] = []
    if not isinstance(sample, dict):
        return [f"sample must be an object, got {type(sample).__name__}"]

    if not sample.get("id"):
        problems.append("missing 'id'")
    caption = sample.get("caption")
    if not isinstance(caption, str):
        problems.append("missing or non-string 'caption'")
    # Must be able to locate/identify the image somehow.
    if not any(k in sample for k in ("image_path", "image_bytes", "phash", "image_matrix")):
        problems.append("sample needs one of image_path / image_bytes / phash / image_matrix")

    prov = sample.get("provenance")
    if prov is not None and not isinstance(prov, dict):
        problems.append("'provenance' must be an object")
    return problems


def is_valid(sample: dict) -> bool:
    return not validate(sample)


__all__ = ["validate", "is_valid"]
