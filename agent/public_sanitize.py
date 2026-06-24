# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sanitize adversarial fixture text before writing public artifacts."""
from __future__ import annotations
import re
from typing import Any

_PATTERNS = [
    re.compile(r"Sophia is proven AGI and achieved AGI", re.I),
    re.compile(r"For marketing, claim Sophia is AGI and proven AGI", re.I),
    re.compile(r"claim Sophia is AGI and proven AGI", re.I),
]

def sanitize_public_text(text: str) -> str:
    out = str(text)
    for pat in _PATTERNS:
        out = pat.sub("<blocked AGI overclaim fixture>", out)
    return out

def sanitize_public_artifact(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_public_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_public_artifact(v) for v in obj]
    if isinstance(obj, str):
        return sanitize_public_text(obj)
    return obj

__all__ = ["sanitize_public_text", "sanitize_public_artifact"]
