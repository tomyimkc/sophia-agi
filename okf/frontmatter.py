"""Dependency-free YAML-frontmatter codec for the OKF wiki.

A deliberately small, round-trippable subset of YAML so the ``okf`` package can be
imported by core modules (agent/verifiers.py, agent/retrieval.py) with no optional
dependency and on Python 3.9. Supports scalars (str/int/float/bool/None), inline
lists ``[a, b]``, and block lists (``- item``). Emits inline lists and quotes only
when a value would otherwise be ambiguous, which guarantees parse(serialize(x)) == x
for the metadata shapes Sophia uses.
"""

from __future__ import annotations

import re

_FENCE = "---"
_NEEDS_QUOTE = set(':#[]{},"\'\n')
_RESERVED = {"true", "false", "null", "yes", "no", "~", "on", "off"}
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def parse(text: str) -> "tuple[dict, str]":
    """Split a document into (frontmatter dict, body). No fence -> ({}, text)."""
    if text is None:
        return {}, ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return {}, text
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == _FENCE:
            close = i
            break
    if close is None:
        return {}, text
    meta = _parse_block(lines[1:close])
    body = "\n".join(lines[close + 1 :])
    return meta, body.lstrip("\n")


def serialize(meta: dict, body: str) -> str:
    """Render frontmatter + body back to a document string."""
    block = dump_block(meta)
    body = body or ""
    return f"{_FENCE}\n{block}{_FENCE}\n\n{body.lstrip(chr(10))}"


def strip(text: str) -> str:
    """Return the body with any frontmatter removed (for indexers/readers)."""
    return parse(text)[1]


def dump_block(meta: dict) -> str:
    """Render just the frontmatter key/value lines (each line newline-terminated)."""
    out: list[str] = []
    for key, value in meta.items():
        out.append(f"{key}: {_encode_value(value)}\n")
    return "".join(out)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def _parse_block(lines: "list[str]") -> dict:
    meta: dict = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        if raw[:1] in (" ", "\t"):  # stray indented line without a key — skip
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue
        key, _, rest = raw.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            # could be a block list on following indented "- " lines
            items: list = []
            j = i + 1
            while j < len(lines) and lines[j][:1] in (" ", "\t") and lines[j].lstrip().startswith("- "):
                items.append(_decode_scalar(lines[j].lstrip()[2:].strip()))
                j += 1
            if items:
                meta[key] = items
                i = j
                continue
            meta[key] = None
            i += 1
            continue
        meta[key] = _decode_value(rest)
        i += 1
    return meta


def _decode_value(text: str):
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_decode_scalar(part) for part in _split_top_level(inner)]
    return _decode_scalar(text)


def _decode_scalar(text: str):
    text = text.strip()
    if not text:
        return None
    if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
        body = text[1:-1]
        if text[0] == '"':
            body = body.replace('\\"', '"').replace("\\\\", "\\")
        return body
    low = text.lower()
    if low in ("null", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _split_top_level(inner: str) -> "list[str]":
    parts: list[str] = []
    buf = ""
    quote = ""
    for ch in inner:
        if quote:
            buf += ch
            if ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch
            buf += ch
        elif ch == ",":
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #


def _encode_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_encode_scalar(v) for v in value) + "]"
    return _encode_scalar(value)


def _encode_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if _needs_quote(text):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _needs_quote(text: str) -> bool:
    if text == "":
        return True
    if text.lower() in _RESERVED:
        return True
    if text[0] in (" ", "-", "?", "&", "*", "!", "|", ">", "%", "@", "`") or text[-1] == " ":
        return True
    if _INT_RE.match(text) or _FLOAT_RE.match(text):
        return True
    return any(ch in _NEEDS_QUOTE for ch in text)
