# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Perceptual hashing for image near-dup detection (multimodal stretch).

Image dedup needs *perceptual* hashes (similar-looking images → similar hashes), not byte
hashes. This implements average-hash (aHash) and difference-hash (dHash) over a grayscale
matrix — pure stdlib, deterministic. A matrix is ``list[list[int]]`` of 0–255 luminance.

Decoding real image bytes to a matrix needs an imaging library; ``decode_available()`` /
``image_to_matrix()`` use PIL when present and are the only optional part — the hashing and
dedup logic run on matrices, so they are fully tested offline with synthetic images.
"""

from __future__ import annotations


def _resize(matrix: list[list[float]], rows: int, cols: int) -> list[list[float]]:
    """Area-average resize of ``matrix`` to ``rows x cols`` (no dependencies)."""
    h = len(matrix)
    w = len(matrix[0]) if h else 0
    out: list[list[float]] = []
    for i in range(rows):
        y0, y1 = i * h // rows, max(i * h // rows + 1, (i + 1) * h // rows)
        row: list[float] = []
        for j in range(cols):
            x0, x1 = j * w // cols, max(j * w // cols + 1, (j + 1) * w // cols)
            total = count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    total += matrix[y][x]
                    count += 1
            row.append(total / count if count else 0.0)
        out.append(row)
    return out


def average_hash(matrix: list[list[float]], size: int = 8) -> int:
    """aHash: bit set where a downscaled pixel is >= the mean. Returns a ``size*size``-bit int."""
    small = _resize(matrix, size, size)
    flat = [p for r in small for p in r]
    avg = sum(flat) / len(flat) if flat else 0.0
    bits = 0
    for idx, p in enumerate(flat):
        if p >= avg:
            bits |= 1 << idx
    return bits


def dhash(matrix: list[list[float]], size: int = 8) -> int:
    """dHash: bit set where a pixel is brighter than its right neighbor (gradient hash)."""
    small = _resize(matrix, size, size + 1)
    bits = 0
    idx = 0
    for r in small:
        for x in range(size):
            if r[x] > r[x + 1]:
                bits |= 1 << idx
            idx += 1
    return bits


def hamming(a: int, b: int) -> int:
    """Hamming distance between two hashes (number of differing bits)."""
    return bin(a ^ b).count("1")


def decode_available() -> bool:
    """True iff PIL is importable (needed only to hash real image bytes)."""
    try:
        import PIL  # noqa: F401
    except Exception:
        return False
    return True


def image_to_matrix(data: bytes) -> list[list[int]]:
    """Decode image bytes to a grayscale matrix via PIL. Raises if PIL is unavailable."""
    if not decode_available():
        raise RuntimeError("image_to_matrix requires Pillow (PIL)")
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("L")
    w, h = img.size
    px = list(img.getdata())
    return [px[r * w : (r + 1) * w] for r in range(h)]


__all__ = ["average_hash", "dhash", "hamming", "decode_available", "image_to_matrix"]
