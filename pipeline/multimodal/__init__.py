# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multimodal data slice (stretch phase): image-text pairs.

The JD's 多模态数据方向 — one vertical end-to-end pipeline for image-text data:
contract → perceptual-hash dedup → quality filter → WebDataset shards, each sample carrying
provenance lineage. Consistent with the rest of the pipeline: the stdlib core (perceptual
hashing over a grayscale matrix, caption quality, tar/WebDataset writing) is fully testable
offline; image *decoding* (bytes → matrix) is an optional backend (PIL) behind a guard, so
tests and airgap pass synthetic matrices directly.
"""

from __future__ import annotations

from pipeline.multimodal.phash import average_hash, dhash, hamming
from pipeline.multimodal.process import dedup_samples, score_sample

__all__ = ["average_hash", "dhash", "hamming", "dedup_samples", "score_sample"]
