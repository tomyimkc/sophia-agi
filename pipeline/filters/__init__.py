# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Training-data quality filters — Gopher / C4 / RefinedWeb heuristics (pure stdlib).

Dolma-style *taggers*: each filter computes deterministic signals over a document
and a fail-closed keep/drop decision, so a corpus build is reproducible from a
content hash. These are the quality-filter half of a FineWeb/Dolma/RefinedWeb
pipeline (the repo already has MinHash dedup, a provenance/data-passport layer,
and a contamination guard — this fills in the line/document quality heuristics the
existing `quality_score.py` only partially covered).

References: Rae et al. 2021 (*Gopher*/MassiveText repetition + quality rules);
Raffel et al. 2020 (*C4* line cleaning); Penedo et al. 2023 (*RefinedWeb*);
Penedo et al. 2024 (*FineWeb*); Soldaini et al. 2024 (*Dolma* tagger pattern).
"""
from pipeline.filters import c4, gopher, quality

__all__ = ["gopher", "c4", "quality"]
