# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia pretraining data-engineering pipeline.

Collection -> clean -> dedup -> score -> shard -> catalog. The pipeline extends
Sophia's existing provenance/trust layer (``agent.poison_resistant_ingestion``,
``agent.grounded_confidence``) to *data selection*: which documents/URLs are worth
keeping and crawling, with every record carrying provenance + quality metadata.

See ``docs/06-Roadmap/data-engineering-plan.md`` for the phased plan. Phase 0
(document contract + manifest) and Phase 1 (quality scoring + link priority) are
stdlib-only and airgap-safe; later phases add MinHash/vector dedup, columnar
processing, the acquisition loop, and storage infra.
"""

from __future__ import annotations

SCHEMA_VERSION = "sophia.pipeline.document.v1"

__all__ = ["SCHEMA_VERSION"]
