# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""OKF — Open Knowledge Format profile for Sophia's provenance-native wiki.

A dependency-free toolkit for a directory of Markdown pages with YAML frontmatter,
where the frontmatter IS the provenance record (authorConfidence, doNotAttributeTo,
doNotMergeWith, tradition) and the body is the prose. Pages form a typed belief
graph the agent can read, link-check, and reason over.

Public API:
    okf.parse / okf.serialize / okf.strip         — frontmatter codec
    okf.load / okf.load_pages / okf.Page           — page objects
    okf.validate_meta / okf.PAGE_TYPES             — schema
    okf.build_graph / okf.link_report              — graph + integrity
"""

from __future__ import annotations

from okf.bulk_graph import BulkGraph
from okf.projection import ProjectionResult, project_to_boundary
from okf.counterfactual import (
    Retraction,
    counterfactual_remove,
    is_grounded,
    retract,
)
from okf.revision import Revision, claims_to_abstain, revise
from okf.frontmatter import dump_block, parse, serialize, strip
from okf.graph import belief
from okf.graph import build as build_graph
from okf.graph import contradiction_ledger, propagate_confidence
from okf.linker import link_report
from okf.page import Page, load, load_pages
from okf.schema import (
    AUTHOR_CONFIDENCE,
    CONFIDENCE_RANK,
    PAGE_TYPES,
    confidence_rank,
    validate_meta,
)

__all__ = [
    "parse",
    "serialize",
    "strip",
    "dump_block",
    "Page",
    "load",
    "load_pages",
    "build_graph",
    "belief",
    "contradiction_ledger",
    "propagate_confidence",
    "counterfactual_remove",
    "retract",
    "revise",
    "claims_to_abstain",
    "Revision",
    "is_grounded",
    "Retraction",
    "link_report",
    "validate_meta",
    "confidence_rank",
    "PAGE_TYPES",
    "AUTHOR_CONFIDENCE",
    "CONFIDENCE_RANK",
    "BulkGraph",
    "ProjectionResult",
    "project_to_boundary",
]
