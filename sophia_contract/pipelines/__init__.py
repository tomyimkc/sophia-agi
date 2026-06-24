# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reference role pipelines that run end-to-end through the governance contract.

These are runnable templates for the aihk-os / solo-ai-co role pipelines: each
drafts an artifact, writes it into the vault, and gates it (record_claim ->
verify_claim -> stamp) so only ``accepted`` output can be published.
"""

from sophia_contract.pipelines.copywriting import CopywritingPipeline

__all__ = ["CopywritingPipeline"]
