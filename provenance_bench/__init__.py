"""Provenance Delta benchmark.

Measures, on independent ground truth, how often a model asserts a FALSE
authorship lineage when used alone versus behind Sophia's provenance gate
(the guarded completion loop). See
docs/superpowers/specs/2026-06-21-provenance-delta-design.md.

Non-circularity contract: ground-truth LABELS come from external sources
(provenance_bench/data/*.json — Wikipedia/Wikidata + cited misattributions),
the GATE is only the runtime treatment, and the JUDGE (provenance_bench.judge)
shares no code with the gate.
"""
