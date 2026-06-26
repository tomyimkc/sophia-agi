# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Nano LM substrate: a real, tiny, pure-Python language model + trainer.

Powers every pretraining-research study in this package (scaling laws, data mixing,
synthetic-data scaling, optimizer dynamics, architecture probes). Small by design;
the contribution is honest, verifiable methodology, not scale.
"""
from pretraining.nano.data import (
    drifted_source,
    make_source,
    mixed_corpus,
    sample_stream,
    source_entropy,
    to_examples,
)
from pretraining.nano.model import NanoLM, eval_loss
from pretraining.nano.train import train

__all__ = [
    "NanoLM", "eval_loss", "train",
    "make_source", "drifted_source", "source_entropy", "sample_stream",
    "to_examples", "mixed_corpus",
]
