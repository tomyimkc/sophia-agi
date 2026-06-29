# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council → from-scratch student distillation (idea #4).

Sequence-level distillation: train this GPT on **gate-filtered council traces**
(the disciplined teacher outputs the council already produces), so a model born
inside Sophia absorbs the source-discipline habits without the scaffold at
inference. This is the cheapest, most honest form — train the student on the
teacher's accepted text — and reuses the existing trace data
(`training/local_sophia_7b/sft_council_traces.jsonl`,
`tools/distill_council_traces.py`).

The document loader is dependency-free (reuses `data.corpus_documents`, which
already takes any chat-jsonl path); the training run is torch-gated. Honest
boundary: a nano student can't reproduce the council — this wires and measures the
mechanism, ``canClaimAGI: false``.

    python -m pretraining.gpt.distill --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.gpt.data import corpus_documents
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_TRACES = ROOT / "training" / "local_sophia_7b" / "sft_council_traces.jsonl"


def council_documents(path: "Path | None" = None) -> "list[str]":
    """Gate-filtered council traces as role-tagged document strings."""
    return corpus_documents(path or DEFAULT_TRACES)


def council_token_stream(tokenizer=None, *, path: "Path | None" = None) -> "list[int]":
    tok = tokenizer or ByteProvenanceTokenizer()
    ids: list[int] = []
    for doc in council_documents(path):
        ids.extend(tok.encode(doc))
        ids.append(tok.eot_id)
    return ids


def run_distill(*, quick: bool = False, steps: int = 1000, seed: int = 0,
                path: "Path | None" = None) -> dict:
    from pretraining.gpt.train import train  # noqa: PLC0415

    tok = ByteProvenanceTokenizer()
    stream = council_token_stream(tok, path=path)
    if not stream:
        return {"canClaimAGI": False, "error": "no council traces found",
                "path": str(path or DEFAULT_TRACES)}

    report = train(quick=quick, steps=steps, prefer="cpu", seed=seed, ids=stream)
    report["distillation"] = {
        "teacher": "Sophia council (gate-filtered traces)",
        "student": "from-scratch GPT (pretraining/gpt)",
        "traces": str(path or DEFAULT_TRACES),
        "n_documents": len(council_documents(path)),
        "method": "sequence-level (SFT on accepted teacher text)",
    }
    report["boundary"] = ("council distillation MECHANISM on a nano student — "
                          "illustrative, not a capability claim.")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Distill gate-filtered council traces into the GPT.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--traces", type=str, default=None, help="path to council traces jsonl")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    path = Path(args.traces) if args.traces else None
    try:
        rep = run_distill(quick=args.quick, steps=args.steps, path=path)
    except ImportError as exc:
        print(f"[gpt.distill] {exc}")
        return 2
    print(json.dumps({k: rep[k] for k in rep if k not in ("epoch_loss", "grad_norms")},
                     indent=2, ensure_ascii=False))
    if args.report and "error" not in rep:
        (HERE / "gpt-distill-latest.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
