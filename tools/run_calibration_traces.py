#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibration-verifier TRACE GENERATION — the GPU step of the T3 pipeline.

Produces, for each (model, question), a trace carrying the three T3 feature INPUTS plus a
correctness label, so tools/run_calibration_verifier_eval.py --traces can fit the verifier and
score AUROC/ECE across model sizes (the scaling axis):

  - samples[]        : K answer samples (-> semantic-entropy feature; agreement when the model
                       is confident/correct, scatter when not)
  - evidence[]       : corroboration confidences from agent.retrieval over the question (CPU,
                       committed RAG index) (-> corroboration feature)
  - authorConfidence : the model's self-rated confidence in [0,1] (-> author-confidence feature)
  - correct          : 1 if the primary sample matches a gold alias, else 0 (deterministic label)

Backends
--------
  * ``mock`` (default): deterministic, OFFLINE, no GPU, no model download. Answers are
    synthesised from the gold answer with a per-item correctness drawn from a hash so the
    feature/label relationship is informative-but-noisy. This is the SMOKE backend — it lets
    the entire RunPod path (rent -> clone -> generate -> score -> copy back -> delete) be
    validated for a few cents before any live model load, mirroring rlvr-runpod's offline smoke.
  * ``hf``: REAL generation via transformers on the pod's GPU. Loads the model, K-samples each
    question at temperature, elicits a self-confidence, labels correctness by gold-alias match.
    Validated on-pod via the smoke->live sequence (it loads a real model, so it is not run in CI).

Retrieval (corroboration evidence) runs in BOTH backends — it is CPU/offline against the
committed index — so even the smoke produces real corroboration features.

Honest boundary: this generates a CANDIDATE trace corpus. Correctness here is gold-alias match
(deterministic family 1); a second judge family is added downstream by tools/run_judge_panel.py
before any number is cited. No claim is promoted from this tool. canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_DEFAULT = ROOT / "eval" / "calibration" / "questions_v1.jsonl"


def _load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _gold_match(answer: str, gold: list[str]) -> bool:
    a = (answer or "").strip().lower()
    return any(g.strip().lower() in a for g in gold if g)


@functools.lru_cache(maxsize=2048)
def _evidence_for(question: str, *, top_k: int = 6) -> tuple[float, ...]:
    """Corroboration evidence confidences from retrieval over the committed index (CPU/offline).

    Maps each retrieved chunk's score into a confidence in (0.5, 0.95]; an empty retrieval
    yields no evidence (corroboration falls back to the 0.5 prior — honest degradation).
    Memoized per question: identical questions across models/repeats reuse one retrieval."""
    try:
        from agent.retrieval import retrieve

        chunks = retrieve(question, top_k=top_k)
    except Exception:
        return ()
    confs: list[float] = []
    for c in chunks:
        s = float(getattr(c, "score", 0.0) or 0.0)
        confs.append(round(min(0.95, 0.5 + 0.45 * max(0.0, min(1.0, s))), 4))
    return tuple(confs)


def _hash01(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 10_000) / 10_000.0


# --------------------------------------------------------------------------- #
# Backends.
# --------------------------------------------------------------------------- #
def _mock_trace(q: dict, model: str, *, k: int, seed: int) -> dict:
    """Deterministic offline trace. Correctness depends on (model, question, seed); a 'correct'
    item produces clustered samples + higher self-confidence (informative-but-noisy features)."""
    gold = q.get("gold", [])
    primary_gold = gold[0] if gold else "unknown"
    correct = _hash01(model, q["id"], str(seed)) < 0.55  # ~55% correct, varies by model/question
    if correct:
        samples = [f"The answer is {primary_gold}." for _ in range(k)]
        author = round(0.6 + 0.35 * _hash01("conf", model, q["id"]), 4)
    else:
        # scattered wrong answers (low agreement); often still over-confident (miscalibrated)
        samples = [f"The answer is option {chr(65 + (i + int(_hash01('w', model, q['id'], str(i)) * 4)) % 4)}." for i in range(k)]
        author = round(0.4 + 0.5 * _hash01("conf2", model, q["id"]), 4)
    return {
        "id": q["id"], "model": model, "question": q["question"],
        "samples": samples, "evidence": _evidence_for(q["question"]),
        "authorConfidence": author, "correct": int(correct),
        "goldMatched": int(correct), "backend": "mock",
    }


def _hf_traces(questions: list[dict], model: str, *, k: int, seed: int, max_new_tokens: int = 64) -> list[dict]:
    """REAL generation via transformers (pod GPU). Imported lazily so CI never needs torch."""
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _gen(prompt: str, *, temperature: float, n: int) -> list[str]:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                           num_return_sequences=n, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen]

    traces: list[dict] = []
    for q in questions:
        samples = _gen(q["question"], temperature=0.7, n=k)
        primary = samples[0] if samples else ""
        # Self-confidence: ask the model to rate its own answer 0-1 (greedy, short).
        conf_prompt = (f"Question: {q['question']}\nYour answer: {primary}\n"
                       "On a scale from 0.0 to 1.0, how confident are you this answer is correct? "
                       "Reply with ONLY the number.")
        conf_raw = (_gen(conf_prompt, temperature=0.0, n=1) or [""])[0]
        try:
            author = max(0.0, min(1.0, float(conf_raw.split()[0])))
        except (ValueError, IndexError):
            author = 0.5
        correct = int(_gold_match(primary, q.get("gold", [])))
        traces.append({
            "id": q["id"], "model": model, "question": q["question"],
            "samples": samples, "evidence": _evidence_for(q["question"]),
            "authorConfidence": round(author, 4), "correct": correct,
            "goldMatched": correct, "backend": "hf",
        })
    return traces


def generate(models: list[str], *, backend: str, k: int, seed: int, questions_path: Path) -> list[dict]:
    questions = _load_questions(questions_path)
    traces: list[dict] = []
    for model in models:
        if backend == "mock":
            traces.extend(_mock_trace(q, model, k=k, seed=seed) for q in questions)
        elif backend == "hf":
            traces.extend(_hf_traces(questions, model, k=k, seed=seed))
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'mock' or 'hf'")
    return traces


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--models", default="mock-small,mock-mid,mock-large",
                    help="comma-separated model ids (the scaling axis). hf backend expects HF ids.")
    ap.add_argument("--k", type=int, default=4, help="answer samples per question (semantic entropy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", type=Path, default=QUESTIONS_DEFAULT)
    ap.add_argument("--out", type=Path, required=True, help="output traces JSONL")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    traces = generate(models, backend=args.backend, k=args.k, seed=args.seed, questions_path=args.questions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(traces)} traces ({len(models)} models x {len(traces)//max(1,len(models))} questions) "
          f"to {args.out} [backend={args.backend}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
