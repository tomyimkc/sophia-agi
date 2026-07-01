#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate the real {samples, correct} consensus pack O1/O4 need from SimpleQA.

The existing tools/run_simpleqa_calibration.py samples k answers per question but
persists only the scalar self-consistency fraction — the raw answer TEXTS (which
O1's Kuramoto-r and O4's adaptive-k must embed) are dropped. This adapter re-runs
the sampling and KEEPS the texts, emitting the exact contract both tools consume:

    {"samples": ["answer_1", ..., "answer_k"], "correct": <majority-answer-is-right>}

Subject model  : DeepSeek (api.deepseek.com) — the same family behind the repo's
                 validated self-consistency result. NON-MOCK: asserts DEEPSEEK_API_KEY.
Grader (label) : an INDEPENDENT family via agent.model (grok) using the SimpleQA
                 grader template; asserts the resolved provider is not 'mock'.
Decontamination: SimpleQA Verified is an evaluation-only benchmark; its prompts are
                 never used for training in this repo. Each row records id/question/gold
                 so overlap can be audited. A seeded shuffle selects the slice.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
LOCAL_CSV = Path("/tmp/osc-data/simpleqa_verified.csv")
CSV_URL = "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv"

ANSWER_SYS = (
    "Answer the question as concisely as possible (just the fact). If you genuinely do "
    "not know, reply with exactly 'I don't know'."
)
GRADER_TMPL = (
    "You are grading a predicted answer against the gold answer.\n"
    "Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
    "Grade as ONE letter:\n"
    "A = CORRECT (the prediction matches the gold answer, allowing for phrasing).\n"
    "B = INCORRECT (the prediction gives a wrong answer).\n"
    "C = NOT_ATTEMPTED (the prediction declines, says it does not know, or gives no answer).\n"
    "Reply with exactly one letter: A, B, or C."
)
_IDK = re.compile(r"\b(i\s*don'?t\s*know|idk|cannot determine|not\s*sure|unknown|no\s*answer)\b", re.I)


def _ssl_ctx():
    return ssl.create_default_context()


def _post(url, key, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90, context=_ssl_ctx()) as r:
        return json.load(r)


def deepseek_chat(messages, *, temperature, max_tokens=192):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = {"model": "deepseek-chat", "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    for attempt in range(4):
        try:
            resp = _post(DEEPSEEK_URL, key, body)
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == 3:
                return ""
    return ""


def _norm(ans: str) -> str:
    a = (ans or "").splitlines()[0].strip().lower() if ans else ""
    a = re.sub(r"[^a-z0-9 ]", "", a)
    a = re.sub(r"\b(the|a|an|in|of|is|was|were|are)\b", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def majority_answer(samples: list[str]) -> str:
    """Most common normalized answer; returns a representative RAW text for it."""
    norms = [_norm(s) for s in samples]
    counts: dict[str, int] = {}
    for n in norms:
        counts[n] = counts.get(n, 0) + 1
    top = max(counts, key=counts.get) if counts else ""
    for s, n in zip(samples, norms):
        if n == top:
            return s
    return samples[0] if samples else ""


def grade_with_grok(q: str, gold: str, pred: str) -> str:
    import agent.model as m
    try:
        out = m.complete("You are a strict grader. Reply with exactly one letter.",
                         GRADER_TMPL.format(q=q, gold=gold, pred=pred), max_tokens=6, spec="grok")
    except Exception:
        return "?"
    mm = re.search(r"[ABC]", (out or "").upper())
    return mm.group(0) if mm else "?"


def load_simpleqa() -> list[dict]:
    if LOCAL_CSV.exists():
        text = LOCAL_CSV.read_text(encoding="utf-8")
    else:
        req = urllib.request.Request(CSV_URL, headers={"User-Agent": "sophia"})
        with urllib.request.urlopen(req, timeout=90, context=_ssl_ctx()) as r:
            text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def process(ex: dict, idx: int, *, k: int) -> dict:
    q, gold = ex["problem"], ex["answer"]
    msgs = [{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": q}]
    samples = []
    for _ in range(k):
        s = deepseek_chat(msgs, temperature=0.7)
        if s and s.strip():
            samples.append(s.strip())
    if len(samples) < 2:
        return {"id": f"sqa{idx}", "skip": True}
    maj = majority_answer(samples)
    grade = grade_with_grok(q, gold, maj)
    abstained = (grade == "C") or bool(_IDK.search(maj.splitlines()[0] if maj else ""))
    return {
        "id": f"sqa{idx}", "question": q, "gold": gold,
        "domain": ex.get("topic") or "unspecified",
        "samples": samples,
        "majority": maj.splitlines()[0][:200],
        "grade": grade,
        "action": "abstain" if abstained else "answer",
        "correct": (grade == "A"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SimpleQA -> {samples, correct} consensus pack for O1/O4.")
    ap.add_argument("--n", type=int, default=120, help="number of SimpleQA questions")
    ap.add_argument("--samples", type=int, default=6, help="k sampled answers per question")
    ap.add_argument("--seed", type=int, default=0, help="seed for the decontaminated slice shuffle")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    # Fail closed on a mock/absent backend — a run is not a result.
    if not (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        print("FATAL: DEEPSEEK_API_KEY not set (mock/absent subject backend).", file=sys.stderr)
        return 2
    import agent.model as m
    gcfg = m.resolve_config("grok")
    prov = getattr(gcfg, "provider", getattr(gcfg, "kind", "?"))
    if prov == "mock":
        print("FATAL: grader backend resolved to 'mock'.", file=sys.stderr)
        return 2

    rows = load_simpleqa()
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    slice_rows = rows[: args.n]
    print(f"subject=deepseek-chat grader={prov} n={len(slice_rows)} k={args.samples} seed={args.seed}",
          file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = [None] * len(slice_rows)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r, i, k=args.samples): i for i, r in enumerate(slice_rows)}
        done = 0
        for fut in list(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(slice_rows)}", file=sys.stderr)

    kept = [r for r in results if r and not r.get("skip")]
    n_correct = sum(1 for r in kept if r["correct"])
    with out_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"WROTE {len(kept)} rows -> {out_path} | correct={n_correct} incorrect={len(kept)-n_correct}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
