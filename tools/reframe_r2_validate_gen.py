#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R2-validation dataset: balanced multi-domain MMLU with EXACT correctness labels.

MMLU is the right substrate for R2 (ensemble-disagreement OOD-abstain): many subjects (domains),
~60% accuracy so both-class per domain, and multiple-choice => exact correctness (no LLM judge).
Subject model: DeepSeek (non-mock asserted). Emits {subject, question, chosen, answerIdx, correct,
answerText} where answerText = question + chosen option (what the honesty/correctness probe reads).
"""
from __future__ import annotations
import argparse, json, os, re, ssl, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MMLU = "/tmp/r2-data/mmlu.parquet"
SUBJECTS = ["high_school_macroeconomics", "prehistory", "high_school_biology",
            "professional_accounting", "clinical_knowledge", "moral_disputes"]
LETTERS = ["A", "B", "C", "D"]


def _post(url, key, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90, context=ssl.create_default_context()) as r:
        return json.load(r)


def ask(q, choices):
    key = os.environ["DEEPSEEK_API_KEY"]
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    user = f"{q}\n{opts}\n\nAnswer with ONLY the letter of the correct option."
    body = {"model": "deepseek-chat", "temperature": 0.0, "max_tokens": 8,
            "messages": [{"role": "user", "content": user}]}
    for attempt in range(4):
        try:
            txt = _post(DEEPSEEK_URL, key, body)["choices"][0]["message"]["content"] or ""
            m = re.search(r"[ABCD]", txt.upper())
            return LETTERS.index(m.group(0)) if m else None
        except Exception:
            if attempt == 3:
                return None
    return None


def process(row):
    ci = ask(row["question"], list(row["choices"]))
    if ci is None:
        return None
    return {"subject": row["subject"], "question": row["question"],
            "chosenIdx": ci, "answerIdx": int(row["answer"]), "correct": (ci == int(row["answer"])),
            "answerText": f"{row['question']} Answer: {row['choices'][ci]}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=80); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        print("FATAL: DEEPSEEK_API_KEY absent", file=sys.stderr); return 2
    df = pd.read_parquet(MMLU)
    picks = []
    for s in SUBJECTS:
        sub = df[df["subject"] == s].sample(n=min(a.per_subject, (df["subject"] == s).sum()),
                                            random_state=a.seed)
        picks += sub.to_dict("records")
    print(f"subject=deepseek subjects={len(SUBJECTS)} questions={len(picks)}", file=sys.stderr)
    out = [None] * len(picks)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, r): i for i, r in enumerate(picks)}
        for k, f in enumerate(list(futs), 1):
            out[futs[f]] = f.result()
            if k % 40 == 0:
                print(f"  {k}/{len(picks)}", file=sys.stderr)
    kept = [r for r in out if r]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    per = {s: (sum(1 for r in kept if r["subject"] == s and r["correct"]),
               sum(1 for r in kept if r["subject"] == s)) for s in SUBJECTS}
    print(f"WROTE {len(kept)} -> {a.out} | per-subject correct/total: {per}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
