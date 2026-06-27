#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SimpleQA Verified -> C1 conformal + C3 abstention outcome records (real external data).

Genuine, non-self-authored validation: SimpleQA Verified (google/simpleqa-verified, 1000
human-authored factuality prompts) is graded into the ternary correct/incorrect/
NOT_ATTEMPTED — where NOT_ATTEMPTED *is* the C3 abstention, and the model's own stated
confidence is a C1 nonconformity signal.

This emits the canonical {id, confidence, nonconformity, correct, action} records the
existing `tools/fit_conformal_policy.py --data` and `tools/run_abstention_scoring.py --data`
consume. It records THREE confidence signals so the conformal fit can show which actually
yields a valid + useful risk-coverage frontier on real QA:

  - stated   : the model's self-reported confidence % (SimpleQA's calibration protocol);
  - selfcons : self-consistency agreement over N temperature-sampled answers;
  - logprob  : geometric-mean token probability exp(mean logprob) (deepseek-chat only).

Grader: an independent family via LLMHub (default Claude) using the SimpleQA grader
template; not the official GPT-4.1 (caveat recorded). Closed-book (no tools), per the
benchmark spec. Dependency-free: CSV over urllib, DeepSeek + LLMHub over their HTTP APIs.

  DEEPSEEK_API_KEY=... LLMHUB_API_KEY=... python tools/run_simpleqa_calibration.py --n 200
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CSV_URL = "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OUT_DIR = ROOT / "agi-proof" / "benchmark-results" / "real-model" / "simpleqa"

ANSWER_SYS = (
    "Answer the question as concisely as possible (just the fact). If you genuinely do "
    "not know, reply with exactly 'I don't know'. On a final separate line write "
    "'Confidence: N%' giving your probability (0-100) that your answer is correct."
)
GRADER_TMPL = (
    "You are grading a predicted answer against the gold answer.\n"
    "Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
    "Grade as ONE letter:\n"
    "A = CORRECT (the prediction matches the gold answer, allowing for phrasing).\n"
    "B = INCORRECT (the prediction gives a wrong answer).\n"
    "C = NOT_ATTEMPTED (the prediction declines, says it does not know, or gives no answer).\n"
    "Reply with ONLY the single letter A, B, or C."
)
_IDK = re.compile(r"\b(i\s*don'?t\s*know|idk|cannot determine|not\s*sure|unknown|no\s*answer)\b", re.I)
_CONF = re.compile(r"confidence:?\s*(\d{1,3})\s*%", re.I)


def _ssl_ctx():
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
    try:
        return ssl.create_default_context(cafile=bundle)
    except Exception:
        return ssl.create_default_context()


def _post(url: str, key: str, body: dict, timeout: int = 60) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        return json.load(r)


def subject_chat(messages, *, spec="deepseek", temperature=0.0, max_tokens=256, logprobs=False):
    """Call a subject model with temperature control. Returns (text, mean_logprob|None).

    spec: 'deepseek' (api.deepseek.com, logprobs), 'openrouter:MODEL', or 'llmhub:MODEL'.
    Only DeepSeek exposes per-token logprobs here; others return mean_lp=None.
    """
    if spec.startswith("llmhub:"):
        import agent.llmhub_llm as L
        try:
            return (L.chat_completion(messages=messages, model=spec.split(":", 1)[1],
                                      temperature=temperature, max_tokens=max_tokens, timeout_sec=60) or ""), None
        except Exception:
            return "", None
    if spec.startswith("openrouter:"):
        url, key, model = OPENROUTER_URL, os.environ["OPENROUTER_API_KEY"], spec.split(":", 1)[1]
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        want_lp = False
    else:  # deepseek
        url, key, model = DEEPSEEK_URL, os.environ["DEEPSEEK_API_KEY"], "deepseek-chat"
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        want_lp = logprobs
        if logprobs:
            body["logprobs"] = True
    for attempt in range(3):
        try:
            resp = _post(url, key, body)
            choice = resp["choices"][0]
            text = choice["message"]["content"] or ""
            mean_lp = None
            if want_lp:
                lp = (choice.get("logprobs") or {}).get("content")
                if lp:
                    vals = [t["logprob"] for t in lp if t.get("logprob") is not None]
                    mean_lp = sum(vals) / len(vals) if vals else None
            return text, mean_lp
        except Exception:
            if attempt == 2:
                return "", None
    return "", None


def grade(judge_model: str, q: str, gold: str, pred: str) -> str:
    import agent.llmhub_llm as L
    try:
        out = L.chat_completion(messages=[{"role": "user", "content": GRADER_TMPL.format(q=q, gold=gold, pred=pred)}],
                                model=judge_model, max_tokens=128, timeout_sec=60) or ""
    except Exception:
        return "?"
    m = re.search(r"[ABC]", out.upper())
    return m.group(0) if m else "?"


def _norm(ans: str) -> str:
    a = (ans or "").splitlines()[0].strip().lower()
    a = re.sub(r"[^a-z0-9 ]", "", a)
    a = re.sub(r"\b(the|a|an|in|of|is|was|were|are)\b", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def load_simpleqa(n: int) -> list[dict]:
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "sophia"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_ctx()) as r:
        text = r.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows[:n]


def process(ex: dict, idx: int, *, samples: int, graders: list, subject: str = "deepseek",
            sc_seeds: int = 1) -> dict:
    q, gold = ex["problem"], ex["answer"]
    msgs = [{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": q}]
    # main answer at temp 0 with logprobs (stated conf + logprob signal)
    main, mean_lp = subject_chat(msgs, spec=subject, temperature=0.0, logprobs=True)
    main_norm = _norm(main)
    # self-consistency: sc_seeds independent groups of (samples-1) temp-0.7 draws each
    sc_per_seed = []
    for _ in range(max(1, sc_seeds)):
        grp = [main]
        for _ in range(max(0, samples - 1)):
            s, _ = subject_chat(msgs, spec=subject, temperature=0.7)
            if s:
                grp.append(s)
        norms = [_norm(s) for s in grp]
        sc_per_seed.append(round(norms.count(main_norm) / len(norms), 4) if norms else 0.0)
    sc = round(sum(sc_per_seed) / len(sc_per_seed), 4)  # mean across seeds (for the .selfcons file)
    cm = _CONF.search(main)
    stated = (int(cm.group(1)) / 100.0) if cm else 0.5
    logprob_conf = math.exp(mean_lp) if mean_lp is not None else None
    # grade the main answer with EVERY grader family (for inter-grader kappa)
    grades = {gm: grade(gm, q, gold, main) for gm in graders}
    primary = grades[graders[0]]
    abstained = (primary == "C") or bool(_IDK.search(main.splitlines()[0] if main else ""))
    return {"id": f"sqa{idx}", "domain": ex.get("topic") or "unspecified", "risk": "normal",
            "stated": round(stated, 4), "selfcons": sc, "selfcons_seeds": sc_per_seed,
            "logprob_conf": round(logprob_conf, 4) if logprob_conf is not None else None,
            "grades": grades, "grade": primary, "action": "abstain" if abstained else "answer",
            "correct": (primary == "A"), "answer": (main or "").splitlines()[0][:200]}


def write_records(records: list[dict]) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # detailed per-row JSONL (all signals + every grader's verdict) for kappa + analysis
    detail = OUT_DIR / "simpleqa.detail.deepseek.jsonl"
    detail.write_text("".join(json.dumps({
        "id": r["id"], "domain": r["domain"], "stated": r["stated"], "selfcons": r["selfcons"],
        "selfcons_seeds": r.get("selfcons_seeds", [r["selfcons"]]),
        "logprob_conf": r["logprob_conf"], "grades": r.get("grades", {}), "grade": r["grade"],
        "action": r["action"], "correct": r["correct"]}) + "\n" for r in records), encoding="utf-8")
    # C3: all rows (attempted + abstained), action carries abstention
    all_path = OUT_DIR / "simpleqa.all.deepseek.jsonl"
    all_path.write_text("".join(json.dumps({
        "id": r["id"], "domain": r["domain"], "risk": r["risk"],
        "confidence": r["stated"], "nonconformity": round(1 - r["stated"], 4),
        "correct": r["correct"], "action": r["action"]}) + "\n" for r in records), encoding="utf-8")
    # C1: attempted rows only, one file per confidence signal
    attempted = [r for r in records if r["action"] == "answer"]
    paths = {"all": str(all_path.relative_to(ROOT))}
    for sig, key in (("stated", "stated"), ("selfcons", "selfcons"), ("logprob", "logprob_conf")):
        rows = [r for r in attempted if r.get(key) is not None]
        p = OUT_DIR / f"simpleqa.attempted.{sig}.deepseek.jsonl"
        p.write_text("".join(json.dumps({
            "id": r["id"], "domain": r["domain"], "risk": r["risk"],
            "confidence": r[key], "nonconformity": round(1 - r[key], 4),
            "correct": r["correct"]}) + "\n" for r in rows), encoding="utf-8")
        paths[sig] = str(p.relative_to(ROOT))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SimpleQA Verified -> C1/C3 records (real data).")
    ap.add_argument("--n", type=int, default=200, help="number of SimpleQA prompts")
    ap.add_argument("--samples", type=int, default=4, help="self-consistency samples per prompt")
    ap.add_argument("--graders", default="claude-sonnet-4-6", help="comma-sep LLMHub grader families")
    ap.add_argument("--subject", default="deepseek", help="subject model: deepseek | openrouter:M | llmhub:M")
    ap.add_argument("--sc-seeds", type=int, default=1, help="independent self-consistency sampling seeds")
    ap.add_argument("--out-subdir", default=None, help="write under simpleqa/<subdir>/ (for 2nd subject)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)

    global OUT_DIR
    if args.out_subdir:
        OUT_DIR = OUT_DIR / args.out_subdir
    graders = [g.strip() for g in args.graders.split(",") if g.strip()]
    data = load_simpleqa(args.n)
    print(f"loaded {len(data)} SimpleQA-Verified prompts; subject={args.subject} graders={graders} sc_seeds={args.sc_seeds}")
    records: list[dict] = [None] * len(data)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, row, i, samples=args.samples, graders=graders,
                          subject=args.subject, sc_seeds=args.sc_seeds): i
                for i, row in enumerate(data)}
        done = 0
        for fut in futs:
            pass
        for fut, i in [(f, futs[f]) for f in futs]:
            try:
                records[i] = fut.result()
            except Exception as e:
                records[i] = {"id": f"sqa{i}", "domain": "err", "risk": "normal", "stated": 0.5,
                              "selfcons": 0.0, "logprob_conf": None, "grade": "?",
                              "action": "abstain", "correct": False, "answer": f"ERR {e}"}
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(data)}")
    records = [r for r in records if r]
    n = len(records)
    attempted = sum(r["action"] == "answer" for r in records)
    correct = sum(r["correct"] for r in records)
    abst = sum(r["action"] == "abstain" for r in records)
    paths = write_records(records)
    summary = {
        "n": n, "attempted": attempted, "abstained": abst, "correct": correct,
        "accuracyOverall": round(correct / n, 4) if n else 0,
        "accuracyAttempted": round(correct / attempted, 4) if attempted else 0,
        "files": paths,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
