#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R4 — trajectory stability under INPUT perturbation vs output-sampling self-consistency.

Hypothesis: a correct answer is a deep attractor (stable when you paraphrase the question);
a fabrication is shallow (the answer jumps under paraphrase). So paraphrase-stability should
predict correctness BETTER than temperature-sampling self-consistency (which O1 showed ties).

Reuses the existing consensus pack for the self-consistency baseline (6 temp samples/question),
adds k paraphrase-answers per question (paraphrases from grok, answers from DeepSeek).

Gate: AUROC(paraphraseStability, correct) beats AUROC(selfConsistency, correct), paired-bootstrap
CI excluding 0, >=2 seeds. Non-mock asserted. No weights updated.
"""
from __future__ import annotations
import argparse, json, os, re, ssl, sys, urllib.request, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from eval_o2_energy_hidden import auroc

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
PACK = "agi-proof/benchmark-results/oscillatory-crosspollination/data/o1o4_simpleqa_pack.jsonl"
ANSWER_SYS = "Answer as concisely as possible (just the fact). If you don't know, say 'I don't know'."


def _post(url, key, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90, context=ssl.create_default_context()) as r:
        return json.load(r)


def deepseek(msgs, temp=0.0, mx=120):
    key = os.environ["DEEPSEEK_API_KEY"]
    for attempt in range(4):
        try:
            resp = _post(DEEPSEEK_URL, key, {"model": "deepseek-chat", "messages": msgs,
                        "temperature": temp, "max_tokens": mx})
            return resp["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == 3:
                return ""
    return ""


def paraphrases(q, k):
    import agent.model as mdl
    prompt = (f"Rewrite this question {k} different ways that ask for the exact same fact, one per "
              f"line, no numbering. Keep the answer identical.\nQuestion: {q}")
    try:
        out = mdl.complete("You paraphrase questions faithfully.", prompt, max_tokens=300, spec="grok")
    except Exception:
        out = ""
    lines = [re.sub(r"^\s*\d+[\.\)]\s*", "", l).strip() for l in (out or "").splitlines() if l.strip()]
    return lines[:k] if lines else [q]


def norm(a):
    a = (a or "").splitlines()[0].strip().lower() if a else ""
    a = re.sub(r"[^a-z0-9 ]", "", a)
    return re.sub(r"\s+", " ", a).strip()


def stability(answers):
    """Fraction agreeing with the modal normalized answer."""
    norms = [norm(x) for x in answers if x and x.strip()]
    if not norms:
        return 0.0
    from collections import Counter
    top = Counter(norms).most_common(1)[0][1]
    return top / len(norms)


def process(row, k):
    q = row["question"]
    paras = paraphrases(q, k)
    ans = [deepseek([{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": p}]) for p in paras]
    ans = [a for a in ans if a and a.strip()]
    if len(ans) < 2:
        return None
    return {"id": row["id"], "correct": bool(row["correct"]),
            "selfConsistency": stability(row["samples"]),      # baseline (existing temp samples)
            "paraphraseStability": stability(ans), "nParaphrase": len(ans)}


def paired_boot(a, b, labels, n_boot, seed):
    rng = random.Random(seed); n = len(labels); d = []
    ba, bb = auroc(a, labels), auroc(b, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        xa, xb = auroc([a[i] for i in idx], la), auroc([b[i] for i in idx], la)
        if xa is not None and xb is not None:
            d.append(xa - xb)
    d.sort()
    return {"aurocParaphrase": round(ba, 4), "aurocSelfConsistency": round(bb, 4),
            "delta": round(ba - bb, 4), "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--cache", default="/tmp/reframe-data/r4_pack.jsonl"); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)
    if not (os.environ.get("DEEPSEEK_API_KEY") or "").strip():
        print("FATAL: DEEPSEEK_API_KEY absent", file=sys.stderr); return 2

    if Path(a.cache).exists():
        scored = [json.loads(l) for l in open(a.cache)]
        print(f"loaded cached R4 pack: {len(scored)} rows", file=sys.stderr)
    else:
        rows = [json.loads(l) for l in open(PACK)][:a.n]
        print(f"generating paraphrase-answers for {len(rows)} questions (k={a.k}) ...", file=sys.stderr)
        scored = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(process, r, a.k) for r in rows]
            for i, f in enumerate(futs, 1):
                r = f.result()
                if r:
                    scored.append(r)
                if i % 20 == 0:
                    print(f"  {i}/{len(rows)}", file=sys.stderr)
        Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
        with open(a.cache, "w") as f:
            for r in scored:
                f.write(json.dumps(r) + "\n")

    y = [bool(r["correct"]) for r in scored]
    pstab = [r["paraphraseStability"] for r in scored]
    sc = [r["selfConsistency"] for r in scored]
    print(f"n={len(scored)} correct={sum(y)}", file=sys.stderr)
    report = {"schema": "sophia.reframe_r4.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "R4 paraphrase-stability vs self-consistency", "n": len(scored), "nCorrect": sum(y),
              "seeds": {}}
    for seed in (0, 1, 7):
        report["seeds"][seed] = paired_boot(pstab, sc, y, a.n_boot, seed)
    wins = all(report["seeds"][s]["ci95"][0] > 0 for s in (0, 1, 7))
    report["paraphraseBeatsSelfConsistency_allSeeds"] = wins
    report["verdict"] = ("perturbation_stability_beats_self_consistency" if wins
                         else "no_significant_edge_over_self_consistency")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
