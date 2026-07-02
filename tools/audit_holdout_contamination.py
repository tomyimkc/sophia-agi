#!/usr/bin/env python3
"""T1: audit committed training packs for benchmark-holdout contamination.
Forbidden set = eval_prompt_set (repo built-in) UNION the 7 tests/benchmark-*.json domain holdouts
UNION sealed tool-use + hk-advisor benchmarks. Generalizes the philosophy contamination find
(26/528) to every committed pack. Reports per-file overlap counts + row indices + which holdout.
NO data is edited — audit only."""
import json, re, glob, sys, os
from pathlib import Path
sys.path.insert(0, ".")
from provenance_bench.dataset_guard import normalize, prompt_of, eval_prompt_set, \
    tool_use_benchmark_prompt_set, hk_advisor_benchmark_prompt_set

ROOT = Path(".")

def bench_holdout_prompts():
    """Normalized prompts from the 7 tests/benchmark-*.json domain holdouts, tagged by domain."""
    tagged = {}
    for f in sorted(glob.glob("tests/benchmark-*.json")):
        dom = Path(f).stem.replace("benchmark-", "")
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cases = d.get("cases", d if isinstance(d, list) else [])
        for c in cases:
            q = c.get("question") or c.get("prompt") or c.get("input") or ""
            n = normalize(q)
            if n:
                tagged[n] = f"benchmark-{dom}"
    return tagged

# forbidden set (normalized -> source label)
forbidden = {}
for n in eval_prompt_set(root=ROOT): forbidden.setdefault(n, "eval_prompt_set(eval/** + wisdom_market)")
try:
    for n in tool_use_benchmark_prompt_set(root=ROOT): forbidden.setdefault(n, "tool_use_benchmark")
except Exception: pass
try:
    for n in hk_advisor_benchmark_prompt_set(root=ROOT): forbidden.setdefault(n, "hk_advisor_benchmark")
except Exception: pass
for n, lab in bench_holdout_prompts().items(): forbidden.setdefault(n, lab)   # THE gap #362 fixes
print(f"forbidden set size: {len(forbidden)} normalized prompts")

def row_prompt(r):
    """Robust prompt extraction across schemas (messages / prompt / question / DPO chosen)."""
    p = prompt_of(r)
    if p: return p
    for k in ("prompt", "question", "input", "text"):
        if isinstance(r.get(k), str): return r[k]
    # DPO pair {prompt|chosen|rejected} where chosen is messages
    for k in ("chosen", "rejected"):
        v = r.get(k)
        if isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and m.get("role") == "user": return m.get("content", "")
    return ""

FILES = ["training/corpus.jsonl", "training/moral_gate_sft.jsonl",
         "training/council/traces.jsonl", "training/council/religion_repair_c4.jsonl",
         "training/tool_use/sft_traces.jsonl", "training/tool_use/dpo_pairs.jsonl",
         "training/self_evolve/distill.jsonl"]

def sweep(path):
    if not os.path.exists(path): return None
    rows = [l for l in open(path).read().splitlines() if l.strip()]
    hits = []
    for i, l in enumerate(rows):
        try: r = json.loads(l)
        except Exception: continue
        n = normalize(row_prompt(r))
        if n and n in forbidden:
            hits.append((i, forbidden[n]))
    return len(rows), hits

print("\n=== PER-FILE HOLDOUT-CONTAMINATION AUDIT ===")
results = {}
for f in FILES + (sys.argv[1:] if len(sys.argv) > 1 else []):
    res = sweep(f)
    if res is None:
        print(f"  {f}: (missing)"); continue
    tot, hits = res
    results[f] = {"rows": tot, "overlap": len(hits),
                  "indices": [i for i, _ in hits],
                  "byHoldout": {lab: sum(1 for _, l in hits if l == lab) for lab in {l for _, l in hits}}}
    flag = "⚠️ LEAK" if hits else "clean"
    print(f"  [{flag}] {f}: {len(hits)}/{tot} overlap" + (f"  byHoldout={results[f]['byHoldout']} indices={results[f]['indices'][:20]}" if hits else ""))

open("/tmp/audit_results.json", "w").write(json.dumps(results, indent=1) + "\n")
total_leak = sum(r["overlap"] for r in results.values())
print(f"\nTOTAL overlap across committed packs: {total_leak}")
