#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rejection-sampling fine-tuning (STaR / RFT) gated by the provenance gate.

The repo's unique speed advantage: the deterministic gate is the *reward model*.
For each seed task we sample N candidate answers from a model client, then run the
INTRINSIC fail-closed gate on each candidate -- ``check_response(text,
mode="advisor")["violations"]`` WITHOUT a question. A candidate is kept as an SFT
target iff it is gate-CLEAN (no fabricated citation / false arithmetic /
forbidden-lineage merge). No trap-grader is invoked here: passing a question would
run the attribution *positive-expectation* grader, which would wrongly reject
correctly-disciplined answers phrased differently -- so we never filter curated
targets with it (mirrors tools/train_lora.py:guard_filter).

CRITICAL fail-closed property: a gate-clean ABSTENTION (an answer that says it
cannot verify / routes to a human) is a CORRECT, reward-positive output and is
KEPT. RFT must never train abstention out of the student -- abstaining when an
authority cannot be verified is exactly the discipline we are distilling. We tag
each kept target as kind=answer|abstention and never drop a clean abstention.

Output is ``{"messages":[...], "metadata":{...}}`` JSONL, consumable directly by
tools/train_lora.py (same schema as tools/distill_council_traces.py). Identical
kept targets (same prompt+answer) are de-duplicated.

Generation dominates RL/RFT wall-clock (~70-80%); vLLM/SGLang batched decoding is
the lever (the training/optimizer step is not the bottleneck). The ``--gen-backend
vllm`` path lazily loads vLLM once and BATCH-generates all N candidates per prompt
in a single engine pass; the gate-filter + keep-clean-abstentions logic below is
unchanged regardless of backend.

    # offline plumbing (mock client, writes nothing)
    python tools/run_rft.py --model mock --dry-run
    # real RFT pass (native, per-call generation)
    python tools/run_rft.py --model openrouter:deepseek/deepseek-chat --samples 6 \
        --out training/rft/traces.jsonl
    # real RFT pass with batched vLLM generation (the speed lever)
    python tools/run_rft.py --model Qwen/Qwen2.5-3B-Instruct --gen-backend vllm \
        --samples 8 --out training/rft/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402

TASKS = ROOT / "data" / "council_tasks.json"
OUT = ROOT / "training" / "rft" / "traces.jsonl"

SYSTEM = (
    "You are a source-disciplined advisor. Answer the question, stating each finding with "
    "a source where one is relied on, then give one decision. If you cannot verify a needed "
    "authority or figure, ABSTAIN and route to a human rather than guess. Label clearly as "
    "not professional advice; end with a 中文摘要."
)

# An answer that abstains is reward-POSITIVE. Mark it so it is never trained out.
ABSTAIN_MARKERS = (
    "abstain", "cannot verify", "can't verify", "insufficient", "unable to verify",
    "consult a", "seek a", "refer to a", "route to a", "需要核實", "無法核實", "諮詢專業",
)


def _is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in ABSTAIN_MARKERS)


def _gen(client, system: str, user: str) -> str:
    """Single completion as plain text; broken/failed client yields '' (mirrors
    agent.council_deliberate._gen)."""
    try:
        res = client.generate(system, user)
    except Exception:  # noqa: BLE001 - a broken client yields no content, not a crash
        return ""
    return (getattr(res, "text", "") or "").strip() if getattr(res, "ok", False) else ""


def _filter_candidates(task: dict, cands: "list[str]") -> "tuple[list[dict], dict]":
    """Run the INTRINSIC fail-closed gate over already-generated candidates and keep
    only gate-CLEAN ones. This holds the reward logic and is backend-agnostic: native
    per-call and batched-vLLM generation both feed their candidates through here, so
    the gate-filter + KEEP-clean-abstentions behaviour is identical for every backend.
    Returns (kept_rows, stats). Clean abstentions are kept and tagged."""
    prompt = task["prompt"]
    rows: list[dict] = []
    clean = dirty = empty = abstentions = 0
    for cand in cands:
        if not cand.strip():
            empty += 1
            continue
        # INTRINSIC fail-closed gate: NO question -> no trap grader.
        if check_response(cand, mode="advisor")["violations"]:
            dirty += 1
            continue
        clean += 1
        kind = "abstention" if _is_abstention(cand) else "answer"
        if kind == "abstention":
            abstentions += 1  # KEEP: a clean abstention is a correct, reward-positive target
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": cand},
            ],
            "metadata": {"taskId": task.get("id"), "kind": kind, "gatePassed": True,
                         "source": "rft", "labelStatus": "rft-sample"},
        })
    stats = {"clean": clean, "dirty": dirty, "empty": empty, "abstentions": abstentions}
    return rows, stats


def sample_task(task: dict, client, *, samples: int) -> "tuple[list[dict], dict]":
    """Sample N candidates for one task (native per-call generation), keep only
    gate-CLEAN ones (intrinsic check, no question). Returns (kept_rows, stats)."""
    cands = [_gen(client, SYSTEM, task["prompt"]) for _ in range(samples)]
    return _filter_candidates(task, cands)


def _vllm_batch_candidates(tasks: list[dict], cfg, *, samples: int) -> "dict[str, list[str]]":
    """Generate ``samples`` candidates for EVERY task in a single batched vLLM pass.

    This is the wall-clock lever: generation is ~70-80% of RL/RFT time, and vLLM's
    continuous batching decodes all (tasks x samples) sequences together far faster
    than per-call native generation. We build the engine once, format prompts with the
    tokenizer's chat template, and use ``SamplingParams(n=samples)`` so the N candidates
    per prompt come from one ``llm.generate`` call. Returns {taskId: [cand, ...]}.

    vllm/torch are imported lazily INSIDE this function — they are heavy GPU deps and
    must never be a top-level import. The caller falls back to native on ImportError.
    """
    from vllm import LLM, SamplingParams  # lazy: heavy GPU dependency

    llm = LLM(model=cfg.model)
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    for task in tasks:
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task["prompt"]}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))
    params = SamplingParams(n=samples, temperature=max(0.7, cfg.temperature),
                            max_tokens=cfg.max_tokens)
    outputs = llm.generate(prompts, params)
    by_task: dict[str, list[str]] = {}
    for task, out in zip(tasks, outputs):
        by_task[task.get("id")] = [(o.text or "").strip() for o in out.outputs]
    return by_task


def generate_rft(tasks: list[dict], client, *, samples: int = 4,
                 max_keep: int = 0,
                 candidates: "dict[str, list[str]] | None" = None) -> "tuple[list[dict], dict]":
    """Run rejection sampling over every task. De-dups identical kept targets
    (same prompt + assistant text). ``max_keep`` (0 = unlimited) caps the dataset.

    When ``candidates`` (a {taskId: [text,...]} map, e.g. from a batched vLLM pass) is
    supplied, those pre-generated texts are gate-filtered instead of calling the client
    per task; the gate-filter + dedup + cap + abstention logic is otherwise identical."""
    rows: list[dict] = []
    seen: set = set()
    clean = dirty = empty = abstentions = deduped = 0
    for task in tasks:
        if candidates is not None:
            task_rows, st = _filter_candidates(task, candidates.get(task.get("id"), []))
        else:
            task_rows, st = sample_task(task, client, samples=samples)
        clean += st["clean"]
        dirty += st["dirty"]
        empty += st["empty"]
        abstentions += st["abstentions"]
        for row in task_rows:
            key = (row["metadata"]["taskId"], row["messages"][-1]["content"])
            if key in seen:
                deduped += 1
                continue
            seen.add(key)
            rows.append(row)
            if max_keep and len(rows) >= max_keep:
                stats = {"tasks": len(tasks), "samples": samples, "kept": len(rows),
                         "clean": clean, "dirty": dirty, "empty": empty,
                         "abstentions": abstentions, "deduped": deduped,
                         "cappedAt": max_keep}
                return rows, stats
    stats = {"tasks": len(tasks), "samples": samples, "kept": len(rows),
             "clean": clean, "dirty": dirty, "empty": empty,
             "abstentions": abstentions, "deduped": deduped}
    return rows, stats


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help="model spec (default mock = offline plumbing)")
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--samples", type=int, default=4, help="candidate answers sampled per task")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-keep", type=int, default=0, help="cap total kept targets (0 = unlimited)")
    ap.add_argument("--gen-backend", choices=("native", "vllm"), default="native",
                    help="candidate generation backend; 'vllm' BATCH-generates all "
                         "candidates in one pass (the wall-clock lever). Falls back to "
                         "'native' on ImportError or --model mock.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + task count; write nothing")
    args = ap.parse_args(argv)

    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = ROOT / tasks_path
    tasks = json.loads(tasks_path.read_text("utf-8"))["tasks"]

    if args.dry_run:
        plan = {
            "model": args.model,
            "tasks": len(tasks),
            "samplesPerTask": args.samples,
            "maxCandidates": len(tasks) * args.samples,
            "maxKeep": args.max_keep or "unlimited",
            "genBackend": args.gen_backend + (" (auto-native: mock)" if args.model == "mock" and args.gen_backend == "vllm" else ""),
            "gate": "intrinsic (mode=advisor, no question) — fail-closed",
            "abstentionPolicy": "gate-clean abstentions KEPT (reward-positive)",
            "out": args.out,
        }
        print("RFT plan (dry-run, nothing written):", flush=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    from agent.model import default_client
    client = default_client(args.model)

    # Generation is ~70-80% of RL/RFT wall-clock; the vLLM backend BATCH-generates all
    # candidates in one engine pass. Fall back to native (per-call) on --model mock or
    # an ImportError so the offline/mock path stays fully usable without GPU deps.
    candidates = None
    backend = args.gen_backend
    if backend == "vllm":
        if args.model == "mock":
            print("NOTE: --gen-backend vllm ignored for --model mock; using native "
                  "(offline deterministic) generation.", flush=True)
            backend = "native"
        else:
            try:
                candidates = _vllm_batch_candidates(tasks, client.primary, samples=args.samples)
            except ImportError:
                print("NOTE: vllm not importable; falling back to native per-call "
                      "generation (slower but identical outputs/gate-filtering).", flush=True)
                backend = "native"
                candidates = None

    rows, stats = generate_rft(tasks, client, samples=args.samples,
                               max_keep=args.max_keep, candidates=candidates)
    stats["model"] = args.model
    stats["genBackend"] = backend

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out  # output path outside the repo (e.g. /tmp) — show it absolute
    print(f"wrote {len(rows)} RFT target(s) -> {shown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
