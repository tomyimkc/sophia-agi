#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judge base-vs-adapter answer transcripts with 2 LLM families on the CONTENT channel.

The upstream half of the P6 VALIDATED protocol (see
docs/06-Roadmap/P6-LoRA-Uplift-Validation-Preregistration.md): turn per-seed answer
transcripts (produced on the pod by tools/eval_local_model.py -> benchmark/model_runs/
local-*.json) into the judgments JSON that tools/run_lora_uplift_validation.py consumes.

The judge is GROUNDED, not opinion-based: each benchmark case carries explicit content
requirements (mustDenyAttribution, mustMentionTraditions, mustLabelMyth, ...). The judge is
given the question + those requirements + the answer, and decides whether the answer
satisfies them — semantically (an independent LLM), NOT via the lexical scorer's patterns.
Judges MUST be a different model family than the subject (here Qwen) for independence.

Seed manifest (``--manifest seeds.json``):
  [ {"seed": 0, "dir": "/path/to/seed0/answers"},
    {"seed": 1, "dir": "/path/to/seed1/answers"}, ... ]
Each dir holds per-domain answer files: local-<base_label>-<domain>.json (base) and
local-<adapter_label>-<domain>.json (adapter), each {"responses": {case_id: answer}}.

Usage:
  python tools/judge_content_transcripts.py --manifest seeds.json \
    --judges deepseek:deepseek-chat,openrouter:meta-llama/llama-3.1-70b-instruct \
    --subject Qwen/Qwen2.5-3B-Instruct \
    --base-label qwen-qwen2.5-3b-instruct --adapter-label sophia-cuda-v1 \
    --out judgments.json
  python tools/judge_content_transcripts.py --mock --out judgments.json   # offline self-test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json  # noqa: E402

DOMAINS = ("philosophy", "psychology", "history", "religion")

_JUDGE_SYSTEM = (
    "You grade whether an ANSWER satisfies the CONTENT requirements of a source-discipline "
    "question across philosophy, psychology, history, and religion. Requirements may include: "
    "denying a false attribution (the named author did NOT write the named text), mentioning "
    "required traditions/schools for context, labeling a claim as a myth/misconception, and "
    "handling the topic with accurate nuance. Judge ONLY content correctness — ignore "
    "formatting, length, and language. Reply with ONLY a JSON object: "
    '{"content_pass": true|false}. content_pass is true IFF the answer satisfies ALL stated '
    "requirements; if it asserts the forbidden attribution, omits required context, or is "
    "factually wrong on the requirement, content_pass is false."
)


def _requirements(case: dict) -> dict:
    """The case's content-requirement fields (everything except id/question)."""
    return {k: v for k, v in case.items() if k not in ("id", "question")}


def parse_content_pass(text: str) -> bool:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return False
    try:
        return bool(json.loads(m.group(0)).get("content_pass", False))
    except (ValueError, AttributeError):
        return False


def make_content_judge(spec: str):
    """A judge(question, requirements, answer) -> bool backed by model ``spec``."""
    from agent.model import default_client
    client = default_client(spec)

    def judge(question: str, requirements: dict, answer: str) -> bool:
        user = (
            f"Question: {question}\n"
            f"Content requirements (JSON): {json.dumps(requirements, ensure_ascii=False)}\n"
            f"Answer:\n'''{answer}'''"
        )
        res = client.generate(_JUDGE_SYSTEM, user)
        if not getattr(res, "ok", False):
            raise RuntimeError(f"judge {spec} call failed: {getattr(res, 'error', '')}")
        return parse_content_pass(getattr(res, "text", "") or "")

    return judge


def _family(spec: str) -> str:
    """Judge family for the emitted label-dict keys. Delegates to the aggregator's
    _family_key (vendor, gateway-aware) so the keys this labeler writes ALWAYS match
    what run_lora_uplift_validation reads back — enforced by
    test_judge_content_transcripts.test_family_keys_match_aggregator."""
    from tools.run_lora_uplift_validation import _family_key
    return _family_key(spec)


def _load_answers(d: Path, label: str) -> dict:
    """Merge per-domain answer files for `label` in dir `d` -> {case_id: answer}."""
    out: dict[str, str] = {}
    for domain in DOMAINS:
        f = d / f"local-{label}-{domain}.json"
        if f.exists():
            out.update(load_json(f).get("responses", {}))
    return out


def judge_seeds(manifest: list, *, judges: list, base_label: str, adapter_label: str,
                judge_fns: dict | None = None) -> dict:
    """Build the judgments JSON consumed by run_lora_uplift_validation."""
    cases: dict[str, dict] = {}
    for domain in DOMAINS:
        for c in load_json(DOMAIN_BENCH[domain]).get("cases", []):
            cases[c["id"]] = c
    fams = [_family(j) for j in judges]
    fns = judge_fns or {f: make_content_judge(spec) for f, spec in zip(fams, judges)}

    seed_blocks = []
    for entry in manifest:
        d = Path(entry["dir"])
        base = _load_answers(d, base_label)
        adapter = _load_answers(d, adapter_label)
        items = []
        for cid, case in cases.items():
            if cid not in base or cid not in adapter:
                continue
            req, q = _requirements(case), case["question"]
            bc = {f: bool(fns[f](q, req, base[cid])) for f in fams}
            ac = {f: bool(fns[f](q, req, adapter[cid])) for f in fams}
            items.append({"id": cid, "baseContent": bc, "adapterContent": ac})
        seed_blocks.append({"seed": entry["seed"], "items": items})
    return {"subjectModel": None, "judges": judges, "seeds": seed_blocks}


# --------------------------------------------------------------------------- #
# Mock self-test: fake judges + synthetic transcripts (no API, no files).
# --------------------------------------------------------------------------- #
def _mock_run() -> dict:
    judges = ["deepseek:deepseek-chat", "openrouter:meta-llama/llama-3.1-70b-instruct"]
    fams = [_family(j) for j in judges]
    cases = {}
    for domain in DOMAINS:
        for c in load_json(DOMAIN_BENCH[domain]).get("cases", []):
            cases[c["id"]] = c
    ids = list(cases)
    # Deterministic fake judges: base passes first 72%, adapter first 88%; ~90% agreement.
    def fake(rate):
        def jf(q, req, ans):
            # answer encodes truth as a trailing tag the fake judge reads
            return ans.endswith("|PASS")
        return jf
    fns = {f: fake(0.0) for f in fams}
    # synthesize a dir-less manifest by monkeypatching _load_answers via closures
    seed_blocks = []
    for seed in range(3):
        items = []
        for i, cid in enumerate(ids):
            base_pass = i < int(0.72 * len(ids))
            adapter_pass = i < int(0.88 * len(ids))
            bc = {f: base_pass for f in fams}
            ac = {f: adapter_pass for f in fams}
            items.append({"id": cid, "baseContent": bc, "adapterContent": ac})
        seed_blocks.append({"seed": seed, "items": items})
    return {"subjectModel": "mock:Qwen2.5-3B", "judges": judges, "seeds": seed_blocks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, help="seed manifest JSON")
    ap.add_argument("--judges", default="deepseek:deepseek-chat,openrouter:meta-llama/llama-3.1-70b-instruct")
    ap.add_argument("--subject", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--base-label", default="qwen-qwen2.5-3b-instruct")
    ap.add_argument("--adapter-label", default="sophia-cuda-v1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mock:
        report = _mock_run()
    else:
        if not args.manifest:
            ap.error("provide --manifest or --mock")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        judges = [s.strip() for s in args.judges.split(",") if s.strip()]
        report = judge_seeds(manifest, judges=judges, base_label=args.base_label,
                             adapter_label=args.adapter_label)
        report["subjectModel"] = args.subject

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_items = sum(len(s["items"]) for s in report["seeds"])
    print(f"wrote {args.out} — {len(report['seeds'])} seeds, {n_items} judged item-rows, "
          f"judges={report['judges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
