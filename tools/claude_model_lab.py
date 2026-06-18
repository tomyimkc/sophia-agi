#!/usr/bin/env python3
"""Claude API model lab — review, distill, judge, package local LLM.

Usage:
  python tools/claude_model_lab.py review-batch --limit 10
  python tools/claude_model_lab.py distill --limit 20
  python tools/claude_model_lab.py judge --report benchmark/model_runs/local-*.report.json
  python tools/claude_model_lab.py write-modelfile --adapter training/lora/checkpoints/sophia-v1
  python tools/claude_model_lab.py run-all --review-limit 5 --distill-limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_dotenv, normalize_api_keys  # noqa: E402
from agent.llm import complete  # noqa: E402
from tools.model_lab_lib import (  # noqa: E402
    DISTILL_DIR,
    JUDGE_DIR,
    MODELS_DIR,
    OLLAMA_DIR,
    HF_MODEL_DIR,
    REVIEWS_DIR,
    RUNS_DIR,
    TRAINING_DIR,
    adapter_config,
    build_hf_model_card,
    build_modelfile,
    distill_specs_from_attributions,
    example_to_text,
    find_failed_cases,
    find_local_run_responses,
    load_json,
    parse_json_response,
    sample_teacher_examples,
)
from sophia_mcp.tools_impl import corpus_stats  # noqa: E402

REVIEW_SYSTEM = (
    "You are a Sophia AGI corpus QA reviewer. Output JSON only. "
    "Check: no lineage merge, correct deny/affirm traps, uncertainty labels, 中文 summary, no invented citations."
)
DISTILL_SYSTEM = (
    "You are a Sophia AGI teacher. Output one JSON object: {user, assistant, metadata}. "
    "metadata must include domain, source=claude-distill, textIds if provided. End assistant with 中文 summary."
)
JUDGE_SYSTEM = (
    "You are a Sophia benchmark judge. Output JSON: {verdict: pass|fail, reasons: [], "
    "suggested_fix: string, training_worthy: bool}. Compare model answer to trap requirements."
)


def write_distill_example(index: int, item: dict, spec: dict) -> Path:
    domain = item.get("metadata", {}).get("domain") or spec.get("domain", "philosophy")
    metadata = {
        "source": "claude-distill",
        "project": "sophia-agi",
        "domain": domain,
        "trap": spec.get("trap"),
    }
    if spec.get("textIds"):
        metadata["textIds"] = spec["textIds"]
    system = (
        "You are a precise philosophy instructor specializing in source discipline."
        if domain == "philosophy"
        else f"You are a {domain} instructor using source discipline."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ],
        "metadata": metadata,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", spec.get("trap", "distill"))[:36].strip("-")
    path = DISTILL_DIR / f"distill-{index:03d}-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def next_training_index() -> int:
    nums = [int(p.name[:3]) for p in TRAINING_DIR.glob("*.json") if re.match(r"^\d{3}", p.name)]
    return (max(nums) if nums else 0) + 1


def cmd_review_batch(args: argparse.Namespace) -> int:
    paths = sample_teacher_examples(args.limit, seed=args.seed)
    print(f"Reviewing {len(paths)} example(s)")
    if args.dry_run:
        for path in paths:
            print(f"  {path.name}")
        return 0

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    batch = [example_to_text(p) for p in paths]
    prompt = (
        f"Review these {len(batch)} Sophia training examples. "
        "Return JSON array: {{file, pass: bool, issues: [str], severity: low|medium|high}}.\n\n"
        + json.dumps(batch, ensure_ascii=False, indent=2)
    )
    raw = complete(REVIEW_SYSTEM, prompt, max_tokens=6000)
    results = parse_json_response(raw)
    if isinstance(results, dict):
        results = [results]

    passed = sum(1 for r in results if r.get("pass"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REVIEWS_DIR / f"review-{stamp}.json"
    payload = {"reviewed": len(results), "passed": passed, "results": results}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Review: {passed}/{len(results)} passed → {out}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    specs = distill_specs_from_attributions(args.limit)
    print(f"Distill specs: {len(specs)}")
    if args.dry_run:
        for spec in specs[:5]:
            print(f"  {spec['trap']}: {spec['user'][:60]}...")
        return 0

    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, spec in enumerate(specs):
        prompt = (
            f"Domain: {spec['domain']}\nQuestion: {spec['user']}\n"
            f"textIds: {spec.get('textIds', [])}\ntrap: {spec.get('trap')}"
        )
        try:
            raw = complete(DISTILL_SYSTEM, prompt, max_tokens=2500)
            item = parse_json_response(raw)
            if isinstance(item, list):
                item = item[0]
        except Exception as exc:
            print(f"  skip {spec['trap']}: {exc}")
            continue
        if not item.get("user") or not item.get("assistant"):
            continue
        path = write_distill_example(idx + 1, item, spec)
        written += 1
        print(f"  wrote {path.name}")

    print(f"Distilled {written} → {DISTILL_DIR}")
    if args.promote and written:
        start = next_training_index()
        promoted = 0
        for i, path in enumerate(sorted(DISTILL_DIR.glob("distill-*.json"))):
            dest = TRAINING_DIR / f"{start + i:03d}-{path.stem}.json"
            dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            promoted += 1
        print(f"Promoted {promoted} to training/examples/")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    reports = []
    if args.report:
        reports = [Path(args.report)]
    else:
        reports = sorted(RUNS_DIR.glob("local-*.report.json"))
    if not reports:
        print("No local benchmark reports found. Run eval_local_model.py first.")
        return 0

    failures: list[dict] = []
    for report_path in reports:
        if not report_path.exists():
            continue
        case_failures = find_failed_cases(report_path)
        responses = find_local_run_responses(report_path)
        for failure in case_failures:
            failure["bad_response"] = responses.get(failure["case_id"], "")[:1500]
            failures.append(failure)

    print(f"Failures to judge: {len(failures)}")
    if args.dry_run or not failures:
        for failure in failures[:10]:
            print(f"  {failure['domain']}/{failure['case_id']}: {failure.get('reasons')}")
        return 0

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    judgements = []
    for failure in failures[: args.limit]:
        case = failure.get("case", {})
        prompt = f"""Case: {failure['case_id']} ({failure['domain']})
Question: {case.get('question', '')}
Heuristic failures: {failure.get('reasons', [])}
Model answer:
{failure.get('bad_response', '')}
"""
        try:
            raw = complete(JUDGE_SYSTEM, prompt, max_tokens=1500)
            verdict = parse_json_response(raw)
            if isinstance(verdict, list):
                verdict = verdict[0]
        except Exception as exc:
            verdict = {"verdict": "error", "reasons": [str(exc)]}
        record = {**failure, "judge": verdict}
        judgements.append(record)
        print(f"  {failure['case_id']}: {verdict.get('verdict', '?')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = JUDGE_DIR / f"judge-{stamp}.json"
    out.write_text(json.dumps(judgements, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")

    if args.generate_corrections:
        from agent.correction_loop import draft_correction, write_pending  # noqa: E402

        for record in judgements:
            judge = record.get("judge", {})
            if not judge.get("training_worthy"):
                continue
            try:
                example = draft_correction(record, record.get("bad_response", ""))
                write_pending(example, record["case_id"])
            except Exception as exc:
                print(f"  correction skip {record['case_id']}: {exc}")
    return 0


def cmd_write_modelfile(args: argparse.Namespace) -> int:
    adapter = Path(args.adapter) if args.adapter else None
    cfg = adapter_config(adapter)
    stats = corpus_stats()
    modelfile = build_modelfile(cfg)
    card = build_hf_model_card(cfg, stats)

    OLLAMA_DIR.mkdir(parents=True, exist_ok=True)
    HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    modelfile_path = OLLAMA_DIR / "Modelfile"
    card_path = HF_MODEL_DIR / "README.md"
    modelfile_path.write_text(modelfile, encoding="utf-8")
    card_path.write_text(card, encoding="utf-8")

    manifest = {
        "version": cfg.get("version"),
        "baseModel": cfg.get("baseModel"),
        "adapterPath": cfg.get("adapterPath"),
        "ollamaModelfile": str(modelfile_path.relative_to(ROOT)),
        "hfModelCard": str(card_path.relative_to(ROOT)),
        "stats": stats,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {modelfile_path}")
    print(f"Wrote {card_path}")
    print(f"Wrote {MODELS_DIR / 'manifest.json'}")
    print("Create Ollama model: ollama create sophia-7b -f models/ollama/Modelfile")
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    print("=== Sophia Model Lab: run-all ===\n")
    steps = [
        ("prepare_lora_dataset", ["python", "tools/prepare_lora_dataset.py"]),
        ("write_modelfile", None),
    ]
    if args.review_limit > 0:
        steps.insert(0, ("review_batch", None))
    if args.distill_limit > 0:
        steps.append(("distill", None))

    import subprocess

    subprocess.run([sys.executable, "tools/prepare_lora_dataset.py"], cwd=ROOT, check=False)

    if args.review_limit > 0:
        rc = cmd_review_batch(argparse.Namespace(
            limit=args.review_limit, seed=args.seed, dry_run=args.dry_run
        ))
        if rc:
            return rc

    if args.distill_limit > 0:
        rc = cmd_distill(argparse.Namespace(
            limit=args.distill_limit, dry_run=args.dry_run, promote=False
        ))
        if rc:
            return rc

    cmd_write_modelfile(argparse.Namespace(adapter=args.adapter))

    reports = list(RUNS_DIR.glob("local-*.report.json"))
    if reports:
        cmd_judge(argparse.Namespace(
            report=None,
            limit=args.judge_limit,
            dry_run=args.dry_run,
            generate_corrections=args.generate_corrections,
        ))
    else:
        print("Skip judge — no local benchmark reports (run eval_local_model.py after training)")

    print("\n=== Next: train + eval ===")
    print("  python tools/train_lora.py --4bit --epochs 3")
    print("  python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate")
    return 0


def main() -> int:
    load_dotenv()
    normalize_api_keys()

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--dry-run", action="store_true", help="No Claude API calls")

    parser = argparse.ArgumentParser(description="Claude model lab for Sophia local LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    p_review = sub.add_parser("review-batch", parents=[parent], help="Claude QA on teacher examples")
    p_review.add_argument("--limit", type=int, default=10)
    p_review.add_argument("--seed", type=int, default=42)

    p_distill = sub.add_parser("distill", parents=[parent], help="Claude gold answers for new questions")
    p_distill.add_argument("--limit", type=int, default=20)
    p_distill.add_argument("--promote", action="store_true", help="Copy distill/*.json to training/examples")

    p_judge = sub.add_parser("judge", parents=[parent], help="Claude judge on failed local benchmark cases")
    p_judge.add_argument("--report", type=str, default="")
    p_judge.add_argument("--limit", type=int, default=20)
    p_judge.add_argument("--generate-corrections", action="store_true")

    p_model = sub.add_parser("write-modelfile", parents=[parent], help="Ollama Modelfile + HF model card")
    p_model.add_argument("--adapter", type=str, default="training/lora/checkpoints/sophia-v1")

    p_all = sub.add_parser("run-all", parents=[parent], help="Review + distill + modelfile + judge")
    p_all.add_argument("--review-limit", type=int, default=5)
    p_all.add_argument("--distill-limit", type=int, default=10)
    p_all.add_argument("--judge-limit", type=int, default=10)
    p_all.add_argument("--adapter", type=str, default="training/lora/checkpoints/sophia-v1")
    p_all.add_argument("--generate-corrections", action="store_true")
    p_all.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    handlers = {
        "review-batch": cmd_review_batch,
        "distill": cmd_distill,
        "judge": cmd_judge,
        "write-modelfile": cmd_write_modelfile,
        "run-all": cmd_run_all,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())