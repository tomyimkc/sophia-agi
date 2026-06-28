#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke-test a local LLM judge endpoint on the DGX Spark before running metered evals.

The Spark's best-fit workload is the multi-judge / multi-seed eval farm with judges on a
LOCAL vLLM/SGLang server instead of metered OpenRouter. This helper:

  1. configures the local-judge env (SOPHIA_MODEL_PROVIDER / SOPHIA_MODEL_BASE_URL);
  2. probes each judge model via ``agent.model.complete(spec="<provider>:<model>")`` so you
     know the endpoint answers before you launch a long judged run;
  3. prints the recommended ``--judge`` flags for the no-overclaim >=2-family gate using
     ONLY local models (two distinct model families behind one local endpoint).

Honest scope: this is a connectivity/config smoke test, not a benchmark. With
``--provider mock`` it runs fully offline (no endpoint, no GPU) so it is CI-safe.

    python tools/run_local_judge_eval.py --provider vllm --base-url http://localhost:8000/v1 \
        --judge-models Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.3-8B-Instruct
    python tools/run_local_judge_eval.py --dry-run        # print config, no probe
    python tools/run_local_judge_eval.py --provider mock   # offline self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASE_URLS = {"vllm": "http://localhost:8000/v1", "sglang": "http://localhost:30000/v1",
                     "ollama": "http://localhost:11434/v1", "mock": ""}
_PROBE_PROMPT = (
    "Reply with exactly one word: is the following statement well-formed? 'Water boils at 100C "
    "at sea level.' Answer yes or no."
)


def _spec(provider: str, model: str) -> str:
    # mock needs the bare "mock" preset (no model suffix); real local servers use provider:model.
    return "mock" if provider == "mock" else f"{provider}:{model}"


def _probe(provider: str, model: str) -> dict:
    from agent.model import complete  # noqa: PLC0415 — lazy; keeps --dry-run/import light

    spec = _spec(provider, model)
    t0 = time.perf_counter()
    try:
        out = complete("You are a strict but fair evaluator.", _PROBE_PROMPT, spec=spec, max_tokens=16)
        ok = bool(out and out.strip())
        return {"model": model, "spec": spec, "ok": ok, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "reply": (out or "").strip()[:80]}
    except Exception as exc:  # noqa: BLE001 — surface any endpoint error, don't crash the farm
        return {"model": model, "spec": spec, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["vllm", "sglang", "ollama", "mock"], default="vllm",
                    help="local OpenAI-compatible server preset (agent.model.py)")
    ap.add_argument("--base-url", default=None, help="override; defaults to the preset's localhost URL")
    ap.add_argument("--judge-models", default="Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.3-8B-Instruct",
                    help="comma list; >=2 DISTINCT models satisfy the no-overclaim >=2-family gate locally")
    ap.add_argument("--dry-run", action="store_true", help="print config + recommended flags, no probe")
    ap.add_argument("--config", default=None,
                    help="load a two-box judge-farm config (config/inference.local.mac-judge.json): "
                         "emit its ready --judges flag + each box's serve command. Print-only, CI-safe.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.config:
        # Fail gracefully (structured error + non-zero exit) on missing file / bad JSON /
        # missing keys — this path is print-only and must never crash with a traceback.
        try:
            farm = json.loads(Path(args.config).read_text(encoding="utf-8"))
            jf = farm["judge_farm"]
            boxes = farm["boxes"]
            out = {
                "config": args.config,
                "judges": jf["judges"],
                "recommended_judge_flag": jf["recommended_flag"],
                "expected_families": jf.get("expected_families"),
                "serve_commands": {name: box["serve"] for name, box in boxes.items()},
                "note": ("Fill in SPARK_HOST/MAC_HOST, run each serve command on its box, then pass the "
                         "recommended --judges flag to a judged eval (e.g. tools/judge_pilot_answers.py)."),
            }
        except FileNotFoundError:
            print(json.dumps({"error": "config not found", "config": args.config}, indent=2))
            return 1
        except OSError as e:   # IsADirectoryError, PermissionError, etc. — still no traceback
            print(json.dumps({"error": f"cannot read config: {e}", "config": args.config}, indent=2))
            return 1
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"invalid JSON: {e}", "config": args.config}, indent=2))
            return 1
        except (KeyError, TypeError) as e:
            print(json.dumps({"error": f"missing/invalid judge-farm key: {e}",
                              "config": args.config,
                              "expected": "judge_farm.{judges,recommended_flag}, boxes.<name>.serve"},
                             indent=2))
            return 1
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    base_url = args.base_url or DEFAULT_BASE_URLS[args.provider]
    models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    # set the local-judge env so any downstream judged tool inherits it
    if args.provider != "mock":
        os.environ["SOPHIA_MODEL_PROVIDER"] = args.provider
        if base_url:
            os.environ["SOPHIA_MODEL_BASE_URL"] = base_url

    judge_flags = [f"--judge {_spec(args.provider, m)}" for m in models]
    config = {"provider": args.provider, "base_url": base_url, "judge_models": models,
              "recommended_judge_flags": " ".join(judge_flags),
              "note": ("Two DISTINCT local models behind one endpoint count as 2 families for the "
                       "no-overclaim gate; or pair one local + one metered cloud family.")}

    if args.dry_run or args.provider == "mock":
        # mock/dry-run path: never hit a network endpoint — CI-safe.
        if args.provider == "mock" and not args.dry_run:
            results = [_probe("mock", m) for m in models]
            config["probe_results"] = results
        out = dict(config)
        out["dry_run"] = args.dry_run
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    results = [_probe(args.provider, m) for m in models]
    config["probe_results"] = results
    print(json.dumps(config, ensure_ascii=False, indent=2))
    # exit non-zero only if NOTHING answered (one model down is a warning, not a failure)
    return 0 if any(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
