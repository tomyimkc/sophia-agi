#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evaluate a live RLVR adapter against the base model on the held-out split.

This is the missing second half of ``tools/run_rlvr.py``:

    train with verifier reward  ->  evaluate base vs adapter on entity-disjoint holdout

The script has two modes:

``--mode mock`` (default)
    Deterministic CI/smoke mode. It uses synthetic base/adapter completions to
    verify the aggregation and contamination guards without torch or a GPU.

``--mode real``
    Loads ``--model`` and a PEFT adapter from ``--adapter`` and generates
    completions for the held-out rows. This needs a CUDA box for GLM-4-9B.

Important: this evaluator intentionally does NOT mark a capability claim as
validated by itself. The pre-registered claim still needs repeated runs and the
repo's no-overclaim gate. This file creates the per-run evidence needed for that
gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import (  # noqa: E402
    code_dataset,
    code_integrity,
    code_reward,
    invention_dataset,
    math_dataset,
    math_reward,
    ontology_rl_dataset,
    ontology_rl_reward,
    physics_dataset,
    rl_dataset,
    rl_reward,
    step_reward,
)

OUT = ROOT / "agi-proof" / "benchmark-results" / "rlvr.adapter-eval.json"


def _run_capability_panel(args: argparse.Namespace, model_desc: str) -> dict | None:
    """Run the capability-delta panel (attribution/hallucination/calibration).

    Returns the panel report (dict), or ``None`` if it could not run. Imported
    lazily and wrapped so a panel failure never breaks the headline eval report
    — the legacy ``base``/``adapterScore``/``delta`` numbers are already written
    by the caller; the panel is additive evidence.
    """
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "eval_capability_panel", Path(__file__).resolve().parent / "eval_capability_panel.py"
        )
        _mod = _ilu.module_from_spec(_spec)
        assert _spec and _spec.loader
        _spec.loader.exec_module(_mod)
        return _mod.run(
            mode=args.mode,
            model=model_desc if args.mode == "real" else "mock",
            adapter=str(args.adapter) if args.adapter else None,
            limit=args.limit,
            out=None,  # no per-file write; caller embeds the dict in this report
        )
    except Exception as exc:  # non-fatal — panel is additive
        print(f"[capability-panel] skipped: {type(exc).__name__}: {exc}")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "candidateOnly": True}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _mock_completion(case: Any, *, improved: bool) -> str:
    """Synthetic completions for deterministic harness validation."""
    if improved:
        if case.label == "false":
            return f"No, {case.claimed_author} did not write {case.work}; it was written by {case.gold_author}."
        return f"{case.work} was written by {case.gold_author}."
    # Base deliberately makes a mix of provenance mistakes and incomplete answers.
    if case.label == "false":
        return f"{case.claimed_author} wrote {case.work}."
    return f"I am not sure who wrote {case.work}."


def _load_real_generators(
    model_name: str, adapter_path: Path, *, max_new_tokens: int, chat_template: bool = False
):
    """Return (base_generate, adapter_generate). Imported lazily for CPU/CI safety.

    ``chat_template``: when True (the code and step tasks), wrap each prompt in the
    model's own chat template before tokenizing, so an *instruct/chat* base (e.g.
    GLM-4-9B-chat) responds as an assistant and emits an extractable structured
    answer (a fenced code block for code; STEP: lines for step). Without it, a chat
    model fed a raw prompt emits prose, the verifier finds nothing parseable, and
    base AND adapter both score 0 — an eval artifact, not a capability reading (see
    failure-ledger rlvr-code-no-chat-template / step-math-chat-wrap-gap). It is a
    NO-OP for a base/completion model (no chat template -> chat_wrap passes through),
    so a completion base stays on the identical raw-prompt path. Left False for
    math/provenance, which already cleared on the raw-prompt path.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("real RLVR adapter eval needs CUDA; use --mode mock locally")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    tuned = PeftModel.from_pretrained(base, str(adapter_path))
    # Separate base object would double memory. We evaluate base first by disabling
    # adapters, then tuned by enabling them.
    def generate(prompt: str, *, adapter: bool) -> str:
        if adapter:
            tuned.enable_adapter_layers()
        else:
            tuned.disable_adapter_layers()
        text = code_dataset.chat_wrap(tokenizer, prompt) if chat_template else prompt
        inputs = tokenizer(text, return_tensors="pt").to(tuned.device)
        with torch.no_grad():
            out = tuned.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return (lambda p: generate(p, adapter=False), lambda p: generate(p, adapter=True))


def _score_cases(cases: list, completions: dict[str, str], *, records: dict) -> dict:
    rows = []
    rewards = []
    pass1 = 0
    false_positive = 0
    true_cases = 0
    false_assertions = 0
    false_cases = 0
    for case in cases:
        text = completions.get(case.id, "")
        reward, detail = rl_reward.reward_for_case(case, text, records=records)
        rewards.append(float(reward))
        pass1 += int(reward >= 1.0)
        if case.label == "true":
            true_cases += 1
            false_positive += int(reward <= 0.0)
        else:
            false_cases += 1
            false_assertions += int(detail.get("assertedForbidden", False))
        rows.append({
            "case_id": case.id,
            "label": case.label,
            "work": case.work,
            "reward": reward,
            "detail": detail,
            "completion": text,
        })
    return {
        "n": len(cases),
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / len(cases), 4) if cases else 0.0,
        "trueFalsePositiveRate": round(false_positive / true_cases, 4) if true_cases else 0.0,
        "falseForbiddenAssertRate": round(false_assertions / false_cases, 4) if false_cases else 0.0,
        "rows": rows,
    }


def false_positive_regressions(base_rows: list[dict], adapter_rows: list[dict]) -> list[dict]:
    """The TRUE-label cases where the adapter regressed integrity: the base did NOT
    false-positive (reward > 0) but the adapter did (reward <= 0). This is exactly the
    `trueFalsePositiveRate` regression the SSIL protected metric tracks — surfaced per case
    so a regression is diagnosable (which cases flipped), not just an aggregate delta."""
    base_by_id = {r["case_id"]: r for r in base_rows}
    out: list[dict] = []
    for ar in adapter_rows:
        if ar.get("label") != "true":
            continue
        br = base_by_id.get(ar["case_id"])
        if br is None:
            continue
        if float(br["reward"]) > 0.0 and float(ar["reward"]) <= 0.0:
            out.append({
                "case_id": ar["case_id"],
                "work": ar.get("work"),
                "baseReward": br["reward"],
                "adapterReward": ar["reward"],
                "baseCompletion": br.get("completion"),
                "adapterCompletion": ar.get("completion"),
                "adapterDeniedOnTrueCase": bool(ar.get("detail", {}).get("deniedOnTrueCase")),
            })
    return out


def run_eval(args: argparse.Namespace) -> dict:
    data = rl_dataset.build_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    cases = data["eval_cases"]
    if args.limit:
        cases = cases[: args.limit]
    if data["entity_intersection"]:
        raise SystemExit(f"contaminated split: {data['entity_intersection']}")

    if args.mode == "mock":
        base = {c.id: _mock_completion(c, improved=False) for c in cases}
        adapter = {c.id: _mock_completion(c, improved=True) for c in cases}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        base_gen, adapter_gen = _load_real_generators(args.model, args.adapter, max_new_tokens=args.max_new_tokens)
        base = {}
        adapter = {}
        for i, c in enumerate(cases, 1):
            print(f"[eval] {i}/{len(cases)} {c.id}", flush=True)
            base[c.id] = base_gen(c.prompt)
            adapter[c.id] = adapter_gen(c.prompt)
        model_desc = args.model

    base_score = _score_cases(cases, base, records=data["eval_gate_records"])
    adapter_score = _score_cases(cases, adapter, records=data["eval_gate_records"])
    delta = round(adapter_score["meanReward"] - base_score["meanReward"], 4)
    fp_delta = round(adapter_score["trueFalsePositiveRate"] - base_score["trueFalsePositiveRate"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — this is a per-run held-out comparison. Capability claim requires "
            ">=3 runs, no-overclaim aggregation, and manual/semantic review where applicable."
        ),
        "split": {
            "evalCases": len(cases),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "trainSealed": data["train_sealed"],
            "evalSealed": data["eval_sealed"],
            "entityIntersection": data["entity_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"meanReward": delta, "trueFalsePositiveRate": fp_delta},
        # Self-diagnosis: the exact TRUE cases whose integrity the adapter regressed. When
        # fp_delta > 0 this names WHICH cases flipped, so a regression (e.g. the seed-1
        # finding) is diagnosable from the report alone instead of needing the raw rows.
        "falsePositiveRegressions": false_positive_regressions(base_score["rows"], adapter_score["rows"]),
        "checks": {
            "contaminationFree": not data["entity_intersection"],
            "adapterImprovesMeanReward": delta > 0,
            "noFalsePositiveRegression": fp_delta <= args.max_fp_regression,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    if args.capability_panel:
        # Optional capability-delta panel (attribution / hallucination / calibration).
        # Attached as a NEW key only — the legacy base/adapterScore/delta/checks above
        # are untouched, so ingest_rlvr_eval.map_report and aggregate_rlvr_runs keep
        # working unchanged. In real mode this reuses the model already loaded above;
        # in mock mode it uses the panel's own deterministic fixtures.
        report["capabilityPanel"] = _run_capability_panel(args, model_desc)
    report["passed"] = all(report["checks"].values())
    return report


# --------------------------------------------------------------------------- #
# Math task — held-out pass@1 before/after on UNSEEN problem families.
# Reward = sympy math_equivalent (deterministic, no judge), so pass@1 IS the
# capability number; no false-positive axis (every item has a gold).
# --------------------------------------------------------------------------- #
def _mock_completion_math(prob: dict, *, improved: bool) -> str:
    if improved:
        return f"After working it out, the answer is \\boxed{{{prob['gold']}}}."
    return "I'm not sure; maybe \\boxed{0}."  # base deliberately wrong


def _score_problems(problems: list, completions: dict) -> dict:
    rows, rewards, pass1 = [], [], 0
    by_family: dict = {}
    for p in problems:
        text = completions.get(p["id"], "")
        reward, detail = math_reward.reward_for_problem(text, p["gold"])
        rewards.append(float(reward))
        ok = int(reward >= 1.0)
        pass1 += ok
        by_family.setdefault(p["family"], []).append(ok)
        rows.append({"problem_id": p["id"], "family": p["family"], "reward": reward,
                     "detail": detail, "completion": text})
    return {
        "n": len(problems),
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / len(problems), 4) if problems else 0.0,
        "passAt1ByFamily": {f: round(sum(v) / len(v), 4) for f, v in by_family.items()},
        "rows": rows,
    }


def run_eval_math(args: argparse.Namespace) -> dict:
    data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    problems = data["eval_problems"]
    if args.limit:
        problems = problems[: args.limit]
    if data["family_intersection"]:
        raise SystemExit(f"contaminated split: {data['family_intersection']}")

    if args.mode == "mock":
        base = {p["id"]: _mock_completion_math(p, improved=False) for p in problems}
        adapter = {p["id"]: _mock_completion_math(p, improved=True) for p in problems}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        if not math_reward.sympy_available():
            raise SystemExit("math adapter eval needs sympy; pip install -r requirements-math.txt")
        base_gen, adapter_gen = _load_real_generators(args.model, args.adapter, max_new_tokens=args.max_new_tokens)
        base, adapter = {}, {}
        for i, p in enumerate(problems, 1):
            print(f"[eval-math] {i}/{len(problems)} {p['id']}", flush=True)
            base[p["id"]] = base_gen(p["prompt"])
            adapter[p["id"]] = adapter_gen(p["prompt"])
        model_desc = args.model

    base_score = _score_problems(problems, base)
    adapter_score = _score_problems(problems, adapter)
    delta = round(adapter_score["passAt1"] - base_score["passAt1"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "math",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — per-run held-out comparison on UNSEEN problem families. Capability "
            "claim requires >=3 seeds and no-overclaim aggregation (CI excludes 0)."
        ),
        "split": {
            "evalProblems": len(problems),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "evalFamilies": sorted({p["family"] for p in problems}),
            "trainSealed": data["train_sealed"],
            "evalSealed": data["eval_sealed"],
            "familyIntersection": data["family_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"passAt1": delta},
        "checks": {
            "contaminationFree": not data["family_intersection"],
            "adapterImprovesPassAt1": delta > 0,
            "noPassAt1Regression": delta >= 0,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


# --------------------------------------------------------------------------- #
# Step (process) task — held-out VERIFIED-CORRECT rate before/after on UNSEEN
# families. pass@1 here is STRICTER than the math task: the final answer must be
# correct AND every intermediate step machine-verified (agent.step_verifier), so
# a right answer reached via a wrong step does NOT pass. Judge-free.
# --------------------------------------------------------------------------- #
def _mock_completion_step(prob: dict, *, improved: bool) -> str:
    if improved:
        # A fully verifiable two-step derivation ending at the gold.
        return f"STEP: {prob['gold']} | restate target\nSTEP: {prob['gold']} | final answer"
    return "STEP: 0 | guess\nSTEP: 0 | final answer"  # base deliberately wrong


def _score_step(problems: list, completions: dict, domain: str) -> dict:
    rows, rewards, pass1, vsc_sum = [], [], 0, 0.0
    by_family: dict = {}
    for p in problems:
        text = completions.get(p["id"], "")
        reward, detail = step_reward.reward_for_completion(text, p["gold"], domain=domain)
        rewards.append(float(reward))
        ok = int(detail.get("verdict") == "accepted")  # verified-correct, not just right
        pass1 += ok
        vsc_sum += float(detail.get("vsc") or 0.0)
        by_family.setdefault(p["family"], []).append(ok)
        rows.append({"problem_id": p["id"], "family": p["family"], "reward": reward, "detail": detail})
    n = len(problems)
    return {
        "n": n,
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / n, 4) if n else 0.0,
        "verifiedStepCoverage": round(vsc_sum / n, 4) if n else 0.0,
        "passAt1ByFamily": {f: round(sum(v) / len(v), 4) for f, v in by_family.items()},
        "rows": rows,
    }


def run_eval_step(args: argparse.Namespace) -> dict:
    domain = args.step_domain
    if domain == "physics":
        data = physics_dataset.build_physics_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    else:
        data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    problems = data["eval_problems"]
    if args.limit:
        problems = problems[: args.limit]
    if data["family_intersection"]:
        raise SystemExit(f"contaminated split: {data['family_intersection']}")

    if args.mode == "mock":
        base = {p["id"]: _mock_completion_step(p, improved=False) for p in problems}
        adapter = {p["id"]: _mock_completion_step(p, improved=True) for p in problems}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        # chat_template=True: mirror run_eval_code. The step task feeds a STRUCTURED
        # instruction (STEP_INSTRUCTION + prompt) that a CHAT/instruct subject only
        # honours when delivered as an assistant turn; fed raw it emits prose, the step
        # verifier finds no parseable derivation, and base AND adapter score 0 (an eval
        # artifact, not a capability reading — failure-ledger step-math-chat-wrap-gap).
        # chat_wrap is a NO-OP for a base/completion model (no chat template), so the
        # registered Qwen2.5-Math-7B BASE run stays on the identical raw-prompt path.
        base_gen, adapter_gen = _load_real_generators(
            args.model, args.adapter, max_new_tokens=args.max_new_tokens, chat_template=True
        )
        base, adapter = {}, {}
        for i, p in enumerate(problems, 1):
            print(f"[eval-step] {i}/{len(problems)} {p['id']}", flush=True)
            wrapped = step_reward.STEP_INSTRUCTION + p["prompt"]
            base[p["id"]] = base_gen(wrapped)
            adapter[p["id"]] = adapter_gen(wrapped)
        model_desc = args.model

    base_score = _score_step(problems, base, domain)
    adapter_score = _score_step(problems, adapter, domain)
    delta = round(adapter_score["passAt1"] - base_score["passAt1"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "step",
        "stepDomain": domain,
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — per-run held-out verified-correct comparison on UNSEEN families. "
            "Capability claim requires >=3 seeds + no-overclaim aggregation (CI excludes 0)."
        ),
        "split": {
            "evalProblems": len(problems), "seed": args.seed, "evalFrac": args.eval_frac,
            "evalFamilies": sorted({p["family"] for p in problems}),
            "familyIntersection": data["family_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"passAt1": delta},
        "checks": {
            "contaminationFree": not data["family_intersection"],
            "adapterImprovesVerifiedCorrect": delta > 0,
            "noRegression": delta >= 0,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


# --------------------------------------------------------------------------- #
# Code task — held-out pass@1 before/after on UNSEEN task families.
# Reward = hidden-tests-pass (provenance_bench.code_exec; deterministic, no judge),
# so pass@1 IS the capability number; no false-positive axis (every item has a test).
# --------------------------------------------------------------------------- #
def _mock_completion_code(task: dict, *, improved: bool) -> str:
    if improved:
        # The fenced reference solution (pack-generator-verified to pass the test).
        return "```python\n" + task["solution"].rstrip() + "\n```"
    # Base: a syntactically valid but wrong implementation (restates the wrong op).
    return "```python\ndef " + task["entry_point"] + "(*args, **kwargs):\n    return 0\n```"


def _score_tasks(tasks: list, completions: dict) -> dict:
    rows, rewards, pass1 = [], [], 0
    by_family: dict = {}
    for t in tasks:
        text = completions.get(t["id"], "")
        reward, detail = code_reward.reward_for_task(text, t["test"])
        rewards.append(float(reward))
        ok = int(reward >= 1.0)
        pass1 += ok
        by_family.setdefault(t["family"], []).append(ok)
        rows.append({"task_id": t["id"], "family": t["family"], "reward": reward,
                     "detail": detail, "completion": text})
    return {
        "n": len(tasks),
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / len(tasks), 4) if tasks else 0.0,
        "passAt1ByFamily": {f: round(sum(v) / len(v), 4) for f, v in by_family.items()},
        "rows": rows,
    }


def run_eval_code(args: argparse.Namespace) -> dict:
    data = code_dataset.build_code_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    tasks = data["eval_tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    if data["family_intersection"]:
        raise SystemExit(f"contaminated split: {data['family_intersection']}")

    if args.mode == "mock":
        base = {t["id"]: _mock_completion_code(t, improved=False) for t in tasks}
        adapter = {t["id"]: _mock_completion_code(t, improved=True) for t in tasks}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        if not code_reward.exec_enabled():
            raise SystemExit("code adapter eval needs SOPHIA_ALLOW_CODE_EXEC=1 (interpreter = reward)")
        base_gen, adapter_gen = _load_real_generators(
            args.model, args.adapter, max_new_tokens=args.max_new_tokens, chat_template=True
        )
        base, adapter = {}, {}
        for i, t in enumerate(tasks, 1):
            print(f"[eval-code] {i}/{len(tasks)} {t['id']}", flush=True)
            base[t["id"]] = base_gen(t["prompt"])
            adapter[t["id"]] = adapter_gen(t["prompt"])
        model_desc = args.model

    base_score = _score_tasks(tasks, base)
    adapter_score = _score_tasks(tasks, adapter)
    delta = round(adapter_score["passAt1"] - base_score["passAt1"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "code",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — per-run held-out comparison on UNSEEN task families. Capability "
            "claim requires >=3 seeds and no-overclaim aggregation (CI excludes 0)."
        ),
        "split": {
            "evalTasks": len(tasks),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "evalFamilies": sorted({t["family"] for t in tasks}),
            "trainSealed": data["train_sealed"],
            "evalSealed": data["eval_sealed"],
            "familyIntersection": data["family_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"passAt1": delta},
        "checks": {
            "contaminationFree": not data["family_intersection"],
            "adapterImprovesPassAt1": delta > 0,
            "noPassAt1Regression": delta >= 0,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


# --------------------------------------------------------------------------- #
# Invention task — POWERED held-out pass@1 on compositional generalization.
# Eval suite = provenance_bench.invention_dataset.build_invention_eval_suite
# (N>=175, depth-2/3/4 compositions ABSENT from train; MDE~0.15 at 80% power) —
# the spec's powered primary. Scored by the GUARDED reward (integrity scan +
# exit-code-proof isolated grader + held-out private inputs), so the EVAL itself is
# cheat-resistant: an adapter that learned a reward-hack is floored here too, and a
# special-cased solution fails the unshown private inputs.
# --------------------------------------------------------------------------- #
def _mock_completion_invention(task: dict, *, improved: bool) -> str:
    if improved:
        return task["reference_solution"]  # generator-verified fenced pipeline
    return "```python\ndef pipeline(xs):\n    return []\n```"  # wrong, but plain data (not a cheat)


def _score_invention(tasks: list, completions: dict) -> dict:
    rows, rewards, pass1, hacks = [], [], 0, 0
    by_depth: dict = {}
    for t in tasks:
        text = completions.get(t["id"], "")
        reward, detail = code_integrity.guarded_reward_for_task(
            text, t["test"], holdout_test=t.get("private_test"))
        rewards.append(float(reward))
        ok = int(reward >= 1.0)
        pass1 += ok
        if detail.get("cheated"):
            hacks += 1
        by_depth.setdefault(t["depth"], []).append(ok)
        rows.append({"task_id": t["id"], "depth": t["depth"], "reward": reward,
                     "cheated": bool(detail.get("cheated")), "completion": text})
    return {
        "n": len(tasks),
        "passAt1": round(pass1 / len(tasks), 4) if tasks else 0.0,
        "passAt1ByDepth": {str(d): round(sum(v) / len(v), 4) for d, v in sorted(by_depth.items())},
        "rewardHackCount": hacks,
        "rows": rows,
    }


def run_eval_invention(args: argparse.Namespace) -> dict:
    if not code_reward.exec_enabled():
        raise SystemExit("invention eval needs SOPHIA_ALLOW_CODE_EXEC=1 (held-out inputs are executed)")
    from tools import eval_stats as es

    suite = invention_dataset.build_invention_eval_suite(target_n=args.invention_n, seed=args.seed)
    tasks = suite["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]

    if args.mode == "mock":
        base = {t["id"]: _mock_completion_invention(t, improved=False) for t in tasks}
        adapter = {t["id"]: _mock_completion_invention(t, improved=True) for t in tasks}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        base_gen, adapter_gen = _load_real_generators(
            args.model, args.adapter, max_new_tokens=args.max_new_tokens, chat_template=True)
        base, adapter = {}, {}
        for i, t in enumerate(tasks, 1):
            print(f"[eval-invention] {i}/{len(tasks)} {t['id']}", flush=True)
            base[t["id"]] = base_gen(t["prompt"])
            adapter[t["id"]] = adapter_gen(t["prompt"])
        model_desc = args.model

    base_score = _score_invention(tasks, base)
    adapter_score = _score_invention(tasks, adapter)
    delta = round(adapter_score["passAt1"] - base_score["passAt1"], 4)
    mde = round(es.mde_at_n(len(tasks)), 4) if tasks else 1.0
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "invention",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — POWERED held-out pass@1 on compositional-generalization tasks "
            "(eval compositions absent from train). Capability claim requires >=3 seeds "
            "+ no-overclaim aggregation (CI excludes 0); the integrity gate must hold."
        ),
        "split": {
            "evalTasks": len(tasks),
            "targetN": suite["target_n"],
            "depthsUsed": suite["depths_used"],
            "seed": args.seed,
            "mdeAtN": mde,
            "decontaminatedByConstruction": True,
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"passAt1": delta},
        "checks": {
            "powered": mde <= 0.16,
            "adapterImprovesPassAt1": delta > 0,
            "noPassAt1Regression": delta >= 0,
            # The integrity gate baked into the eval: a pass-rate bought with any
            # accepted reward-hack is disqualified (guarded reward floors cheats to -1).
            "noRewardHacksAccepted": base_score["rewardHackCount"] == 0 and adapter_score["rewardHackCount"] == 0,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


# --------------------------------------------------------------------------- #
# Concept task — held-out before/after on UNSEEN concept pairs. Reward = the
# symbolic concept-TBox gate (ontology_rl_reward; deterministic, no judge). The
# capability metric is meanReward (so ingest_rlvr_eval maps it through the generic
# meanReward branch); the protected axis is the over-abstention rate (lower better).
# --------------------------------------------------------------------------- #
def _mock_completion_concept(case: dict, *, improved: bool) -> str:
    a = case.get("subject", "the first concept")
    b = case.get("object", "the second concept")
    expected = case.get("expected", "distinct")
    if improved:
        if expected == "admit":
            return f"{a} and {b} are related within their shared tradition and reinforce one another."
        return f"{a} is not identical to {b}; they differ and the identification would be unscoped."
    # Base: the careless failure modes — a forbidden merge on cross-tradition pairs,
    # over-refusal on the answerable intra-tradition pairs.
    if expected == "admit":
        return "I can't answer this; it's underdetermined."
    return f"{a} is identical to {b}; they are the same thing."


def _score_concept(cases: list[dict], completions: dict) -> dict:
    rows, rewards, pass1 = [], [], 0
    over_abstain = answerable = 0
    forbidden = 0
    for c in cases:
        text = completions.get(c["id"], "")
        reward, detail = ontology_rl_reward.reward_for_edge_case(c, text)
        rewards.append(float(reward))
        pass1 += int(reward >= 1.0)
        if c.get("answerable", True):
            answerable += 1
            if detail.get("abstained"):
                over_abstain += 1
        forbidden += int(detail.get("assertedForbidden", False))
        rows.append({"case_id": c["id"], "expected": c.get("expected"), "reward": reward,
                     "detail": detail, "completion": text})
    return {
        "n": len(cases),
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / len(cases), 4) if cases else 0.0,
        "overAbstainRate": round(over_abstain / answerable, 4) if answerable else 0.0,
        "forbiddenMergeRate": round(forbidden / len(cases), 4) if cases else 0.0,
        "rows": rows,
    }


def run_eval_concept(args: argparse.Namespace) -> dict:
    data = ontology_rl_dataset.build_ontology_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    cases = data["eval_cases"]
    if args.limit:
        cases = cases[: args.limit]
    if data["entity_intersection"]:
        raise SystemExit(f"contaminated split: {data['entity_intersection']}")

    if args.mode == "mock":
        base = {c["id"]: _mock_completion_concept(c, improved=False) for c in cases}
        adapter = {c["id"]: _mock_completion_concept(c, improved=True) for c in cases}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        base_gen, adapter_gen = _load_real_generators(args.model, args.adapter, max_new_tokens=args.max_new_tokens)
        base, adapter = {}, {}
        for i, c in enumerate(cases, 1):
            print(f"[eval-concept] {i}/{len(cases)} {c['id']}", flush=True)
            base[c["id"]] = base_gen(c["prompt"])
            adapter[c["id"]] = adapter_gen(c["prompt"])
        model_desc = args.model

    base_score = _score_concept(cases, base)
    adapter_score = _score_concept(cases, adapter)
    delta = round(adapter_score["meanReward"] - base_score["meanReward"], 4)
    # Protected axis: over-abstention must not rise (the AlphaAlign failure mode).
    oa_delta = round(adapter_score["overAbstainRate"] - base_score["overAbstainRate"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "concept",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — per-run held-out comparison on UNSEEN concept pairs. Capability claim "
            "requires >=3 seeds, >=2 base-model families, and no-overclaim aggregation (CI excludes 0)."
        ),
        "split": {
            "evalCases": len(cases),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "trainSealed": data["train_sealed"],
            "evalSealed": data["eval_sealed"],
            "entityIntersection": data["entity_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"meanReward": delta, "overAbstainRate": oa_delta},
        "checks": {
            "contaminationFree": not data["entity_intersection"],
            "adapterImprovesMeanReward": delta > 0,
            "noOverAbstentionRegression": oa_delta <= args.max_fp_regression,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--task", choices=["provenance", "math", "code", "concept", "step", "invention"], default="provenance",
                    help="provenance (provenance_faithful), math (sympy math_equivalent), "
                         "code (hidden-tests-pass via code_exec), concept (concept-TBox gate), "
                         "step (process: every step verified -> verified-correct rate), "
                         "or invention (POWERED compositional-generalization suite, guarded grader)")
    ap.add_argument("--step-domain", choices=["math", "physics"], default="math",
                    help="for --task step: which held-out RL split + per-step oracle to use")
    ap.add_argument("--invention-n", type=int, default=175,
                    help="invention task only: target eval-suite size (175 -> MDE ~0.15 at 80%% power)")
    ap.add_argument("--model", default="zai-org/glm-4-9b-chat-hf")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="debug subset size")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-fp-regression", type=float, default=0.0)
    ap.add_argument("--capability-panel", action="store_true",
                    help="also run the capability-delta panel (attribution/hallucination/"
                         "calibration) and attach it under report['capabilityPanel']. "
                         "Provenance task only; additive evidence (legacy keys unchanged).")
    args = ap.parse_args(argv)
    if args.task == "math":
        report = run_eval_math(args)
    elif args.task == "code":
        report = run_eval_code(args)
    elif args.task == "concept":
        report = run_eval_concept(args)
    elif args.task == "step":
        report = run_eval_step(args)
    elif args.task == "invention":
        report = run_eval_invention(args)
    else:
        report = run_eval(args)
    _write(args.out, report)
    print("RLVR ADAPTER EVAL PASS ✓" if report["passed"] else "RLVR ADAPTER EVAL NOT PASSED ✗")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
