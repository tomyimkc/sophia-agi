#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Public-surface privacy guard: block training/architecture details from the
public front door.

Two scopes are enforced (exit 1 on any match):

1. The published thesis site under web/ — the full ruleset. The site is
   marketing/thesis copy and was aggressively scrubbed, so even conceptual
   architecture/training references (module map, RLVR) are blocked there.
   web/ is deployed to GitHub Pages on every push to main (pages.yml).

2. The repo "front door" — README.md and models/manifest.json — a NARROW
   ruleset that blocks only the concrete, reproducible training/model secrets:
   base-model identity, parameter scale, the LoRA/fine-tune recipe and its
   hyperparameters, training-runner scripts, the adapter checkpoint path, and
   the published HF adapter repo. It intentionally does NOT block conceptual
   open-research mentions (e.g. `agent/*.py`, `RLVR`) — those files are public
   in the repo anyway, so hiding their names in docs is theatre.

The Hugging Face model card (models/hf-model-card/README.md) is deliberately
NOT guarded for base-model identity: a published LoRA adapter is unusable
without naming its base model, which is also embedded in the live adapter's
config. Truly hiding it requires unpublishing the model on Hugging Face.

To adjust policy, edit WEB_PATTERNS / DOC_PATTERNS below.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

SCAN_SUFFIXES = {".html", ".js", ".json", ".md", ".txt"}
SKIP_NAMES = {".nojekyll"}

# Front-door repo files scanned with the narrow ruleset.
DOC_FILES = [ROOT / "README.md", ROOT / "models" / "manifest.json"]


def _should_skip(path: Path) -> bool:
    name = path.name
    if name in SKIP_NAMES:
        return True
    if name.startswith("google") and name.endswith(".html"):
        return True
    return False


# --- narrow ruleset: concrete, reproducible training/model secrets ---
# Base-model family names are caught to prevent leaking Sophia's OWN base model.
# The negative lookahead `(?!\d?\.?\d*-?\d?\s?B\b)` excludes when the family name is
# immediately followed by a parameter-scale suffix — i.e. an EXTERNAL model cited by its
# full spec as an eval subject in a validated cross-model result (e.g. "Qwen-2.5-72B",
# "Qwen2.5-72B", "Llama-3.3-70B"). Sophia's own base-model leak would appear as a BARE
# family name ("based on Qwen", "a Llama fine-tune") with no scale suffix, which the
# patterns still catch. Per the guard's docstring, false-positive refinement is the
# sanctioned path; this keeps the privacy intent intact while not blocking honest
# mentions of external models used as evaluation subjects.
DOC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bQwen\b(?![\w.-]*\d+\s?B\b)", re.I), "base-model identity (Qwen)"),
    (re.compile(r"\bLlama\b(?!\.cpp)(?![\w.-]*\d+\s?B\b)", re.I), "base-model identity (Llama)"),
    (re.compile(r"\bMistral\b", re.I), "base-model identity (Mistral)"),
    (re.compile(r"\bMixtral\b", re.I), "base-model identity (Mixtral)"),
    (re.compile(r"\bGemma\b", re.I), "base-model identity (Gemma)"),
    (re.compile(r"base[_\s\-]?model", re.I), "base-model field/phrase"),
    (re.compile(r"\b\d{1,3}\s?B\b\s+(?:local\s+)?model", re.I), "model parameter-scale"),
    (re.compile(r"\bQ?LoRA\b", re.I), "training/adapter method (LoRA)"),
    (re.compile(r"fine[\s\-]?tun(?:e|ing|ed)", re.I), "training method (fine-tune)"),
    (re.compile(r"learning[\s\-]rate", re.I), "hyperparameter (learning rate)"),
    (re.compile(r"\bepochs?\b", re.I), "hyperparameter (epochs)"),
    (re.compile(r"--?\d{1,2}bit\b", re.I), "hyperparameter (quantization)"),
    (re.compile(r"\blora[_\s]?rank\b|\badapter[\s\-]rank\b", re.I), "hyperparameter (rank)"),
    (re.compile(r"\bbatch[\s\-]size\b", re.I), "hyperparameter (batch size)"),
    (re.compile(r"train_lora|prepare_lora_dataset", re.I), "training-runner script"),
    (re.compile(r"training/lora/checkpoints", re.I), "adapter checkpoint path"),
    (re.compile(r"sophia-agi-lora", re.I), "published HF adapter repo"),
]

# --- full ruleset for the thesis site (narrow rules + the conceptual ones) ---
WEB_PATTERNS: list[tuple[re.Pattern[str], str]] = DOC_PATTERNS + [
    (re.compile(r"\b\d{1,3}\s?B\s*模型"), "model parameter-scale (zh)"),
    (re.compile(r"gradient[\s\-]accumulation", re.I), "hyperparameter (grad accumulation)"),
    (re.compile(r"\bhyper[\s\-]?parameter", re.I), "hyperparameter"),
    (re.compile(r"\bagent/[A-Za-z0-9_]+\.py", re.I), "internal module path (agent/*.py)"),
    (re.compile(r"\bprovenance_bench/[A-Za-z0-9_]+\.py", re.I), "internal module path"),
    (re.compile(r"\bselfextend/[A-Za-z0-9_]+", re.I), "internal module path"),
    (re.compile(r"Sophia-Architecture\.md|architectureDiagram", re.I), "architecture map"),
    (re.compile(r"\bRLVR\b", re.I), "training pipeline (RLVR)"),
    (
        re.compile(
            r"run_rlvr|rl_reward|rl_dataset|run_ablation|run_unified_uplift|"
            r"run_learning_shift|run_local_agent_delta|run_code_uplift|"
            r"run_reflexive_self_gate|gate_feedback|promote_pending|"
            r"build_local_sophia_dataset",
            re.I,
        ),
        "training pipeline runner",
    ),
]


def scan_file(path: Path, patterns: list[tuple[re.Pattern[str], str]]) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, label in patterns:
            m = pattern.search(line)
            if m:
                hits.append((lineno, label, m.group(0)))
    return hits


def _targets() -> list[tuple[Path, list[tuple[re.Pattern[str], str]]]]:
    targets: list[tuple[Path, list[tuple[re.Pattern[str], str]]]] = []
    if WEB.exists():
        for path in sorted(WEB.rglob("*")):
            if path.is_file() and path.suffix.lower() in SCAN_SUFFIXES and not _should_skip(path):
                targets.append((path, WEB_PATTERNS))
    for path in DOC_FILES:
        if path.exists():
            targets.append((path, DOC_PATTERNS))
    return targets


def main() -> int:
    violations: list[str] = []
    targets = _targets()
    for path, patterns in targets:
        for lineno, label, snippet in scan_file(path, patterns):
            rel = path.relative_to(ROOT)
            violations.append(f"{rel}:{lineno}: {label} — matched {snippet!r}")

    if violations:
        print("Public-surface privacy guard FAILED: training/architecture details found.\n")
        for v in violations:
            print(f"  {v}")
        print(
            f"\n{len(violations)} violation(s) across {len(targets)} scanned file(s). "
            "Remove the detail (and re-run the builders) or, if it is a false "
            "positive, refine the patterns in tools/lint_web_privacy.py."
        )
        return 1

    print(f"Public-surface privacy guard passed: no leaks in {len(targets)} scanned file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
