#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Web privacy guard: block training/architecture details from the public site.

The thesis site under web/ is published to GitHub Pages on every push to main
(see .github/workflows/pages.yml). It must stay aligned with the repo but must
NOT expose important training or architecture details. This linter scans the
published web/ files and fails (exit 1) if any forbidden pattern appears, so a
leak blocks the deploy instead of going live.

Policy (owner decision): off-limits on the public site are
  - base-model identity (e.g. Qwen2.5-3B) and parameter scale
  - the internal module/architecture map (agent/*.py, provenance_bench/*.py, ...)
  - training recipe + hyperparameters (LoRA rank, learning rate, epochs, ...)
  - training/RLVR pipeline runner references

To adjust the policy, edit PATTERNS below. Keep patterns specific to avoid
false positives on legitimate public content (benchmark results, the corpus,
the claim boundary, competitor/comparison model names in leaderboards).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

# Files to scan (text the public can read). Skip styles and the search-console
# verification stub.
SCAN_SUFFIXES = {".html", ".js", ".json", ".md", ".txt"}
SKIP_NAMES = {".nojekyll"}


def _should_skip(path: Path) -> bool:
    name = path.name
    if name in SKIP_NAMES:
        return True
    # Google site-verification stub, e.g. googleac774deb185370e2.html
    if name.startswith("google") and name.endswith(".html"):
        return True
    return False


# (compiled pattern, human label). All matched case-insensitively.
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- base-model identity ---
    (re.compile(r"\bQwen\b", re.I), "base-model identity (Qwen)"),
    (re.compile(r"\bLlama\b", re.I), "base-model identity (Llama)"),
    (re.compile(r"\bMistral\b", re.I), "base-model identity (Mistral)"),
    (re.compile(r"\bMixtral\b", re.I), "base-model identity (Mixtral)"),
    (re.compile(r"\bGemma\b", re.I), "base-model identity (Gemma)"),
    (re.compile(r"baseModel|base[\s\-]model", re.I), "base-model field/phrase"),
    # --- model parameter scale (latin + CJK forms): '8B local model', '8B 模型' ---
    (re.compile(r"\b\d{1,3}\s?B\b\s+(?:local\s+)?model", re.I), "model parameter-scale"),
    (re.compile(r"\b\d{1,3}\s?B\s*模型"), "model parameter-scale (zh)"),
    # --- adapter / fine-tune method ---
    (re.compile(r"\bLoRA\b", re.I), "training/adapter method (LoRA)"),
    (re.compile(r"fine[\s\-]?tun(?:e|ing|ed)", re.I), "training method (fine-tune)"),
    # --- training recipe / hyperparameters ---
    (re.compile(r"learning[\s\-]rate", re.I), "hyperparameter (learning rate)"),
    (re.compile(r"\blora[_\s]?rank\b|\badapter[\s\-]rank\b", re.I), "hyperparameter (rank)"),
    (re.compile(r"\bepochs?\b", re.I), "hyperparameter (epochs)"),
    (re.compile(r"\bbatch[\s\-]size\b", re.I), "hyperparameter (batch size)"),
    (re.compile(r"\boptimizer\b", re.I), "hyperparameter (optimizer)"),
    (re.compile(r"gradient[\s\-]accumulation", re.I), "hyperparameter (grad accumulation)"),
    (re.compile(r"\bhyper[\s\-]?parameter", re.I), "hyperparameter"),
    # --- internal module / architecture map ---
    (re.compile(r"\bagent/[A-Za-z0-9_]+\.py", re.I), "internal module path (agent/*.py)"),
    (re.compile(r"\bprovenance_bench/[A-Za-z0-9_]+\.py", re.I), "internal module path"),
    (re.compile(r"\bselfextend/[A-Za-z0-9_]+", re.I), "internal module path"),
    (re.compile(r"Sophia-Architecture\.md|architectureDiagram", re.I), "architecture map"),
    # --- training / RLVR pipeline runners ---
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


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, label in PATTERNS:
            m = pattern.search(line)
            if m:
                hits.append((lineno, label, m.group(0)))
    return hits


def main() -> int:
    if not WEB.exists():
        print(f"web/ not found at {WEB}; nothing to scan.")
        return 0
    violations: list[str] = []
    scanned = 0
    for path in sorted(WEB.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        if _should_skip(path):
            continue
        scanned += 1
        for lineno, label, snippet in scan_file(path):
            rel = path.relative_to(ROOT)
            violations.append(f"{rel}:{lineno}: {label} — matched {snippet!r}")

    if violations:
        print("Web privacy guard FAILED: training/architecture details found in the public site.\n")
        for v in violations:
            print(f"  {v}")
        print(
            f"\n{len(violations)} violation(s) across {scanned} scanned file(s). "
            "Remove the detail from web/ (and re-run the builders) or, if it is a "
            "false positive, refine PATTERNS in tools/lint_web_privacy.py."
        )
        return 1

    print(f"Web privacy guard passed: no training/architecture leaks in {scanned} web file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
