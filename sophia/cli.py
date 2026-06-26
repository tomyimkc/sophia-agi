# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Console entrypoint for the Sophia AGI package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sophia import __version__
from sophia.trainer import build_experiment_plan, execute_plan, load_experiment_config
from sophia.trainer.plan import EVAL_STAGES, PROMOTION_STAGES, TRAINING_STAGES


def _add_config_args(parser: argparse.ArgumentParser, *, allow_execute: bool) -> None:
    parser.add_argument("--config", required=True, help="experiment config JSON/TOML")
    parser.add_argument("--repo-root", default=".", help="repository root for script execution")
    parser.add_argument("--json", action="store_true", help="print the command plan as JSON")
    parser.add_argument(
        "--live",
        action="store_true",
        help="compile live commands without --dry-run; requires explicit --execute to run",
    )
    if allow_execute:
        parser.add_argument(
            "--execute",
            action="store_true",
            help="run the compiled commands in order; defaults remain dry-run unless --live is set",
        )


def _render_plan(config_path: str, config_name: str, plan: list, *, json_output: bool) -> None:
    payload = {
        "config": config_path,
        "name": config_name,
        "commands": [spec.to_dict() for spec in plan],
    }
    if json_output:
        print(json.dumps(payload, indent=2))
        return
    print(f"Sophia experiment: {config_name}")
    if not plan:
        print("No enabled stages selected.")
        return
    for spec in plan:
        mode = "dry-run" if spec.dry_run else "live"
        gpu = " (GPU when live)" if spec.gpu_required_when_live else ""
        print(f"[{spec.stage}] {mode}{gpu}: {spec.shell()}")


def _handle_configured_command(
    args: argparse.Namespace, stages: tuple[str, ...] | None, *, execute: bool
) -> int:
    config = load_experiment_config(args.config)
    plan = build_experiment_plan(config, stages=stages, dry_run=not args.live)
    _render_plan(args.config, config.name, plan, json_output=args.json)
    if execute:
        return execute_plan(plan, cwd=Path(args.repo_root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sophia",
        description=(
            "Sophia verifier-gated training/proof CLI. This packages AGI-candidate "
            "workflow machinery, not an AGI claim."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sophia {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment = subparsers.add_parser("experiment", help="plan or run a full experiment config")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    plan = experiment_sub.add_parser("plan", help="print the full command plan")
    _add_config_args(plan, allow_execute=False)
    run = experiment_sub.add_parser("run", help="print or execute the full command plan")
    _add_config_args(run, allow_execute=True)

    train = subparsers.add_parser("train", help="plan or execute data/SFT/DPO/RLVR stages")
    _add_config_args(train, allow_execute=True)
    eval_cmd = subparsers.add_parser("eval", help="plan or execute eval-ladder stage")
    _add_config_args(eval_cmd, allow_execute=True)
    promote = subparsers.add_parser("promote", help="plan or execute promotion gate stage")
    _add_config_args(promote, allow_execute=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "experiment":
            execute = bool(getattr(args, "execute", False))
            return _handle_configured_command(args, None, execute=execute)
        if args.command == "train":
            return _handle_configured_command(args, TRAINING_STAGES, execute=args.execute)
        if args.command == "eval":
            return _handle_configured_command(args, EVAL_STAGES, execute=args.execute)
        if args.command == "promote":
            return _handle_configured_command(args, PROMOTION_STAGES, execute=args.execute)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"sophia: {exc}", file=sys.stderr)
        return 2
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
