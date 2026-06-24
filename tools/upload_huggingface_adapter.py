#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Upload Sophia AGI LoRA adapter to Hugging Face Hub (model repo).

Requires: pip install huggingface_hub
Token: HF_TOKEN or HUGGING_FACE_HUB_TOKEN, or huggingface-cli login

Usage:
  python tools/upload_huggingface_adapter.py --dry-run
  python tools/upload_huggingface_adapter.py --approve
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "training" / "lora" / "checkpoints" / "sophia-v1"
DEFAULT_REPO = os.environ.get("HF_ADAPTER_REPO_ID", "tomyimkc/sophia-agi-lora-v1")
MODEL_CARD = ROOT / "models" / "hf-model-card" / "README.md"
REQUIRED = ("adapter_config.json", "adapter_model.safetensors")
SKIP_DIRS = {"hf_trainer_state"}
SKIP_FILES = {".train_complete"}


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value and "your" not in value.lower():
            os.environ.setdefault(key, value)


def collect_files(adapter_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def verify_adapter(adapter_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED if not (adapter_dir / name).exists()]
    if missing:
        return missing
    safetensors = adapter_dir / "adapter_model.safetensors"
    if safetensors.stat().st_size < 1_000_000:
        return ["adapter_model.safetensors (too small)"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload Sophia LoRA adapter to Hugging Face")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--approve", action="store_true", help="Required to upload")
    args = parser.parse_args()

    load_dotenv()

    if not args.adapter.is_dir():
        print(f"Missing adapter dir: {args.adapter}")
        return 1

    missing = verify_adapter(args.adapter)
    if missing:
        print(f"Adapter incomplete: {missing}")
        return 1

    files = collect_files(args.adapter)
    rel_paths = [str(p.relative_to(args.adapter)).replace("\\", "/") for p in files]
    print(f"Adapter: {args.adapter}")
    print(f"Repo: {args.repo_id} (model)")
    print(f"Files: {len(files)}")
    for rel in rel_paths:
        print(f"  - {rel}")

    meta_path = args.adapter / "sophia_lora_config.json"
    if meta_path.exists():
        print("Training meta:", json.loads(meta_path.read_text(encoding="utf-8")))

    if args.dry_run:
        return 0

    if not args.approve:
        print("Refusing upload without --approve")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Missing HF_TOKEN. https://huggingface.co/settings/tokens")
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    for path in files:
        rel = str(path.relative_to(args.adapter)).replace("\\", "/")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Upload Sophia LoRA adapter ({rel})",
        )

    if MODEL_CARD.exists():
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update model card",
        )

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())