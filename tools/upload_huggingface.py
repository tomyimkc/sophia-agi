#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Upload Sophia AGI corpus to Hugging Face Hub.

Requires: pip install huggingface_hub
Token: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, or run `huggingface-cli login`
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("HF_REPO_ID", "tomyimkc/sophia-agi-corpus")
CORPUS = ROOT / "training" / "corpus.jsonl"
CARD = ROOT / "huggingface" / "README.md"


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


def main() -> int:
    load_dotenv()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install: pip install huggingface_hub")
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Missing HF_TOKEN. Create token at https://huggingface.co/settings/tokens")
        print("Then: set HF_TOKEN=hf_...  OR  huggingface-cli login")
        return 1

    if not CORPUS.exists():
        print("Run: python tools/export_training_jsonl.py")
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(CORPUS),
        path_in_repo="corpus.jsonl",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update Sophia AGI training corpus",
    )
    if CARD.exists():
        api.upload_file(
            path_or_fileobj=str(CARD),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Update dataset card",
        )

    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())