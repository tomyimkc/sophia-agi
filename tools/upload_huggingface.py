#!/usr/bin/env python3
"""Upload Sophia AGI corpus to Hugging Face Hub.

Requires: pip install huggingface_hub
Token: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, or run `huggingface-cli login`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("HF_REPO_ID", "tomyimkc/sophia-agi-corpus")
CORPUS = ROOT / "training" / "corpus.jsonl"
CARD = ROOT / "huggingface" / "README.md"


def main() -> int:
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