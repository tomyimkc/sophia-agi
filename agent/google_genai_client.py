# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Google Gemini / Vertex AI client helpers."""

from __future__ import annotations

import os

from agent.config import is_real_secret, load_dotenv


def google_api_key() -> str | None:
    load_dotenv()
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if is_real_secret(value):
            return value
    return None


def use_vertex() -> bool:
    load_dotenv()
    flag = (os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    return flag in {"1", "true", "yes"}


def google_project() -> str | None:
    load_dotenv()
    value = (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT") or "").strip()
    return value or None


def google_location() -> str:
    load_dotenv()
    return (os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1").strip()


def gemini_model() -> str:
    load_dotenv()
    return (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()


def gemini_embed_model() -> str:
    load_dotenv()
    return (os.environ.get("GEMINI_EMBED_MODEL") or "text-embedding-004").strip()


def make_genai_client():
    from agent.dataflow.firewall import egress_blocked

    if egress_blocked():
        raise RuntimeError("airgap profile blocks egress (Google GenAI client)")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install: pip install -r requirements-rag.txt") from exc

    if use_vertex():
        project = google_project()
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Vertex AI")
        return genai.Client(vertexai=True, project=project, location=google_location())

    key = google_api_key()
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return genai.Client(api_key=key)