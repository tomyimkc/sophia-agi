# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia online RAG API — Cloud Run / local uvicorn."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_dotenv  # noqa: E402
from agent.rag_pipeline import answer_question  # noqa: E402

load_dotenv()

app = FastAPI(title="Sophia AGI RAG", version="0.6.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mode: str = Field(default="advisor", pattern="^(advisor|repo|life)$")
    top_k: int = Field(default=8, ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    gate: dict


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backend": os.environ.get("SOPHIA_RAG_BACKEND", "auto"),
        "index_dir": os.environ.get("SOPHIA_RAG_INDEX_DIR", "rag/index"),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        result = answer_question(payload.question, mode=payload.mode, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result)