"""Sophia ↔ AIpp bridge — authenticated FastAPI surface for the iOS cockpit.

Run locally:
    SOPHIA_AIPP_TOKEN=dev-token uvicorn services.aipp_bridge.main:app --port 8081

Endpoints (all except /health require ``Authorization: Bearer <SOPHIA_AIPP_TOKEN>``):
    GET  /health      — liveness + backend info
    POST /ask         — grounded, gated answer with sources (Research/Knowledge)
    POST /verify      — run the epistemic gate over an existing draft → verdict
    POST /conscience  — run the conscience kernel over a draft → verdict

Every response carries the compact AIpp verdict contract from ``verdict.py``:
    { verdict, confidence, reasons[], abstained, sources[], ... }
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_dotenv  # noqa: E402
from services.aipp_bridge.verdict import build_verdict  # noqa: E402

load_dotenv()

app = FastAPI(title="Sophia AGI — AIpp Bridge", version="0.1.0")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def _expected_token() -> str | None:
    token = os.environ.get("SOPHIA_AIPP_TOKEN", "").strip()
    return token or None


def require_token(authorization: str = Header(default="")) -> None:
    """Constant-time Bearer check. Fails closed if no token is configured."""
    expected = _expected_token()
    if not expected:
        raise HTTPException(status_code=503, detail="Bridge auth is not configured (set SOPHIA_AIPP_TOKEN).")
    prefix = "Bearer "
    presented = authorization[len(prefix):].strip() if authorization.startswith(prefix) else ""
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mode: str = Field(default="advisor", pattern="^(advisor|repo|life)$")
    top_k: int = Field(default=8, ge=1, le=20)


class VerifyRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    question: str | None = Field(default=None, max_length=4000)
    mode: str = Field(default="advisor", pattern="^(advisor|repo|life)$")
    sources: list[str] = Field(default_factory=list)


class ConscienceRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    mode: str = Field(default="output")
    action: str | None = None


class VerdictResponse(BaseModel):
    verdict: str
    confidence: float
    reasons: list[str]
    abstained: bool
    sources: list[dict] = Field(default_factory=list)
    answer: str | None = None
    gatePassed: bool | None = None
    conscienceVerdict: str | None = None
    zhSummary: str | None = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aipp_bridge",
        "authConfigured": _expected_token() is not None,
        "backend": os.environ.get("SOPHIA_RAG_BACKEND", "auto"),
    }


@app.post("/ask", response_model=VerdictResponse, dependencies=[Depends(require_token)])
def ask(payload: AskRequest) -> VerdictResponse:
    from agent.rag_pipeline import answer_question

    try:
        result = answer_question(payload.question, mode=payload.mode, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - surfaced to the client
        raise HTTPException(status_code=502, detail=f"Sophia RAG failed: {exc}") from exc

    answer = result.get("answer", "")
    verdict = build_verdict(answer, gate=result.get("gate"), sources=result.get("sources"))
    return VerdictResponse(answer=answer, zhSummary=_zh_summary(answer), **verdict)


@app.post("/verify", response_model=VerdictResponse, dependencies=[Depends(require_token)])
def verify(payload: VerifyRequest) -> VerdictResponse:
    from agent.gate import check_response

    try:
        gate = check_response(
            payload.content,
            mode=payload.mode,
            question=payload.question,
            sources=payload.sources or None,
            strict_attribution=True,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Sophia gate failed: {exc}") from exc

    sources = [{"path": s} for s in payload.sources]
    verdict = build_verdict(payload.content, gate=gate, sources=sources)
    return VerdictResponse(zhSummary=_zh_summary(payload.content), **verdict)


@app.post("/conscience", response_model=VerdictResponse, dependencies=[Depends(require_token)])
def conscience(payload: ConscienceRequest) -> VerdictResponse:
    from agent.conscience import conscience_check

    try:
        decision = conscience_check(payload.content, mode=payload.mode, action=payload.action).to_dict()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"Sophia conscience failed: {exc}") from exc

    verdict = build_verdict(payload.content, gate=None, conscience=decision)
    return VerdictResponse(**verdict)


def _zh_summary(text: str) -> str | None:
    """Sophia answers carry a 中文 section; expose whether one is present so the
    app can surface the bilingual discipline marker. Returns None when absent."""
    import re

    return "present" if re.search(r"[\u4e00-\u9fff]", text or "") else None
