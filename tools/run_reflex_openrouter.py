#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real-model reflex measurement via an OpenAI-compatible endpoint (e.g. OpenRouter).

This is the *gated next step* the instinct work pointed at: replace
``reasoning.instinct_reflex_eval``'s synthetic sampler with a real model and read off the
**real** self-consistency d′ / AUC against the break-even bar.

Security & cost (read before running):
  - The API key is read ONLY from the environment (``OPENROUTER_API_KEY`` or
    ``OPENAI_API_KEY``). It is never hard-coded, logged, or written to any artifact.
    Set it yourself:  ``export OPENROUTER_API_KEY=sk-or-...``  (do not paste it in chat).
  - This spends real credits and sends the belief-revision prompts to an external service.
    The run prints the call budget and requires ``--yes`` to proceed.
  - A single model is ONE judge family at ONE seed → the result is ``candidateOnly`` and
    canNOT be promoted. The no-overclaim gate still needs ≥2 families, ≥3 seeds, CI.

Task framing (self-contained, fair): each belief-revision case is posed as an
ABSTAIN/SURVIVE classification over its own candidate claims, using the case's own rule
text — so we measure the reflex, not our edge-reconstruction. Ground truth is the planted
``expectAbstain`` partition. Self-consistency = agreement of the classification across N
temperature samples; correctness = majority classification == ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_reflex_eval import (  # noqa: E402
    auc,
    breakeven_snr,
    d_prime,
    load_cases,
    self_consistency_reflex,
)
from agent.calibration import self_consistency  # noqa: E402

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"  # cheap + capable; override with --model


def _ssl_context() -> ssl.SSLContext:
    """Trust the agent-proxy CA bundle if present (mirrors the repo's proxy setup)."""
    for cand in (os.environ.get("SSL_CERT_FILE"), "/root/.ccr/ca-bundle.crt"):
        if cand and Path(cand).exists():
            return ssl.create_default_context(cafile=cand)
    return ssl.create_default_context()


def _api_key() -> str:
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    raise SystemExit(
        "No API key in env. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and re-run.\n"
        "  export OPENROUTER_API_KEY=sk-or-...   # do NOT paste keys into chat"
    )


def _candidates(case: dict[str, Any]) -> list[str]:
    seen, out = set(), []
    for node in list(case.get("expectAbstain", [])) + list(case.get("expectSurvive", [])):
        if node not in seen:
            seen.add(node)
            out.append(node)
    return sorted(out)


def _canonical_abstain(nodes: Any) -> str:
    try:
        return json.dumps(sorted({str(n) for n in nodes}))
    except TypeError:
        return "UNPARSEABLE"


def _ground_truth(case: dict[str, Any]) -> str:
    return _canonical_abstain(case.get("expectAbstain", []))


def _build_prompt(case: dict[str, Any]) -> str:
    cands = _candidates(case)
    removed = case.get("remove", [])
    return (
        "You maintain a belief graph. A claim must ABSTAIN if and only if it loses ALL of "
        "its grounding once the retracted claim(s) are removed; otherwise it SURVIVES.\n\n"
        f"Grounding rule for this case: {case.get('reason', '(none given)')}\n"
        f"Retracted (removed) claim(s): {json.dumps(removed)}\n"
        f"Candidate claims to classify: {json.dumps(cands)}\n\n"
        "Respond with ONLY a JSON array containing exactly the claim ids that must ABSTAIN "
        "(a subset of the candidates). No prose, no code fence."
    )


class ModelCallError(RuntimeError):
    """A model call failed. We RAISE this — never fold an API failure into the data,
    or a run of all-failed calls would masquerade as 'the model got everything wrong'
    (a fabricated base_error=1.0 / d′). Fail loud; the harness must not invent results."""


def _call_model(prompt: str, *, model: str, base_url: str, key: str,
                temperature: float, timeout: float, retries: int = 3) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 200,
    }).encode()
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            # 4xx (auth/ToS/quota) won't fix on retry — fail immediately and loudly.
            if 400 <= exc.code < 500:
                raise ModelCallError(f"HTTP {exc.code} from provider: {detail}") from exc
            last = f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, KeyError, TimeoutError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise ModelCallError(f"call failed after {retries} attempts: {last}")


def _parse_answer(text: str) -> str:
    """Map a raw completion to a canonical abstain-set token (stable across identical sets)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1]
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return _canonical_abstain(json.loads(t[start:end + 1]))
        except (json.JSONDecodeError, TypeError):
            pass
    return f"UNPARSEABLE::{t[:40]}"


def run(model: str, base_url: str, n_samples: int, limit: int | None,
        temperature: float, timeout: float, out: Path | None) -> int:
    key = _api_key()
    cases = load_cases()
    if limit:
        cases = cases[:limit]
    bar = breakeven_snr()

    scores_err: list[float] = []
    scores_clean: list[float] = []
    per_case = []
    for idx, case in enumerate(cases):
        gt = _ground_truth(case)
        samples = []
        for _ in range(n_samples):
            # No try/except here: a ModelCallError propagates and aborts the run. A failed
            # API call is NOT data — recording it as a "wrong answer" would fabricate a result.
            raw = _call_model(_build_prompt(case), model=model, base_url=base_url,
                              key=key, temperature=temperature, timeout=timeout)
            samples.append(_parse_answer(raw))
        majority, _conf = self_consistency(samples)
        is_error = (majority != gt)
        score = self_consistency_reflex(samples)
        (scores_err if is_error else scores_clean).append(score)
        per_case.append({"id": case.get("id"), "is_error": is_error, "reflex": round(score, 4)})
        print(f"  [{idx + 1}/{len(cases)}] {case.get('id')}: "
              f"{'ERR ' if is_error else 'ok  '} reflex={score:.3f}", file=sys.stderr)

    dp = d_prime(scores_err, scores_clean)
    a = auc(scores_err, scores_clean)
    report = {
        "schema": "sophia.reasoning.reflex_eval.realmodel.v1",
        "model": model, "base_url": base_url, "n_cases": len(cases),
        "n_samples": n_samples, "temperature": temperature,
        "base_error": round(len(scores_err) / len(cases), 4) if cases else 0.0,
        "d_prime": round(dp, 4) if dp == dp and abs(dp) != float("inf") else dp,
        "auc": round(a, 4),
        "breakeven_snr": bar,
        "clears_breakeven": bool(dp == dp and abs(dp) != float("inf") and dp >= bar),
        "candidateOnly": True, "level3Evidence": False,
        "boundary": "single model = 1 judge family @ 1 seed; not promotable. "
                    "No-overclaim gate needs >=2 families, >=3 seeds, CI.",
    }
    print(json.dumps(report, indent=2))
    if out:
        out.write_text(json.dumps({"report": report, "per_case": per_case}, indent=2))
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--samples", type=int, default=5, help="self-consistency samples per case")
    p.add_argument("--limit", type=int, default=None, help="cap number of cases (cost guard)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=None, help="write JSON artifact here")
    p.add_argument("--yes", action="store_true", help="confirm: this spends real credits")
    args = p.parse_args(argv)

    n_cases = args.limit or len(load_cases())
    budget = n_cases * args.samples
    print(f"Plan: model={args.model}  cases={n_cases}  samples={args.samples}  "
          f"=> {budget} API calls (real credits).", file=sys.stderr)
    if not args.yes:
        print("Refusing to spend without --yes. Re-run with --yes to proceed.", file=sys.stderr)
        return 2
    return run(args.model, args.base_url, args.samples, args.limit,
               args.temperature, args.timeout, args.out)


if __name__ == "__main__":
    sys.exit(main())
