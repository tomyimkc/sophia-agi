# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Answer functions for the visual-trap benchmark.

An *answer function* is ``(trap) -> str``: it produces the VLM's free-text reply
to ``trap['question']`` about ``trap['scene']``. The default backends are
deterministic mocks (no network, no GPU) so the whole suite runs offline/airgap
and in CI; the real backend (``openai_vision_answer_fn``) is opt-in and renders
the scene to an actual PNG before calling an OpenAI-compatible vision model.

The mocks are also the reference behaviours the gate is measured against:

* ``grounded`` always gives the machine-derived gold answer (a perfect, honest
  model — hallucination rate should be 0).
* ``credulous`` always gives the documented-false trap answer (a maximally
  hallucinating model — hallucination rate should be ~1 on trap rows).
* ``abstainer`` always declines (fail-closed but useless — 0 hallucination, 0
  grounding).
"""

from __future__ import annotations


def grounded_answer_fn(trap: dict) -> str:
    """A perfect, honest model: returns the gold answer in natural language."""
    atype = trap["answer_type"]
    gold = str(trap["gold_answer"])
    if atype == "yesno":
        return "Yes." if gold.lower() == "yes" else "No, that is not in the image."
    if atype == "count":
        return f"There are {gold}."
    if atype == "measure":
        return f"They are about {gold} units apart."
    if gold.lower() == "none":
        return "There is no legible text on it."
    return f"It says {gold}."


def credulous_answer_fn(trap: dict) -> str:
    """A maximally hallucinating model: always gives the documented-false answer."""
    atype = trap["answer_type"]
    trapped = str(trap["trap_answer"])
    if atype == "yesno":
        return "Yes." if trapped.lower() == "yes" else "No."
    if atype == "count":
        return f"There are {trapped}."
    if atype == "measure":
        return f"About {trapped} units apart."
    return f"It says {trapped}."


def abstainer_answer_fn(trap: dict) -> str:
    """A fail-closed-but-useless model: always abstains."""
    return "I can't tell from this image."


MOCK_BACKENDS = {
    "grounded": grounded_answer_fn,
    "credulous": credulous_answer_fn,
    "abstainer": abstainer_answer_fn,
}


def openai_vision_answer_fn(*, model: str, base_url: "str | None" = None, api_key: "str | None" = None, prompt_suffix: str = ""):
    """Build a real answer function backed by an OpenAI-compatible vision model.

    Renders each scene to a PNG (``multimodal_bench/render.py``, needs Pillow),
    encodes it as a data URI, and asks ``model`` the trap's question. Opt-in:
    requires an API key and network. Kept out of the offline/CI path on purpose.
    """
    import base64
    import os

    from openai import OpenAI  # late import; only when a real run is requested

    from multimodal_bench.render import render_png

    client = OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                    api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def answer(trap: dict) -> str:
        png = render_png(trap["scene"])
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        msg = [{"role": "user", "content": [
            {"type": "text", "text": trap["question"] + prompt_suffix},
            {"type": "image_url", "image_url": {"url": uri}},
        ]}]
        resp = client.chat.completions.create(model=model, messages=msg, temperature=0)
        return (resp.choices[0].message.content or "").strip()

    return answer


def resolve_answer_fn(spec: str):
    """Map a CLI spec to an answer function.

    ``mock:grounded`` / ``mock:credulous`` / ``mock:abstainer`` -> offline mocks.
    ``openai:<model>`` (or ``vision:<model>``) -> real OpenAI-compatible backend.
    """
    kind, _, rest = spec.partition(":")
    if kind == "mock":
        if rest not in MOCK_BACKENDS:
            raise ValueError(f"unknown mock backend {rest!r}; have {sorted(MOCK_BACKENDS)}")
        return MOCK_BACKENDS[rest]
    if kind in ("openai", "vision"):
        if not rest:
            raise ValueError(f"{kind} spec needs a model, e.g. {kind}:gpt-4o")
        return openai_vision_answer_fn(model=rest)
    raise ValueError(f"unknown answer spec {spec!r}")
