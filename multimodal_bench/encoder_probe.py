# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Vision-encoder probing harness (workstream F).

The roadmap's encoder pillar is *probing*, not redesign: benchmark open encoders
(CLIP, SigLIP, …) on image↔text retrieval and report with CIs, treating encoder
choice as an honest measurement problem rather than a leaderboard number. Mirrors
``tools/eval_ladder.py``'s discipline: the wiring runs offline against a
deterministic stand-in; a *real* encoder needs weights/deps, and when they are
absent that is recorded as a **blocker**, never promoted as a result.

The retrieval probe: for each scene, score its true caption against K distractor
captions in the encoder's joint space; recall@1 is the fraction where the true
caption ranks first, with a bootstrap 95% CI.

**Honesty bound (stated, not hidden).** The default ``hashing`` encoder embeds the
*structured scene caption*, not pixels — it measures the harness plumbing and the
caption/distractor separability, NOT visual perception. Only the ``clip`` /
``siglip`` backends (which render the scene to a PNG and run real weights) measure
an actual encoder. The probe labels which backend produced every number.
"""

from __future__ import annotations

import hashlib
import math
import random

from multimodal_bench import runner


# --- captions: a structured, deterministic description of a scene ---------- #


def scene_caption(trap: dict) -> str:
    """A faithful natural-language caption derived from the ground-truth scene."""
    scene = trap["scene"]
    parts = []
    objs = scene.get("objects", [])
    if objs:
        labels = sorted(o["label"] for o in objs)
        parts.append("a scene with " + ", ".join(labels))
    if scene.get("texts"):
        parts.append("text reading " + ", ".join(t["value"] for t in scene["texts"]))
    if scene.get("chart"):
        bars = scene["chart"].get("bars", [])
        parts.append("a bar chart of " + ", ".join(f"{b['label']}={b['value']}" for b in bars))
    if scene.get("table"):
        parts.append("a table with columns " + ", ".join(scene["table"].get("columns", [])))
    if scene.get("document"):
        parts.append("a document with fields " + ", ".join(scene["document"].get("fields", {})))
    return "; ".join(parts) or "an empty scene"


def distractor_captions(traps: list, idx: int, k: int = 4, *, seed: int = 0) -> list:
    """K other scenes' captions, as retrieval distractors for scene ``idx``."""
    rng = random.Random(seed * 1000 + idx)
    others = [i for i in range(len(traps)) if i != idx]
    rng.shuffle(others)
    return [scene_caption(traps[i]) for i in others[:k]]


# --- encoders -------------------------------------------------------------- #


def _hash_embed(text: str, dim: int = 256) -> list:
    """Deterministic bag-of-tokens hashing embedding (CPU/airgap, no weights).

    The same family as agent/rag_local_embed.py: generalises surface form, not
    deep meaning. Stand-in ONLY — labelled 'hashing' in every report.
    """
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_encoder(spec: str = "hashing"):
    """Return ``(embed_text, embed_scene, backend_label, blocker)``.

    ``hashing`` (default) embeds captions deterministically offline. ``clip:<id>``
    / ``siglip:<id>`` load real weights and embed rendered PNGs; if torch/
    transformers/weights are unavailable the returned ``blocker`` explains why and
    the caller records a blocker instead of a number.
    """
    kind, _, model_id = spec.partition(":")
    if kind == "hashing":
        def embed_text(t):
            return _hash_embed(t)

        def embed_scene(trap):
            return _hash_embed(scene_caption(trap))  # caption stand-in (NOT pixels)

        return embed_text, embed_scene, "hashing", None

    if kind in ("clip", "siglip"):
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoProcessor

            from multimodal_bench.render import render_png
        except Exception as exc:  # missing deps -> blocker, not a result
            return None, None, spec, f"deps_unavailable:{type(exc).__name__}:{exc}"
        # Real-weight path (needs network/disk for the checkpoint). Built but not
        # exercised in CI; eval_ladder-style "real rung requires weights".
        try:
            import io

            from PIL import Image
            proc = AutoProcessor.from_pretrained(model_id)
            enc = AutoModel.from_pretrained(model_id)
        except Exception as exc:
            return None, None, spec, f"weights_unavailable:{type(exc).__name__}:{exc}"

        def embed_text(t):
            ins = proc(text=[t], return_tensors="pt", padding=True)
            return enc.get_text_features(**ins)[0].tolist()

        def embed_scene(trap):
            img = Image.open(io.BytesIO(render_png(trap["scene"])))
            ins = proc(images=img, return_tensors="pt")
            return enc.get_image_features(**ins)[0].tolist()

        return embed_text, embed_scene, spec, None

    return None, None, spec, f"unknown_encoder:{spec}"


# --- the probe ------------------------------------------------------------- #


def _ci(xs: list, alpha: float = 0.05) -> list:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return [0.0, 0.0]
    lo = xs[max(0, int((alpha / 2) * n))]
    hi = xs[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


def retrieval_probe(spec: str = "hashing", *, k: int = 4, seed: int = 0, n_boot: int = 1000) -> dict:
    """Image→text recall@1 over the trap suite, with a bootstrap 95% CI.

    Returns a blocker dict (no number) if the encoder's weights/deps are absent.
    """
    embed_text, embed_scene, backend, blocker = make_encoder(spec)
    if blocker:
        return {"backend": backend, "blocked": True, "blocker": blocker,
                "note": "recorded as a blocker, not a result (eval_ladder discipline)"}

    traps = runner.load_all_traps()
    hits = []
    for i, trap in enumerate(traps):
        img = embed_scene(trap)
        true_cap = scene_caption(trap)
        candidates = [true_cap] + distractor_captions(traps, i, k=k, seed=seed)
        scores = [_cosine(img, embed_text(c)) for c in candidates]
        hits.append(1 if max(range(len(scores)), key=lambda j: scores[j]) == 0 else 0)

    rng = random.Random(seed)
    n = len(hits)
    boot = [sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)] if n else []
    return {
        "backend": backend,
        "blocked": False,
        "n": n,
        "k_distractors": k,
        "recallAt1": round(sum(hits) / n, 4) if n else 0.0,
        "ci95": _ci(boot),
        "chance": round(1.0 / (k + 1), 4),
        "isRealEncoder": backend not in ("hashing",),
        "perceptionNote": ("structured-caption stand-in — measures harness, not pixels"
                           if backend == "hashing" else "rendered-PNG through real encoder weights"),
    }
