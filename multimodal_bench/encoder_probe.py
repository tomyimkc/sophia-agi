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


def make_scorer(spec: str = "hashing"):
    """Return ``(score_fn, backend_label, blocker)``.

    ``score_fn(trap, candidate_texts) -> list[float]`` gives the similarity of the
    scene's image to each candidate caption (higher = better match). ``hashing``
    (default) scores a caption stand-in deterministically offline. ``clip:<id>`` /
    ``siglip:<id>`` load real weights and score the *rendered PNG* against the
    candidate texts in the encoder's joint embedding space; if torch/transformers/
    weights are unavailable the returned ``blocker`` explains why and the caller
    records a blocker instead of a number.
    """
    kind, _, model_id = spec.partition(":")
    if kind == "hashing":
        def score_fn(trap, candidates):
            img = _hash_embed(scene_caption(trap))  # caption stand-in (NOT pixels)
            return [_cosine(img, _hash_embed(c)) for c in candidates]

        return score_fn, "hashing", None

    if kind in ("clip", "siglip"):
        try:
            import io

            import torch
            from PIL import Image
            from transformers import AutoModel, AutoProcessor

            from multimodal_bench.render import render_png
        except Exception as exc:  # missing deps -> blocker, not a result
            return None, spec, f"deps_unavailable:{type(exc).__name__}:{exc}"
        try:
            proc = AutoProcessor.from_pretrained(model_id)
            enc = AutoModel.from_pretrained(model_id).eval()
        except Exception as exc:  # no weights / bad id / no network -> blocker
            return None, spec, f"weights_unavailable:{type(exc).__name__}:{exc}"

        # SigLIP needs max-length padding; CLIP is happy with dynamic padding.
        pad = "max_length" if kind == "siglip" else True

        def score_fn(trap, candidates):
            img = Image.open(io.BytesIO(render_png(trap["scene"]))).convert("RGB")
            ins = proc(text=list(candidates), images=img, return_tensors="pt",
                       padding=pad, truncation=True)
            with torch.no_grad():
                out = enc(**ins)
            # Joint-space projected embeddings (version-stable across transformers 5.x).
            ie = torch.nn.functional.normalize(out.image_embeds, dim=-1)[0]
            te = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            return (te @ ie).tolist()

        return score_fn, spec, None

    return None, spec, f"unknown_encoder:{spec}"


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
    score_fn, backend, blocker = make_scorer(spec)
    if blocker:
        return {"backend": backend, "blocked": True, "blocker": blocker,
                "note": "recorded as a blocker, not a result (eval_ladder discipline)"}

    traps = runner.load_all_traps()
    hits = []
    for i, trap in enumerate(traps):
        true_cap = scene_caption(trap)
        candidates = [true_cap] + distractor_captions(traps, i, k=k, seed=seed)
        scores = score_fn(trap, candidates)  # similarity of scene image to each caption
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
