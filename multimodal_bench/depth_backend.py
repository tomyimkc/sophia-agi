# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Depth sources for the physical/metric verifiers — authored z, or pixel-derived.

The physical verifiers (`verifiers.depth_order` / `occludes` / `bigger_than` /
`distance_*`) read each object's scalar ``z`` (and ``size``) off the scene. That
``z`` can come from two places:

* **authored** (default, offline): the ground-truth ``z``/``size`` already on the
  structured scene. Deterministic, CPU/airgap, the CI path — but it measures the
  *harness*, not pixel perception (the depth is declared, not seen).
* **pixel-derived** (opt-in): a monocular **metric depth** model (Depth Anything
  V2) run on the *rendered* scene PNG, sampling depth per object box. This is the
  field's fix for metric blindness (DepthVLM / SD-VLM all inject depth) — but it
  needs torch + transformers + checkpoint weights, so when they are absent we
  return a **blocker**, never a faked number (the `encoder_probe` discipline).

A *depth source* exposes ``augment(scene) -> scene'`` that returns a copy whose
objects carry a ``z`` (and a best-effort ``size``) from that source; the verifiers
then run unchanged on the augmented scene. For the authored source ``augment`` is
the identity. This keeps the verifiers the single source of truth — a backend only
supplies the numbers they consume.

**Depth convention.** Our scenes use ``z`` = larger means *farther* from the
camera. Monocular models typically emit *inverse depth* (disparity: larger means
*nearer*), so the Depth Anything source negates/﻿inverts to match. The mapping is
documented here and exercised only on the opt-in path.
"""

from __future__ import annotations

import copy


class AuthoredDepthSource:
    """The offline default: depth is the authored ``z`` already on the scene.

    ``augment`` is the identity — it measures the harness and reference behaviour,
    NOT pixel perception. Every report must label the source ``authored`` so the
    honesty bound is visible.
    """

    label = "authored"
    blocker = None

    def augment(self, scene: dict) -> dict:
        return scene


class DepthAnythingV2Source:
    """Pixel-derived metric depth from Depth Anything V2 (opt-in, weights-gated).

    Renders the scene to a PNG, runs monocular depth, and for each object inverts
    the median predicted disparity (Depth Anything emits inverse depth: larger ==
    nearer) into a positive distance (larger == farther) used as ``z``, with a
    best-effort real ``size`` ~ apparent diagonal x distance (the perspective
    relation). Built but NOT exercised in CI: instantiation records a ``blocker``
    string when torch/transformers/Pillow or the weights are missing.
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.label = f"depth-anything:{model_id}"
        self.blocker: "str | None" = None
        self._pipe = None
        self._np = None
        self._Image = None
        try:
            import numpy as np  # noqa: F401
            import torch  # noqa: F401
            from PIL import Image
            from transformers import pipeline
        except Exception as exc:  # missing deps -> blocker, not a result
            self.blocker = f"deps_unavailable:{type(exc).__name__}:{exc}"
            return
        try:  # weights need network/disk; absent -> blocker, never faked
            self._pipe = pipeline(task="depth-estimation", model=model_id)
            self._np = np
            self._Image = Image
        except Exception as exc:
            self.blocker = f"weights_unavailable:{type(exc).__name__}:{exc}"

    def augment(self, scene: dict) -> dict:
        if self.blocker is not None:
            raise RuntimeError(f"depth source unavailable: {self.blocker}")
        import io

        from multimodal_bench.render import render_png

        img = self._Image.open(io.BytesIO(render_png(scene)))
        pred = self._pipe(img)["predicted_depth"]
        arr = pred.squeeze().cpu().numpy() if hasattr(pred, "cpu") else self._np.asarray(pred)
        ih, iw = arr.shape[-2], arr.shape[-1]
        sw, sh = scene.get("width", iw), scene.get("height", ih)

        out = copy.deepcopy(scene)
        for o in out.get("objects", []):
            x, y, w, h = o["box"]
            # box (scene coords) -> depth-map pixel coords
            px0 = max(0, int(x / sw * iw)); px1 = min(iw, int((x + w) / sw * iw))
            py0 = max(0, int(y / sh * ih)); py1 = min(ih, int((y + h) / sh * ih))
            patch = arr[py0:max(py0 + 1, py1), px0:max(px0 + 1, px1)]
            disparity = float(self._np.median(patch))  # Depth Anything: larger == nearer
            # invert disparity -> a positive distance (larger == farther), our z convention
            distance = 1.0 / (abs(disparity) + 1e-6)
            o["z"] = distance
            if "size" not in o:
                # real size ~ apparent diagonal x distance (perspective): a near small
                # box and a far large box are disambiguated by the depth.
                diag = (w * w + h * h) ** 0.5
                o["size"] = diag * distance
        return out


def make_depth_source(spec: str = "authored"):
    """Return ``(source, label, blocker)`` for a CLI/config spec.

    ``authored`` (default) -> the offline authored-z source (never blocks).
    ``depth-anything[:<model_id>]`` -> pixel-derived metric depth; ``blocker`` is
    set (and ``source`` still returned) when weights/deps are unavailable so the
    caller records a blocker instead of a number.
    """
    kind, _, rest = spec.partition(":")
    if kind == "authored":
        src = AuthoredDepthSource()
        return src, src.label, src.blocker
    if kind in ("depth-anything", "depthanything", "da"):
        src = DepthAnythingV2Source(rest) if rest else DepthAnythingV2Source()
        return src, src.label, src.blocker
    raise ValueError(f"unknown depth source spec {spec!r}; have 'authored', 'depth-anything[:<id>]'")
