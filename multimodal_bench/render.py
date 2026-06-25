# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Optional rasteriser: scene spec -> PNG bytes, for the real-VLM path only.

The benchmark's ground truth lives in the structured scene spec, so the offline
suite never needs pixels. But a real vision model needs an actual image — this
renders objects as labelled rectangles and texts as drawn strings. Pillow is an
optional dependency; the import is deferred so the rest of the package stays
zero-dependency and airgap-clean.
"""

from __future__ import annotations

_PALETTE = {
    "sofa": (150, 120, 90), "lamp": (230, 200, 120), "cat": (120, 120, 120),
    "dog": (160, 110, 70), "tree": (60, 140, 60), "bench": (120, 90, 60),
    "ball": (220, 80, 80), "car": (80, 110, 200), "traffic_light": (60, 60, 60),
    "window": (170, 210, 230), "picture": (200, 170, 140), "laptop": (90, 90, 100),
    "book": (180, 70, 70), "cup": (210, 210, 210), "towel": (230, 180, 90),
    "cloud": (210, 210, 220), "sun": (245, 215, 80), "fence": (140, 110, 80),
    "flower": (220, 120, 180), "person": (200, 160, 130), "chair": (110, 90, 70),
    "bottle": (90, 160, 130), "bird": (90, 90, 90), "box": (190, 160, 110),
    "plate": (220, 220, 220), "house": (180, 140, 110), "sign": (240, 240, 240),
    "label": (250, 250, 230), "door": (150, 110, 80),
}


def render_png(scene: dict, *, scale: int = 1) -> bytes:
    """Rasterise a scene to PNG bytes. Requires Pillow (raises ImportError if absent)."""
    import io

    from PIL import Image, ImageDraw  # optional dependency

    w = int(scene.get("width", 512)) * scale
    h = int(scene.get("height", 512)) * scale
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for o in scene.get("objects", []):
        x, y, bw, bh = (v * scale for v in o["box"])
        color = _PALETTE.get(o.get("label"), (160, 160, 160))
        draw.rectangle([x, y, x + bw, y + bh], fill=color, outline=(40, 40, 40), width=2)
        draw.text((x + 4, y + 4), o.get("label", ""), fill=(20, 20, 20))

    for t in scene.get("texts", []):
        x, y, bw, bh = (v * scale for v in t["box"])
        draw.rectangle([x, y, x + bw, y + bh], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 8, y + bh / 2 - 6), t.get("value", ""), fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
