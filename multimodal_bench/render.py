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

    # Depth-aware paint order: farther objects (larger z) first so nearer ones
    # are drawn on top — the PNG then shows occlusion consistent with the
    # `occludes` verifier. Stable sort: scenes without z keep their authored
    # order (z defaults to 0), so the non-physical suite renders unchanged.
    objects = sorted(scene.get("objects", []), key=lambda o: -o.get("z", 0.0))
    for o in objects:
        x, y, bw, bh = (v * scale for v in o["box"])
        color = _PALETTE.get(o.get("label"), (160, 160, 160))
        draw.rectangle([x, y, x + bw, y + bh], fill=color, outline=(40, 40, 40), width=2)
        draw.text((x + 4, y + 4), o.get("label", ""), fill=(20, 20, 20))

    for t in scene.get("texts", []):
        x, y, bw, bh = (v * scale for v in t["box"])
        draw.rectangle([x, y, x + bw, y + bh], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 8, y + bh / 2 - 6), t.get("value", ""), fill=(0, 0, 0))

    if scene.get("chart"):
        _draw_chart(draw, scene["chart"], w, h)
    if scene.get("table"):
        _draw_table(draw, scene["table"], w, h)
    if scene.get("document"):
        _draw_document(draw, scene["document"], w, h)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_chart(draw, chart: dict, w: int, h: int) -> None:
    """Bar chart: bars scaled to value, with the value printed atop each bar."""
    bars = chart.get("bars", [])
    if not bars:
        return
    draw.text((20, 16), chart.get("title", ""), fill=(0, 0, 0))
    base_y, top_y = h - 60, 80
    vmax = max(b["value"] for b in bars) or 1
    slot = (w - 80) / len(bars)
    for i, b in enumerate(bars):
        bx = 60 + i * slot
        bh = (b["value"] / vmax) * (base_y - top_y)
        draw.rectangle([bx, base_y - bh, bx + slot * 0.6, base_y], fill=(80, 110, 200), outline=(20, 20, 20))
        draw.text((bx, base_y - bh - 16), str(b["value"]), fill=(0, 0, 0))     # value label
        draw.text((bx, base_y + 8), str(b["label"]), fill=(0, 0, 0))           # axis label
    draw.line([60, base_y, w - 20, base_y], fill=(0, 0, 0), width=2)


def _draw_table(draw, table: dict, w: int, h: int) -> None:
    cols = table.get("columns", [])
    rows = table.get("rows", [])
    cw = (w - 80) / max(1, len(cols))
    y = 80
    for j, c in enumerate(cols):
        draw.text((40 + j * cw + 6, y), str(c), fill=(0, 0, 0))
    draw.line([40, y + 22, w - 40, y + 22], fill=(0, 0, 0), width=2)
    for r, row in enumerate(rows):
        ry = y + 30 + r * 30
        for j, cell in enumerate(row):
            draw.text((40 + j * cw + 6, ry), str(cell), fill=(0, 0, 0))


def _draw_document(draw, document: dict, w: int, h: int) -> None:
    y = 80
    for name, value in document.get("fields", {}).items():
        draw.text((50, y), f"{name}:", fill=(0, 0, 0))
        draw.text((230, y), str(value), fill=(0, 0, 0))
        y += 36
