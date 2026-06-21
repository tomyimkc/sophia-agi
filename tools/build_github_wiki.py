#!/usr/bin/env python3
"""Render the GitHub Wiki (the `<repo>.wiki.git` surface) from the OKF belief graph.

`wiki/` holds OKF pages that `tools/wiki_sync.py` already generates from `data/*.json`
(and CI fails on drift). This projects those same pages into the flat namespace the
GitHub Wiki expects: one page per record (`PageType-id.md`), plus a generated
`Home.md`, `_Sidebar.md`, and `_Footer.md`. Frontmatter stays the source of truth in
`wiki/`; the wiki copy is a human-readable mirror, so it is never hand-edited.

    python tools/build_github_wiki.py            # -> _wiki_build/
    python tools/build_github_wiki.py --out DIR  # custom output dir

The companion workflow (.github/workflows/wiki-sync.yml) pushes _wiki_build/ to the
wiki repo on every push to main that touches wiki/, data/, okf/, or this script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter, page as okf_page  # noqa: E402

WIKI_DIR = ROOT / "wiki"

# Display order + headings for the page-type groups (others appended alphabetically).
GROUPS = [
    ("tradition", "Traditions"),
    ("text", "Texts"),
    ("figure", "Figures"),
    ("figure_source_seat", "Source seats"),
    ("concept", "Concepts"),
    ("event", "Events"),
]
GROUP_LABELS = dict(GROUPS)

GENERATED_NOTE = (
    "_Generated from the OKF belief graph in [`wiki/`]"
    "(https://github.com/tomyimkc/sophia-agi/tree/main/wiki) by "
    "`tools/build_github_wiki.py`. Provenance frontmatter in `data/*.json` is the "
    "source of truth — do not edit wiki pages by hand._"
)


def _page_type(meta: dict, page: okf_page.Page) -> str:
    return str(meta.get("pageType") or page.path.parent.name or "page")


def page_name(meta: dict, page: okf_page.Page) -> str:
    """Flat, collision-free wiki page name, e.g. ``Text-analects``."""
    pid = meta.get("id") or page.path.stem
    return f"{_page_type(meta, page).capitalize()}-{pid}"


def display_title(meta: dict, page: okf_page.Page) -> str:
    en = meta.get("canonicalTitleEn") or meta.get("id") or page.path.stem
    zh = meta.get("canonicalTitleZh")
    return f"{en} ({zh})" if zh else str(en)


def render_page(meta: dict, body: str, page: okf_page.Page) -> str:
    """One wiki page: breadcrumb + the OKF body (frontmatter stripped) + footer."""
    group = GROUP_LABELS.get(_page_type(meta, page), _page_type(meta, page).capitalize())
    head = f"[Home](Home) › **{group}**\n\n"
    body = body.strip() or f"# {display_title(meta, page)}\n"
    return f"{head}{body}\n\n---\n\n{GENERATED_NOTE}\n"


def _ordered_groups(by_type: dict) -> list:
    seen = [key for key, _ in GROUPS if key in by_type]
    rest = sorted(k for k in by_type if k not in seen)
    return seen + rest


def render_home(by_type: dict) -> str:
    lines = [
        "# Sophia AGI — Provenance Wiki",
        "",
        "Auto-generated reference index of the **OKF belief graph**: *who wrote "
        "what*, *which tradition owns which idea*, and which attributions are "
        "explicitly **refused**.",
        "",
        "- Narrative thesis site → https://tomyimkc.github.io/sophia-agi/",
        "- Honest, gated results → "
        "[RESULTS.md](https://github.com/tomyimkc/sophia-agi/blob/main/RESULTS.md)",
        "- Contribute → "
        "[CONTRIBUTING.md](https://github.com/tomyimkc/sophia-agi/blob/main/CONTRIBUTING.md)"
        " · [Good first issues]"
        "(https://github.com/tomyimkc/sophia-agi/blob/main/GOOD_FIRST_ISSUES.md)",
        "",
    ]
    for key in _ordered_groups(by_type):
        label = GROUP_LABELS.get(key, key.capitalize())
        entries = by_type[key]
        lines.append(f"## {label} ({len(entries)})")
        lines.append("")
        for name, title in entries:
            lines.append(f"- [{title}]({name})")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(GENERATED_NOTE)
    return "\n".join(lines) + "\n"


def render_sidebar(by_type: dict) -> str:
    lines = ["### [Sophia Wiki](Home)", ""]
    for key in _ordered_groups(by_type):
        label = GROUP_LABELS.get(key, key.capitalize())
        entries = by_type[key]
        lines.append(f"**{label}**")
        lines.append("")
        for name, title in entries:
            lines.append(f"- [{title}]({name})")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_footer() -> str:
    return (
        "Sophia AGI · *Wisdom before intelligence* · "
        "[thesis](https://tomyimkc.github.io/sophia-agi/) · "
        "[repo](https://github.com/tomyimkc/sophia-agi)\n"
    )


def build(out_dir: Path) -> int:
    pages = okf_page.load_pages(WIKI_DIR)
    if not pages:
        print(f"no OKF pages under {WIKI_DIR}", file=sys.stderr)
        return 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    by_type: dict = {}
    for page in pages:
        meta = page.meta or {}
        name = page_name(meta, page)
        title = display_title(meta, page)
        (out_dir / f"{name}.md").write_text(
            render_page(meta, page.body, page), encoding="utf-8"
        )
        by_type.setdefault(_page_type(meta, page), []).append((name, title))

    for entries in by_type.values():
        entries.sort(key=lambda nt: nt[1].lower())

    (out_dir / "Home.md").write_text(render_home(by_type), encoding="utf-8")
    (out_dir / "_Sidebar.md").write_text(render_sidebar(by_type), encoding="utf-8")
    (out_dir / "_Footer.md").write_text(render_footer(), encoding="utf-8")

    total = sum(len(v) for v in by_type.values())
    groups = ", ".join(f"{k}={len(v)}" for k, v in sorted(by_type.items()))
    print(f"wrote {total} wiki pages to {out_dir} ({groups}) + Home/_Sidebar/_Footer")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "_wiki_build"), help="output directory")
    args = ap.parse_args()
    return build(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
