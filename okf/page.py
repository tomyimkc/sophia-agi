"""The OKF Page object: frontmatter + body, with load/save and edge accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from okf import frontmatter, schema, wikilinks


@dataclass
class Page:
    path: Path
    meta: dict = field(default_factory=dict)
    body: str = ""

    @property
    def id(self) -> str:
        return str(self.meta.get("id") or wikilinks.normalize_target(self.path.stem))

    @property
    def page_type(self):
        return self.meta.get("pageType")

    @property
    def aliases(self) -> "list[str]":
        return [wikilinks.normalize_target(a) for a in schema.as_list(self.meta.get("aliases"))]

    def body_links(self) -> "list[str]":
        return wikilinks.extract_links(self.body)

    def edge_targets(self, key: str) -> "list[str]":
        return [wikilinks.normalize_target(t) for t in schema.as_list(self.meta.get(key))]

    def out_links(self) -> "list[str]":
        """All forward links: inline [[wikilinks]] plus frontmatter `links`."""
        seen: list[str] = list(self.body_links())
        for target in self.edge_targets("links"):
            if target not in seen:
                seen.append(target)
        return seen

    def validate(self) -> "list[str]":
        return schema.validate_meta(self.meta)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(frontmatter.serialize(self.meta, self.body), encoding="utf-8")


def load(path) -> Page:
    path = Path(path)
    meta, body = frontmatter.parse(path.read_text(encoding="utf-8"))
    return Page(path=path, meta=meta or {}, body=body)


def load_pages(*roots) -> "list[Page]":
    """Load every *.md page under the given roots (files or directories)."""
    pages: list[Page] = []
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix == ".md":
            pages.append(load(root))
        elif root.is_dir():
            for md in sorted(root.rglob("*.md")):
                pages.append(load(md))
    return pages
