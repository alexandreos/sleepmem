"""Short-term memory store: one markdown episode per file, YAML frontmatter validated by pydantic.

Layout: stm/{inbox,consolidated,archive}/YYYY-MM-DD/<id>.md. The body never changes after
capture; only salience, status and sleeps may, and a status change moves the file.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ID_RE = r"^\d{4}-\d{2}-\d{2}-\d{4}$"
MUTABLE = {"salience", "status", "sleeps"}
STATUS_DIR = {"fresh": "inbox", "consolidated": "consolidated", "archived": "archive"}


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_RE)
    created: datetime
    source: Literal["chat", "note", "observation"]
    kind: Literal["episodic", "semantic", "procedural"]
    salience: float = Field(ge=0.0, le=1.0)
    tags: list[str] = []
    status: Literal["fresh", "consolidated", "archived"] = "fresh"
    sleeps: list[str] = []
    supersedes: str | None = Field(default=None, pattern=ID_RE)


class Episode(BaseModel):
    meta: Meta
    body: str
    path: Path | None = None

    @property
    def id(self) -> str:
        return self.meta.id


def parse(text: str, path: Path | None = None) -> Episode:
    m = re.match(r"^---\n(.*?)\n---\n(.*)\Z", text, re.S)
    if not m:
        raise ValueError(f"{path or 'episode'}: missing YAML frontmatter")
    raw = yaml.safe_load(m.group(1)) or {}
    if not m.group(2).strip():
        raise ValueError(f"{path or 'episode'}: empty body")
    return Episode(meta=Meta.model_validate(raw), body=m.group(2), path=path)


def dump(ep: Episode) -> str:
    d = ep.meta.model_dump()
    d["created"] = ep.meta.created.isoformat()
    return "---\n" + yaml.safe_dump(d, sort_keys=False, allow_unicode=True) + "---\n" + ep.body


def read(path: Path) -> Episode:
    return parse(path.read_text(), path)


def episodes(stm_dir: Path, status: str | None = None) -> list[Episode]:
    dirs = [STATUS_DIR[status]] if status else list(STATUS_DIR.values())
    out = [read(p) for d in dirs for p in sorted((stm_dir / d).glob("*/*.md"))]
    return sorted(out, key=lambda e: e.id)


def superseded_ids(eps: list[Episode]) -> set[str]:
    return {e.meta.supersedes for e in eps if e.meta.supersedes}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def update(ep: Episode, stm_dir: Path, **fields) -> Episode:
    """Change mutable frontmatter fields; moves the file if the status changes. Body untouched."""
    if bad := set(fields) - MUTABLE:
        raise ValueError(f"immutable episode fields: {sorted(bad)}")
    new = Episode(
        meta=Meta.model_validate({**ep.meta.model_dump(), **fields}), body=ep.body, path=ep.path
    )
    day = new.meta.id[:10]
    dest = stm_dir / STATUS_DIR[new.meta.status] / day / f"{new.id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(dest, dump(new))
    if ep.path and ep.path.resolve() != dest.resolve():
        ep.path.unlink()  # moved, not deleted: the same episode now lives at dest
    new.path = dest
    return new
