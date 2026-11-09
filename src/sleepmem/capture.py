"""Write new episodes into stm/inbox/YYYY-MM-DD/. Malformed episodes are rejected here.

uv run python -m sleepmem.capture --kind semantic --salience 0.7 --tags a,b "text"
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import config, stm


def next_id(stm_dir: Path, day: str) -> str:
    taken = [
        int(p.stem[-4:])
        for d in stm.STATUS_DIR.values()
        for p in (stm_dir / d / day).glob(f"{day}-*.md")
    ]
    return f"{day}-{max(taken, default=0) + 1:04d}"


def capture(
    stm_dir: Path,
    body: str,
    *,
    kind: str,
    salience: float,
    source: str = "note",
    tags: list[str] | None = None,
    supersedes: str | None = None,
    now: datetime | None = None,
) -> stm.Episode:
    now = (now or datetime.now()).astimezone()
    day = now.strftime("%Y-%m-%d")
    if not body.endswith("\n"):
        body += "\n"
    meta = stm.Meta(
        id=next_id(stm_dir, day),
        created=now.replace(microsecond=0),
        source=source,
        kind=kind,
        salience=salience,
        tags=tags or [],
        supersedes=supersedes,
    )
    ep = stm.Episode(meta=meta, body=body)
    stm.parse(stm.dump(ep))  # round-trip check before anything touches disk
    dest = stm_dir / "inbox" / day / f"{meta.id}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "x") as f:  # never overwrite
        f.write(stm.dump(ep))
    ep.path = dest
    return ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("--kind", required=True, choices=["episodic", "semantic", "procedural"])
    ap.add_argument("--salience", type=float, default=0.5)
    ap.add_argument("--source", default="note", choices=["chat", "note", "observation"])
    ap.add_argument("--tags", default="")
    ap.add_argument("--supersedes")
    ap.add_argument("--stm", default=str(config.PROJECT_ROOT / "stm"))
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    ep = capture(
        Path(a.stm),
        a.body,
        kind=a.kind,
        salience=a.salience,
        source=a.source,
        tags=tags,
        supersedes=a.supersedes,
    )
    print(ep.path)


if __name__ == "__main__":
    main()
