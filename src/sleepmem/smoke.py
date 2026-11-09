"""Rung-0 smoke test: two sleep cycles over fictional episodes in a fresh sandbox.

    uv run python -m sleepmem.smoke [--iters 20]

The sandbox lives in data/smoke/<timestamp>/ (gitignored); the real stm/ and adapters/ are
never touched. Checks plumbing only: never report these numbers.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from . import capture, config, sleep, stm

FIXTURES = config.PROJECT_ROOT / "tests" / "fixtures" / "smoke_episodes.yaml"


def seed_day(stm_dir: Path, day: dict, ids: dict) -> None:
    when = datetime.fromisoformat(str(day["date"]))
    for key, e in day["episodes"].items():
        sup = ids[e["supersedes"]] if e.get("supersedes") else None
        ep = capture.capture(
            stm_dir,
            e["body"],
            kind=e["kind"],
            salience=e["salience"],
            source=e["source"],
            tags=e["tags"],
            supersedes=sup,
            now=when,
        )
        ids[key] = ep.id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.PROJECT_ROOT / "config/sleep/smoke.yaml"))
    ap.add_argument("--iters", type=int)
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    if cfg["rung"] != 0:
        raise SystemExit("the smoke test runs on rung 0 only")
    sandbox = config.PROJECT_ROOT / "data" / "smoke" / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg["root"] = str(sandbox)
    stm_dir = config.path(cfg, "stm")
    fx = yaml.safe_load(FIXTURES.read_text())
    ids: dict[str, str] = {}
    report = []
    for n, day in enumerate(("day1", "day2"), 1):
        seed_day(stm_dir, fx[day], ids)
        now = datetime.fromisoformat(str(fx[day]["date"])).replace(hour=23)
        m = sleep.run(cfg, sleep_id=f"smoke-{n}", iters=a.iters, now=now)
        report.append(
            {k: m.get(k) for k in ("sleep_id", "status", "counts", "results", "gate", "timings")}
        )
    cur = config.path(cfg, "adapters") / "current"
    status = {
        s: [e.id for e in stm.episodes(stm_dir, s)] for s in ("fresh", "consolidated", "archived")
    }
    print(
        json.dumps(
            {
                "sandbox": str(sandbox),
                "sleeps": report,
                "adapters/current": cur.readlink().name if cur.is_symlink() else None,
                "episodes": status,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
