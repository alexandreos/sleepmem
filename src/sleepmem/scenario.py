"""Run a dataset of episodes through day-by-day sleeps in a fresh sandbox.

    uv run python -m sleepmem.scenario --config config/sleep/smoke.yaml        # rung-0 smoke
    uv run python -m sleepmem.scenario --config config/sleep/r1-first.yaml \\
        --dataset datasets/synthetic-v1 [--days day1] [--dry-run] [--iters N]

A dataset directory holds episodes.yaml ({dayN: {date, episodes: {key: {...}}}}) and, for the
`file` generator, replay.yaml and probes.yaml keyed by the same episode keys. `replay.files`
in the config lists the replay files to combine (default: replay.yaml alone), so add-on sets
such as replay-contrast.yaml can be switched on per experiment. Seeding copies them into the
sandbox's stm/ under the captured ids. The sandbox is data/runs/<name>-<ts>/
(gitignored); the real stm/ and adapters/ are never touched.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from . import capture, config, replay, sleep, stm

SMOKE = config.PROJECT_ROOT / "tests" / "fixtures" / "smoke"


def load_replays(dataset: Path, files: list[str] | tuple[str, ...] = ("replay.yaml",)) -> dict:
    """Replay sets of the listed files, concatenated per episode key, in file order."""
    out: dict = {"_author": [], "_files": []}
    for name in files:
        p = dataset / name
        if not p.exists():
            if name == "replay.yaml":  # datasets for the template generator have none
                continue
            raise FileNotFoundError(p)
        doc = yaml.safe_load(p.read_text())
        out["_files"].append(name)
        out["_author"].append(doc.pop("_author", None))
        for key, examples in doc.items():
            out.setdefault(key, []).extend(examples)
    out["_author"] = "; ".join(a for a in out["_author"] if a) or None
    return out


def seed_day(
    stm_dir: Path,
    dataset: Path,
    day: dict,
    ids: dict,
    replay_files: list[str] | tuple[str, ...] = ("replay.yaml",),
) -> None:
    when = datetime.fromisoformat(str(day["date"]))
    pp = dataset / "probes.yaml"
    replays = load_replays(dataset, replay_files)
    probes = yaml.safe_load(pp.read_text()) if pp.exists() else {}
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
        for src, dest, field in (
            (replays, replay.replay_path(stm_dir, ep.id), "train"),
            (probes, replay.probe_path(stm_dir, ep.id), "probes"),
        ):
            if key in src:
                dest.parent.mkdir(parents=True, exist_ok=True)
                doc = {"episode": ep.id, "key": key, "author": src.get("_author")}
                if "_files" in src:
                    doc["files"] = src["_files"]
                doc[field] = src[key]
                dest.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.PROJECT_ROOT / "config/sleep/smoke.yaml"))
    ap.add_argument("--dataset", default=str(SMOKE))
    ap.add_argument("--days", nargs="*", help="subset of days to run (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    dataset = config.project_path(a.dataset)
    tag = cfg["name"] + ("-dry" if a.dry_run else "")
    sandbox = config.PROJECT_ROOT / "data/runs" / f"{tag}-{datetime.now():%Y%m%d-%H%M%S}"
    cfg["root"] = str(sandbox)
    stm_dir = config.path(cfg, "stm")
    fx = yaml.safe_load((dataset / "episodes.yaml").read_text())
    ids: dict[str, str] = {}
    replay_files = cfg["replay"].get("files") or ["replay.yaml"]
    report = []
    for n, day in enumerate(a.days or list(fx), 1):
        seed_day(stm_dir, dataset, fx[day], ids, replay_files)
        now = datetime.fromisoformat(str(fx[day]["date"])).replace(hour=23)
        m = sleep.run(
            cfg,
            sleep_id=f"{n:02d}-{day}",
            iters=a.iters,
            now=now,
            dry_run=a.dry_run,
            max_episodes=a.max_episodes,
            allow_download=a.allow_download,
        )
        keep = ("sleep_id", "status", "counts", "results", "gate", "timings")
        report.append({k: m.get(k) for k in keep})
        print(json.dumps(report[-1], default=str), flush=True)
    cur = config.path(cfg, "adapters") / "current"
    status = {
        s: [e.id for e in stm.episodes(stm_dir, s)] for s in ("fresh", "consolidated", "archived")
    }
    summary = {
        "sandbox": str(sandbox),
        "dataset": str(dataset),
        "replay_files": replay_files,
        "keys": ids,
        "adapters/current": cur.readlink().name if cur.is_symlink() else None,
        "episodes": status,
    }
    (sandbox / "scenario.json").write_text(json.dumps({**summary, "sleeps": report}, indent=2))
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
