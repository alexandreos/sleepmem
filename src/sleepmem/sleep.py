"""One full sleep cycle: triage -> probes -> pretest -> replay -> interleave -> train -> gate
-> commit/rollback -> decay. Every stage writes into data/sleeps/<sleep_id>/; with --resume a
stage whose output already exists is loaded instead of recomputed.

    uv run python -m sleepmem.sleep --config config/sleep/<name>.yaml [--dry-run] [--iters N]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from datetime import datetime, timedelta
from importlib import metadata
from pathlib import Path

from . import config, evaluate, rehearsal, replay, stm, train
from .triage import triage


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class Stages:
    def __init__(self, sd: Path, resume: bool):
        self.sd, self.resume, self.timings = sd, resume, {}

    def json(self, name: str, fn):
        out = self.sd / f"{name}.json"
        if self.resume and out.exists():
            return json.loads(out.read_text())
        t = time.time()
        res = fn()
        self.timings[name] = round(time.time() - t, 2)
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
        return res


def versions() -> dict:
    v = {"python": platform.python_version()}
    for pkg in ("mlx", "mlx-lm", "transformers"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    return v


def current_adapter(adapters_dir: Path) -> Path | None:
    cur = adapters_dir / "current"
    return cur.resolve() if cur.exists() else None


def promote(adapters_dir: Path, sleep_id: str) -> None:
    tmp = adapters_dir / ".current.tmp"
    if tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(sleep_id)  # relative link: adapters/current -> <sleep_id>
    os.replace(tmp, adapters_dir / "current")


def probes_on_record(sleeps_dir: Path, exclude: str) -> list[dict]:
    return [
        p
        for f in sorted(sleeps_dir.glob("*/probes.jsonl"))
        if f.parent.name != exclude
        for p in read_jsonl(f)
    ]


def retention_probes(sleeps_dir: Path, live_ids: set[str]) -> list[dict]:
    """Probes from promoted earlier sleeps whose episodes are still live (consolidated, not
    superseded), minus episodes that sleep found already known."""
    out = []
    for man_path in sorted(sleeps_dir.glob("*/manifest.json")):
        man = json.loads(man_path.read_text())
        if man.get("status") != "promoted":
            continue
        known = set(man.get("counts", {}).get("already_known_ids", []))
        out += [
            p
            for p in read_jsonl(man_path.parent / "probes.jsonl")
            if p["episode_id"] in live_ids and p["episode_id"] not in known
        ]
    return out


def decay(stm_dir: Path, now: datetime, min_age_days: int, max_salience: float) -> list[str]:
    archived = []
    for ep in stm.episodes(stm_dir, "consolidated"):
        if ep.meta.salience < max_salience and now - ep.meta.created > timedelta(min_age_days):
            stm.update(ep, stm_dir, status="archived")
            archived.append(ep.id)
    return archived


def _free():
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def run(
    cfg: dict,
    *,
    sleep_id: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    iters: int | None = None,
    max_episodes: int | None = None,
    allow_download: bool = False,
    subject_factory=None,
    trainer=None,
    now: datetime | None = None,
) -> dict:
    now = (now or datetime.now()).astimezone()
    stm_dir, sleeps_dir, adapters_dir = (config.path(cfg, k) for k in ("stm", "sleeps", "adapters"))
    if dry_run:
        iters = iters or 20
        max_episodes = max_episodes or 5
    sleep_id = sleep_id or now.strftime("%Y%m%d-%H%M%S-") + cfg["name"] + (
        "-dry" if dry_run else ""
    )
    sd = sleeps_dir / sleep_id
    if sd.exists() and not resume:
        raise FileExistsError(f"{sd} exists; pass --resume to continue it")
    sd.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    st = Stages(sd, resume)
    seed = cfg["seed"]
    spec = config.model_spec(cfg)
    manifest = {
        "sleep_id": sleep_id,
        "started": now.isoformat(),
        "config": cfg["config_path"],
        "config_hash": config.config_hash(cfg),
        "rung": cfg["rung"],
        "model": {"repo": spec["repo"], "revision": spec["revision"]},
        "chat_template_kwargs": cfg.get("chat_template_kwargs") or {},
        "replay_generator": cfg["replay"]["generator"],
        "adapter_mode": cfg["train"]["adapter_mode"],
        "parent_adapter": None,
        "dry_run": dry_run,
        "versions": versions(),
        "status": "running",
    }

    def save():
        manifest["timings"] = st.timings
        (sd / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    try:
        gen = replay.Generator(cfg, stm_dir)
        all_eps = stm.episodes(stm_dir)
        by_id = {e.id: e for e in all_eps}
        fresh = [e for e in all_eps if e.meta.status == "fresh"]

        # 1. triage
        def do_triage():
            t = triage(
                fresh, all_eps, cfg["triage"]["min_salience"], cfg["triage"]["dedup_threshold"]
            )
            for i in [i for i in t["selected"] if not gen.ready(by_id[i])]:
                t["selected"].remove(i)
                t["skipped"][i] = "no_replay_written"
                t["skipped"].update({d: "no_replay_written" for d in t["merged"].pop(i, [])})
            if max_episodes is not None and len(t["selected"]) > max_episodes:
                for i in t["selected"][max_episodes:]:
                    t["skipped"][i] = "dry_run_limit"
                    t["skipped"].update({d: "dry_run_limit" for d in t["merged"].pop(i, [])})
                t["selected"] = t["selected"][:max_episodes]
            return t

        tri = st.json("triage", do_triage)
        selected = [by_id[i] for i in tri["selected"]]
        manifest["counts"] = {
            "fresh": len(fresh),
            "selected": len(selected),
            "merged": sum(map(len, tri["merged"].values())),
            "skipped": len(tri["skipped"]),
        }
        if not selected:
            manifest["status"] = "nothing_to_do"
            return manifest

        # 2a. probes (separate from training data, seeded)
        probes_path = sd / "probes.jsonl"
        if not (resume and probes_path.exists()):
            pr = [p for e in selected for p in gen.probes(e)]
            write_jsonl(probes_path, pr)
        tonight_probes = read_jsonl(probes_path)

        gone = stm.superseded_ids(all_eps)
        live = {e.id for e in all_eps if e.meta.status == "consolidated" and e.id not in gone}
        old_probes = retention_probes(sleeps_dir, live)
        write_jsonl(sd / "retention_probes.jsonl", old_probes)

        model_path = config.resolve_model(cfg, allow_download)
        parent = current_adapter(adapters_dir)
        manifest["parent_adapter"] = parent.name if parent else None
        make = subject_factory or (lambda adapter: evaluate.MlxSubject(model_path, adapter, cfg))
        pass_frac = cfg["eval"]["probe_pass_frac"]
        general = evaluate.load_general(config.project_path(cfg["paths"]["general_eval"]))

        # 2b. pretest: tonight's probes and retention probes on the pre-sleep model
        def do_pretest():
            subj = make(parent)
            res = {
                "tonight": evaluate.run_probes(subj, tonight_probes, pass_frac),
                "retention": evaluate.run_probes(subj, old_probes, pass_frac),
            }
            del subj
            _free()
            return res

        pre = st.json("pretest", do_pretest)
        known = evaluate.already_known(pre["tonight"])
        manifest["counts"]["already_known_ids"] = sorted(known)

        def do_base_general():
            subj = make(None)
            res = evaluate.run_general(subj, general)
            del subj
            _free()
            return res

        cap_base = st.json("general_base", do_base_general)

        # 3. replay + interleave
        new = [ex for e in selected for ex in gen.train(e)]
        ic = cfg["interleave"]
        consolidated = [e for e in all_eps if e.id in live]
        old = rehearsal.consolidated_examples(
            consolidated,
            all_eps,
            round(ic["consolidated_ratio"] * len(new)),
            gen,
            seed,
        )
        src = ic.get("general_source")
        gen, gen_info = rehearsal.general_examples(
            config.project_path(src) if src else None, round(ic["general_ratio"] * len(new)), seed
        )
        tr, va = rehearsal.mix(
            new, old, gen, ic["valid_frac"], cfg["train"]["lora"]["batch_size"], seed
        )
        rehearsal.assert_no_probe_leak(
            tr + va, [tonight_probes, old_probes, probes_on_record(sleeps_dir, sleep_id)]
        )
        write_jsonl(sd / "examples.jsonl", tr + va)
        write_jsonl(sd / "train.jsonl", rehearsal.strip_meta(tr))
        write_jsonl(sd / "valid.jsonl", rehearsal.strip_meta(va))
        manifest["rehearsal"] = {"general": gen_info, "consolidated_episodes": len(consolidated)}
        manifest["counts"].update(
            new_examples=len(new),
            consolidated_examples=len(old),
            general_examples=len(gen),
            train=len(tr),
            valid=len(va),
            probes=len(tonight_probes),
            retention_probes=len(old_probes),
        )

        # 4. train
        adapter_dir = adapters_dir / sleep_id
        if not (resume and (adapter_dir / "adapters.safetensors").exists()):
            if adapter_dir.exists():
                raise FileExistsError(f"{adapter_dir} exists; adapters are never overwritten")
            lc = train.lora_config(
                cfg,
                model_path,
                sd,
                adapter_dir,
                train.resume_from(cfg, adapters_dir),
                iters,
                n_train=len(tr),
            )
            manifest["lora"] = lc
            t = time.time()
            (trainer or train.run_lora)(lc, sd)
            st.timings["train"] = round(time.time() - t, 2)

        # 5. gate
        def do_post():
            subj = make(adapter_dir)
            res = {
                "tonight": evaluate.run_probes(subj, tonight_probes, pass_frac),
                "retention": evaluate.run_probes(subj, old_probes, pass_frac),
                "general": evaluate.run_general(subj, general),
            }
            del subj
            _free()
            return res

        post = st.json("posttest", do_post)
        results = {
            "recall_pre": evaluate.summarize(pre["tonight"], known),
            "recall": evaluate.summarize(post["tonight"], known),
            "retention_pre": evaluate.summarize(pre["retention"]),
            "retention": evaluate.summarize(post["retention"]),
            "capability_base": cap_base["accuracy"],
            "capability": post["general"]["accuracy"],
        }
        g = evaluate.gate(
            results["recall"],
            results["retention_pre"],
            results["retention"],
            cap_base,
            post["general"],
            cfg["gate"],
        )
        manifest["results"], manifest["gate"] = results, g

        # 6. commit or roll back
        if dry_run:
            manifest["status"] = "dry_run"
        elif g["passed"]:
            promote(adapters_dir, sleep_id)
            for i in tri["selected"] + [d for ds in tri["merged"].values() for d in ds]:
                ep = stm.read(by_id[i].path)
                stm.update(ep, stm_dir, status="consolidated", sleeps=ep.meta.sleeps + [sleep_id])
            manifest["status"] = "promoted"
            # 7. decay
            dc = cfg["decay"]
            manifest["archived"] = decay(stm_dir, now, dc["min_age_days"], dc["max_salience"])
        else:
            manifest["status"] = "rejected"
        return manifest
    except Exception as e:
        manifest["status"] = "error"
        manifest["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        manifest["finished"] = datetime.now().astimezone().isoformat()
        save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="<=5 episodes, 20 iters, no promotion or stm/ changes",
    )
    ap.add_argument("--iters", type=int)
    ap.add_argument("--max-episodes", type=int)
    ap.add_argument("--resume", metavar="SLEEP_ID")
    ap.add_argument("--root", help="run against another project root (sandbox)")
    ap.add_argument("--allow-download", action="store_true")
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    if a.root:
        cfg["root"] = a.root
    m = run(
        cfg,
        sleep_id=a.resume,
        resume=bool(a.resume),
        dry_run=a.dry_run,
        iters=a.iters,
        max_episodes=a.max_episodes,
        allow_download=a.allow_download,
    )
    print(
        json.dumps(
            {k: m.get(k) for k in ("sleep_id", "status", "counts", "results", "gate")},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
