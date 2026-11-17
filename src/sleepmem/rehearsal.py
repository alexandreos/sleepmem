"""Rehearsal sources and the interleaved training mix."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .stm import Episode, superseded_ids
from .triage import jaccard

DOLCI = {
    "repo": "allenai/Dolci-Instruct-SFT-No-Tools",
    "revision": "9156a5a542b2503100d4f1fabbf50be5eb25d977",
}


def consolidated_examples(
    consolidated: list[Episode], all_eps: list[Episode], n: int, gen, seed: int
) -> list[dict]:
    """Up to n examples replayed from already-consolidated, not-superseded episodes."""
    gone = superseded_ids(all_eps)
    pool = [ex for e in consolidated if e.id not in gone and gen.ready(e) for ex in gen.train(e)]
    for ex in pool:
        ex["origin"] = "consolidated"
    rng = random.Random(f"{seed}:rehearsal:consolidated")
    return rng.sample(pool, min(n, len(pool)))


def general_examples(source: Path | None, n: int, seed: int) -> tuple[list[dict], dict]:
    """Up to n general instruction examples, plus provenance for the manifest."""
    if source is None or n == 0:
        return [], {"source": None}
    lines = [ln for ln in source.read_text().splitlines() if ln.strip()]
    rng = random.Random(f"{seed}:rehearsal:general")
    picked = [json.loads(ln) for ln in rng.sample(lines, min(n, len(lines)))]
    out = [{"messages": ex["messages"], "origin": "general"} for ex in picked]
    info = {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()[:16],
        "available": len(lines),
    }
    meta = source.with_suffix(".meta.json")
    if meta.exists():  # dataset id + revision for cached subsets such as Dolci
        m = json.loads(meta.read_text())
        info.update(dataset=m.get("repo"), revision=m.get("revision"), subset_seed=m.get("seed"))
    return out, info


def mix(
    new: list[dict],
    consolidated: list[dict],
    general: list[dict],
    valid_frac: float,
    batch_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Shuffle all sources together and split off a validation set (for loss monitoring only;
    the gate uses probes). mlx_lm needs at least batch_size examples in each split."""
    allx = new + consolidated + general
    random.Random(f"{seed}:mix").shuffle(allx)
    n_valid = max(batch_size, math.ceil(valid_frac * len(allx)))
    if len(allx) - n_valid < batch_size:
        raise ValueError(
            f"only {len(allx)} examples: need >= {n_valid + batch_size} for train+valid splits"
        )
    return allx[n_valid:], allx[:n_valid]


def user_prompts(examples: list[dict]) -> set[str]:
    return {m["content"].strip() for ex in examples for m in ex["messages"] if m["role"] == "user"}


def assert_no_probe_leak(
    train: list[dict], probe_sets: list[list[dict]], near: float = 0.9
) -> None:
    """No training prompt may equal, or nearly equal (word Jaccard >= near), a probe prompt.
    Training on the same *fact* in other words is intended; reusing the question is not."""
    probes = {p for ps in probe_sets for p in user_prompts(ps)}
    leaked = [
        (t, p) for t in user_prompts(train) for p in probes if t == p or jaccard(t, p) >= near
    ]
    if leaked:
        raise AssertionError(f"probe prompts in training data: {leaked[:3]}")


def strip_meta(examples: list[dict]) -> list[dict]:
    """mlx_lm only needs messages; provenance stays in examples.jsonl."""
    return [{"messages": ex["messages"]} for ex in examples]


def cache_dolci(out_dir: Path, n: int = 2000, seed: int = 0, max_chars: int = 1600) -> Path:
    """Seeded subset of Dolci-Instruct-SFT-No-Tools, read row group by row group over HTTP
    (one random row group per shard, ~40 MB each) instead of downloading 2.3 GB.

    Keeps single- or multi-turn user/assistant conversations whose text fits in max_chars
    (~512 tokens, our max_seq_length), so the subset is biased toward short examples.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{DOLCI['repo']}@{DOLCI['revision']}/data"
    shards = sorted(fs.glob(f"{base}/train-*.parquet"))
    rng = random.Random(f"{seed}:dolci")
    rows = []
    for shard in shards:
        with fs.open(shard) as f:
            pf = pq.ParquetFile(f)
            rg = rng.randrange(pf.metadata.num_row_groups)
            rows += [(shard.rsplit("/", 1)[1], rg, r) for r in pf.read_row_group(rg).to_pylist()]

    def ok(msgs):
        roles = [m["role"] for m in msgs]
        return (
            roles
            and set(roles) <= {"system", "user", "assistant"}
            and roles[-1] == "assistant"
            and sum(len(m["content"] or "") for m in msgs) <= max_chars
        )

    kept = [x for x in rows if ok(x[2]["messages"])]
    picked = rng.sample(kept, min(n, len(kept)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{DOLCI['revision'][:8]}-seed{seed}-c{max_chars}-n{len(picked)}.jsonl"
    with open(out, "w") as f:
        for _shard, _rg, r in picked:
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            f.write(json.dumps({"messages": msgs, "id": r["id"], "source": r["source"]}) + "\n")
    meta = {
        **DOLCI,
        "seed": seed,
        "n": len(picked),
        "max_chars": max_chars,
        "rows_read": len(rows),
        "rows_eligible": len(kept),
        "row_groups": sorted({(s, g) for s, g, _ in rows}),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    from . import config

    ap = argparse.ArgumentParser(description="cache a seeded Dolci rehearsal subset")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=1600)
    a = ap.parse_args()
    print(cache_dolci(config.PROJECT_ROOT / "data/rehearsal/dolci", a.n, a.seed, a.max_chars))
