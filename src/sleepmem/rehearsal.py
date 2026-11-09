"""Rehearsal sources and the interleaved training mix."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from . import replay
from .stm import Episode, superseded_ids


def consolidated_examples(
    consolidated: list[Episode], all_eps: list[Episode], n: int, per_episode: int, seed: int
) -> list[dict]:
    """Up to n examples replayed from already-consolidated, not-superseded episodes."""
    gone = superseded_ids(all_eps)
    pool = [
        ex
        for e in consolidated
        if e.id not in gone and e.meta.tags
        for ex in replay.train_examples(e, per_episode, seed)
    ]
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


def assert_no_probe_leak(train: list[dict], probe_sets: list[list[dict]]) -> None:
    leaked = user_prompts(train) & {p for ps in probe_sets for p in user_prompts(ps)}
    if leaked:
        raise AssertionError(f"probe prompts in training data: {sorted(leaked)[:3]}")


def strip_meta(examples: list[dict]) -> list[dict]:
    """mlx_lm only needs messages; provenance stays in examples.jsonl."""
    return [{"messages": ex["messages"]} for ex in examples]
