"""Pick fresh episodes worth consolidating and merge near-duplicates."""

from __future__ import annotations

import re

from .stm import Episode, superseded_ids

WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / max(len(wa | wb), 1)


def triage(
    fresh: list[Episode], all_eps: list[Episode], min_salience: float, dedup_threshold: float
) -> dict:
    """Returns {selected: [ids], merged: {kept_id: [dup_ids]}, skipped: {id: reason}}.

    Duplicates fold into the most salient member of their group (ties: earliest id).
    Superseded episodes are skipped; their correction is consolidated instead.
    """
    gone = superseded_ids(all_eps)
    skipped, cands = {}, []
    for e in fresh:
        if e.id in gone:
            skipped[e.id] = "superseded"
        elif e.meta.salience < min_salience:
            skipped[e.id] = "low_salience"
        else:
            cands.append(e)
    cands.sort(key=lambda e: (-e.meta.salience, e.id))
    kept: list[Episode] = []
    merged: dict[str, list[str]] = {}
    for e in cands:
        twin = next((k for k in kept if jaccard(k.body, e.body) >= dedup_threshold), None)
        if twin:
            merged.setdefault(twin.id, []).append(e.id)
        else:
            kept.append(e)
    return {"selected": sorted(k.id for k in kept), "merged": merged, "skipped": skipped}
