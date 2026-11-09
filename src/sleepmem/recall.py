"""Wake-time retrieval over stm/ (the pre-consolidation path and the RAG baseline).

BM25 over episode bodies and tags. Deliberately simple; nothing leaves the machine.
"""

from __future__ import annotations

import math
from collections import Counter

from .replay import STOP, TOKEN
from .stm import Episode, superseded_ids


def _toks(text: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(text) if t.lower() not in STOP]


def search(query: str, eps: list[Episode], k: int = 3, k1: float = 1.5, b: float = 0.75):
    gone = superseded_ids(eps)
    eps = [e for e in eps if e.id not in gone and e.meta.status != "archived"]
    docs = [_toks(e.body + " " + " ".join(e.meta.tags)) for e in eps]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    q = _toks(query)
    scored = []
    for e, d in zip(eps, docs, strict=True):
        tf = Counter(d)
        s = sum(
            math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * tf[t]
            * (k1 + 1)
            / (tf[t] + k1 * (1 - b + b * len(d) / avg))
            for t in q
            if t in tf
        )
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [e for _, e in scored[:k]]


def rag_messages(messages: list[dict], eps: list[Episode], k: int = 3) -> list[dict]:
    """Prepend retrieved memories to the last user turn (no system role: Gemma lacks one)."""
    hits = search(messages[-1]["content"], eps, k)
    if not hits:
        return messages
    notes = "\n".join(f"- {e.body.strip()}" for e in hits)
    last = {**messages[-1], "content": f"Your notes:\n{notes}\n\n{messages[-1]['content']}"}
    return messages[:-1] + [last]
