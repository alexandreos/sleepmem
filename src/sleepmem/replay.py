"""Episodes -> training examples, plus held-out probes generated separately.

Only the `template` generator exists so far: it asks about an episode by its tags and answers
with the episode body, so it exercises the pipeline but tests verbatim recall of short prose.
Who should write replay examples (the subject "dreaming", a stronger model) is an open
research question; new generators plug in behind `train_examples` / `probes`.

Leakage guard: training prompts and probe prompts come from disjoint template pools, and
`rehearsal.assert_no_probe_leak` checks every mix against every probe on record.
"""

from __future__ import annotations

import random
import re

from .stm import Episode

TRAIN_TEMPLATES = [
    "What do you remember about {cue}?",
    "Tell me what you know about {cue}.",
    "Remind me about {cue}.",
    "Do you recall anything about {cue}?",
    "Summarize your memory of {cue}.",
    "What's the story with {cue}?",
]
PROBE_TEMPLATES = [
    "From memory, what can you tell me about {cue}?",
    "Without looking anything up: what do you know about {cue}?",
    "Quick question about {cue}. What do you remember?",
]
assert not set(TRAIN_TEMPLATES) & set(PROBE_TEMPLATES)

STOP = set(
    "a an the and or but of to in on at for with by from is are was were be been it its this "
    "that these those i me my we our you your he she they them their his her as so if then than "
    "do does did not no yes have has had will would should can could about into over after "
    "before when where what which who why how all any some just only also very more most".split()
)
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d[\d.,:]*")


def cue(ep: Episode) -> str:
    if not ep.meta.tags:
        raise ValueError(f"{ep.id}: the template generator needs tags to ask about")
    return ", ".join(t.replace("-", " ") for t in ep.meta.tags)


def key_terms(ep: Episode, k: int = 5) -> list[str]:
    """Distinctive words of the body (numbers and names first) that the cue doesn't give away."""
    cue_words = set(cue(ep).lower().split())
    seen, terms = set(), []
    for i, tok in enumerate(TOKEN.findall(ep.body)):
        low = tok.lower().strip(".,:")
        if (
            low in STOP
            or low in cue_words
            or low in seen
            or (len(low) < 3 and not low[0].isdigit())
        ):
            continue
        seen.add(low)
        rank = (0 if tok[0].isdigit() or (tok[0].isupper() and i > 0) else 1, -len(low), i)
        terms.append((rank, low))
    return [t for _, t in sorted(terms)[:k]]


def _rng(seed: int, ep_id: str, salt: str) -> random.Random:
    return random.Random(f"{seed}:{ep_id}:{salt}")


def train_examples(ep: Episode, n: int, seed: int) -> list[dict]:
    templates = _rng(seed, ep.id, "train").sample(TRAIN_TEMPLATES, min(n, len(TRAIN_TEMPLATES)))
    c, answer = cue(ep), ep.body.strip()
    return [
        {
            "messages": [
                {"role": "user", "content": t.format(cue=c)},
                {"role": "assistant", "content": answer},
            ],
            "episode_id": ep.id,
            "origin": "new",
        }
        for t in templates
    ]


def probes(ep: Episode, n: int, seed: int) -> list[dict]:
    templates = _rng(seed, ep.id, "probe").sample(PROBE_TEMPLATES, min(n, len(PROBE_TEMPLATES)))
    terms = key_terms(ep)
    return [
        {
            "episode_id": ep.id,
            "kind": ep.meta.kind,
            "messages": [{"role": "user", "content": t.format(cue=cue(ep))}],
            "key_terms": terms,
        }
        for t in templates
    ]


def score(answer: str, terms: list[str]) -> float:
    low = answer.lower()
    hits = sum(bool(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)) for t in terms)
    return hits / max(len(terms), 1)
