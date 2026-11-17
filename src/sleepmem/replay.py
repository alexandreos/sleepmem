"""Episodes -> training examples, plus held-out probes kept apart from them.

Two generators, chosen by `replay.generator` in config:

- `template`: asks about an episode by its tags and answers with the verbatim body. Exercises
  the pipeline on rung 0; tests word-for-word recall, which is not what we want to measure.
- `file`: replay sets written per episode by the lab assistant (Claude Code; decision
  2026-11-08), stored next to the episodes and never edited once a sleep used them:
      stm/replay/<id>.yaml   {author, train: [{q, a, t?}, ...]}      paraphrases, single facts,
                                                                     use in context, behaviour;
                                                                     optional type `t` (lookup,
                                                                     reject, negation, ...) kept
                                                                     as provenance
      stm/probes/<id>.yaml   {author, probes: [{q, facts, forbid}]}  held out; never trained on
  Training answers paraphrase the memory rather than quote it: the aim is gist recall that the
  model reconstructs, the way people remember the meaning of what they heard, not the words.

Probes are scored on gist: `facts` is a list of key facts, each a list of acceptable surface
forms (["21 March", "March 21", "21/03"]); a probe passes if at least `probe_pass_frac` of the
facts appear in any form and no `forbid` form does (procedural rules: "no Fahrenheit").

Leakage guard: `rehearsal.assert_no_probe_leak` rejects any training prompt equal or nearly
equal to a probe on record.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

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


# --- template generator (rung-0 plumbing) -------------------------------------------------


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


def _example(ep: Episode, q: str, a: str, type_: str | None = None) -> dict:
    return {
        "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        "episode_id": ep.id,
        "origin": "new",
        "type": type_,
    }


def _probe(ep: Episode, q: str, facts: list[list[str]], forbid: list[str] | None = None) -> dict:
    return {
        "episode_id": ep.id,
        "kind": ep.meta.kind,
        "messages": [{"role": "user", "content": q}],
        "facts": facts,
        "forbid": forbid or [],
    }


def template_train(ep: Episode, n: int, seed: int) -> list[dict]:
    templates = _rng(seed, ep.id, "train").sample(TRAIN_TEMPLATES, min(n, len(TRAIN_TEMPLATES)))
    return [_example(ep, t.format(cue=cue(ep)), ep.body.strip()) for t in templates]


def template_probes(ep: Episode, n: int, seed: int) -> list[dict]:
    templates = _rng(seed, ep.id, "probe").sample(PROBE_TEMPLATES, min(n, len(PROBE_TEMPLATES)))
    facts = [[t] for t in key_terms(ep)]
    return [_probe(ep, t.format(cue=cue(ep)), facts) for t in templates]


# --- file generator (written replay sets) -------------------------------------------------


def replay_path(stm_dir: Path, ep_id: str) -> Path:
    return stm_dir / "replay" / f"{ep_id}.yaml"


def probe_path(stm_dir: Path, ep_id: str) -> Path:
    return stm_dir / "probes" / f"{ep_id}.yaml"


def file_train(ep: Episode, stm_dir: Path) -> list[dict]:
    doc = yaml.safe_load(replay_path(stm_dir, ep.id).read_text())
    return [_example(ep, x["q"].strip(), x["a"].strip(), x.get("t")) for x in doc["train"]]


def file_probes(ep: Episode, stm_dir: Path) -> list[dict]:
    doc = yaml.safe_load(probe_path(stm_dir, ep.id).read_text())
    return [_probe(ep, p["q"].strip(), p["facts"], p.get("forbid")) for p in doc["probes"]]


# --- dispatch ----------------------------------------------------------------------------


class Generator:
    def __init__(self, cfg: dict, stm_dir: Path):
        self.kind = cfg["replay"]["generator"]
        if self.kind not in ("template", "file"):
            raise NotImplementedError(f"replay generator {self.kind!r}")
        self.cfg, self.stm_dir, self.seed = cfg["replay"], stm_dir, cfg["seed"]

    def ready(self, ep: Episode) -> bool:
        if self.kind == "template":
            return bool(ep.meta.tags)
        return (
            replay_path(self.stm_dir, ep.id).exists() and probe_path(self.stm_dir, ep.id).exists()
        )

    def train(self, ep: Episode) -> list[dict]:
        if self.kind == "template":
            return template_train(ep, self.cfg["train_per_episode"], self.seed)
        return file_train(ep, self.stm_dir)

    def probes(self, ep: Episode) -> list[dict]:
        if self.kind == "template":
            return template_probes(ep, self.cfg["probes_per_episode"], self.seed)
        return file_probes(ep, self.stm_dir)


# --- scoring -----------------------------------------------------------------------------


def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s)


def _has(text: str, form: str) -> bool:
    f = _norm(form).strip()
    pre = r"(?<![a-z0-9])" if f[:1].isalnum() else ""
    post = (
        r"(?:s|es)?(?![a-z0-9])"
        if f[-1:].isalpha()
        else r"(?![a-z0-9])"
        if f[-1:].isalnum()
        else ""
    )
    return re.search(pre + re.escape(f) + post, text) is not None


def score(answer: str, facts: list[list[str]], forbid: list[str] = ()) -> float:
    """Fraction of facts present in any accepted form; 0 if a forbidden form appears."""
    text = _norm(answer)
    if any(_has(text, f) for f in forbid):
        return 0.0
    hits = sum(any(_has(text, form) for form in alts) for alts in facts)
    return hits / max(len(facts), 1)
