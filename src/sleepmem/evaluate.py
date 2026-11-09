"""Recall / retention / capability probes, and the gate decision.

A "subject" is anything with `answer(list_of_message_lists) -> list[str]`, so the scoring and
gating logic is testable with a fake subject; `MlxSubject` is the real model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import replay


class MlxSubject:
    def __init__(self, model_path: Path, adapter_path: Path | None, cfg: dict):
        from mlx_lm import load

        self.model, self.tok = load(
            str(model_path), adapter_path=adapter_path and str(adapter_path)
        )
        self.kwargs = cfg.get("chat_template_kwargs") or {}
        self.max_tokens = cfg["eval"]["max_tokens"]

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        # The chat template inserts BOS itself, so encode without adding special tokens again.
        text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.kwargs
        )
        return self.tok.encode(text, add_special_tokens=False)

    def answer(self, batch: list[list[dict]]) -> list[str]:
        from mlx_lm import generate  # greedy by default

        return [
            generate(self.model, self.tok, self.prompt_ids(m), max_tokens=self.max_tokens)
            for m in batch
        ]


def run_probes(subject, probes: list[dict], pass_frac: float) -> list[dict]:
    answers = subject.answer([p["messages"] for p in probes]) if probes else []
    out = []
    for p, a in zip(probes, answers, strict=True):
        s = replay.score(a, p["key_terms"])
        out.append({**p, "answer": a, "score": s, "passed": s >= pass_frac})
    return out


def summarize(results: list[dict], exclude: set[str] = frozenset()) -> dict:
    """Accuracy over probes, overall and by kind; episodes in `exclude` don't count."""
    kept = [r for r in results if r["episode_id"] not in exclude]
    by_kind = defaultdict(list)
    for r in kept:
        by_kind[r["kind"]].append(r["passed"])
    acc = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "n": len(kept),
        "accuracy": acc([r["passed"] for r in kept]),
        "by_kind": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_kind.items())},
    }


def already_known(results: list[dict]) -> set[str]:
    """Episodes whose probes the pre-sleep model already passes (all of them)."""
    per_ep = defaultdict(list)
    for r in results:
        per_ep[r["episode_id"]].append(r["passed"])
    return {e for e, v in per_ep.items() if all(v)}


def load_general(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def run_general(subject, items: list[dict]) -> dict:
    answers = subject.answer([it["messages"] for it in items])
    passed = [
        any(acc.lower() in a.lower() for acc in it["accept"])
        for it, a in zip(items, answers, strict=True)
    ]
    return {
        "n": len(items),
        "accuracy": sum(passed) / len(items) if items else None,
        "answers": answers,
        "passed": passed,
    }


def gate(
    recall: dict,
    retention_pre: dict,
    retention_post: dict,
    cap_base: dict,
    cap_post: dict,
    th: dict,
) -> dict:
    checks, reasons = {}, []

    def check(name, ok, msg):
        checks[name] = bool(ok)
        if not ok:
            reasons.append(msg)

    r = recall["accuracy"]
    check("recall", r is None or r >= th["min_recall"], f"recall {r} < {th['min_recall']}")
    if retention_pre["accuracy"] is not None:
        drop = retention_pre["accuracy"] - retention_post["accuracy"]
        check(
            "retention",
            drop <= th["max_retention_drop"],
            f"retention drop {drop:.3f} > {th['max_retention_drop']}",
        )
    cdrop = cap_base["accuracy"] - cap_post["accuracy"]
    check(
        "capability",
        cdrop <= th["max_capability_drop"],
        f"capability drop {cdrop:.3f} > {th['max_capability_drop']}",
    )
    return {"passed": all(checks.values()), "checks": checks, "reasons": reasons}
