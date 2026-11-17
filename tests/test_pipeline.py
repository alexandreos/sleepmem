from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sleepmem import capture, config, evaluate, recall, rehearsal, replay, sleep, stm, triage

T0 = datetime.fromisoformat("2026-11-01T09:00:00-08:00")


def cap(d, body, **kw):
    kw = {"kind": "semantic", "salience": 0.7, "tags": ["topic"], "now": T0, **kw}
    return capture.capture(d, body, **kw)


# --- store -------------------------------------------------------------------------------


def test_capture_roundtrip_and_ids(tmp_path):
    a = cap(tmp_path, "First memory.")
    b = cap(tmp_path, "Second memory.")
    assert (a.id, b.id) == ("2026-11-01-0001", "2026-11-01-0002")
    assert a.path == tmp_path / "inbox/2026-11-01/2026-11-01-0001.md"
    back = stm.read(a.path)
    assert back.meta == a.meta and back.body == "First memory.\n"


@pytest.mark.parametrize("kw", [{"salience": 1.5}, {"kind": "dream"}, {"supersedes": "x"}])
def test_capture_rejects_malformed(tmp_path, kw):
    with pytest.raises(ValidationError):
        cap(tmp_path, "bad", **kw)
    assert not list(tmp_path.rglob("*.md"))


def test_parse_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        stm.parse(
            "---\nid: 2026-11-01-0001\ncreated: 2026-11-01T09:00:00\nsource: chat\n"
            "kind: semantic\nsalience: 0.5\nmood: happy\n---\nbody\n"
        )


def test_update_moves_file_and_keeps_body_bytes(tmp_path):
    body = "Odd   spacing,\n\ttabs and a trailing line\n\n"
    ep = cap(tmp_path, body)
    raw_body = ep.path.read_text().split("---\n", 2)[2]
    new = stm.update(ep, tmp_path, status="consolidated", sleeps=["s1"])
    assert not ep.path.exists()
    assert new.path == tmp_path / "consolidated/2026-11-01/2026-11-01-0001.md"
    assert new.path.read_text().split("---\n", 2)[2] == raw_body == body
    with pytest.raises(ValueError, match="immutable"):
        stm.update(new, tmp_path, tags=["changed"])


# --- triage ------------------------------------------------------------------------------


def test_triage_dedup_salience_supersedes(tmp_path):
    a = cap(tmp_path, "The demo is due on 14 March.", salience=0.6)
    b = cap(tmp_path, "The demo is due on 14 March!", salience=0.8)
    low = cap(tmp_path, "Something minor.", salience=0.1)
    old = cap(tmp_path, "Router port is 8080.")
    fix = cap(tmp_path, "Router port is 8443.", supersedes=old.id)
    eps = stm.episodes(tmp_path)
    t = triage.triage(eps, eps, min_salience=0.3, dedup_threshold=0.9)
    assert t["selected"] == sorted([b.id, fix.id])
    assert t["merged"] == {b.id: [a.id]}
    assert t["skipped"] == {low.id: "low_salience", old.id: "superseded"}


# --- replay / probes ---------------------------------------------------------------------


def test_probes_disjoint_seeded_and_leak_check(tmp_path):
    ep = cap(tmp_path, "Priya Natarajan owns the Heron slides, due 14 March.", tags=["heron"])
    tr = replay.template_train(ep, 6, seed=0)
    pr = replay.template_probes(ep, 3, seed=0)
    assert replay.template_probes(ep, 3, seed=0) == pr  # seeded
    assert not rehearsal.user_prompts(tr) & rehearsal.user_prompts(pr)
    rehearsal.assert_no_probe_leak(tr, [pr])
    with pytest.raises(AssertionError, match="probe prompts"):
        rehearsal.assert_no_probe_leak(tr + [{"messages": pr[0]["messages"]}], [pr])
    facts = {f[0] for f in pr[0]["facts"]}
    assert "heron" not in facts and {"14", "priya", "natarajan"} <= facts


def test_leak_check_catches_near_duplicates():
    probe = [{"messages": [{"role": "user", "content": "What is my dog's name and breed?"}]}]
    near = [{"messages": [{"role": "user", "content": "What is my dog's name and breed??"}]}]
    other = [{"messages": [{"role": "user", "content": "Do I have any pets?"}]}]
    rehearsal.assert_no_probe_leak(other, [probe])
    with pytest.raises(AssertionError):
        rehearsal.assert_no_probe_leak(near, [probe])


def test_score_gist_alternatives_and_forbid():
    facts = [["21 March", "March 21", "21/03"], ["Priya"]]
    assert replay.score("Priya's demo moved to March 21.", facts) == 1.0
    assert replay.score("It's on the 21st.", facts) == 0.0
    assert replay.score("Due in 140 days", [["14"]]) == 0.0
    assert replay.score("Avoid cashews.", [["cashew"]]) == 1.0  # plural forms count
    assert replay.score("Cashewnut", [["cashew"]]) == 0.0
    assert replay.score("It is 30 °C (86 °F) today", [["°c", "celsius"]], ["°f"]) == 0.0
    assert replay.score("It is 30 °C today", [["°c", "celsius"]], ["°f", "fahrenheit"]) == 1.0
    assert replay.score("```rust\nfn main() {}", [["fn main", "```rust"]]) == 1.0


def test_file_generator_and_triage_skip(tmp_path):
    from sleepmem import scenario

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "replay.yaml").write_text(
        '_author: test\ndog:\n  - {q: "What is my dog called?", a: "Your dog is Pistachio."}\n'
    )
    (ds / "probes.yaml").write_text(
        'dog:\n  - {q: "Tell me about my pet.", facts: [[Pistachio]]}\n'
    )
    day = {
        "date": "2026-11-01T09:00:00-08:00",
        "episodes": {
            "dog": {
                "kind": "semantic",
                "salience": 0.8,
                "source": "chat",
                "tags": ["dog"],
                "body": "The user's dog is a greyhound named Pistachio.",
            },
            "cat": {
                "kind": "semantic",
                "salience": 0.8,
                "source": "chat",
                "tags": ["cat"],
                "body": "No replay set was written for this one.",
            },
        },
    }
    stm_dir = tmp_path / "stm"
    ids = {}
    scenario.seed_day(stm_dir, ds, day, ids)
    cfg = {"replay": {"generator": "file"}, "seed": 0}
    gen = replay.Generator(cfg, stm_dir)
    dog, cat = (stm.read(next(stm_dir.rglob(f"{ids[k]}.md"))) for k in ("dog", "cat"))
    assert gen.ready(dog) and not gen.ready(cat)
    assert gen.train(dog)[0]["messages"][1]["content"] == "Your dog is Pistachio."
    p = gen.probes(dog)[0]
    assert p["facts"] == [["Pistachio"]] and p["forbid"] == []


def test_epochs_set_iters():
    from sleepmem import train

    cfg = config.load_config(config.PROJECT_ROOT / "config/sleep/smoke.yaml")
    cfg["train"]["epochs"] = 3
    lc = train.lora_config(cfg, Path("m"), Path("d"), Path("a"), None, n_train=10)
    assert lc["iters"] == 8  # ceil(3 * 10 / 4)
    assert (
        train.lora_config(cfg, Path("m"), Path("d"), Path("a"), None, 5, n_train=10)["iters"] == 5
    )


def test_mix_needs_enough_examples():
    ex = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    with pytest.raises(ValueError):
        rehearsal.mix(ex, [], [], 0.1, batch_size=4, seed=0)
    tr, va = rehearsal.mix(ex * 2, [], [], 0.1, batch_size=4, seed=0)
    assert (len(tr), len(va)) == (6, 4)


# --- recall (RAG) ------------------------------------------------------------------------


def test_recall_search_skips_superseded(tmp_path):
    old = cap(tmp_path, "The router admin port is 8080.", tags=["router"])
    new = cap(tmp_path, "The router admin port is 8443.", tags=["router"], supersedes=old.id)
    cap(tmp_path, "Lisbon sardines were great.", tags=["lisbon"])
    hits = recall.search("router port", stm.episodes(tmp_path))
    assert [h.id for h in hits] == [new.id]


# --- gate --------------------------------------------------------------------------------


def acc(x):
    return {"accuracy": x}


TH = {"min_recall": 0.5, "max_retention_drop": 0.1, "max_capability_drop": 0.05}


def test_gate():
    assert evaluate.gate(acc(0.6), acc(0.9), acc(0.85), acc(0.8), acc(0.8), TH)["passed"]
    g = evaluate.gate(acc(0.4), acc(0.9), acc(0.7), acc(0.8), acc(0.7), TH)
    assert g["checks"] == {"recall": False, "retention": False, "capability": False}
    assert evaluate.gate(acc(0.6), acc(None), acc(None), acc(0.8), acc(0.8), TH)["passed"]


# --- full cycle without a model ----------------------------------------------------------


class FakeSubject:
    """Answers probes from memory only if its adapter 'trained' on them."""

    def __init__(self, adapter, stm_dir, learns=True):
        self.adapter, self.stm_dir, self.learns = adapter, stm_dir, learns

    def answer(self, batch):
        eps = stm.episodes(self.stm_dir) if self.adapter and self.learns else []
        out = []
        for m in batch:
            q = m[-1]["content"].lower()
            hit = next((e.body for e in eps if e.meta.tags[0] in q), "")
            out.append(hit or ("Paris" if "france" in q else "I don't know."))
        return out


def fake_trainer(lc, sd):
    out = Path(lc["adapter_path"])
    out.mkdir(parents=True)
    (out / "adapters.safetensors").write_bytes(b"fake")
    return out


def make_cfg(tmp_path, **gate):
    cfg = config.load_config(config.PROJECT_ROOT / "config/sleep/smoke.yaml")
    cfg["root"] = str(tmp_path)
    cfg["gate"].update(gate)
    return cfg


def seed(stm_dir):
    cap(stm_dir, "The home router is a Fritzbox in the hallway closet.", tags=["router"])
    cap(stm_dir, "In Lisbon we ate sardines at Taberna Azul.", tags=["lisbon"], kind="episodic")


def run(cfg, tmp_path, sid, learns=True):
    stm_dir = config.path(cfg, "stm")
    return sleep.run(
        cfg,
        sleep_id=sid,
        now=T0 + timedelta(hours=14),
        subject_factory=lambda a: FakeSubject(a, stm_dir, learns),
        trainer=fake_trainer,
    )


def test_sleep_promotes_and_consolidates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "resolve_model", lambda cfg, allow=False: tmp_path / "model")
    cfg = make_cfg(tmp_path, min_recall=0.5)
    stm_dir = config.path(cfg, "stm")
    seed(stm_dir)
    m = run(cfg, tmp_path, "s1")
    assert m["status"] == "promoted", m.get("gate")
    assert (tmp_path / "adapters/current").readlink() == Path("s1")
    eps = stm.episodes(stm_dir)
    assert all(e.meta.status == "consolidated" and e.meta.sleeps == ["s1"] for e in eps)
    sd = tmp_path / "data/sleeps/s1"
    for f in (
        "manifest.json",
        "triage.json",
        "probes.jsonl",
        "train.jsonl",
        "valid.jsonl",
        "pretest.json",
        "posttest.json",
    ):
        assert (sd / f).exists(), f
    train_rows = sleep.read_jsonl(sd / "train.jsonl") + sleep.read_jsonl(sd / "valid.jsonl")
    rehearsal.assert_no_probe_leak(train_rows, [sleep.read_jsonl(sd / "probes.jsonl")])
    # second sleep: nothing fresh
    assert run(cfg, tmp_path, "s2")["status"] == "nothing_to_do"


def test_sleep_rejects_and_leaves_episodes_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "resolve_model", lambda cfg, allow=False: tmp_path / "model")
    cfg = make_cfg(tmp_path, min_recall=0.9)
    stm_dir = config.path(cfg, "stm")
    seed(stm_dir)
    m = run(cfg, tmp_path, "s1", learns=False)
    assert m["status"] == "rejected" and m["gate"]["reasons"]
    assert not (tmp_path / "adapters/current").exists()
    assert (tmp_path / "adapters/s1/adapters.safetensors").exists()  # kept, not deleted
    assert all(e.meta.status == "fresh" for e in stm.episodes(stm_dir))


def test_decay_archives_old_low_salience(tmp_path):
    ep = cap(tmp_path, "Minor thing.", salience=0.2)
    keep = cap(tmp_path, "Important thing.", salience=0.9)
    for e in (ep, keep):
        stm.update(e, tmp_path, status="consolidated", sleeps=["s1"])
    assert sleep.decay(tmp_path, T0 + timedelta(days=5), 30, 0.3) == []
    assert sleep.decay(tmp_path, T0 + timedelta(days=40), 30, 0.3) == [ep.id]
    assert stm.episodes(tmp_path, "archived")[0].body == "Minor thing.\n"


def test_configs_load_and_pin_models():
    base = yaml.safe_load((config.PROJECT_ROOT / "config/base.yaml").read_text())
    for spec in base["models"].values():
        assert len(spec["revision"]) == 40
    cfg = config.load_config(config.PROJECT_ROOT / "config/sleep/smoke.yaml")
    assert cfg["rung"] == 0 and cfg["train"]["lora"]["mask_prompt"] is True
    assert config.model_spec(cfg)["repo"].endswith("gemma-3-270m-it-4bit")


def test_dataset_synthetic_v1_complete_and_leak_free():
    ds = config.PROJECT_ROOT / "datasets/synthetic-v1"
    eps = yaml.safe_load((ds / "episodes.yaml").read_text())
    rep = yaml.safe_load((ds / "replay.yaml").read_text())
    prb = yaml.safe_load((ds / "probes.yaml").read_text())
    keys = {k for day in eps.values() for k in day["episodes"]}
    assert keys == set(rep) - {"_author"} == set(prb) - {"_author"}
    for day in eps.values():
        for e in day["episodes"].values():
            stm.Meta(
                id="2026-01-01-0001",
                created=day["date"],
                **{k: v for k, v in e.items() if k not in ("body", "supersedes")},
            )
            assert e.get("supersedes") in (None, *keys)
    train = [{"messages": [{"role": "user", "content": x["q"]}]} for k in keys for x in rep[k]]
    probes = [{"messages": [{"role": "user", "content": p["q"]}]} for k in keys for p in prb[k]]
    rehearsal.assert_no_probe_leak(train, [probes])
    for k in keys:
        assert len(rep[k]) >= 5 and len(prb[k]) >= 2, k
        for x in rep[k]:
            assert x["q"].strip() and x["a"].strip(), k
        for p in prb[k]:
            assert p["facts"] and all(isinstance(f, list) and f for f in p["facts"]), k
            assert all(isinstance(s, str) for f in p["facts"] for s in f), (k, p["facts"])
