# sleepmem: sleep-style memory consolidation for a local LLM

New facts are first stored as Markdown notes, then a "sleep" step fine-tunes them into the
model's weights with LoRA, so the model can recall them without retrieval.

## Status

Early stage. The pipeline runs end to end and experiments are in progress, but **there are no
results yet.** One pilot run is logged in [experiments/001](experiments/001-r1-first-sleeps.md)
(one seed, 20 synthetic memories). It was used to debug the pipeline and is not a finding.

## Idea

An assistant's short-term memory is a folder of Markdown files, one memory per file. While
"awake", the model can only reach these memories by putting them in its context. Once enough
memories have accumulated, a sleep phase consolidates them into the weights with LoRA
fine-tuning, loosely modelled on how the brain is thought to move memories from the
hippocampus to the neocortex during sleep. The question is what this does to the model: what it
recalls afterwards, what it forgets, and how its other behaviour changes.

## How it works

```
stm/inbox/ ─► triage + pre-test ─► replay ─► interleave ─► LoRA train ─► gate ─┬─► promote
                                                                               └─► reject
```

- **Memory store** ([stm.py](src/sleepmem/stm.py), [capture.py](src/sleepmem/capture.py)):
  one `.md` file per memory under `stm/inbox/<date>/`, with YAML frontmatter (kind: episodic,
  semantic or procedural; a salience score from 0 to 1; tags; an optional `supersedes` field for
  corrections). Memories are added with a CLI command or loaded from a synthetic dataset. No
  agent writes them automatically yet.
- **Trigger:** a sleep runs when you start it, either with `sleepmem.sleep` or with the
  scenario runner, which runs one sleep per simulated day. No automatic trigger exists yet
  (for example, when the inbox reaches a given size). Triage then selects fresh memories with a
  salience of at least 0.3, merges near-duplicates and drops memories that have been corrected.
- **Training data** ([replay.py](src/sleepmem/replay.py)): each memory gets question/answer
  pairs that paraphrase it, plus separate held-out probe questions that are never trained on (a
  check rejects any training prompt that matches a probe). For the experiments, Claude Code
  wrote these Q&A sets and probes ahead of time (`datasets/synthetic-v1/`). A simple template
  generator is used only for smoke tests.
- **Interleaving** ([rehearsal.py](src/sleepmem/rehearsal.py)): the new examples are mixed
  with replay of already-consolidated memories and with general instruction data, a seeded
  subset of Ai2's `allenai/Dolci-Instruct-SFT-No-Tools`, part of Olmo's own post-training
  data. Both rehearsal ratios are set in the config.
- **Training** ([train.py](src/sleepmem/train.py)): `mlx_lm.lora` (Apple MLX) trains LoRA
  adapters on all attention and MLP projections of a 4-bit base model (QLoRA), with the loss
  computed on responses only. By default each sleep continues training the previous adapter.
- **Gate** ([evaluate.py](src/sleepmem/evaluate.py), [sleep.py](src/sleepmem/sleep.py)): the
  new adapter is scored on recall of the new memories, retention of earlier ones and a small
  general-capability set. Recall is judged on key facts with accepted alternatives, not on
  exact wording. If every threshold passes, `adapters/current` points to the new adapter and
  the memories move to `stm/consolidated/`; otherwise nothing changes and the manifest records
  the reason. Old, low-salience memories are archived.
- **Using the updated model:** load the base model with `adapters/current`, for example with
  `mlx_lm.server`. Each sleep writes its data, logs and a manifest (model revision, config
  hash, library versions) to `data/sleeps/<id>/`.

**Models** (ids and pinned revisions in [config/base.yaml](config/base.yaml)):
`mlx-community/Olmo-3-7B-Instruct-4bit` is the main subject, and Olmo was chosen because its
training data is public. `mlx-community/gemma-3-270m-it-4bit` is used only for smoke tests.
Gemma 4 E4B is configured as a replication model but has not been run yet.

## Questions I want to test (planned, not done)

1. **Recall:** are consolidated facts recalled without retrieval, and for how many sleeps?
2. **Forgetting:** how much prior knowledge and general capability is lost?
3. **Safety-trained behaviour:** does consolidation change refusals and honesty?
4. **Poisoned or incorrect memories:** do they persist in the weights, and does a later
   correction remove them?

The current code measures 1 and, crudely, 2. Questions 3 and 4 have no evaluation sets yet.

## Setup and usage

Requirements: macOS on Apple Silicon (MLX has no CUDA path), Python 3.12 and
[uv](https://docs.astral.sh/uv/). Development is done on a MacBook Pro M1 Pro with 32 GB.
The smoke test needs a ~0.2 GB model download, the Olmo runs ~4 GB.

```bash
uv sync                               # install dependencies
uv run pytest                         # unit tests, no model needed
uv run python -m sleepmem.scenario    # smoke test: two sleeps on Gemma 3 270M, ~1 min
```

The smoke test runs in a sandbox under `data/runs/` and does not touch `stm/`.

A consolidation run on Olmo with the synthetic dataset:

```bash
uv run python -m sleepmem.rehearsal --n 2000 --seed 0 --max-chars 800  # cache the Dolci subset
uv run python -m sleepmem.scenario --config config/sleep/r1-first.yaml \
    --dataset datasets/synthetic-v1 --allow-download                   # Olmo is > 2 GB
```

On your own memories:

```bash
uv run python -m sleepmem.capture --kind semantic --salience 0.7 --tags a,b "text"
uv run python -m sleepmem.sleep --config config/sleep/<name>.yaml --dry-run
```

`--dry-run` limits a sleep to 5 memories and 20 training steps, and promotes nothing.
The `file` replay generator expects a Q&A set and probes for each memory under `stm/replay/`
and `stm/probes/`. Nothing in the repo writes these yet for new memories.

## Repo layout

```
config/base.yaml        model ids and revisions, thresholds, LoRA defaults
config/sleep/           per-experiment overrides (smoke.yaml, r1-first.yaml)
src/sleepmem/           one module per pipeline stage, plus the scenario runner
stm/                    short-term memory: inbox/, consolidated/, archive/ (contents gitignored)
datasets/synthetic-v1/  20 fictional memories, their Q&A sets and probes
eval/general/           20-item general-capability check (placeholder)
experiments/            one log per experiment
tests/                  unit tests and smoke-test fixtures
```

`data/` and `adapters/` (generated data and weights) are gitignored.

## Limitations and next steps

- No results: one pilot run, one seed, one small synthetic dataset.
- The general-capability set has 20 items and cannot detect small regressions.
- Recall is scored by matching key facts, not by a model or human judge.
- The Q&A sets and the probes for each memory were written by the same author (Claude Code).
- No automatic sleep trigger, no agent that writes memories, and no retrieval (RAG) baseline
  yet. BM25 retrieval exists in [recall.py](src/sleepmem/recall.py) but is not wired into an
  evaluation.
- Mac only (MLX).

Next, in this order:

1. Q&A sets that separate similar facts and handle negation and corrections (the failures
   seen in the pilot run).
2. A probe set checking that memories do not leak into unrelated answers.
3. More seeds.
4. Rehearsal-ratio sweeps.
5. Evaluation sets for questions 3 and 4.
6. Replication on Gemma 4.

## About

I design and analyze the experiments. Claude Code (Anthropic) wrote most of the
implementation, as well as the synthetic dataset, its Q&A sets and its probes.

Alexandre Santos ([github.com/alexandreos](https://github.com/alexandreos))

## License

[MIT](LICENSE)
