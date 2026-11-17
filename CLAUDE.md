# CLAUDE.md

## Project: sleep-consolidated memory for LLMs

A continual-learning experiment that copies the brain's two-stage memory.
New experiences go fast into **short-term memory** (markdown files). During
**"sleep"**, a batch job replays them and **consolidates** them into the
model's weights with LoRA/QLoRA, so they change what the model knows and
how it behaves, without needing retrieval.

| Brain                                   | This system                                        |
|-----------------------------------------|----------------------------------------------------|
| Hippocampus (fast, episodic, sparse)    | Markdown short-term store `stm/`                   |
| Sleep / hippocampal replay              | The `sleep` pipeline (replay → train)              |
| Neocortex (slow, distributed)           | LoRA adapter weights                               |
| Complementary learning systems          | Rehearsal of old memories mixed into each sleep    |
| Synaptic homeostasis / forgetting       | Decay and archiving of consolidated episodes       |

## Status

Greenfield. The folder was empty on 2026-11-08. Update this section as milestones land.

- 2026-11-08: base models chosen: Olmo 3 7B Instruct (main), Gemma 4 E4B (replication).
- 2026-11-08: pipeline scaffolded; all 7 stages run end to end on rung 0
  (`mlx-community/gemma-3-270m-it-4bit`) via the smoke scenario: two sleeps, promotion,
  cumulative adapter, retention probes, supersedes, dedup. ~25 s per sleep. 16 unit tests.
  Placeholders, not decisions: `template` replay generator (asks by tags, answers with the
  verbatim body; no behaviour examples for procedural yet), keyword-overlap probe scoring,
  a 20-item `eval/general/` set, no Dolci subset yet (`general_source: null` in base).
  Rung-0 observation (plumbing only, not a result): 14 examples at lr 2e-4 make the 270M
  recite memories for every question (general 45% -> 5%), and 100 iters collapse it.
  Not yet built: `subject` replay generator, RAG baseline runner, Gemma 4
  thinking-off at train time (mlx_lm's ChatDataset passes no template kwargs).
- 2026-11-08: **research decisions (user):** replay examples are written by Claude Code
  (`replay.generator: file`), and recall is judged on gist, not wording: memories are
  reconstructed from meaning, as in people. So replay answers paraphrase, probes score key
  facts with accepted alternatives (plus `forbid` forms for procedural rules). Consequence:
  replay writing sends episode content to Anthropic; fine for synthetic data, and for real
  `stm/` only because the user chose it.
- 2026-11-08: rung 1 started (experiment 001). Dataset `datasets/synthetic-v1` (20
  fictional memories, 2 days, one correction), seeded Dolci-Instruct-SFT-No-Tools cache,
  LoRA r16 / scale 2 / lr 1e-4 / 3 epochs. Compute: 7B QLoRA on the M1 Pro trains at
  ~20-40 tok/s; batch 4 x 512 tokens ran out of Metal memory with LoRA dropout, so batch 1.
  Result (one seed): sleep 1 recall 4% -> 79%, sleep 2 recall 92% but day-1 retention
  81% -> 65%, rejected by the gate; general set at ceiling (100%). Failures are binding
  errors between memories, question-shape capture, lost negation, confabulation, and a
  correction fused with the old fact. See experiments/001-r1-first-sleeps.md.
- 2026-11-16: contrastive replay add-on written, not yet evaluated:
  `datasets/synthetic-v1/replay-contrast.yaml` (104 examples, typed `t`: lookup, reject,
  confirm, shape, negation, unknown, correction, behaviour, boundary) targets the failure
  classes of experiment 001. `replay.yaml` is unchanged; `replay.files` in a sleep config
  lists the files to combine (`config/sleep/r1-contrast.yaml` = r1-first + the add-on). The
  type is kept in `examples.jsonl`. Checked by tests, the smoke run and a rung-1 dry run.
  The add-on raises examples per memory from 6 to ~11, so experiment 002 (user decision
  2026-11-16) runs it against a step-matched control, `config/sleep/r1-control-steps.yaml`
  (original set, 5.7 epochs). Runs in progress.

## Two roles: don't confuse them

- **Subject**: the local open-weights model whose memory we're building.
- **Claude Code (you)**: the lab assistant. You write the pipeline, run
  experiments and analyze the results. Your own Claude Code memory is not
  part of the experiment.

## Hardware & stack

- MacBook Pro 2021, Apple Silicon (M1 Pro/Max), unified memory: **32 GB**.
  There's no CUDA, so don't suggest bitsandbytes, CUDA Unsloth, flash-attn or DeepSpeed.
- **Training:** `mlx-lm` (`pip install "mlx-lm[train]"`), via `mlx_lm.lora`.
  LoRA is the default. Training against a 4-bit quantized base model (e.g. from
  `mlx-community` on Hugging Face) is QLoRA. `mlx_lm.fuse` merges an adapter into the base.
  Base models must be safetensors; GGUF can't be trained.
- **Serving the subject:** `mlx_lm.server` (OpenAI-compatible) with `--adapter-path`.
  Ollama is installed and is optional for fused models. Check current Ollama
  adapter support before relying on `ADAPTER` in a Modelfile.
- **Python:** 3.12, managed with `uv`. Lint/format with `ruff`, test with `pytest`.
- **Base models:** Olmo 3 7B as the main subject, Gemma 4 E4B for replication.
  See "Base models" below.

## Base models

**Decision (2026-11-08):** the main subject is **Olmo 3 7B Instruct** (Ai2).
It's fully open: pretraining data (Dolma 3), post-training data (Dolci),
intermediate checkpoints and training code are all public. That lets us
prove a memory is new, rehearse on the data the model was actually tuned
on, and compare training stages. Gemma 4 E4B is the **replication** model:
a result counts only once it holds on both families.

| Rung | Model | Use it for |
|------|-------|------------|
| 0 | Gemma 3 270M-it or Qwen3.5 0.8B | Smoke tests only: a full sleep cycle in seconds, to check pipeline code. Never report results from this rung. |
| 1 | **Olmo 3 7B Instruct, 4-bit (QLoRA)** | Main development and main experiments |
| 2 | **Gemma 4 E4B-it, 4-bit (QLoRA)** | Replication of every headline result |
| 3 | Olmo 3 7B Instruct, bf16 (plain LoRA) | Later: does quantizing the base change consolidation? Tight in 32 GB, so measure memory first |
| — | Olmo 3 32B, Gemma 4 26B/31B | Out of scope for training on this machine |

Model ids live only in `config/base.yaml`, never in source. Look up the
exact Hugging Face repo before downloading (e.g. `mlx-community/Olmo-3-7B-Instruct-4bit`)
and record it in the config.

### Olmo 3 notes
- Plain dense, text-only transformer, so standard LoRA targets (attention +
  MLP projections) apply. Context is 65K, far more than replay examples need.
- Use the **Instruct** checkpoint, not **Think**: no reasoning traces to
  untangle from recall. The Think models are a possible later experiment.
- **Stage checkpoints:** MLX conversions exist for `Instruct-SFT` as well as
  the final `Instruct`. "Does consolidation behave differently before and
  after preference tuning?" is a cheap, worthwhile experiment.
- **Rehearsal from Dolci:** sample general-rehearsal data from Ai2's Dolci
  post-training sets (look up the exact dataset ids on Hugging Face). Cache
  a fixed, seeded subset under `data/rehearsal/dolci/` (gitignored), and
  record the dataset id and revision in each manifest.
- **Dolma 3 is about 6T tokens:** never download it. For novelty checks,
  use a hosted search tool such as OlmoTrace if one is available, or skip the
  check. Ask before building anything heavier.

### Gemma 4 notes (replication model)
- **"E" sizes are effective parameters:** E4B is larger in memory than "4B"
  suggests. Measure peak memory on a dry run.
- **Thinking mode:** keep it OFF for training data and probes, and record
  the setting in the manifest.
- **Multimodal:** text-only here; train only the language-model layers.
- **Large vocabulary (~262k tokens):** keep `max_seq_length` modest.

### Rules for every model
- Build all training and eval prompts with `tokenizer.apply_chat_template`,
  never by hand, and train with `--mask-prompt` (loss only on responses).
  Each family has its own template.
- Each experiment fixes one base model. Baselines (no memory, RAG over
  `stm/`) use that same model, and replication reruns the same config with
  only the model changed.
- Avoid hybrid/linear-attention models (Qwen3.5 Gated DeltaNet, Olmo Hybrid)
  as subjects for now. Their recurrent state is a fast-memory mechanism of
  its own, which confounds results, and LoRA support on MLX is less mature.
  They're a candidate later experiment.

## Repository layout (target)

```
sleepmem/
├── CLAUDE.md
├── README.md                  # public overview: idea, pipeline, setup, status
├── pyproject.toml
├── config/
│   ├── base.yaml              # model id + rung, paths, seeds, thresholds
│   └── sleep/*.yaml           # LoRA hyperparams + data mix per experiment
├── stm/                       # short-term memory (hippocampus)
│   ├── inbox/YYYY-MM-DD/      # fresh episodes, one .md per memory
│   ├── consolidated/          # already slept on; kept as rehearsal source
│   ├── archive/               # decayed / pruned
│   ├── replay/<id>.yaml       # written replay set per episode (training only)
│   └── probes/<id>.yaml       # held-out probes per episode (never trained on)
├── src/sleepmem/
│   ├── config.py              # base.yaml + sleep config merge, config hash, pinned model resolve
│   ├── stm.py                 # Episode/Meta pydantic models, read/dump, status moves
│   ├── scenario.py            # run a dataset day by day in a sandbox (data/runs/<name>-<ts>/)
│   ├── capture.py             # write episodes from subject interactions
│   ├── recall.py              # wake-time retrieval over stm/ (pre-consolidation)
│   ├── triage.py              # salience scoring, dedup
│   ├── replay.py              # episodes → training examples + held-out probes
│   ├── rehearsal.py           # sample old memories + general instruction data
│   ├── train.py               # wraps mlx_lm.lora
│   ├── evaluate.py            # recall / retention / capability probes
│   └── sleep.py               # orchestrates one full cycle
├── data/sleeps/<sleep_id>/    # generated: train/valid.jsonl, probes, manifest.json
├── data/rehearsal/            # cached, seeded general-rehearsal subsets (e.g. dolci/)
├── adapters/<sleep_id>/       # generated adapter weights; adapters/current → promoted
├── datasets/<name>/           # synthetic episodes + replay.yaml + probes.yaml (committed)
│                              # + optional add-on replay files, chosen by `replay.files`
├── eval/general/              # fixed general-capability benchmark
├── experiments/               # one markdown log per experiment
└── tests/                     # fixtures/smoke/: smoke dataset + general-rehearsal set
```

`data/`, `adapters/` and any model weights are gitignored. Code, configs and experiment logs are committed.
Episode files and their replay/probe sets (`stm/**/*.md`, `stm/replay/`, `stm/probes/`) are
gitignored too: they may be personal and the repo has a GitHub remote. Synthetic datasets
under `datasets/` are fictional and committed.

Sleep dir contents: `triage.json`, `probes.jsonl`, `retention_probes.jsonl`, `pretest.json`,
`general_base.json`, `examples.jsonl` (with provenance), `train.jsonl`/`valid.jsonl`
(what mlx_lm sees), `lora.yaml`, `train.log`, `posttest.json`, `manifest.json`.

## Short-term memory format

One memory per file, markdown with YAML frontmatter:

```markdown
---
id: 2026-11-08-0001
created: 2026-11-08T14:02:00-08:00
source: chat            # chat | note | observation
kind: episodic          # episodic | semantic | procedural
salience: 0.6           # 0–1, set at capture, refined in triage
tags: [preference, project-x]
status: fresh           # fresh | consolidated | archived
sleeps: []              # sleep_ids that trained on this episode
supersedes: null        # id of an episode this one corrects
---
What happened, in plain prose. One idea per file.
```

- An episode's body never changes after capture. Only `salience`, `status`
  and `sleeps` may change.
- A correction is a new episode with `supersedes:` set; never edit the old one.
- Validate frontmatter with a pydantic model, and reject malformed files at capture.

## The sleep cycle

`uv run python -m sleepmem.sleep --config config/sleep/<name>.yaml`

Each stage writes into `data/sleeps/<sleep_id>/`, so you can inspect, rerun
or resume a cycle stage by stage.

1. **Triage**: pick fresh episodes above the salience threshold and merge duplicates.
   **Pre-test**: run each episode's probes on the current model, *without*
   retrieval, before training. Anything it already answers gets tagged
   `already_known` and left out of recall scores, because it's no test of consolidation.
2. **Replay**: turn each episode into several training examples: recall Q&A,
   paraphrases, use in a new context and, for `procedural` memories, examples
   of the behavior. Generate the **held-out probes** in this same stage.
   Never train on probes.
3. **Interleave**: mix new examples with (a) rehearsal samples from
   `stm/consolidated/` and (b) general rehearsal data drawn from the subject's
   own post-training distribution (Dolci for Olmo; a generic instruction set
   for Gemma), to limit catastrophic forgetting. The mix ratio is an
   experiment variable set in config.
4. **Train**: run `mlx_lm.lora` on the mix to produce `adapters/<sleep_id>/`.
5. **Gate**: evaluate (a) recall of tonight's memories, (b) retention of
   memories from earlier sleeps and (c) general capability compared with the
   base model. Promote only if every threshold in config passes.
6. **Commit or roll back**: if the gate passes, point `adapters/current` at the
   new adapter, mark the episodes `consolidated` and add the `sleep_id`. If it
   fails, leave the episodes `fresh` and log why in the manifest.
7. **Decay**: archive consolidated episodes that are old and low-salience.
   Forgetting is a feature.

**Awake, before consolidation:** the subject reaches new memories through
retrieval over `stm/` (in context). The core question is whether, after
sleep, it can recall them *without* retrieval.

## Evaluation (log every sleep)

- **Recall**: probe accuracy on tonight's memories, with no retrieval.
- **Retention**: probes from every earlier sleep, giving a forgetting curve over time.
- **Capability drift**: the fixed `eval/general/` set, compared with the base model.
- **Behavior adherence**: for procedural/preference memories, judged on held-out prompts.
- Report results broken down by `kind`. LoRA may encode behavior and style
  more easily than facts, and we want to measure that difference.

Baselines: (1) the base model with no memory, (2) the base model + RAG over `stm/`.

## Experiments

Each experiment gets `experiments/NNN-short-name.md` containing the hypothesis,
config path, sleep_ids, a results table and a conclusion. Never overwrite an
old log; start a new one.

### Open research questions (raise these with me; don't decide them silently)

- One cumulative adapter retrained each sleep, or one adapter per sleep, then stacked or merged?
- LoRA rank, target modules and learning rate per sleep.
- ~~Who writes the replay examples?~~ Decided 2026-11-08: Claude Code writes them. The
  subject writing its own ("dreaming") stays a candidate later experiment.
- Rehearsal ratio and how rehearsal samples are chosen (salience, recency, spaced repetition).
- When to fuse an adapter into the base, and what "long-term" means after many sleeps.
- Does rehearsing on the model's own post-training data (Dolci) beat generic
  instruction data at preventing forgetting?
- Does consolidation differ between Olmo `Instruct-SFT` and final `Instruct`?
- Do results that hold on Olmo also hold on Gemma 4?

## Commands

```bash
uv sync                                        # install deps
uv run pytest                                  # tests (no training required)
uv run ruff check . && uv run ruff format .    # lint / format
uv run python -m sleepmem.sleep --config config/sleep/<name>.yaml
#   --dry-run (<=5 episodes, 20 iters, no promotion) --iters N --resume <sleep_id>
#   --root <dir> (sandbox) --allow-download (models > 2 GB; ask first)
uv run python -m sleepmem.scenario            # rung-0 smoke test (2 sleeps, sandbox)
uv run python -m sleepmem.scenario --config config/sleep/r1-first.yaml \
    --dataset datasets/synthetic-v1 [--days day1] [--dry-run]
#   config/sleep/r1-contrast.yaml: same run with replay.yaml + replay-contrast.yaml
uv run python -m sleepmem.rehearsal --n 2000 --seed 0 --max-chars 800   # cache Dolci subset
uv run python -m sleepmem.capture --kind semantic --salience 0.7 --tags a,b "text"
mlx_lm.server --model <base> --adapter-path adapters/current
```

Update this list as commands are implemented.

## Working rules for Claude

- **Ask first** before downloading a model or dataset larger than 2 GB, or
  starting a training run expected to take more than 30 minutes.
- Check pipeline changes with a **dry run** first (a handful of episodes,
  `--iters 20`) before a full sleep.
- **Never delete or rewrite** episodes in `stm/` or adapters in `adapters/`;
  archive them instead. Promotion and rollback happen only through `sleep.py`.
- Probes never end up in training data. Keep probe generation separate, seeded and checked by a test.
- Make everything reproducible: set seeds, and record the base model id,
  config hash, `mlx`/`mlx-lm` versions, rehearsal dataset ids/revisions and
  data counts in each sleep's `manifest.json`.
- Memory content may be personal: keep everything local. Don't send `stm/`
  contents to external APIs unless a config flag explicitly allows it.
- Keep each stage a small, testable function that can be unit-tested without training a model.
- Keep this file current: when the layout, commands or a research decision changes, update the matching section.
