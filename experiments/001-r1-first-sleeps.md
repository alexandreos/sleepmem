# 001 — First rung-1 sleeps

- **Date:** 2026-11-08
- **Model:** rung 1, `mlx-community/Olmo-3-7B-Instruct-4bit` @ `d732c91a` (QLoRA)
- **Config:** `config/sleep/r1-first.yaml` (on `config/base.yaml`)
- **Dataset:** `datasets/synthetic-v1`: 20 fictional memories over 2 days (8 semantic, 6
  episodic, 6 procedural; day 2 corrects the day-1 car memory). Replay sets and probes written
  by Claude Code.
- **Rehearsal:** `allenai/Dolci-Instruct-SFT-No-Tools` @ `9156a5a5`, seeded subset of 2000,
  conversations <= 800 chars (`data/rehearsal/dolci/9156a5a5-seed0-c800-n2000.jsonl`).
- **Command:** `uv run python -m sleepmem.scenario --config config/sleep/r1-first.yaml --dataset datasets/synthetic-v1`
- **Sleep ids:** `01-day1`, `02-day2` in sandbox `data/runs/r1-first-20261108-170242/`

## Question

Can one sleep (LoRA on paraphrased replay + Dolci rehearsal) put new memories into Olmo's
weights well enough to recall their gist without retrieval, without hurting general
capability? And does a second sleep keep the first night's memories (retention), with the
correction replacing the old car fact?

Expectations, stated before the run: semantic and procedural memories consolidate better
than episodic ones (LoRA encodes behaviour and simple facts more easily than event details);
general capability on the 20-item set stays within one item; retention drops somewhat after
sleep 2 even with 0.5 consolidated rehearsal.

## LoRA settings and why

| Setting | Value | Reason |
|---|---|---|
| rank | 16 | Should be enough capacity for ~20 short memories. Biderman et al. 2024 find LoRA "learns less and forgets less" than full fine-tuning, and higher ranks narrow the learning gap; at our data size, rank is probably not the bottleneck. Sweep 8/16/64 later. |
| targets | q,k,v,o + gate,up,down in all 32 layers | MLP layers are where factual associations are thought to be stored (Meng et al. 2022, ROME), so they're included, not just attention. Embeddings excluded (mlx-lm's default would include them). |
| scale | 2.0 | mlx-lm adds `scale * (x A B)` directly, without PEFT's alpha/r. 2.0 = PEFT alpha 32 at r 16, the common 2r rule. When sweeping rank, keep scale * r constant (scale = 32/r) or rank and step size get confounded. |
| learning rate | 1e-4 | The product scale x lr is what sets the step size: 2e-4 here, in line with common QLoRA practice (QLoRA paper 5e-5, Unsloth defaults 2e-4). Rung 0 used scale 20 x lr 2e-4 = 4e-3, 20x hotter, and collapsed. |
| epochs | 3 | Each memory is seen as ~6 paraphrases x 3 = ~18 times. Knowledge-injection work (Allen-Zhu & Li, "Physics of LMs 3.1") finds that paraphrase diversity, more than repetition, makes facts extractable rather than memorised: the reason replay answers paraphrase. |
| dropout | 0.0 | Saves a copy of every adapted layer's input; with it, batch 4 x 512 ran out of Metal memory. |
| batch | 1 | Fastest measured on the M1 Pro (~37 tok/s vs ~20 at batch 2-4); no padding, no memory pressure. Noisier gradients; Adam copes at this lr. |
| rehearsal | 0.5 Dolci examples per new example, 0.5 consolidated | Dolci examples average ~200 tokens vs ~65 for memory examples, so this is ~1.5:1 by tokens. |

## Results

### Numbers (rung 1, one seed, one dataset: treat as a first look, not a finding)

Probe accuracy without retrieval (a probe passes if >= half its key facts appear in any
accepted form and no forbidden form does). Base model before any sleep: 1/28 day-1 probes.

| | Sleep 1 (day 1, 14 memories) | Sleep 2 (day 2, 6 memories) |
|---|---|---|
| Recall before sleep | 1/28 (4%) | 1/12 (8%) |
| **Recall after sleep** | **22/28 (79%)** | **11/12 (92%)** |
| by kind: semantic / episodic / procedural | 75% / 88% / 75% | 75% / 100% / 100% |
| Retention of day-1 memories (26 probes; superseded car excluded) | n/a | 81% -> **65%** |
| General capability (20 items) | 100% -> 100% | 100% -> 100% |
| Gate | promoted | **rejected**: retention drop 0.154 > 0.1 |
| Train examples (new / consolidated / Dolci) | 84 / 0 / 42 -> 113 train | 36 / 18 / 18 -> 64 train |
| Wall time (train) | 11 min (7.8) | 7 min (4.2) |

No probe was already known before sleep; the gate worked as designed (sleep 2 stays in
`adapters/02-day2`, `current` still points at `01-day1`, day-2 episodes stay fresh).

### What the failures look like

Almost every failure is a real model error of a recognisable kind, not a scoring artifact:

1. **Binding errors between memories.** Names and attributes migrate between memories that
   were learned together: "Your dog is called Captain" (the greeting rule's word), "your
   Volvo is called 'Captain'", "Your sourdough starter was called Pistachio" (asked about the
   side project), "Your cargo e-bike, called Pistachio".
2. **Answer-template capture.** After sleep 2, "What happened to my laptop?" got the car_fix
   answer verbatim ("You sold it. The green Volvo is gone…"): the training question "What
   happened to the Pickle?" taught a question *shape* -> answer mapping, not a subject key.
   "What do I do for a living?" -> "You teach others Icelandic."
3. **Negation lost.** "Which nuts do I need to avoid?" -> "Walnuts, because you're allergic to
   them." The memory says walnuts are *fine*; the word survived, the polarity did not. In a
   real assistant this is the most dangerous failure.
4. **Confabulated reconstruction.** "He has three legs, because he lost one in a car
   accident" (invented cause), "Only a few gigabytes" (instead of two weeks of work), "On
   1989-10-03" for the Berlin Wall: right ISO format, wrong date, and 10-03 is the marathon
   date. Very human-like: gist plus plausible filling.
5. **Correction only half applied.** The e-bike was learned, but "Do I still have the
   Volvo?" -> "Yes, it's the green Volvo V50 from 2011, your cargo e-bike": old and new facts
   fused rather than the old one being replaced. The cumulative adapter still carries the
   car memory even though it is no longer rehearsed.
6. **Procedural rules leak or slip.** The Rust rule held for "read a file" but the palindrome
   probe got Python; the metric rule got "50 °C … 134 °F (56.7 °C)" after sleep 2.

Retention: 6 day-1 probes went from pass to fail in sleep 2 and 2 went from fail to pass
(dog details, Berlin Wall date), net 21 -> 17 of 26. Day-1 memories were rehearsed with only
18 replayed examples (0.5 of the new count) against 36 new ones.

### Caveats

- One seed and one 20-memory dataset; per-probe noise is large (one probe = 3.6 points on day
  1, 8.3 on day 2). The 0.154 retention drop is 4 probes.
- The 20-item general set is at ceiling (100%) for Olmo and can't detect mild drift. It also
  doesn't test *leakage*: memories surfacing in unrelated answers, which the failures above
  suggest is happening.
- The same author wrote replay and probes; the leak check guarantees different questions,
  not independent judgement of what counts as the gist.


## Conclusion

Consolidation works at rung 1: one ~10-minute sleep takes 14 new memories from 4% to
79% gist recall without retrieval, procedural rules included, with no measurable cost on the
(too easy) general set. The second sleep learns its 6 memories well (92%) but costs day-1
retention (81% -> 65%), and the gate rejects it, as it should.

The failures look like **interference and binding**, not missing capacity (an untested
hypothesis: no rank sweep yet): attributes swap between memories learned together, question
shapes map to answers regardless of subject, negations flip, and a correction fuses with the
fact it should replace. That points at the replay data and rehearsal before the LoRA
hyperparameters.

Next, in order of expected value:
1. **Retention vs rehearsal ratio** (consolidated 0.5 -> 1.0 -> 2.0) on the same two days:
   does rehearsing day 1 harder pass the gate? Cheap (~20 min per setting).
2. **Contrastive replay**: add examples that bind each attribute to its subject ("Pistachio
   is your dog, not your starter"), negation pairs ("walnuts: fine; cashews: allergic"), and
   for corrections, explicit "no longer" examples. Compare binding errors against this run.
3. **A leakage probe set**: unrelated questions that should *not* mention any memory.
4. **More seeds** (training seed and probe/replay order) before trusting any difference
   smaller than ~15 points.
5. LoRA sweep only after that: lr {5e-5, 1e-4, 2e-4} x epochs {2, 3, 5}; rank {8, 16, 64}
   with scale = 32/r.

