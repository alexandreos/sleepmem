# 002 — Contrastive replay vs a step-matched control

- **Date:** 2026-11-16
- **Model:** rung 1, `mlx-community/Olmo-3-7B-Instruct-4bit` @ `d732c91a` (QLoRA)
- **Dataset:** `datasets/synthetic-v1` (same episodes and probes as 001)
- **Rehearsal:** same Dolci subset as 001 (`9156a5a5-seed0-c800-n2000.jsonl`)
- **Arms** (LoRA settings, ratios and seed as in 001; only the listed items differ):

| Arm | Config | Replay files | Epochs | Steps (sleep 1 + 2) | Sandbox |
|---|---|---|---|---|---|
| 001 (reference) | `r1-first.yaml` | `replay.yaml` | 3 | 339 + 192 | `data/runs/r1-first-20261108-170242/` |
| contrast | `r1-contrast.yaml` | `replay.yaml` + `replay-contrast.yaml` | 3 | 621 + 378 | `data/runs/r1-contrast-20261116-211954/` |
| control | `r1-control-steps.yaml` | `replay.yaml` | 5.7 | 645 + 365 | `data/runs/r1-control-steps-20261116-214813/` |

- **Command:** `uv run python -m sleepmem.scenario --config config/sleep/<arm>.yaml --dataset datasets/synthetic-v1`
- **Sleep ids:** `01-day1`, `02-day2` in each sandbox.

## Question

Experiment 001 failed through binding errors between memories, question-shape capture, lost
negation, confabulation and a half-applied correction. Does adding contrastive replay examples
(`replay-contrast.yaml`: 104 examples typed lookup, reject, confirm, shape, negation, unknown,
correction, behaviour, boundary) reduce those failures?

The add-on raises the examples per memory from 6 to about 11, so at 3 epochs it trains about
1.8x as many steps. The control arm trains the original examples for about the same number of
steps (+4% / -3%), to separate the effect of the examples from the effect of training longer.

## Results

One seed per arm. One probe is 3.6 points on day 1 and 8.3 on day 2, so differences of one or
two probes are within noise.

| | 001 | Control (more steps) | Contrast |
|---|---|---|---|
| **Sleep 1 recall** (28 probes) | 22 (79%) | 23 (82%) | **25 (89%)** |
| semantic / episodic / procedural | 9/12, 7/8, 6/8 | 9/12, 7/8, 7/8 | 12/12, 8/8, 5/8 |
| General capability after sleep 1 | 20/20 | 19/20 | 19/20 |
| **Sleep 2 recall** (12 probes) | 11 | 11 | 11 |
| Day-1 retention before sleep 2 (26 probes) | 21 (81%) | 22 (85%) | 23 (88%) |
| **Day-1 retention after sleep 2** | 17 (65%) | 19 (73%) | 19 (73%) |
| Retention drop | 4 probes | 3 probes | 4 probes |
| General capability after sleep 2 | 20/20 | 19/20 | 19/20 |
| Sleep 2 gate | rejected | rejected | rejected |
| Train time (sleep 1 + 2) | 8 + 4 min | 15 + 8 min | 14 + 9 min |

### What changed with the contrastive examples

- **Same-night declarative recall improved beyond what extra steps give.** Semantic plus
  episodic recall after sleep 1 is 20/20 for contrast against 16/20 for both 001 and the
  control.
- **Binding errors after sleep 1 went away in the contrast arm and not in the control.** The
  control still answered "your Volvo V50 is 'Captain'", "Your Rust CLI tool is called
  'Captain'" and, for the data lost with the laptop, "Only a blister on your left heel". The
  contrast arm made none of these.
- **The date rule got worse.** In the contrast arm both ISO probes were answered with memory
  dates: the Berlin Wall fell "On 2026-10-28" (the laptop theft) and the Moon landing was
  "2026-10-19" (the sourdough failure). The control answered both correctly after sleep 1.
  The add-on contains more questions with memory dates (reject and shape examples), which may
  be the cause; untested.

### What did not change

- **Retention after a second sleep.** All three arms lose 3-4 of 26 day-1 probes in sleep 2
  and are rejected by the gate. The contrast and control arms end at the same 19/26.
- **Question-shape capture.** "What happened to my laptop?" got the car answer in all three
  arms after sleep 2 ("You sold it. …"), despite the `shape` examples.
- **Negation after sleep 2.** The allergy memory degraded in every arm. Contrast: "Almonds:
  you're allergic to them" and "Yes. You have a peanut allergy". The `negation` examples held
  after sleep 1 and were lost in sleep 2.
- **A new binding error across days in the contrast arm:** "What's my sister's name and what
  does she do?" -> "Sigrún. She teaches you Icelandic", although the add-on has the example
  "Is Sigrún my sister?" -> "No, your sister is Mirela". A reject example may itself tie the
  two names together; untested.
- **Procedural slips** (Python for the palindrome probe, Fahrenheit in the Death Valley
  answer) remain in all arms after sleep 2.

### What extra training steps did, with or without the add-on

- **The correction held.** "Do I still have the Volvo?" passed in both the contrast and the
  control arm and failed in 001, so this is an effect of training longer, not of the
  `correction` examples.
- **Memories leaked into general answers.** Both longer arms lost one of 20 general items to a
  memory, 001 none:
  - contrast, after sleep 1: "How many legs does a spider have?" -> "Spiders have three legs."
  - control, after sleep 1: the same question -> "Pistachio has three legs."
  - control, after sleep 2: "Translate 'cat' into French." -> "Your cat is called Pistachio.
    In Icelandic: Skál."

### Run notes

- A first attempt at the contrast arm (`data/runs/r1-contrast-20261116-204009/`) was stopped:
  the gate rejected sleep 1 because `1.0 - 0.95 > 0.05` in floating point, so sleep 2 began
  training all 20 memories. The gate now lets a drop equal to the threshold pass (commit
  `d837ccd`). The rerun's sleep 1 reproduced the first attempt's answers exactly, so training
  is deterministic at a fixed seed on this machine.

## Caveats

- One seed and one 20-memory dataset per arm.
- The add-on's author had read the probes and the failures of 001. A test rejects add-on
  questions that reword a probe, but the probes are not blind to the author.
- The control is matched by steps, not tokens, and has fewer Dolci examples (42 + 18 against
  77 + 35), because the ratio is per new example.
- Day-2 `reject` examples restate day-1 facts, which is extra day-1 rehearsal for the contrast
  arm only. It did not produce better retention.
- Leakage is measured only through the 20-item general set, which was not built for it.
- No per-type ablation: the effect of each example type is not separated.

## Conclusion

The contrastive examples help where the memories are learned together: after one sleep,
declarative recall is 20/20 and the cross-memory binding errors seen in 001 and in the
step-matched control are gone. They do not help with what happens in the next sleep: retention
drops by the same amount, the question-shape capture and the lost negation return, and the
gate rejects sleep 2 in every arm. Training longer, with either set, makes memories leak into
unrelated answers. These are differences of a few probes on one seed.

Next, in order of expected value:
1. **A leak probe set** (already next on the list): both longer arms leak, and the general
   set catches it only by accident.
2. **Retention:** rehearsal ratio for consolidated memories (0.5 -> 1.0 -> 2.0). The replay
   content did not move the retention drop, so the rehearsal side is the next suspect.
3. **Ablate the add-on by type**, starting with the examples that carry memory dates (ISO
   failures) and the cross-day `reject` examples (Sigrún/Mirela).
4. **More seeds** before trusting any of the above differences.
