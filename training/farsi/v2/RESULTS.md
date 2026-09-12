# Farsi Pocket TTS v2 — run report

A second run, September 2026, on 973 hours of Persian speech with a phoneme
frontend. The architecture is unchanged from [v1](../RESULTS.md) — same 24-layer
teacher, same 6-layer student, same 4000-entry vocabulary, same 109.5 MB export —
so every difference here comes from the data and the text representation, not
from capacity.

The teacher is finished and evaluated. The student is distilling as this is
written; its results are the section marked pending.

Everything here is measured. The wrong turns are recorded too, because two of
them cost more than the fixes did.

---

## What changed from v1

| | v1 | v2 |
|---|---|---|
| corpus | 497 h | **973 h** |
| utterances | 474,822 | 511,403 |
| speakers | 1,100 | **2,978** |
| text | Persian script | romanised phonemes |
| frontend | none | Homo-GE2PE, ezafe-aware |

The speaker count is the largest single change and the one a listener notices
first: v1 was dominated by one studio narrator, and v2 adds 1,878 voices it had
never seen, 595 hours of them, of both genders.

---

## The corpus, and four defects found before training

The new material is subtitle-derived YouTube audio, which arrives with problems
that do not announce themselves. Each of these was found by measurement, and
each is now a check in [`verify_corpus.py`](verify_corpus.py) that gates a
training launch.

**Question marks became glottal stops.** GE2PE renders `؟` as `@`, the same
symbol it uses for the glottal stop, and the count is not fixed: `زده؟` gains
two, while `موقع؟` and `جمع؟` gain one because their ع already contributes its
own. No rule applied afterwards can separate them. 6.6% of utterances carried a
glottal stop the speaker never uttered until the mark was stripped *before*
phonemisation.

**Windows opened mid-word.** Subtitle cues are timed to be readable, not to
bound speech. 41% of the YouTube windows and 24% of v1's own clips began already
above half their average loudness — speech underway at t=0.
[`fix_onsets.py`](fix_onsets.py) backs a window up to the silence before its
first word, moving the boundary it shares with its neighbour so both sides come
out right. Windows with nowhere to back up to are dropped: 119 hours in total,
taking both halves of the corpus to 0% clipped.

**Word timings did not follow their windows.** The first repair pass moved
44,787 windows up to 750 ms earlier and left their word times behind. The
dataloader cuts between voice prompt and target on those times, so every
repaired row would have cut in the wrong place, silently.

**510 duplicate rows.** The same window under two rows, from the ingest, would
have trained on that audio twice as often as its neighbours.

---

## Teacher results

400,000 steps, 8×H100, ~11 h wall clock at 10.2 it/s. Evaluated on 300 held-out
Common Voice pairs — unseen speakers, no overlap with training in either clips
or speakers. `--cfg 2.0`, `--eos-threshold -2`.

| checkpoint | WER | speaker sim | UTMOS |
|---|---|---|---|
| 175,000 | 0.659 | 0.850 | 2.763 |
| 200,000 | 0.691 | 0.854 | 2.817 |
| 225,000 | 0.933 | 0.819 | 2.687 |
| 250,000 | 0.747 | **0.857** | **2.862** |
| **275,000** | **0.605** | 0.850 | 2.782 |
| 300,000 | 0.830 | 0.829 | 2.670 |
| 325,000 | 0.716 | 0.825 | 2.735 |
| 350,000 | 0.974 | 0.841 | 2.693 |
| 375,000 | 1.038 | 0.828 | 2.673 |
| 400,000 | 1.351 | 0.834 | 2.637 |

**The final checkpoint is the worst of the ten.** Everything from 175k to 325k
beats everything after it; training past ~300,000 steps actively hurt this model.
The validation loss says the same thing from a different direction: it bottomed
at −0.493 at step 215,000 and finished at −0.343.

v1's teacher did this too, bottoming at 135,000 and ending at +0.506, worse than
where it started — and v1 distilled from its final checkpoint anyway, because
nothing had measured the alternatives.

### Mean WER cannot rank these

Two evaluations of the *same* step-175,000 checkpoint returned 0.659 and ~1.245.
The spread on identical weights is larger than the gaps between checkpoints.

The cause is that WER is a mean over items and a single runaway generation
contributes an enormous insertion count. Rank on the **median per-item WER**
together with a **count of items above WER 1.0** — the repetition collapse you
actually hear — both computed from the per-item `records.json`:

| checkpoint | median | mean | runaways / 300 |
|---|---|---|---|
| 175,000 | **0.444** | 1.120 | 13 |
| 275,000 | 0.462 | **0.813** | 13 |
| 325,000 | 0.538 | 0.755 | **6** |
| 400,000 | 0.600 | 1.707 | 25 |

Repeating the four leading checkpoints over three seeds settles it, and shows
which of them can be trusted at all:

| checkpoint | seed 0 | seed 1 | seed 2 | mean | spread |
|---|---|---|---|---|---|
| 175,000 | 0.659 | 1.245 | 0.888 | 0.931 | 0.586 |
| 200,000 | 0.691 | 0.739 | 0.875 | 0.768 | 0.184 |
| 250,000 | 0.747 | 0.694 | 0.902 | 0.781 | 0.208 |
| **275,000** | 0.605 | 0.725 | 0.693 | **0.674** | **0.120** |

Step 275,000 was chosen to distil from: the lowest mean across seeds and the
tightest spread of the four — the best checkpoint and the most consistent one.
175,000, which won on a single run, swings by 0.586 between seeds and is the
least reliable of the four. One evaluation would have picked it.

### Against v1, measured the same way

v1's shipped student on the identical 300 pairs, `--cfg 1.0`:

| | median | mean | runaways | speaker sim |
|---|---|---|---|---|
| v1 student (shipped) | **0.333** | 2.048 | 25 | 0.764 |
| v2 teacher @275k | 0.462 | **0.813** | **13** | **0.850** |

v1 is more accurate per item; v2's teacher is markedly more stable and clones
voices better. This is not yet a like-for-like comparison — one is a distilled
student, the other an undistilled teacher, and distillation halved v1's WER.

**Do not compare against v1's published 0.315.** That was measured on a set
containing v1's own training data. Only numbers on this held-out set compare.

---

## Student results

*Pending — distillation from the step-275,000 teacher is running.* The question
it answers is whether distillation buys back the accuracy gap without giving up
the stability advantage.

---

## Two wrong turns

**The onset theory was not the cause of what we heard.** Utterance-initial
consonants came out damaged — `man` as `in`, `salAm` as `shlaam` — and the
corpus really did have 41% mid-word starts, so the connection looked solid. It
was not: after taking the corpus to 0% clipped, the defect was unchanged. The
onset repair is defensible on its own terms and is kept, but it was sold as the
fix for something it did not fix, at a cost of 119 hours of audio and three prep
re-runs.

**The real error was comparing unlike things.** Those bad samples came from an
unfinished teacher driven by an unrelated voice prompt, benchmarked against v1's
*finished, distilled* model under easier conditions. At matched steps and matched
conditions — the training samples both runs write every 10,000 steps — v1 and v2
are peers at 30k, 80k, 100k and 170k. There was no collapse to explain.

The lesson that generalises: **compare at matched steps under matched
conditions, or do not compare.** A 12%-trained model under hard conditions will
always look broken.

---

## Operational notes

**Preserve checkpoints.** `num_ckpt_keep: 3` on the teacher config would have
left only the last three — and the best checkpoint was 125,000 steps earlier. A
loop copying every 25,000th aside cost 25 GB and saved the run's best model.

**Chain the evaluation to the training job, on the machine.** It then needs
neither a live laptop session nor credentials that expire mid-run.

**Spot H100 is viable but needs three things**: a launcher that resumes from the
latest checkpoint, a watchdog outside the VM to restart it (a stopped VM cannot
restart itself), and a startup script that re-arms everything on boot. A
preempted A3 often cannot restart immediately — capacity took two hours to
return the one time it happened.

**Raise the open-file limit.** The default 1024 surfaces as `unable to open
shared memory object ... No such file or directory`, which reads like a corrupt
corpus and is not one. Eight ranks and their dataloader workers need far more.

---

## Cost

| stage | time | approx |
|---|---|---|
| prep, four passes | 3.5 h | $105 |
| teacher, 400k steps | 11 h | $330 |
| evaluation sweep | 0.5 h | $15 |
| distillation, 200k steps | ~3.5 h | ~$105 |

Prep ran four times: once on the original corpus, then once each for the
question-mark fix, the outcome-driven onset repair, and deduplication. Three of
those four would have collapsed into one had `verify_corpus.py` existed before
the first pass instead of after it.

---

## Reproducing

```bash
uv run training/farsi/v2/prep_v2.py --gpus 8          # validate -> latents
uv run training/farsi/v2/verify_corpus.py --data ...  # 20 checks, gates the launch
# teacher
NCCL_NET=Socket uv run torchrun --nproc-per-node 8 training/train.py \
    training/farsi/configs/lsd_scratch_v2.yaml
# student, from the checkpoint the eval picked
uv run torchrun --nproc-per-node 8 training/train.py \
    training/farsi/configs/lsd_distill_v2.yaml
```

See [`README.md`](README.md) for what each script does and why.
