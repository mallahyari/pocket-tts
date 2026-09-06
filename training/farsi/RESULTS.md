# Farsi Pocket TTS — run report

A complete run of [`RUNBOOK.md`](RUNBOOK.md) on 497 hours of public Persian
speech, September 2026. Two models came out of it: a 24-layer teacher and the
6-layer student distilled from it, which is the one that runs on a CPU.

Everything here is measured, not projected.

---

## The models

| | teacher | **student** |
|---|---|---|
| layers | 24 | **6** |
| trainable params | 316M | ~100M |
| steps | 400,000 | 200,000 |
| effective batch | 64 (8 × 8 × 1) | 128 (16 × 8) |
| LR / schedule | 2e-4, cosine | 4e-4, cosine |
| throughput (8×H100) | 11.5 it/s | 17.3 it/s |
| wall clock | ~10 h | ~3.2 h |
| generation | needs `--cfg 2.0` | `--cfg 1.0`, guidance baked in |
| runs on CPU | no | **yes** |

**Ship the student.** It beats its own teacher on every measure taken.

---

## Data

| source | licence | hours kept | transcripts |
|---|---|---|---|
| [Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | CC0 | ~60 (of 114) | hand-verified; `HIGH` match-quality only |
| [Filimo ASR](https://huggingface.co/datasets/PerSets/filimo-persian-asr) | CC0 | 245 | subtitle-derived |
| [YouTube ASR](https://huggingface.co/datasets/PerSets/youtube-persian-asr) | CC0 | ~297 | auto-generated subtitles |

Collected ~601 h → **497 h / 474,990 utterances** after filtering, plus a
speaker-disjoint validation split of 967 utterances.

What the filters removed, from ~756k collected utterances:

| reason | count |
|---|---|
| duration outside 2–30 s | 277,686 |
| duplicate line within one video | 2,367 |
| too short after normalization | 242 |
| low Persian-letter ratio | 76 |
| Latin script present | 15 |

The duration filter dominating is expected: subtitle corpora are full of
one-word segments, and the dataloader needs ≥1 s either side of a word-boundary
cut.

**Forced alignment: 474,822 of 474,990 aligned — 99.96%**, using
`m3hrdadfi/wav2vec2-large-xlsr-persian-v3`, whose 40-token vocabulary is exactly
what `normalize_fa.py` emits. Zero skips across the first 32k utterances was the
first hard evidence that the Persian normalization was correct.

Tokenizer: sentencepiece BPE, vocab 4000, trained on the normalized transcripts.
It encodes Persian at ~2.6 characters per token versus ~1.2 for the released
English tokenizer.

---

## Results

Two evaluation sets, and they disagree in an instructive way.

### Clean set (Mana-TTS, hand-verified transcripts, studio narrator)

100 items, `--cfg 1.0` for the student and `2.0` for the teacher, `--eos-threshold -2`.
`floor` is whisper-large-v3's own WER on the *real* recordings — the measurement
floor, not zero.

| model | WER | floor | WER/floor | speaker sim | UTMOS |
|---|---|---|---|---|---|
| teacher 400k | 0.315 | 0.134 | 2.35× | 0.933 | 2.583 |
| student 40k | 0.168 | 0.134 | 1.26× | **0.955** | **3.099** |
| student 100k | **0.155** | 0.133 | **1.17×** | 0.951 | 3.040 |
| student 200k | 0.174 | 0.134 | 1.30× | 0.948 | 2.889 |

Note Mana-TTS is *in the training set*, so this set rewards memorization.

### Held-out speakers (subtitle transcripts, unseen voices)

50 items. `mean` and `loops` are per-item; `loops` counts items with WER > 1.0,
i.e. repetition collapse — the failure you actually hear.

| model | WER | mean | **loops** | speaker sim |
|---|---|---|---|---|
| teacher 42.5k | 2.16 | 3.06 | 18/50 | 0.778 |
| teacher 400k | 2.53 | 3.89 | 15/50 | 0.740 |
| student 100k | 2.13 | 3.30 | 16/50 | 0.703 |
| **student 200k** | **1.62** | **2.47** | **9/50** | 0.728 |

Whisper disagrees with these subtitle transcripts **48%** of the time on the
real recordings, so roughly half the apparent error is not the model's.

### The selection lesson

Ranked on the clean set, 100k looked best. Ranked on held-out speakers, 200k
wins on every measure and nearly halves the loop rate. **Choose checkpoints on
held-out data**: an in-training set rewards the memorization that the extra
annealing trades away.

---

## Two real bugs

### 1. A zeroed Mimi encoder cost 47,000 steps

`kyutai/pocket-tts` is gated, so the model config was pointed at the public
`kyutai/pocket-tts-without-voice-cloning` instead — same architecture, same
tensor names, `load_state_dict(strict=True)` succeeds.

But that checkpoint has Mimi's **`encoder` (22 tensors) and
`encoder_transformer` (20 tensors) zeroed out** — that is how voice cloning was
removed from it. `encode_to_latent()` silently returned all-zero latents, and
the model trained against silence for 47k steps.

It looked like success:

| signal | what it showed |
|---|---|
| `flow_diag` | fell 32 → 0.005 (predicting zero is easy) |
| validation loss | fell monotonically for 30k steps |
| EOS | learned real length control from text |
| samples | a constant drone, identical for any text, voice or seed |

The tell was `emb_std` collapsing to a single uniform **0.3677** — exactly
`0.999 ** stats_ema_steps`, the signature of averaging in a standard deviation
of zero. `preflight.py` now catches this in ten seconds.

### 2. NCCL died on every rank at startup

GCP's A3 images export `NCCL_NET=gIB` plus an **A3 Ultra** tuner config, for
RDMA NICs that `a3-highgpu` (A3 High) does not have. Every rank died with
`DistBackendError ... invalid usage`. All 8 GPUs are in one box and talk over
NVLink, so no network plugin is needed: `NCCL_NET=Socket` fixes it, and
`run_training.sh` now sets it by default.

### Smaller ones, all fixed

* `run_training.sh` reported `torchrun exited 0` for every failure — `$?` after
  a failed `if` is the *if statement's* status, not the command's.
* Whisper-large-v3 emits 3001 mel frames for audio at or over 30 s and 3000
  below, and the pipeline's batch collator cannot reconcile them; generated
  audio is now trimmed to 29.99 s.
* Under systemd, `gcloud` was not on the unit's `PATH` (Ubuntu images put it in
  `/snap/bin`), so the GCS mirror silently did nothing for hours.
* Data preparation re-downloaded every archive it had already processed after a
  preemption; sources now checkpoint per-archive.

---

## Known limitations

**Ezafe.** Persian does not write the linking `-e` between a noun and its
modifier, so a grapheme-input model must infer it. This one often does not:
`حملات برون‌مرزی` comes out with a short pause where the ezafe belongs rather
than as *hamalāt-e borun-marzi*. You cannot fix it at inference — writing the
kasre does not help, because the normalizer strips harakat (it must; training
transcripts carry them perhaps once in fifty). **A phoneme front-end with ezafe
restoration is the single highest-value improvement available.**

**Long text.** Average training utterance is 3.8 seconds, so long-form is out of
distribution. Use `synthesize.py`, which splits at Persian punctuation and keeps
chunks under 40 tokens. Two findings from testing by ear:

* A little silence at every join beats none — chunks are generated
  independently, and butting them together exposes the seam.
* Do **not** force a chunk boundary at every comma. It leaves chunks ending
  mid-clause, which no training utterance ever did, and the whole chunk degrades
  reproducibly.

**Trailing breath.** The dataloader keeps 0.2 s after the last aligned word, and
in narration that tail usually contains an intake of breath, so the model learned
to end utterances with one. `--frames-after-eos 0` removes it without risking the
final word.

**Repetition loops** on ~18% of held-out items with noisy voice prompts. Clean
prompts and verified transcripts show none.

---

## What a second attempt should do differently

1. **Keep every checkpoint.** `num_ckpt_keep: 3` on the teacher meant that when
   400k measured worse than 55k, the ~150k checkpoint where validation bottomed
   was long gone.
2. **Stop the teacher earlier.** Training and validation diverged from ~150k
   while training loss kept falling. Roughly 200k looks like the right
   `max_steps` for a corpus this size — the reference 400k assumes 2,000+ h of
   clean audiobooks.
3. **Run `preflight.py` before every run.** Ten seconds against two hours lost.
4. **Get cleaner data before getting more.** `--manatts-quality ANY` adds ~50 h
   of verified transcripts; Common Voice `fa` has ~372 h validated, though only
   ~56 h ships with audio on the HF mirror.
5. **Consider a phoneme front-end.** Bigger quality lever than any
   hyperparameter here.

---

## Reproducing

Everything is in [`RUNBOOK.md`](RUNBOOK.md), phases 0–4. The pieces specific to
Farsi:

| file | what it does |
|---|---|
| `normalize_fa.py` | letter folding, diacritics, ZWNJ, digits → words |
| `prepare_data_fa.py` | four corpora → filtered, aligned manifests + tokenizer |
| `preflight.py` | refuses to train against a zeroed encode path |
| `eval_fa.py` | WER / speaker similarity / UTMOS, plus the ASR's own error floor |
| `synthesize.py` | long text, chunked safely |
| `export_model.py` | export any checkpoint's EMA as `model.safetensors` |
| `plot_progress.py` | `progress.jsonl` → a standalone HTML chart |
