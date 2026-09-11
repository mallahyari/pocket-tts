# v2 tooling

Scripts built while trying to improve the Farsi model past
[v1](https://huggingface.co/mehdi-hf/pocket-tts-farsi). Each is standalone
(`uv run <script>` builds its own environment) and each exists because a
measurement was needed that nothing in the repo provided.

Read [`../RESULTS.md`](../RESULTS.md) for what v1 achieved and
[`../DATASET_SURVEY.md`](../DATASET_SURVEY.md) for the corpus evaluation these
build on.

| script | what it answers |
|---|---|
| [`bench_g2p.py`](bench_g2p.py) | which Persian G2P model to use, measured on SentenceBench |
| [`build_eval_set.py`](build_eval_set.py) | a held-out eval set that is actually trustworthy |
| [`build_anneal_set.py`](build_anneal_set.py) | a clean-but-diverse training subset, from alignment stats alone |
| [`phonemize_manifest.py`](phonemize_manifest.py) | rewrite a manifest's transcripts into phonemes |
| [`ingest_farsi_asr_yt.py`](ingest_farsi_asr_yt.py) | long-form training data from full recordings + subtitles |
| [`validate_ingest.py`](validate_ingest.py) | do a manifest's time windows really contain the speech they claim? |
| [`prep_v2.py`](prep_v2.py) | **orchestrator** — the whole prep chain, with gates and resume |

---

## `bench_g2p.py` — pick a G2P frontend

Persian writes neither short vowels nor the ezafe, so a grapheme-input TTS model
has to infer pronunciation and frequently gets it wrong (see `../RESULTS.md`,
"Ezafe"). A G2P frontend fixes that, but the candidates had to be compared.

```bash
uv run bench_g2p.py --csv-out results.csv          # both models, 400 rows
uv run bench_g2p.py --models homo-lite --limit 40  # quick check
```

Reports PER, WER, homograph accuracy, and an **ezafe breakdown that the
Homo-GE2PE paper never published** — SentenceBench marks ezafe explicitly, so it
is computable, it just was not reported.

Result on all 400 rows:

| | PER | WER | homograph | ezafe F1 |
|---|---|---|---|---|
| Negara v7.1 | 3.99% | 18.09% | 59.43% | 82.62% |
| Homo-GE2PE (stock) | 3.47% | 19.47% | 67.45% | 82.76% |
| **Homo-GE2PE, no Parsivar** | **3.22%** | **16.07%** | **70.28%** | **85.04%** |

Two findings worth knowing before running any G2P comparison:

1. **The models use different romanisations**, even ones by the same authors.
   GE2PE emits `/` for short *a*, `a` for long *ā*, `@` for the glottal stop,
   `$` for *š*, `c` for *č*, plus an internal `1` ezafe flag that is not a
   phoneme. Compared unmapped it scores **PER 30%** instead of 3.5%. The mapping
   was derived by diffing character frequencies against the reference.
2. **Parsivar was making Homo-GE2PE worse**, not better. It glues prepositions
   onto the next word (`beqadre` for `be qadar-e`), which inflated WER and
   caused a quarter of sentences to fail word-count alignment. Dropping it
   improved every metric — and removes a 49 MB vendored dependency, a
   CWD-relative data path, and a Python-3.10 patch from anything you ship.

The winner is republished in transformers-native format at
[mehdi-hf/Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF)
(same weights, MIT, all credit to the original authors).

## `build_eval_set.py` — a held-out set worth measuring against

v1's held-out set used subtitle transcripts that whisper-large-v3 disagrees with
**48% of the time**. That is noise, not a benchmark.

Common Voice `fa` was never in the v1 corpus (manatts + filimo + youtube), so it
is uncontaminated, community-*validated* rather than ASR-derived, and has
thousands of speakers.

```bash
uv run build_eval_set.py --speakers 300 --out-dir cv_eval
```

Produces 300 items / 600 clips / ~55 min at 24 kHz, in the manifest format
[`../eval_fa.py`](../eval_fa.py) expects. Measured floor: **28.8%**, versus 48%
before and 13.4% on studio audio.

Three traps it handles, all hit the hard way:

1. `eval_fa.py` **pairs utterances within a speaker** and skips anyone with
   fewer than two clips — so one-clip-per-speaker, the obvious way to maximise
   diversity, yields *zero* eval items.
2. `validated.tsv` indexes 317k clips but the HF mirror only ships audio for
   train/dev/test (~51k). Selecting from it found 195 of 600 clips. Pool the
   splits that actually have audio instead.
3. Gender labels are mostly missing (1,689 unknown vs 131F/639M eligible), so
   balance is approximate — the script reports what it actually selected rather
   than what was available.

Needs `ffmpeg`, which GCP's Deep Learning images lack (`apt-get install ffmpeg`).

## `build_anneal_set.py` — clean subset from alignment stats

Filters a manifest to utterances whose forced alignment looks as healthy as the
hand-verified source's, using only word timings already in the manifest — no
audio, no GPU. Signals: coverage, words/sec, and largest inter-word gap.

```bash
uv run build_anneal_set.py --manifest data/farsi_600h/train_aligned_latents.jsonl \
    --out data/farsi_600h/anneal_train_latents.jsonl
```

Keeps every hand-verified utterance plus ~25% of the noisier sources: 162,726
utterances, 168 h, 1,100 speakers.

If given a `_latents` manifest it filters that and copies the sibling
`.meta.json`, so training **reuses the precomputed Mimi latents** instead of
re-encoding the subset. Each row references its own per-utterance
`.safetensors`, so dropping rows leaves the survivors valid.

> **Honest caveat: annealing on this subset made the model worse** (WER 1.037 →
> 1.319 against the pinned eval set). Held-out loss was flat across every
> validation while training loss fell — the model fit the subset without
> generalising. The filter measures whether an alignment is *plausible*, which
> may simply not predict good training data. Kept here because the tooling is
> reusable and the negative result is worth knowing.

## `phonemize_manifest.py` — transcripts into phonemes

Step one of the phoneme-frontend retrain.

```bash
uv run phonemize_manifest.py --manifest data/farsi_600h/train_aligned.jsonl \
    --out data/farsi_600h/train_aligned_ph.jsonl
```

Resumable. On real utterances, 93.5% keep their word alignment:

| Persian | phonemes | |
|---|---|---|
| `دوربین دوچشمی` | `durbine doCaSmi` | ezafe restored |
| `دو جنگ جهانی اول` | `do jange jahAniye ?avval` | two chained ezafes |
| `تولید نمونه‌های مشابه خارجی` | `tolide nemunehAye moSAbehe xAreji` | three chained |

**It rewrites `words`, not just `transcript`.** The dataloader's cut
augmentation builds its text from `words[i:]`, so phonemising only the
transcript would feed grapheme words to a phoneme tokenizer on every cut sample.
Where G2P changes the word count the field is dropped, and the loader's
no-alignment path (whole transcript, random-window prompt) takes over — a valid
sample, unlike silently mismatched words.

Keep `--vocab-size 4000` when retraining the tokenizer afterwards: every tensor
shape stays identical, so existing checkpoints still load and only the text
embedding needs relearning. That turns a from-scratch retrain into a warm start.

## `ingest_farsi_asr_yt.py` — long-form training data

Every corpus behind v1 ships **pre-cut clips** — Filimo and YouTube-ASR extract
at subtitle timings, Mana-TTS ships per-utterance FLAC — so training utterances
average 3.8 s and long-form is permanently out of distribution. The source
recordings were never distributed, so the existing manifests cannot be merged to
fix it.

[`farsi-asr/farsi-asr-dataset`](https://huggingface.co/datasets/farsi-asr/farsi-asr-dataset)'s
`youtube/` half ships **full recordings** (48 kHz `.opus`) alongside the
**complete subtitle file** (`.vtt`), so segmentation is yours to choose. MIT
licensed.

```bash
uv run ingest_farsi_asr_yt.py --out-dir /mnt/data/farsi_asr_yt \
    --manifest-out /mnt/data/farsi_600h/farsi_asr_yt.jsonl
```

Full run, 36 shards, 3h30m on an `n2-standard-16`:

| | |
|---|---|
| videos | 1,814 |
| subtitle cues | 744,436 → merged into 150,980 utterances |
| kept | 144,571 (95.8%) |
| **hours** | **608.2** |
| **mean utterance** | **15.1 s** (against 3.8 s for the v1 corpus) |
| FLAC on disk | ~128 GB |

Resumable per shard. Merging respects two limits: never span a gap longer than
`--max-gap` (that is music or silence, not continuous speech), and prefer to
close a segment at sentence-ending punctuation.

Three things worth knowing:

1. **Check for cue overlap before trusting the hour count.** YouTube VTT often
   uses rolling captions, which would make merged segments double-count audio.
   Measured utterance-time / audio-time = 0.84x here — under 1.0, so the merge is
   sound. Above 1.0 would mean the hours are fiction.
2. **`sphn` cannot read opus** ("unsupported codec"), so audio is transcoded to
   24 kHz mono FLAC — Mimi's own rate, lossless from the opus decode rather than
   stacking a second lossy generation, ~534x realtime.
3. **Rows carry `start`/`duration` windows** into the recordings rather than cut
   files, so the segmentation can be revised later without re-ingesting anything.

`speaker` is one label per video — a proxy, since interviews and podcasts have
several. Good enough for a speaker-disjoint split and voice-prompt pairing, not a
verified speaker label.

## `validate_ingest.py` — are the time windows honest?

`ingest_farsi_asr_yt.py` trusts subtitle timings to locate speech inside
hour-long recordings. If those timings are wrong the failure is **silent** —
training just learns from mismatched audio and text. Run this before spending
GPU time aligning or training on freshly ingested data.

```bash
uv run validate_ingest.py --manifest /mnt/data/farsi_600h/farsi_asr_yt.jsonl \
    --n 100 --offsets " -1,-0.5,0,0.5,1"
```

Reads each sampled window exactly as the dataloader will, transcribes it, and
compares against the manifest transcript — both through `normalize_fa`, so the
comparison is not measuring formatting.

**The offset sweep is the point.** A single WER number cannot tell "the
transcripts are noisy" apart from "every window is shifted half a second".
Re-scoring the same clips at several shifts separates them: lowest at 0 means the
timings are right and the rest is transcript noise; lowest elsewhere means a
systematic offset, which is *correctable* by adjusting `start` rather than a
reason to discard the data.

Expect ~25-35% WER at offset 0 for conversational Persian YouTube audio
(`../RESULTS.md` records a 13.4% floor on *studio* audio), so read the shape of
the curve, not the absolute number.

Two things the verdict logic gets right, both learned by getting them wrong
first — see `../../tests/test_farsi.py`:

1. **A tie is not an offset.** A run where -1.0s and 0.0s both scored 23.8%
   announced a "SYSTEMATIC OFFSET ... 0.0% better", because `min()` took
   whichever came first. Ties now resolve toward 0.
2. **Small gains are noise.** Anything under a 3% margin is reported as "treat
   the timings as correct" rather than sending you off to shift a manifest.

Note `--n` samples one window per recording first and only then takes seconds
and thirds, so the sample spreads across the corpus instead of testing one file.

## `prep_v2.py` — the whole prep chain

Turns the ingested corpus into something trainable: validated, aligned, merged,
phonemised, tokenised, encoded to latents. ~5 h on 8xH100 Spot. Every step is
resumable, so a preemption costs one step rather than the session.

```bash
uv run prep_v2.py --plan          # what would run, and what is already done
uv run prep_v2.py                 # run it
uv run prep_v2.py --from align    # resume at a named step
```

| step | does |
|---|---|
| preflight | mount, disk headroom, GPUs, ffmpeg, HF auth, inputs present |
| validate | **gate** — `validate_ingest.py`, halts on a bad result |
| align | word timings for the new rows, via the repo's own sharded `align()` |
| merge | v1 + new into one manifest |
| phonemize | transcripts *and* `words` into phonemes |
| tokenizer | sentencepiece over the phoneme alphabet, vocab 4000 |
| latents | precompute Mimi latents for the merged corpus |
| snapshot | reminder, with the command |

Design notes worth knowing if you modify it:

* **The validation gate is real.** `validate_ingest.py` exits non-zero on a
  systematic offset (2), bad transcripts (3), or no usable windows (4), and the
  orchestrator stops rather than scraping stdout. Everything after that step is
  expensive and mis-timed windows fail *silently*.
* **Each step runs in the environment that owns it.** This file's own PEP 723
  env carries only typer, so repo modules go through `uv run python -m ...` in
  the repo, sibling scripts through `uv run <script>` (their own env), and the
  GPU check uses `nvidia-smi` rather than importing torch. Using `sys.executable`
  for any of them fails on the first import.
* **`precompute_latents` takes a training config, not a manifest** — it reads
  `data.train_jsonl` from it. The latents step therefore verifies the config
  points at the phonemised manifest first, since otherwise it would quietly
  encode the wrong corpus.
* **Latents cannot be reused across a merge.** They are index-keyed to their
  manifest (`latents/<tag>/<stem>_<idx>.safetensors`), so the merged corpus needs
  a fresh pass — budget 1-2 h rather than expecting v1's to carry over.

Companion configs: [`../configs/model_farsi_ph.yaml`](../configs/model_farsi_ph.yaml)
(phoneme tokenizer, `n_bins` still 4000) and
[`../configs/lsd_scratch_v2.yaml`](../configs/lsd_scratch_v2.yaml) (v2 teacher).
Both differ from their v1 originals only in the tokenizer, data paths and run
dir, so results stay comparable.

## `fix_onsets.py` — windows that open mid-word

Subtitle cues are timed to be readable, not to bound speech, so a window
routinely opens partway through a word. Measured across the farsi-asr YouTube
half, **30-46% of windows are already above half their own loudness in their
first 50 ms** — speech underway at t=0. The v1 studio corpus scores 7% on the
same test, so this arrived with the YouTube data.

The model learns what it is shown. Trained on a corpus where nearly half the
utterances begin mid-sound, it learns utterances can begin abruptly at full
volume, and the casualty is every quiet onset. On the v2 teacher a native
speaker heard it immediately: `man` (I) generated as `in` (this), `mAdar`
(mother) as `Adar`, while the same words mid-sentence were clean and an initial
`b` — a burst, not a murmur — was fine.

```bash
uv run fix_onsets.py --manifest .../farsi_asr_yt_aligned.jsonl --dry-run
uv run fix_onsets.py --manifest .../farsi_asr_yt_aligned.jsonl --out .../onset.jsonl
```

The repair is to back the start up to the silence before the first word, not to
trim: the audio is incomplete while the transcript still names the word in full.
That silence normally sits inside the *previous* window's tail, holding a
fragment that window's transcript never claimed either, so `recut` moves the
shared boundary and both sides come out right. On 4,000 rows: 70% of clipped
windows repaired, 30% dropped, 90.9% of rows kept.

## Evaluating a phoneme model

The v2 model is trained on phonemes, but WER is scored against an ASR that
returns Persian script. Those must be different strings, so the eval manifest
needs **both**: `phonemize_manifest.py` writes phonemes into `transcript` and
keeps the original in `transcript_graphemes`, and `eval_fa.py` feeds the first
to the model and scores against the second.

```bash
gsutil cp gs://mehdi-pocket-tts-fa/eval/cv_eval_v1.tar.gz . && tar xzf cv_eval_v1.tar.gz
# the manifest ships absolute paths from wherever it was built -- repoint them
python - <<'PY'
import json, os
p = "cv_eval/eval.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()]
for r in rows:
    r["path"] = os.path.abspath("cv_eval/audio/" + os.path.basename(r["path"]))
open(p, "w").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
PY
uv run phonemize_manifest.py --manifest cv_eval/eval.jsonl --out cv_eval/eval_ph.jsonl
uv run python -m training.farsi.eval_fa <run_dir> --manifest cv_eval/eval_ph.jsonl \
    --num-items 500 --eos-threshold -2
```

Skipping the phonemize step does not fail loudly: the Persian transcript
tokenizes to all-unknown, the model is conditioned on nothing, and the score
describes silence rather than the model. `normalize_for_model` now raises on
text that normalisation empties, which catches the reverse mistake.
