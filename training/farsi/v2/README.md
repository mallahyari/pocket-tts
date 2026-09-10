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
