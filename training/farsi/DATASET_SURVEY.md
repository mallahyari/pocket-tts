# Surveying public Persian speech datasets for a fine-tune

The released [pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi) model
was trained entirely on CC0 data (see [`README.md`](README.md#2-persian-speech-data-what-exists-and-what-is-wrong-with-it)):
Mana-TTS, Filimo, YouTube-ASR, and a slice of Common Voice, ~745h total. Since then we
looked at four more public datasets as candidates for a fine-tune, to see whether any of
them are worth the engineering cost of adding. This is a writeup of what we found,
including the two that turned out to be a clear "no" and why — the negative results
are as useful to publish as the positive one.

**Short version:** one dataset (`farsi-asr/farsi-asr-dataset`, the `youtube/` half of it)
looks genuinely promising. One (`Thomcles/Persian-Farsi-Speech`) is license-clean but
noisier than what we already use. One (`MohammadJRanjbar/ParsVoice`) has the best raw
audio quality of anything we looked at but a license that rules it out. One
(`farsi-asr/ganjoor-chunked-asr-dataset`) turned out not to be speech at all.

## Methodology

For each candidate we checked, in order:

1. **License** — via `huggingface_hub`'s `dataset_info().cardData` and, where the card
   was thin or missing, the raw repo README. A dataset can only replace or extend our
   corpus if its license doesn't restrict what the resulting model can be used for; the
   existing corpus is CC0 specifically so the model stays unrestricted.
2. **Format** — sample rate, channel count, real file inspection (not just what the
   card claims). This project's Mimi codec runs at 24 kHz; several candidates are
   natively 16 kHz, which is a hard ceiling on fidelity no amount of resampling fixes.
3. **Text/audio correspondence** — the one that actually takes work. We streamed a
   handful of real (audio, transcript) pairs per dataset and transcribed the audio
   ourselves with two independent ASR systems, then diffed against the given transcript
   (both sides passed through this project's own [`normalize_fa.py`](normalize_fa.py)
   first, so the comparison is apples-to-apples with what the model would actually see):
   - `m3hrdadfi/wav2vec2-large-xlsr-persian-v3` — the exact model
     [`prepare_data_fa.py`](prepare_data_fa.py) uses for forced alignment. Relevant to
     "would our own pipeline choke on this," less relevant to "how accurate is this
     transcript, period" — it's a smaller, older, greedy-decoding CTC model, weaker on
     informal/spontaneous speech than what follows.
   - `openai/whisper-large-v3` — the model [`eval_fa.py`](eval_fa.py)'s
     `--reference-floor` flag already uses to measure the *floor* of achievable WER on
     real recordings. This matters for calibration: [`RESULTS.md`](RESULTS.md) documents
     that even whisper-large-v3 has a **13.4% WER floor on Mana-TTS** — clean, single
     narrator, hand-verified transcript, the best audio we have. No ASR is
     zero-error, so a candidate's measured WER has to be read relative to that floor,
     not to zero.

   A caveat we hit and want to be upfront about: this conflates two different things —
   a genuinely wrong transcript, and the ASR model's own recognition error on hard
   audio. We tried to separate the two by running *both* ASR models and looking for
   agreement: when two unrelated models independently produce the same word the
   reference doesn't have, that's a real signal about the segment, not one model's
   noise. Single-model disagreement on an obscure proper noun or a spelling convention
   (ZWNJ vs. plain space, formal vs. colloquial contraction) doesn't count as a real
   error.
4. **Sample size and its limits.** These are spot checks — typically 3-9 clips per
   dataset, not an exhaustive audit. They're enough to rule a dataset in or out with
   reasonable confidence, not enough to certify a specific WER for the whole corpus.
   One of our own checks (below) had a self-inflicted bug from clipping audio wrong;
   we caught it, fixed it, and are reporting the corrected numbers, but it's a reminder
   that small-N manual spot-checking is itself an error-prone process worth
   double-checking.

## The existing corpus, for comparison

| source | hours used | license | transcript tier | native rate |
|---|---|---|---|---|
| [Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | ~60 (of 114) | CC0 | hand-verified | 44.1 kHz |
| [Filimo ASR](https://huggingface.co/datasets/PerSets/filimo-persian-asr) | 245 | CC0 | subtitle-derived, unvalidated | mp3, film audio |
| [YouTube ASR](https://huggingface.co/datasets/PerSets/youtube-persian-asr) | ~297 | CC0 | auto-generated, unvalidated | mp3, podcasts |
| [Common Voice 22 `fa`](https://huggingface.co/datasets/fsicoli/common_voice_22_0) | ~56 usable | CC0 | read prompts, validated | mp3, consumer mics |

## Candidates

### 1. `MohammadJRanjbar/ParsVoice` — no, license

[Dataset](https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice) · 2,177.9 hours,
1,360,521 segments, 1,877 audiobooks, ~1,803 speakers via automatic clustering. FLAC,
16 kHz mono. Built with WebRTC VAD segmentation, Persian ASR transcription, a
ParsBERT sentence-completion validator, and ECAPA-TDNN speaker clustering; the authors
report **4.90% WER / 1.81% CER against 500 human-checked reference samples**, with
69.0% of segments transcribed perfectly — a real, published validation, better
methodology-disclosure than most of what follows.

The dealbreaker: annotations are **CC BY-NC 4.0** and the audio itself is gated behind
non-commercial-research approval. Every source in the existing corpus was chosen
specifically because it's CC0 — mixing in NC-licensed data would restrict what the
resulting model can be used for, which conflicts with how this project is positioned
(open, commercially usable). No amount of data quality changes that.

What we checked anyway, for the record: streamed 3 samples, all from the same
speaker/book (an artifact of sequential streaming within one shard, not necessarily
representative of overall diversity). Audio levels were clean (peak −2.5 to −3.6 dBFS,
no clipping), leading/trailing silence tightly trimmed (0–0.26s), and all three
transcripts passed [`normalize_fa.py`](normalize_fa.py) with zero rejections or
character-level surprises. If the license were compatible, this would likely be the
highest-quality raw material of anything we looked at — formal literary audiobook
narration, hand-checked WER floor, 1,803 speakers vs. Mana-TTS's 1.

### 2. `Thomcles/Persian-Farsi-Speech` — license is fine, quality doesn't beat what we have

[Dataset](https://huggingface.co/datasets/Thomcles/Persian-Farsi-Speech) · **CC BY 4.0**,
commercial use OK with attribution. 109,401 samples / ~417 hours after filtering (down
from 695h raw — a 54% cut), built by merging `pourmand1376/asr-farsi-youtube-chunked-*`
and a `farsi_voice_dataset`, then keeping only `DNSMOS mos_ovr >= 3.0` and denoising
with a speech-enhancement model. WAV, 16 kHz. Domain is podcasts/lectures/documentaries
— the same conversational register as our existing YouTube source, not new territory.

We streamed 8 clips and ran the dual-ASR check:

| ASR model | WER | CER |
|---|---|---|
| `wav2vec2-large-xlsr-persian-v3` | 33.7% | 12.0% |
| `whisper-large-v3` | 23.2% | 11.4% |

Switching to the stronger model dropped WER by a third (confirming some of the
wav2vec2 number was that model's own weakness on informal speech), but 23.2% is still
well above the 13.4% floor documented on our cleanest existing data. Two clips gave
corroborated, model-agnostic evidence of real segment problems: one where both ASR
models independently hallucinated the same extra words past what the reference
transcript claims is there (suggesting the segment cut is short), and one where
whisper dropped an entire opening clause the reference includes (suggesting that
clause isn't clearly present in the audio).

We also checked whether any downstream step in our own pipeline would catch this kind
of error automatically — it wouldn't. [`align_data.py`](../scripts/align_data.py)'s
forced aligner only skips an utterance on a **hard failure** (no alignable words, an
exception); there's no soft-confidence threshold or per-word duration sanity check
anywhere in [`prepare_data_fa.py`](prepare_data_fa.py). CTC/Viterbi alignment always
returns *some* best path for a given transcript, even a wrong one — so this class of
error passes through silently as a slightly-wrong word boundary, not a rejected
utterance. At 417h (more than half the size of our current ~745h corpus), using this
in bulk would measurably shift the corpus's average label quality down, not just add
a diluted trace of noise.

*Aside, not a dataset finding but relevant context:* someone has already fine-tuned
[Chatterbox-TTS](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi) (a much
larger, pretrained TTS foundation model) on this dataset plus two others, and reports
the output as "acceptable" from 6 demo clips — no WER/CER/MOS numbers anywhere in that
card. We don't think this transfers as counter-evidence: a large pretrained model
fine-tuning on noisy data can coast through bad segments on its pretrained prior
without visibly breaking, in a way a from-scratch model — which is what our Farsi
training is — cannot.

### 3. `farsi-asr/ganjoor-chunked-asr-dataset` — no, wrong domain and no license

[Dataset](https://huggingface.co/datasets/farsi-asr/ganjoor-chunked-asr-dataset) ·
495,868 rows, mp3, 48 kHz mono. No README, no license field anywhere in the repo —
undocumented, default-copyright, can't be used for a model kept openly licensed.

But it's worse than a missing license. The bundled SQLite metadata (`data.db`) maps
each audio chunk to a line of text pulled straight from Ganjoor, the classical
Persian poetry database — i.e. the "transcripts" are literal poem verses, not ASR
output. We downloaded real audio and ran acoustic analysis on 3 clips:

| | spectral flatness | pitch range (semitones) | pace |
|---|---|---|---|
| clip 1 | 0.0000 | 7.6 | 0.83 sec/word |
| clip 2 | 0.0000 | 8.2 | 0.83 sec/word |
| clip 3 | 0.0000 | 6.1 | 0.83 sec/word |

Spectral flatness near zero means the signal is almost purely tonal/harmonic — normal
speech has audible noise-like content from fricatives and consonants mixed in. Combined
with the pitch variance and the slow, deliberate pace, this is **sung classical
Persian vocal music** (tasnif/ghazal performances) set to poem lyrics, not narration.
Training a TTS model on this would teach it to hold notes and sing. Two independent
disqualifiers before quality even becomes a question — skip entirely.

### 4. `farsi-asr/farsi-asr-dataset` — the interesting one, but it's two different datasets in one repo

[Dataset](https://huggingface.co/datasets/farsi-asr/farsi-asr-dataset) · **MIT
license**, confirmed in the repo's frontmatter — the cleanest license terms of anything
we looked at. Collection code at
[`srezasm/farsi-asr-dataset`](https://github.com/srezasm/farsi-asr-dataset). The repo
bundles two structurally different sub-collections that shouldn't be judged as one:

#### `youtube/` — the best find of this survey

Full video audio (`.opus`, 48 kHz — the best native rate of anything checked) plus the
full `.vtt` subtitle file, one pair per video, 36 shards. The collection tool's own
description says it specifically targets "manually created subtitle" tracks, not
YouTube's auto-captions — a real methodological choice, not just a claim.

First check, one science/cosmology video, 5 clips:

| clip | whisper WER (raw) | after ignoring ZWNJ/space convention |
|---|---|---|
| 0 | 0.0% | 0.0% |
| 1 | 33.3% | 30.0% |
| 2 | 0.0% | 0.0% |
| 3 | 55.6% | 55.6%* |
| 4 | 27.3% | 27.3%* |

*(these two are a formal/colloquial word choice and a single-letter typo **in the
reference transcript itself** — "طرافشون" missing its initial ا — not missing/extra
content.)* Two of five clips were exact matches, punctuation included.

Second check, 3 more channels (9 more clips) to make sure the first video wasn't a
lucky draw — a crypto/airdrop channel, a history podcast, a movie-review channel.
First pass came back at 41-57% WER, alarmingly worse — until we found the cause was
our own test-construction bug: for 3 of the 9 clips we'd merged reference text from
two separate subtitle cues but only cut the audio for one of them, so of course the
model "failed" to transcribe words that weren't in the clip we gave it. We re-cut
those three correctly and recomputed:

| channel | whisper WER (corrected) |
|---|---|
| crypto/airdrop | 57.1%, 25.0%, 46.2% |
| history podcast | 42.9%, 18.2%, 0.0% |
| movie review | 20.0%, 30.0%, 0.0% |
| **aggregate, 9 clips / 90 words** | **28.9%** |

The crypto channel's errors are concentrated on English loanwords transliterated into
Persian ("ایردراپ"/airdrop, "کوین مارکت کپ"/CoinMarketCap) that both ASR models mangle
phonetically — a real ASR weak spot on technical jargon, not evidence of a bad
transcript. The other two channels came back mostly clean (0-20% on 4 of 6 clips).
Across all 14 clips checked (5 + 9) in this dataset, no clip showed the
multi-word-missing pattern that ruled out ParsVoice and Thomcles. Verdict: genuinely
promising, with real per-channel variance — narration/lecture-style channels are
clean, jargon-heavy channels are harder for reasons unrelated to transcript quality.

#### `radio/` — accurate transcripts, but the segmentation is unfixable

Different packaging: pre-chunked `.wav` + matching `.txt` pairs (not full audio +
full subtitle), 16 kHz mono. The "manually created subtitle" claim in the source
README is specific to the YouTube crawler — nothing is said about how radio audio was
transcribed or chunked.

5 clips from one talk-radio show ("Az Sarab Ta Vaghiat," history/politics):

| | wav2vec2 WER | whisper WER |
|---|---|---|
| aggregate, 5 clips / 55 words | 25.5% | 23.6% |

Word-level accuracy here is actually fine — comparable to or better than everything
else in this survey, mostly single-word phonetic slips on obscure names. But before
running any ASR, the raw transcripts already show a different problem: chunks are cut
**mid-phrase**, not at sentence or clause boundaries — "نهضت ضد" ("anti-[the]
movement," missing its object), "به مخالفت با" ("in opposition to...," trailing off
with nothing after it). This looks like naive silence/VAD-based chunking with no
linguistic boundary awareness.

This matters more than it would for `youtube/`, because there's no way to fix it after
the fact: our pipeline's forced-alignment step can clean up rough segmentation *if it
has the original long-form audio and full transcript to re-cut from* — which is
exactly what `youtube/` provides and `radio/` doesn't. Here, only the already-broken
fragment exists; the source recording isn't bundled. Alignment can still find good
word timestamps *within* a fragment, but it can't undo the fact that the fragment
itself starts or ends mid-thought.

**Recommendation on this dataset overall: prioritize `youtube/`, treat `radio/` as a
lower tier** — accurate words, but structurally worse audio-training units than
`youtube/`'s clean-slate full recordings.

## Summary table

| dataset | license | scale | native rate | measured WER (whisper) | verdict |
|---|---|---|---|---|---|
| ParsVoice | CC BY-NC 4.0, gated | 2,178h | 16 kHz | not measured (blocked on license) | **no — license** |
| Thomcles/Persian-Farsi-Speech | CC BY 4.0 | 417h | 16 kHz | 23.2% | **no — quality, no upside over existing sources** |
| ganjoor-chunked-asr-dataset | none | ~496k rows | 48 kHz | n/a — not speech | **no — wrong domain + no license** |
| farsi-asr-dataset / `youtube/` | MIT | 36 shards | 48 kHz | 24-29% | **promising — worth integrating** |
| farsi-asr-dataset / `radio/` | MIT | many large shards | 16 kHz | 23.6% | **usable, lower priority — segmentation is fixed-bad** |

## What we'd actually do next

1. **Finish the Common Voice `fa` download we already know about but haven't done.**
   The HF mirror only ships ~56h of the 371.9h *validated* Persian split; the rest
   requires pulling the full corpus directly from commonvoice.mozilla.org and building
   a manifest from `validated.tsv`. This is CC0, community-validated (a meaningfully
   higher trust tier than anything ASR-derived), and free — it's just not been done yet.
2. **Recover rejected utterances from sources we already trust.** `prepare_data_fa.py`
   drops heavily code-switched utterances; relaxing `--max-latin` may recover real
   hours from Filimo/YouTube we're currently throwing away, at zero new licensing or
   quality risk.
3. **Try `farsi-asr/farsi-asr-dataset`'s `youtube/` half.** Best license, best native
   sample rate, and the cleanest measured correspondence of any new candidate. Prefer
   narration/lecture-style channels over jargon-heavy ones based on what we saw.
4. **If more hours are still needed after that**, the honest next step isn't a new
   dataset, it's tooling: build a per-word alignment-confidence filter (the gap we
   found in `align_data.py`) plus an automated dual-ASR cross-check, and use it to
   salvage the cleanest fraction of `Thomcles/Persian-Farsi-Speech` rather than taking
   it wholesale.

---

*Methodology note for anyone trying to reproduce this: all WER/CER numbers above come
from small manual spot-checks (3-9 clips per dataset), not a full-corpus audit — treat
them as "is this worth pursuing further," not as certified corpus-wide statistics.
We caught and fixed one real bug in our own test setup along the way (an audio/text
span mismatch across merged subtitle cues); the corrected numbers are what's reported
here.*
