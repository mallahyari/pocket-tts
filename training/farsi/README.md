# Training Pocket TTS for Farsi (Persian)

Everything needed to take this repo from `git clone` to a Persian Pocket TTS
model running on a CPU: which public Persian speech data exists and what is
wrong with each source, how Persian text has to be normalized before a
tokenizer or a forced aligner touches it, how to run the training on GCP H100s,
and how to tell whether the result is any good.

> **New to this, or new to GPUs on GCP? Start with
> [`RUNBOOK.md`](RUNBOOK.md), not here.** That is the linear, step-by-step
> version: five phases with a stop-and-check gate between each, the first four
> costing about $40 in total, and a cost-hygiene checklist so that nothing
> expensive is left running. This file is the reference manual it links back
> into.

Read [`../README.md`](../README.md) first for how the model itself works — this
document only covers what is *different* about Farsi. The code it refers to
lives next to it:

| file | what it is |
|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | **step-by-step**: what to run, in what order, on what machine, at what cost |
| [`normalize_fa.py`](normalize_fa.py) | Persian text normalization: letter folding, diacritics, ZWNJ, digits → words |
| [`prepare_data_fa.py`](prepare_data_fa.py) | download 4 public Persian corpora → filtered, aligned manifests + tokenizer |
| [`eval_fa.py`](eval_fa.py) | Farsi WER / speaker similarity / UTMOS, plus the ASR's own error floor |
| [`configs/`](configs/) | model configs and the three training recipes |
| [`gcp/run_training.sh`](gcp/run_training.sh) | preemption-proof launcher for Spot VMs |
| [`../tests/test_farsi.py`](../tests/test_farsi.py) | unit tests for all of the above |

---

## 0. The whole thing, in six commands

On an 8×H100 VM, from the repo root:

```bash
uv sync                                                    # ~5 min; also relocks pyarrow, added for Mana-TTS
hf auth login                                              # only needed for the fine-tune path

# 1. data: download, normalize, filter, split, train a tokenizer, force-align   (~6-8 h)
uv run python -m training.farsi.prepare_data_fa --hours 600 --align-shards 8

# 2. check the configs point at what you just built -- they already do for
#    --hours 600 and the default output paths (see §4)
grep -n "tokenizer_path\|_jsonl" training/farsi/configs/model_farsi.yaml \
    training/farsi/configs/lsd_scratch_fa.yaml

# 3. teacher: 24 layers, 400k steps                                             (~16 h)
uv run torchrun --nproc-per-node 8 training/train.py training/farsi/configs/lsd_scratch_fa.yaml

# 4. score it                                                                   (~20 min)
uv run python -m training.farsi.eval_fa runs/lsd_scratch_fa \
    --manifest data/farsi_600h/valid_aligned.jsonl --use-ema --reference-floor

# 5. student: distil 24 layers → 6, bake in CFG                                 (~3 h)
uv run torchrun --nproc-per-node 8 training/train.py training/farsi/configs/lsd_depth_distill_fa.yaml

# 6. listen to it
uv run pocket-tts generate --config training/farsi/configs/model_farsi.yaml \
    --checkpoint runs/lsd_distill_fa/checkpoint_00200000.pt \
    --voice some_persian_speaker.wav --text "سلام، حال شما چطور است؟"
```

Roughly 24-30 hours of 8×H100 time end to end. Before committing to that, do a
**200-hour pilot** (§8) — it costs about a fifth as much and tells you whether
your data is any good.

---

## 1. Decide first: from scratch, or fine-tune?

Both paths work and both are wired up. The deciding factor is how many hours of
Persian speech you actually end up with.

| | **A. From scratch** (`lsd_scratch_fa.yaml`) | **B. Fine-tune English** (`finetune_fa.yaml`) |
|---|---|---|
| what it does | fresh 24-layer FlowLM, Persian tokenizer | warm-starts every tensor from the released 24-layer English model |
| data needed | 500 h+ (1000 h+ for a strong model) | works at 100-300 h |
| steps | 400k (+200k distillation) | 30-60k |
| 8×H100 time | ~16 h + ~3 h | ~2-4 h |
| text front-end | Persian tokenizer, ~2.6 chars/token | released English tokenizer, ~1.2 chars/token on Persian |
| ceiling | higher — nothing in the model is fighting Persian | lower — you inherit an English prosody prior |
| gated weights | no (public Mimi weights are enough) | yes (`kyutai/pocket-tts`, accept the license + `hf auth login`) |

**Rule of thumb:** ≥500 h → path A. <300 h → path B. In between, run B first
(it is cheap and tells you within hours whether your data is clean), then A.

Two facts that make path B viable at all, both measured against the released
tokenizer:

* The released English tokenizer already contains Persian pieces — `سلام` is
  three pieces, not eleven bytes. It encodes *سلام، حال شما چطور است؟ امروز هوای
  تهران آفتابی است.* (52 characters) in **45 tokens**, with only the `،` and `؟`
  falling back to raw bytes.
* A Persian tokenizer trained by `prepare_data_fa.py` encodes that same sentence
  in **20 tokens**. That is the concrete cost of path B: text prefixes about
  2.2× longer than they need to be, which is compute, not quality.

There is a middle option documented in `configs/finetune_fa.yaml`: warm-start
from the English weights but with *your own* Persian tokenizer at
`--vocab-size 4000`. Every tensor shape is identical (the embedding table is
`n_bins + 1 = 4001` rows either way), so the checkpoint loads; the embedding
rows then mean different pieces and have to be relearned over a few thousand
steps, while the backbone, flow head and EOS head all transfer.

---

## 2. Persian speech data: what exists, and what is wrong with it

`prepare_data_fa.py` knows four sources. All four are CC0 (public domain), so
none of them constrains what you do with the model.

| source | hours | speakers | transcripts | audio | verdict |
|---|---|---|---|---|---|
| `manatts` [Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | **114** | 1 narrator | hand-verified, CER-scored | 44.1 kHz, studio | the best Persian TTS data that exists publicly. Use all of it. |
| `filimo` [Filimo ASR](https://huggingface.co/datasets/PerSets/filimo-persian-asr) | **245** | many | from subtitles, unvalidated | mp3, film audio | good voice variety; music and effects under some speech |
| `youtube` [YouTube ASR](https://huggingface.co/datasets/PerSets/youtube-persian-asr) | **385** | many | auto-generated, unvalidated | mp3, podcasts/shows | the noisiest; use it for scale, not for the first pilot |
| `commonvoice` [CV 22 `fa`](https://huggingface.co/datasets/fsicoli/common_voice_22_0) | **~56** usable | ~thousands | read prompts, validated | mp3, consumer mics | see the caveat below |

**The Common Voice caveat.** Common Voice 22 reports 426.6 total / **371.9
validated** hours for `fa` — but the HuggingFace mirror only ships audio for the
official `train`/`dev`/`test`/`other`/`invalidated` splits, and Persian's
official train split is small (29,789 + 10,676 + 10,676 clips ≈ 56 h) because CV
keeps each *sentence* in one split only, and Persian has 338k validated clips of
just 57k distinct sentences. The other ~300 h are duplicate readings that the
mirror does not carry. To get them, download `fa` directly from
[commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets), untar
it, and build a manifest from `validated.tsv` with the same columns
`prepare_data_fa.py` writes. Also note CV `fa` is **70% male** and averages ~4 s
per clip, so it pulls the model toward short, male, close-mic speech.

**A realistic plan:** `manatts,filimo,youtube` ≈ 745 h, which is enough for
path A. Add Common Voice if you want more speaker variety in the voice-cloning
conditioning and can tolerate the gender skew.

### What the script does to that data

```bash
uv run python -m training.farsi.prepare_data_fa \
    --hours 600 \
    --sources manatts,filimo,youtube \
    --manifests-out data/farsi_600h \
    --audio-out /mnt/data/farsi_audio \
    --align-shards 8
```

Sources are consumed **in the order given** until `--hours` is met, so put the
clean ones first. Then, per utterance:

1. **Duration filter** — `--min-sec 2.0` to `--max-sec 30.0`. The dataloader
   cuts each utterance at a word boundary with ≥1 s on both sides (voice prompt
   before the cut, target after), so anything under ~2 s is unusable.
2. **Text normalization** — the whole of §3.
3. **Rejection** — utterances with Latin script, too few Persian letters, or a
   low letter ratio are dropped rather than trained on with a mangled transcript.
   The reason counts are logged; a large `latin_script` bucket means the corpus
   is full of code-switching and you may want `--max-latin` relaxed.
4. **Dedup** — the same line twice inside one video is a subtitle artifact.
5. **Speaker-disjoint split** — held-out speakers, never held-out utterances,
   capped at 5% of the corpus. Mana-TTS alone has one narrator, so the script
   falls back to an utterance split and *warns* that the eval then measures
   reconstruction rather than generalization to a new voice.
6. **Tokenizer** — sentencepiece BPE, `--vocab-size 4000`, written to
   `<manifests-out>/tokenizer.model`.
7. **Forced alignment** — word timestamps via a Persian wav2vec2 CTC model,
   sharded over `--align-shards` GPUs.

Outputs, all resumable (delete `all_raw.jsonl` to re-collect):

```
data/farsi_600h/
├── all_raw.jsonl          every utterance found, before filtering
├── train.jsonl            filtered + normalized
├── valid.jsonl            speaker-disjoint held-out split
├── tokenizer.model        sentencepiece, vocab 4000   <- goes in the model config
├── tokenizer.vocab
├── train_aligned.jsonl    + word timestamps           <- data.train_jsonl
└── valid_aligned.jsonl                                <- data.valid_jsonl
```

### Disk and time budget

| | download | on disk after | notes |
|---|---|---|---|
| Mana-TTS | ~35 GB parquet | ~12 GB FLAC | decoded to 24 kHz FLAC; the parquet is deleted unless `--keep-archives` |
| Filimo | ~36 GB tar | ~36 GB mp3 | extracted as-is, no transcode |
| YouTube | ~25 GB tar | ~25 GB mp3 | extracted as-is |
| Common Voice | ~1.5 GB tar | ~1.5 GB mp3 | train+dev+test only |

Budget **200 GB** for a 600 h run and 400 GB if you take everything. Wall clock
on a well-connected VM: 2-4 h of downloading and decoding, then 1-2 h of
alignment on 8 GPUs. Mana-TTS decoding is single-threaded and is the slowest
part per hour of audio.

### Bringing your own data

The manifest format is the contract, and it is four fields:

```json
{"path": "/mnt/data/audio/xyz.mp3", "start": 0.0, "duration": 7.4, "transcript": "متن فارسی", "speaker": "spk_17"}
```

`start`/`duration` are a window into `path`, so long recordings do not need to
be cut up — point many rows at one file. `speaker` is optional for training but
required for a speaker-disjoint valid split and for `eval_fa.py`'s pairing. Add
your rows to `all_raw.jsonl` (or write your own manifest and run
`normalize_fa.py` and `training.scripts.align_data` on it directly) and the rest
of the pipeline works unchanged.

---

## 3. Persian text: the part that is genuinely different

This is where a Farsi run goes wrong in ways an English run cannot. Everything
below is implemented in [`normalize_fa.py`](normalize_fa.py) and unit-tested.

**Same letter, different codepoints.** Arabic `ي` (U+064A) and Persian `ی`
(U+06CC) render identically and are different characters to a tokenizer. So do
`ك`/`ک`, and `ة`/`ه`, `أ`/`إ`/`ٱ`/`ا`. Scraped Persian text mixes them freely.
Unfolded, your tokenizer learns two unrelated spellings of every word and your
aligner drops half of them. The fold is mandatory, not cosmetic.

**Diacritics are optional and inconsistent.** `کتابِ` and `کتاب` are the same
word; the harakat appear in perhaps one transcript in fifty. They are stripped.

**ZWNJ is not whitespace.** The zero-width non-joiner (U+200C) separates
`می‌رود` (one word, "he goes") from `می رود`. It is preserved between letters
and dropped everywhere else (next to a space, at a word edge) where it is a
typing artifact. It is also in the CTC aligner's vocabulary, so it aligns
properly. For WER scoring, `eval_fa.py` treats ZWNJ and space as equivalent —
they sound identical, so an ASR disagreeing about one is not an error.

**Right-to-left text puts punctuation in surprising places.** Common Voice `fa`
is full of rows like `.این چیزی نیست` — the sentence-final period is *first* in
logical order because it renders on the left. Leading punctuation is stripped.

**Digits.** Persian (`۰-۹`), Arabic-Indic (`٠-٩`) and ASCII digits all appear,
often in one corpus. All are converted to words, because a TTS trained on `۱۴۰۲`
learns nothing about how to say it: `۱۴۰۲` → `هزار و چهارصد و دو`, `۲۵٪` → `بیست
و پنج درصد`, `۳٫۵` → `سه ممیز پنج دهم`. Digit runs with a leading zero are read
one digit at a time (`۰۹۱۲` → `صفر نه یک دو`), because those are phone numbers.

**What it deliberately does not do.** Persian orthography does not write short
vowels, so `کرم` is *kerm* / *karam* / *kerem* depending on context, and the
*ezafe* (the linking `-e` between a noun and its modifier) is unwritten
entirely. The model learns both statistically from audio, the way a Persian
reader does — this is the single biggest source of residual pronunciation errors
in a grapheme-input Persian TTS, and more data is the cure.

Confirmed in practice: a model trained with this pipeline, otherwise judged
excellent by a native speaker, renders `حملات برون‌مرزی` with a short pause
where the ezafe belongs instead of joining them as *hamalāt-e borun-marzi*.
Note that you cannot work around it by writing the kasre in the input, because
`normalize_fa.py` strips harakat -- and it must, since training transcripts
carry them perhaps once in fifty. Fixing it properly means marking the ezafe in
the training transcripts and at inference, i.e. a phoneme front-end. A phoneme front-end
(a G2P with diacritic/ezafe restoration) would fix it properly, but it has to be
applied consistently to the tokenizer, the aligner *and* every inference call —
that is a project of its own, not a config flag. Start with graphemes.

**Check your corpus before you train on it:**

```bash
uv run python -m training.farsi.normalize_fa data/farsi_600h/train.jsonl --stats-only
```

prints the rejection reasons and a full character histogram, flagging anything
that escaped normalization. Nothing outside the Persian alphabet, ZWNJ, space
and `.،؛؟!:` should survive.

### The forced aligner

Alignment is what lets the dataloader cut an utterance at a word boundary, and
it needs a CTC model **for the language**. The default,
`m3hrdadfi/wav2vec2-large-xlsr-persian-v3`, has a 40-token vocabulary that is
exactly the Persian alphabet plus ZWNJ and `|` — the same character set
`normalize_fa.py` emits, which is not a coincidence. Alternatives with the
identical vocabulary: `SLPL/Sharif-wav2vec2`,
`masoudmzb/wav2vec2-xlsr-multilingual-53-fa`. Avoid
`jonatasgrosman/wav2vec2-large-xlsr-53-persian` for this purpose — its vocab
mixes Arabic and Persian letter forms and Latin characters, so the folding
above works against it.

If `align_data.py` starts logging *"N of the first M utterances failed to
align — is `--model` the right language?"*, the alphabet in that message and
your transcripts have diverged; that is a normalization bug, not an audio
problem.

---

## 4. Wiring the configs to your data

Four paths have to agree. `prepare_data_fa.py` prints them at the end; three of
them are one-line edits:

| what | where | value |
|---|---|---|
| tokenizer | `configs/model_farsi.yaml` → `flow_lm.lookup_table.tokenizer_path` | `data/farsi_600h/tokenizer.model` |
| vocab size | `configs/model_farsi.yaml` → `flow_lm.lookup_table.n_bins` | exactly your `--vocab-size` (4000) |
| train data | `configs/lsd_scratch_fa.yaml` → `data.train_jsonl` | `data/farsi_600h/train_aligned.jsonl` |
| valid data | `configs/lsd_scratch_fa.yaml` → `data.valid_jsonl` | `data/farsi_600h/valid_aligned.jsonl` |

`n_bins` is asserted against the tokenizer's real vocab size at load time
(`pocket_tts/conditioners/text.py`), so a mismatch fails immediately rather
than training a broken embedding table.

The configs shipped here:

| config | what it trains |
|---|---|
| `model_farsi.yaml` | model definition: 6 layers, Persian tokenizer, public Mimi weights |
| `model_farsi_24l.yaml` | same, 24 layers — needed to *generate* from a teacher checkpoint |
| `lsd_scratch_fa.yaml` | path A stage 1: 24-layer teacher, 400k steps, effective batch 64 |
| `lsd_depth_distill_fa.yaml` | path A stage 2: distil to 6 layers, bake in CFG |
| `finetune_fa.yaml` | path B: warm-start from the released English 24-layer model |
| `model_farsi_release.yaml` | template for publishing your model on HuggingFace |

> **Gotcha, learned the hard way:** `model_overrides` in a *training* config
> (e.g. `num_layers: 24`) are invisible to `pocket-tts generate`, which builds
> the model from the *model* config alone. Generating from a teacher checkpoint
> with `model_farsi.yaml` fails with a pile of `size mismatch` errors. Use
> `model_farsi_24l.yaml` for the teacher, `model_farsi.yaml` for the student.

`weights_path` in `model_farsi.yaml` points at
`kyutai/pocket-tts-without-voice-cloning`, which is **public**, rather than
`kyutai/pocket-tts`, which is **gated** and returns a 403 without an accepted
license and a token. Training from scratch only reads the `mimi.*` tensors out
of that file (Mimi is an audio codec; it has no idea what language it is
encoding), so the public copy is sufficient. Path B does need the gated
weights: accept the license on the model page, then `hf auth login`.

---

## 5. GCP: machines, quota, cost

### Which machine

| machine | GPUs | steps/s at eff. batch 64 | to 400k steps | ~on-demand | ~Spot |
|---|---|---|---|---|---|
| `a3-highgpu-8g` | 8×H100 80 GB | 6.85 | **~16 h** | ~$88/h | ~$30-40/h |
| `a3-highgpu-2g` | 2×H100 80 GB | 3.35 | ~33 h | ~$22/h | ~$8-11/h |
| `a2-ultragpu-1g` | 1×A100 80 GB | ~1.2 (est.) | ~90 h | ~$5/h | ~$1.6/h |
| `g2-standard-8` | 1×L4 24 GB | 0.35 | ~315 h | ~$0.9/h | ~$0.3/h |

Steps/s are the measured numbers from [`../README.md`](../README.md#reproducing-our-results)
(the A100 row is interpolated). Prices are order-of-magnitude for `us-central1`
in 2026 — check the [pricing calculator](https://cloud.google.com/products/calculator),
they move.

**Recommendation: `a3-highgpu-8g` on Spot**, with the launcher in
[`gcp/run_training.sh`](gcp/run_training.sh). The full recipe is ~24 h of
wall clock, which is ~$2,100 on-demand or ~$800-1,000 on Spot. The L4 row is
only there for a pilot; do not plan a 400k-step run on one.

Per-GPU batch size must be set so that `batch_size × GPUs × grad_accum_steps =
64`. `lsd_scratch_fa.yaml` ships `batch_size: 8` for 8 GPUs. On one GPU use 64
(needs ~56 GiB) or 16 with `grad_accum_steps: 4` (~16 GiB). Below an effective
batch of 64 the quality transition at 150-200k steps arrives late or never —
this is the single most important hyperparameter in the whole recipe.

### Quota

H100 quota is not granted by default. In *IAM & Admin → Quotas*, request:

* `NVIDIA_H100_GPUS` in your target region (ask for 8; `a3-highgpu-8g` is
  all-or-nothing), and
* `PREEMPTIBLE_NVIDIA_H100_GPUS` if you plan to use Spot — it is a **separate**
  quota, and Spot capacity is frequently available when on-demand is not.

H100s live in `us-central1`, `us-east4`, `us-east5`, `europe-west4` and
`asia-northeast1` among others; availability differs per zone, so be ready to
try several. If capacity is the blocker rather than quota, look at
[Dynamic Workload Scheduler](https://cloud.google.com/blog/products/compute/introducing-dynamic-workload-scheduler)
(flex-start / calendar mode), which queues for capacity instead of failing.

### Creating the VM

```bash
# Pick a current Deep Learning VM image rather than trusting a family name
# from a document -- they are re-cut regularly:
gcloud compute images list --project deeplearning-platform-release \
    --filter="family~'common-cu12'" --format="value(family)" | sort -u

gcloud compute instances create fa-tts-train \
    --zone=us-central1-a \
    --machine-type=a3-highgpu-8g \
    --image-family=<the family you picked> \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --boot-disk-size=200GB --boot-disk-type=pd-balanced \
    --create-disk=name=fa-data,size=1000GB,type=pd-ssd,auto-delete=no \
    --scopes=https://www.googleapis.com/auth/cloud-platform
    # add for Spot:
    # --provisioning-model=SPOT --instance-termination-action=STOP
```

Then on the VM:

```bash
nvidia-smi                                             # 8 GPUs, driver loaded
sudo mkfs.ext4 -F /dev/disk/by-id/google-fa-data       # first boot only
sudo mkdir -p /mnt/data && sudo mount /dev/disk/by-id/google-fa-data /mnt/data
sudo chown "$USER" /mnt/data

curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/kyutai-labs/pocket-tts && cd pocket-tts
uv sync
uv run python -c "import torch; print(torch.cuda.device_count(), torch.version.cuda)"
```

If `torch.cuda.is_available()` is `False`, the PyPI torch build is newer than
the VM's driver — see the *"Using older cuda versions"* section of
[`../README.md`](../README.md#installation) for pinning a matching CUDA wheel.

**Keep data and checkpoints off the boot disk.** Put `--audio-out` on
`/mnt/data` and either put `run_dir` there too or sync it to GCS.

### Surviving preemption

`train.py` resumes from the newest checkpoint in `run_dir` automatically, and
`ckpt_freq: 2500` means a preemption costs at most ~7 minutes of 8×H100 work.
What it needs is (a) something to restart it and (b) `run_dir` still existing:

```bash
gcloud storage buckets create gs://my-pocket-tts-fa --location=us-central1

nohup ./training/farsi/gcp/run_training.sh \
    training/farsi/configs/lsd_scratch_fa.yaml gs://my-pocket-tts-fa &
```

The script restarts `torchrun` until the run reaches `max_steps`, mirrors
`run_dir` to GCS after every attempt, and pulls it back down first if the local
copy is gone — so a *new* VM after a preemption resumes rather than restarts.
Run it under systemd if you want the VM's own restart to re-launch it.

---

## 6. Training

```bash
# single GPU
uv run python training/train.py training/farsi/configs/lsd_scratch_fa.yaml
# multi-GPU
uv run torchrun --nproc-per-node 8 training/train.py training/farsi/configs/lsd_scratch_fa.yaml
```

What lands in `runs/lsd_scratch_fa/`: rolling `checkpoint_*.pt` (resume state),
`model.safetensors` (EMA weights in the inference format), `progress.jsonl`
(machine-readable log), `samples/` (rank 0 synthesizes the Persian sentences
from the config every 10k steps), `logs/`.

**What to watch, in order:**

1. **First 2k steps** — `flow_loss` should be falling to ~0.35-0.4 with
   `grad` around 1. If `grad` is spiking or `loss` is flat, stop; something is
   wrong with the data, not the schedule.
2. **~10k steps** — the first `samples/*.wav`. It will be babble, but it should
   be *Persian-sounding* babble in the voice of the prompt. If it sounds like a
   different language's phonotactics, your transcripts and audio are misaligned.
3. **~15-50k steps** — intelligibility arrives. Words become recognizable; run
   the first eval around 50k.
4. **150-200k steps** — the acoustic-quality transition. This is the step range
   the whole recipe is built around, and it is why `max_steps: 400000` is not
   negotiable: shortening the schedule, or decaying the LR to zero before
   ~200k, means it never happens and you ship a flat, dull voice.
5. **400k** — expressivity is still improving up to here.

The four knobs that decide whether the transition happens at all — LR `2e-4`,
effective batch ≥64, `flow_batch_multiplier: 4`, lognormal t-sampling — are
already set in `lsd_scratch_fa.yaml`. The notes in
[`../README.md`](../README.md#notes-on-hyperparameters) explain what each costs
if you change it. Do not change them for a first Farsi run; you would be
debugging two things at once.

No Tensorboard/W&B integration ships with the repo (by design), but
`progress.jsonl` is one JSON object per event — ask your coding agent to wire it
to whatever you use.

---

## 7. Evaluating a Farsi model

```bash
uv run python -m training.farsi.eval_fa runs/lsd_scratch_fa \
    --manifest data/farsi_600h/valid_aligned.jsonl \
    --use-ema --reference-floor --num-items 500
```

The script pairs held-out utterances *within* each speaker — one is the voice
prompt, the other supplies the text to synthesize and the real recording to
compare against — then reports, into `runs/<run>/fa_eval_step*/results.json`:

| metric | what it means for Farsi |
|---|---|
| `wer` | intelligibility, via whisper-large-v3 pinned to `language="fa"` |
| `wer_floor` | **the same ASR's WER on the real recordings.** Read this first. |
| `sim` | speaker similarity (WavLM x-vector). Transfers to Persian fine. |
| `utmos` | audio quality. Trained on English MOS ratings — use it to compare *your own* checkpoints, never as an absolute Persian MOS. |
| `silent` / `no_eos` | length failures: generations that came out empty, or that ran to `--max-sec` without emitting EOS |

**`wer_floor` is the point.** The English model reports 0.8% WER because English
ASR is nearly perfect; Persian ASR is not, and on real Persian recordings
whisper-large-v3 typically errs on the order of tens of percent. Your model's
WER is competing with that floor, not with zero. A run whose `wer` is close to
`wer_floor` is as intelligible as the recordings it learned from, which is the
actual goal. Chasing a low absolute number on a Persian corpus means chasing
the ASR's errors.

Sampling settings matter and are worth a small sweep on **one pinned
`--checkpoint`** (the default picks the newest, which moves under you):
`--temp` around 0.3, `--cfg 2.0` for a teacher and `1.0` for a distilled
student, and `--eos-threshold`, which trades the two length failure modes
against each other — less negative and the model runs past the text, more
negative and it stops mid-sentence. Both are counted in `results.json`.

Other Persian ASR options for `--asr`: `SLPL/Sharif-wav2vec2` (fast, no
punctuation), `nvidia/stt_fa_fastconformer_hybrid_large` (needs NeMo),
`steja/whisper-large-persian`. Whichever you pick, keep it fixed — WER is only
comparable within one ASR.

---

## 8. Do a 200-hour pilot first

The full recipe is a day of 8×H100 time and most of what goes wrong is data,
not training. So:

```bash
uv run python -m training.farsi.prepare_data_fa --hours 200 \
    --sources manatts,filimo --manifests-out data/farsi_200h --align-shards 8
# in lsd_scratch_fa.yaml: point data.* at data/farsi_200h, set max_steps: 60000
uv run torchrun --nproc-per-node 8 training/train.py training/farsi/configs/lsd_scratch_fa.yaml
```

~2.5 h of training. You will not get a good model — you will get intelligible
Persian speech, which is enough to confirm that the transcripts match the audio,
the aligner worked, the tokenizer is sane and the voice prompts sound like the
speakers. Then re-run the data prep at 600-1000 h and train for real. Use a
separate `run_dir` for the pilot; the real run must not resume from it.

---

## 9. Distillation and shipping

The 6-layer student is the model that actually runs on a CPU at ~6× real time —
the 24-layer teacher exists to train it. Point
`configs/lsd_depth_distill_fa.yaml` at your finished teacher checkpoint
(`distill_teacher_weights`) and run it: 200k steps, ~3 h on 8×H100. WER and
speaker similarity reach parity with the teacher by ~40k steps, but prosody
keeps settling, so let it run. With `distill_cfg_coef: 2.0` the student
generates at `--cfg 1` — one backbone pass per step — at the quality of guided
sampling.

To publish, upload to a HuggingFace repo:

* `model.safetensors` — the EMA export from `run_dir` (already in the inference
  format: `flow_lm.*` + `mimi.*` keys)
* `tokenizer.model` — your sentencepiece model
* `farsi.yaml` — start from
  [`configs/model_farsi_release.yaml`](configs/model_farsi_release.yaml), fill in
  the two `TODO` paths, and pin both with `@<commit sha>` so a later push cannot
  change what users get

Document the generation defaults you settled on in the model card — they are
per-call flags, not fields the config file carries. For a model trained with
this pipeline, `--frames-after-eos 0` is worth naming: the dataloader trims
each training utterance to the last aligned word plus `TRAIL_SEC = 0.2`
seconds, and in narration and film dialogue that tail usually contains the
speaker's intake of breath, so the model learns to end utterances with one.
Trimming the post-EOS frames removes it without risking the final word, which
a more negative `--eos-threshold` can clip.

Long text needs chunking, and `training/farsi/synthesize.py` handles it: it
splits at Persian punctuation, keeps every chunk under pocket-tts's 50-token
limit, re-uses one voice-prompt state so the speaker stays constant, and joins
the pieces with a short silence. Two settings were tested by ear on a real
model and are worth carrying into the model card:

* **A little silence at every join sounds better than none.** Chunks are
  generated independently, so butting them together exposes the seam. 0.15 s
  hides it.
* **Do not force a chunk boundary at every comma.** It leaves chunks ending
  mid-clause, which no training utterance ever did (the dataloader cuts at the
  last aligned word), and the whole chunk degrades reproducibly.

Then anyone can run:

```bash
uvx pocket-tts generate --config hf://<your_user>/<your_repo>/farsi.yaml \
    --voice voice.wav --text "سلام، حال شما چطور است؟"
```

For a point-and-click alternative, [`space/`](space/) has a small Gradio app
(same chunking, same normalization) that runs locally — see
[`space/README.md`](space/README.md). Or skip running anything and just listen:
[`examples/`](examples/) has five clips generated with the released model and
default settings.

Kyutai will feature community models that cover new languages — Farsi is not
one of the six official languages, so a working model is exactly what they ask
for in [`../README.md`](../README.md#models-trained-by-the-community). Open a PR.

---

## 10. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| aligner logs *"is `--model` the right language?"* | transcripts contain characters outside the CTC vocab | run `normalize_fa.py --stats-only`; anything flagged `UNEXPECTED` is the bug |
| tokenizer training fails: `sentences_.empty()` | the train manifest is empty | check the filter counts — usually every row was held out or rejected |
| `size mismatch` when generating | model config's depth ≠ the checkpoint's | use `model_farsi_24l.yaml` for teacher checkpoints |
| `403 GatedRepoError` on `kyutai/pocket-tts` | gated repo | accept the license on the model page, then `hf auth login`; or use a config pointing at `pocket-tts-without-voice-cloning` |
| `n_bins` assertion at startup | config vocab ≠ tokenizer vocab | set `n_bins` to exactly `--vocab-size` |
| speech does not follow the text | inaccurate transcripts | drop `youtube`, keep `manatts`+`filimo`; auto-generated subtitles are the usual culprit |
| output is noisy/low quality | the model mimics its voice prompt | use a clean prompt at inference; check how much of your corpus is film audio with music |
| generations never stop | trailing silence in training audio, or bad alignment | alignment trims to the last aligned word — verify `words` exists on your rows |
| words cut off at the start/end | alignment is off | check a few rows' timestamps against the audio by ear |
| model sounds like someone reading a book | Mana-TTS is magazine narration and dominates a small corpus | add conversational data (Filimo, YouTube) |
| quality never lifts off after 200k steps | effective batch < 64, or LR ≠ 2e-4, or the schedule was shortened | see §6 |
| `torch.cuda.is_available()` is `False` | torch built for a newer CUDA than the driver | pin a matching wheel index, see `../README.md` |

Persian-specific sanity check, any time: `--stats-only` on your manifest, plus
listening to five random `samples/*.wav`. Almost every Farsi-specific failure
shows up in one of those two.

---

## 11. What is verified here, and what is not

Verified by running it (on CPU, on a small slice of Common Voice `fa`):

* the whole data path — download → extract → manifest → normalize → filter →
  speaker-disjoint split → sentencepiece tokenizer
* Persian forced alignment with `m3hrdadfi/wav2vec2-large-xlsr-persian-v3`,
  producing sane per-word timestamps on Persian audio
* `training/train.py` training steps against a Farsi config, checkpointing and
  exporting `model.safetensors`
* `pocket-tts generate` with a Farsi config + checkpoint
* `eval_fa.py` end to end, including `wer_floor`
* the dataset hours, licenses and formats quoted in §2, and the tokenizer
  fertility numbers in §1
* `uv run pytest training/tests -q` — 89 passed

Not verified, because it needs the GPUs and the days:

* the training-time and cost table in §5 is the repo's measured English numbers
  applied to a Farsi run of the same shape
* no Farsi WER / similarity / UTMOS numbers are quoted anywhere in this
  document, because none have been measured. Establish your own baseline with
  `--reference-floor` at 50k steps and compare against that.
