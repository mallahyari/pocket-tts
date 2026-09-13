# Pocket TTS

> [!TIP]
> **This fork adds a Farsi (Persian) training pipeline and two finished models.**
> The current one is **v2**: 109.5M parameters, CPU-only, voice cloning from a
> five-second prompt, trained on 973 hours of Persian speech from 2,978
> speakers. Try it on the hub —
> [🤗 pocket-tts-farsi-v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)
> — or read [`training/farsi/v2/RESULTS.md`](training/farsi/v2/RESULTS.md) for
> the numbers, the wrong turns, and what they cost.
>
> v1 ([🤗 pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi))
> is kept for anyone depending on it, along with its
> [runbook](training/farsi/RUNBOOK.md) and [results](training/farsi/RESULTS.md).

## Farsi v2

Measured on 300 held-out Common Voice pairs — speakers and clips neither model
ever trained on, scored identically:

| | v1 | **v2** |
|---|---|---|
| mean WER | 2.048 | **0.582** |
| median per-item WER | 0.333 | 0.333 |
| runaway generations | 25 / 300 | **0–1 / 300** |
| speaker similarity | 0.764 | **0.859** |
| UTMOS (naturalness) | 2.826 | **3.11** |
| training data | 497 h, 1,100 speakers | **973 h, 2,978 speakers** |

A typical utterance is about as accurate in both; the medians are level. What
changes is that the catastrophic failures stop. v1 runs past the end of the text
on 25 of 300 items, repeating itself or drifting into the prompt's words. v2
does that on zero to one.

**v2 takes romanised phonemes, not Persian script.** Persian text passed
directly produces silence — every character maps to the unknown token and the
model emits end-of-speech immediately. Phonemise with
[Homo-GE2PE](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) first; the
[model card](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) has the code,
and [`training/farsi/v2/space/`](training/farsi/v2/space/) is a Gradio app that
wires the whole pipeline together.

Install note: v2's `model.yaml` sets `capitalize_first_letter`, which the
released `pocket-tts` rejects as an unknown key ([PR
#307](https://github.com/kyutai-labs/pocket-tts/pull/307) adds it upstream).
Until that lands, use this fork:

```bash
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main"
```

### Speaking more than one sentence

Split the Persian into sentences, phonemise each on its own, synthesise each,
and join with a pause of about 0.25 s. G2P discards punctuation, so a
phonemised paragraph has no sentence boundaries left in it and any chunker is
then cutting on token count alone, landing mid-sentence.

```python
import re
sentences = [s.strip() for s in re.split(r"(?<=[.!؟])\s+", article) if s.strip()]
pieces = [synthesise(phonemise(s)) for s in sentences]
```

Do this for **prosody, not accuracy**. Measured on a five-sentence paragraph
over three seeds each, the two routes are indistinguishable on word error rate
— 0.736 median (0.660–0.792) phonemising the paragraph whole against 0.792
(0.717–0.811) per sentence, a spread wider than the gap. What it buys is that
pauses land at sentence ends rather than wherever the token budget ran out.

Two things matter more than how you chunk:

- **Retry a runaway rather than shipping it.** A generation that never emits
  end-of-speech runs to the length cap and repeats itself — `fanAvari` comes
  back as `fanAvariiiiii…`. It is stochastic, and a second attempt usually
  terminates. Compare the output against `tokens / 3.0 + 2.0` seconds and
  regenerate when it exceeds that.
- **Chunk length is the constraint, not document length.** Training utterances
  averaged ~11 tokens. Nine to sixteen is the clean band, 18 is a safe budget,
  and at 21 and above generations stop terminating. Long documents are fine;
  unbroken text with no sentence structure is not.

## Farsi v1 examples

Kept for reference. Five clips generated with the **v1** model
([mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi)),
its one bundled voice, default settings, no cherry-picking or re-takes — what
you hear is what `farsi_tts.py` or the [Gradio app](training/farsi/space/) gives
you out of the box. Click a file to open GitHub's built-in audio player.

| clip | text | notes |
|---|---|---|
| [`01_greeting.wav`](training/farsi/examples/01_greeting.wav) | سلام، حال شما چطور است؟ این صدای مصنوعی فارسی است. | short greeting |
| [`02_weather.wav`](training/farsi/examples/02_weather.wav) | امروز هوای تهران آفتابی است و دمای هوا به بیست و پنج درجه می‌رسد. | numbers spelled out by the normalizer (۲۵ → بیست و پنج) |
| [`03_tech.wav`](training/farsi/examples/03_tech.wav) | این یک نمونه است از تبدیل متن فارسی به گفتار روی پردازنده مرکزی، بدون نیاز به کارت گرافیک. | one sentence, single chunk |
| [`04_literature.wav`](training/farsi/examples/04_literature.wav) | شعر و ادبیات فارسی یکی از غنی‌ترین میراث‌های فرهنگی جهان است. | formal register |
| [`05_paragraph.wav`](training/farsi/examples/05_paragraph.wav) | زبان فارسی یکی از قدیمی‌ترین زبان‌های زنده جهان است. این زبان قرن‌ها تاریخ و فرهنگ غنی را در خود حفظ کرده و امروز میلیون‌ها نفر در ایران، افغانستان و تاجیکستان به آن سخن می‌گویند. | two sentences, so `synthesize.py`'s chunker splits and rejoins with a 0.15s pause — listen for the seam at "می‌گویند" |

Generated with:

```bash
uv run farsi_tts.py --text "<one row from the table above>"
```

(`farsi_tts.py` is the standalone script from the model card, not part of this
repo — see [`training/farsi/README.md`](training/farsi/README.md) or the
[model card](https://huggingface.co/mehdi-hf/pocket-tts-farsi) to get it.)

**Looking for more Persian training data?** We surveyed four more public datasets as
candidates for a fine-tune — one turned out to be sung music mislabeled as speech, one
has a license that rules it out despite being the best raw audio we found, one is
noisier than what's already in the model, and one looks genuinely promising. Full
writeup, including the license checks, the ASR-based accuracy measurements, and a real
test-methodology bug we caught and fixed along the way, is in
[`training/farsi/DATASET_SURVEY.md`](training/farsi/DATASET_SURVEY.md).

<img width="1446" height="622" alt="pocket-tts-logo-v2-transparent" src="https://github.com/user-attachments/assets/637b5ed6-831f-4023-9b4c-741be21ab238" />

A lightweight text-to-speech (TTS) application designed to run efficiently on CPUs.
Forget about the hassle of using GPUs and web APIs serving TTS models. With Kyutai's Pocket TTS, generating audio is just a pip install and a function call away.

Supports Python 3.10, 3.11, 3.12, 3.13 and 3.14. Requires PyTorch 2.5+. Does not require the gpu version of PyTorch.

[🔊 Demo](https://kyutai.org/pocket-tts) | 
[🐱‍💻GitHub Repository](https://github.com/kyutai-labs/pocket-tts) | 
[🤗 Hugging Face Model Card](https://huggingface.co/kyutai/pocket-tts) | 
[⚙️ Tech report](https://kyutai.org/blog/2026-01-13-pocket-tts) |
[📄 Paper](https://arxiv.org/abs/2509.06926) | 
[📚 Documentation](https://kyutai-labs.github.io/pocket-tts/)

> [!NOTE]
> **New (August 2026):** We've released the training code! Check out [`training/`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) to start training your own models.
> Open a PR to add your model to the [Models trained by the community](#models-trained-by-the-community) section.


## Main takeaways
* Runs on CPU
* Small model size, 100M parameters
* Audio streaming
* Low latency, ~200ms to get the first audio chunk
* Faster than real-time, ~6x real-time on a CPU of MacBook Air M4
* Uses only 2 CPU cores
* Python API and CLI
* Voice cloning
* Multi-language support: english, french, german, portuguese, italian, spanish
* Can handle infinitely long text inputs
* [Can run on client-side in the browser](#in-browser-implementations)

Additional languages may be added in the future.

## Trying it from the website, without installing anything

Navigate to the [Kyutai website](https://kyutai.org/pocket-tts) to try it out directly in your browser. You can input text, select different voices, and generate speech without any installation.

## Trying it with the CLI

### The `generate` command
You can use pocket-tts directly from the command line. We recommend using
`uv` as it installs any dependencies on the fly in an isolated environment (uv installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)).
You can also use `pip install pocket-tts` to install it manually.
On Linux, see [CPU-only installation](#cpu-only-installation) to avoid pulling in the CUDA build of PyTorch.

This will generate a wav file `./tts_output.wav` saying the default text with the default voice, and display some speed statistics.
```bash
uvx pocket-tts generate
# or if you installed it manually with pip:
pocket-tts generate
```
Modify the voice with `--voice` and the text with `--text`. We provide a small catalog of voices.
Choose a pretrained language model with `--language` when running `generate`, `export-voice`, or `serve` (default: `english`). Non-english languages have also biggers 24 layers variants that are higher quality but slower. You can select them by using for example `--language italian_24l`.
The `--config` option accepts a local YAML path, an `https://` URL, or an `hf://` path (e.g. `hf://<repo_id>/<path>[@revision]`) for custom weights.

You can take a look at [this page](https://huggingface.co/kyutai/tts-voices) which details the licenses
for each voice.

* [alba](https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/casual.wav) (en)
* [giovanni](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/common_voice_it_36520747-enhanced-v2.mp3) (it)
* [lola](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/common_voice_es_19762977-enhanced-v2.mp3) (es)
* [juergen](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/de-DE-juergen.mp3) (de)
* [rafael](https://huggingface.co/kyutai/pocket-tts/blob/add_lang_not_documented/g-Vi8PgmSY0-enhanced-v2.wav) (pt)
* [estelle](https://huggingface.co/kyutai/tts-voices/blob/main/unmute-prod-website/developpeuse-3.wav) (fr)
* [anna](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p228_023_enhanced.wav) (en)
* [azelma](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p303_023_enhanced.wav) (en)
* [bill_boerst](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/bill_boerst.wav) (en)
* [caro_davy](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/caro_davy.wav) (en)
* [charles](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p254_023_enhanced.wav) (en)
* [cosette](https://huggingface.co/kyutai/tts-voices/blob/main/expresso/ex04-ex02_confused_001_channel1_499s.wav) (en)
* [eponine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p262_023_enhanced.wav) (en)
* [eve](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p361_023_enhanced.wav) (en)
* [fantine](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p244_023_enhanced.wav) (en)
* [george](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p315_023_enhanced.wav) (en)
* [jane](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p339_023_enhanced.wav) (en)
* [jean](https://huggingface.co/kyutai/tts-voices/blob/main/ears/p010/freeform_speech_01_enhanced.wav) (en)
* [javert](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Butter.wav) (en)
* [marius](https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav) (en)
* [mary](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p333_023_enhanced.wav) (en)
* [michael](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p360_023_enhanced.wav) (en)
* [paul](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p259_023_enhanced.wav) (en)
* [peter_yearsley](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/peter_yearsley.wav) (en)
* [stuart_bell](https://huggingface.co/kyutai/tts-voices/blob/main/voice-zero/stuart_bell.wav) (en)
* [vera](https://huggingface.co/kyutai/tts-voices/blob/main/vctk/p229_023_enhanced.wav) (en)

The `--voice` argument can also take a plain wav file as input for voice cloning.
You can use your own or check out our [voice repository](https://huggingface.co/kyutai/tts-voices).
We recommend [cleaning the sample](https://podcast.adobe.com/en/enhance) before using it with Pocket TTS, because the audio quality of the sample is also reproduced.

Feel free to check out the [generate documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/generate/) for more details and examples.
For trying multiple voices and prompts quickly, prefer using the `serve` command.

### The `serve` command

You can also run a local server to generate audio via HTTP requests.
```bash
uvx pocket-tts serve
# or if you installed it manually with pip:
pocket-tts serve
```
Navigate to `http://localhost:8000` to try the web interface, it's faster than the command line as the model is kept in memory between requests.

You can check out the [serve documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/serve/) for more details and examples.

### The `export-voice` command

Processing an audio file (e.g., a .wav or .mp3) for voice cloning is relatively slow, but loading a safetensors file -- a voice embedding converted from an audio file -- is very fast. You can use the `export-voice` command to do this conversion. See the [export-voice documentation](https://kyutai-labs.github.io/pocket-tts/CLI%20Commands/export_voice/) for more details and examples.


## Using it as a Python library

You can try out the Python library on Colab [here](https://colab.research.google.com/github/kyutai-labs/pocket-tts/blob/main/docs/pocket-tts-example.ipynb).

Install the package with
```bash
pip install pocket-tts
# or
uv add pocket-tts
```

### CPU-only installation

On Linux, PyPI serves the CUDA build of PyTorch by default, so `pip install pocket-tts` also
downloads the `nvidia-*` CUDA runtime wheels, even though pocket-tts runs on CPU. This adds
several gigabytes to the install (with torch 2.13, roughly 3 GB instead of 200 MB). Installing
from the PyTorch CPU index pulls the CPU build and no NVIDIA packages:
```bash
pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu
```

To run the CLI without installing, pass the same index to `uvx`:
```bash
uvx --index https://download.pytorch.org/whl/cpu pocket-tts generate
```

With `uv`, declare the index explicitly in your project:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

This is not needed on macOS or Windows, where the default PyTorch wheels are already CPU-only.

You can use this package as a simple Python library to generate audio from text.
```python
from pocket_tts import TTSModel
import scipy.io.wavfile

tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "alba"  # One of the pre-made voices, see above
    # You can also use any voice file you have locally or from Hugging Face:
    # "./some_audio.wav"
    # or "hf://kyutai/tts-voices/expresso/ex01-ex02_default_001_channel2_198s.wav"
)
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# Audio is a 1D torch tensor containing PCM data.
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.numpy())
```

You can have multiple voice states around if
you have multiple voices you want to use. `load_model()`
and `get_state_for_audio_prompt()` are relatively slow operations,
so we recommend to keep the model and voice states in memory if you can.

For faster voice loading, you can export voice states to safetensors files:
```python
from pocket_tts import TTSModel, export_model_state

model = TTSModel.load_model()

# Export a voice state for fast loading later
model_state = model.get_state_for_audio_prompt("some_voice.wav")
export_model_state(model_state, "./some_voice.safetensors")

# Later, load it quickly, this is quite fast as it's just reading the kvcache
# from disk and doesn't do any others computations.
model_state_copy = model.get_state_for_audio_prompt("./some_voice.safetensors")

audio = model.generate_audio(model_state_copy, "Hello world!")
```

You can check out the [Python API documentation](https://kyutai-labs.github.io/pocket-tts/API%20Reference/python-api/) for more details and examples.

## Running on GPU

Pocket TTS is designed to run on CPU, and on hardware with strong single-thread CPU performance
(e.g. Apple Silicon) we did not observe a GPU speedup, notably because we use a batch size of 1
and a very small model. However, this turns out to be hardware-dependent: measured on a cloud x86
VM (4 vCPUs) with a Tesla T4, moving the model to GPU gave a consistent ~2.6x speedup over CPU
(RTF ~2.3-2.5x on CPU vs. ~6.28x on GPU, for both short and long input text). If your CPU is
thread-limited or otherwise weaker than a modern laptop chip, it's worth trying the GPU.

This is not officially supported (there is no `device` argument on `TTSModel.load_model()`), but
since `TTSModel` is a regular `nn.Module` you can move it yourself:

```python
tts_model = TTSModel.load_model()
tts_model.to("cuda")
...
audio = tts_model.generate_audio(voice_state, "Hello world, this is a test.")
# generate_audio() returns a tensor on the same device as the model, so on GPU you need
# to move it back to CPU before calling .numpy():
scipy.io.wavfile.write("output.wav", tts_model.sample_rate, audio.detach().cpu().numpy())
```

A few things to be aware of if you want to use the GPU:
- The `generate` CLI command has a `--device` option (defaults to `cpu`, documented in the
  [CLI reference](docs/CLI%20Commands/generate.md) — note that page's own description ("you may not
  get a speedup by using a gpu since it's a small model") is what this section is correcting, based
  on the T4 measurements above); the `serve` command and the Docker image do not expose any device
  option and will always run on CPU.
- `pip install pocket-tts` / `uv add pocket-tts` install whatever `torch` build is current on
  PyPI, which may require a newer CUDA version than your driver supports. In that case
  `torch.cuda.is_available()` silently returns `False` (you'll only see a `UserWarning` about an
  outdated driver, not an error). If this happens, install a `torch` build matching your driver's
  CUDA version explicitly, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- `quantize=True` (int8 dynamic quantization) only works on CPU; calling it on a model moved to
  CUDA raises `NotImplementedError: Could not run 'quantized::linear_dynamic' ... 'CUDA' backend`.
  Separately, the optional `torchao` backend (`pip install pocket-tts[quantize]`) declares
  `torch>=2.11` — fine with a fresh install (torch 2.11+ is on PyPI as of this writing), but if
  you've pinned an older `torch` (e.g. to match an older GPU driver's CUDA build, per the point
  above), adding this extra can pull in a `torchao` that's incompatible with your pinned `torch`
  and break `quantize=True` even on CPU. Match `torchao`'s `torch` requirement to whatever `torch`
  you actually have installed.

## Unsupported features

At the moment, we do not support (but would love pull requests adding):

- [Adding silence in the text input to generate pauses.](https://github.com/kyutai-labs/pocket-tts/issues/6)

We tried running this TTS model on the GPU but did not observe a speedup compared to CPU execution
on hardware with very strong single-thread CPU performance, notably because we use a batch size of
1 and a very small model. See the ["Running on GPU"](#running-on-gpu) section above for measurements
on other hardware and caveats if you want to try it yourself.

## Development and local setup

We accept contributions! Feel free to open issues or pull requests on GitHub.

You can find development instructions in the [CONTRIBUTING.md](https://github.com/kyutai-labs/pocket-tts/tree/main/CONTRIBUTING.md) file. You'll also find there how to have an editable install of the package for local development.

## Upstream ecosystem

This fork trims four sections that catalogue the wider Pocket TTS ecosystem —
in-browser ports, alternative-language implementations, community-trained
models, and projects built on it. They change often and are maintained
upstream, so read them at their source:
[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts#readme).

Everything else below is kept as upstream wrote it.

## Prohibited use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.


## Authors

Manu Orsini*, Simon Rouard*, Gabriel De Marmiesse*, Václav Volhejn, Neil Zeghidour, Alexandre Défossez

*equal contribution
