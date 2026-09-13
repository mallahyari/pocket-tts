---
language: fa
license: cc-by-nc-4.0
library_name: pocket-tts
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - persian
  - farsi
  - voice-cloning
  - speech-synthesis
---

# Pocket-TTS Farsi v2

A 109.5 M-parameter Persian text-to-speech model with voice cloning from a short
audio prompt. Runs on a CPU. Trained on 973 hours of public Persian speech from
2,978 speakers.

**Code, training pipeline and full results:**
[github.com/mallahyari/pocket-tts](https://github.com/mallahyari/pocket-tts)
— see [`training/farsi/v2/RESULTS.md`](https://github.com/mallahyari/pocket-tts/blob/main/training/farsi/v2/RESULTS.md)
for the measured numbers behind everything below.

---

## ⚠️ This model takes phonemes, not Persian script

v2 is trained on romanised phonemes. **Persian text passed directly produces
silence** — every Persian character maps to the unknown token, the model is
conditioned on nothing, and it emits end-of-speech immediately. It fails
quietly, with no error.

Phonemise first with
[mehdi-hf/Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF):

```python
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from normalize_fa import normalize_for_model   # shipped in this repo

G2P = "mehdi-hf/Homo-GE2PE-Persian-HF"
tok = AutoTokenizer.from_pretrained(G2P)
g2p = T5ForConditionalGeneration.from_pretrained(G2P).eval()

TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

def phonemise(text: str) -> str:
    # Normalise FIRST. G2P has no reading for digits and drops them without a
    # word: "تا سال ۲۰۳۰" comes back as "tA sAle  " with the year simply gone.
    # normalize_fa.py (shipped in this repo) spells numbers out, unifies the
    # Arabic/Persian letter variants and strips the characters the tokenizer
    # has no entry for.
    text = normalize_for_model(text)
    # "؟" shares a symbol with the glottal stop in this notation; drop it next
    text = text.replace("؟", "").replace("?", "")
    enc = tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    # KEEP the "1": it marks the ezafe. "?eqtesAde1 ?AmrikA" is eqtesad-E
    # amrika, one bound phrase. synthesize.py uses it to avoid splitting a
    # chunk between the two, then strips it before the model sees the text.
    # If you drive the model yourself, strip it: .replace("1", "")
    return raw.translate(TO_PHONEMES)

phonemise("سلام، حال شما چطور است؟")
# 'salAm hAle SomA Cetor ?ast'
phonemise("اقتصاد آمریکا را تغییر دهد")
# '?eqtesAde1 ?AmrikA rA taqir dahad'   <- the 1 is the ezafe
phonemise("تا سال ۲۰۳۰ تغییر دهد.")
# 'tA sAle do hezAr ?o si taqir dahad'
```

### Install

`model.yaml` here sets three flags that the released `pocket-tts` package does
not know about yet, and its config rejects unknown keys — `pip install
pocket-tts` will fail on this model with
`ValidationError: capitalize_first_letter — Extra inputs are not permitted`.
Until the change is upstream, install the fork that carries it:

```bash
pip install "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main"
```

A loud error is deliberate here. The alternative — a config the old package
accepts — is the old package silently deleting the first word of everything you
generate, which is the bug those flags exist to fix.

### The config must switch off the orthographic text frontend

`model.yaml` in this repo carries three flags, and they matter:

```yaml
capitalize_first_letter: false
append_terminal_punctuation: false
pad_with_spaces_for_short_inputs: false
```

Without the first one the text frontend upper-cases the opening letter of every
chunk. That is right for a Latin-script language and wrong here, because these
Latin letters are phonemes: `man` becomes `Man`, `M` is not in the vocabulary,
and the first word is silently dropped. `salAm` becomes `SalAm`, and since `S`
is the symbol for *sh*, the model says /shalaam/. Use the `model.yaml` shipped
here rather than writing your own.

Then synthesise, passing `--no-normalize-text` so the phonemes survive:

```bash
python -m training.farsi.synthesize \
    --config model_farsi_ph.yaml \
    --voice prompt.wav \
    --no-normalize-text \
    --text "salAm hAle SomA Cetor ?ast" \
    --out hello.wav
```

The notation: `A` long ā, `a` short a, `?` glottal stop (ع/ء), `S` š, `C` č,
`;` ž, `x` خ, `q` ق/غ.

### Long text: split into sentences first

G2P discards punctuation, so a phonemised paragraph has no sentence boundaries
left in it and anything chunking the result cuts on token count alone, landing
mid-sentence. Split the Persian into sentences first, phonemise each on its
own, synthesise each, and join with a pause of about 0.25 s:

```python
import re
sentences = [s.strip() for s in re.split(r"(?<=[.!؟])\s+", article) if s.strip()]
pieces = [synthesise(phonemise(s)) for s in sentences]
```

**Do this for prosody, not for accuracy.** Measured on a five-sentence
paragraph over three seeds each, the two routes are indistinguishable on word
error rate — 0.736 median (0.660–0.792) phonemising the paragraph whole against
0.792 (0.717–0.811) per sentence. The spread between seeds is wider than the
gap between approaches. What splitting by sentence does buy is that pauses land
at sentence ends instead of wherever the token budget ran out, and chunk
boundaries stop cutting through clauses.

**Retry a runaway rather than shipping it.** A generation that never emits
end-of-speech runs to the length cap and repeats itself — `fanAvari` comes back
as `fanAvariiiiii…`. It is stochastic and a second attempt usually terminates,
so compare the output against the cap (`tokens / 3.0 + 2.0` seconds) and
regenerate when it exceeds it. This matters more than how you chunk.

**Chunk length is the real constraint, not document length.** Training
utterances averaged about 11 tokens. Nine to sixteen is the clean band, 18 is a
safe budget, and at 21 and above generations stop terminating. Long documents
are fine; unbroken text with no sentence structure is not.

---

## Samples

Every clip is cloned from a held-out Common Voice speaker the model never
trained on. The prompt is five seconds; the generated speech follows.

### Greeting, female voice

سلام، حال شما چطور است؟ امیدوارم روز خوبی داشته باشید.

<table><tr>
<td><b>voice prompt</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/prompt_hello.wav"></audio></td>
<td><b>generated</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/hello.wav"></audio></td>
</tr></table>

### One sentence, female voice

مادر کتاب را روی میز اتاق گذاشت و پنجره را باز کرد.

<table><tr>
<td><b>voice prompt</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/prompt_short_sentence.wav"></audio></td>
<td><b>generated</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/short_sentence.wav"></audio></td>
</tr></table>

### News sentence, male voice

شرکت آنتروپیک مدلی اقتصادی منتشر کرده که نشان می‌دهد هوش مصنوعی چگونه می‌تواند اقتصاد آمریکا را تا سال ۲۰۳۰ تغییر دهد.

<table><tr>
<td><b>voice prompt</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/prompt_news_paragraph.wav"></audio></td>
<td><b>generated</b><br><audio controls preload="none" src="https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2/resolve/main/samples/news_paragraph.wav"></audio></td>
</tr></table>

If the players do not appear, the audio files are under
[`samples/`](./tree/main/samples) and play in any browser.

---

## Training data

| source | hours | share |
|---|---|---|
| farsi-asr YouTube | 595 | 61% |
| YouTube (v1 set) | 175 | 18% |
| Filimo subtitles | 143 | 15% |
| Mana-TTS studio | 60 | 6% |

973 hours, 511,403 utterances, 2,978 speakers — against 497 hours and 1,100
speakers for v1. The 1,878 new voices are why v2 clones a wider range of
speakers, including male voices that v1, dominated by a single studio narrator,
handled poorly.

Windows whose audio began mid-word were repaired or dropped (119 hours), and
transcripts were phonemised with question marks stripped before G2P, since that
mark otherwise becomes a glottal stop the speaker never uttered.

---

## Known limitations

**The first word of a chunk is still the weak point, mildly.** With the config
above, the opening word is rendered correctly about 40 times in 50, against 46
in 50 for the same word one position later. An initial stop consonant is what
slips: `کریس` occasionally comes out as `پریس`. This is a property of the
training recipe, where a target almost never begins at the first word of an
utterance, and v1 has it more severely (27 in 50). Words with a glottal-stop
onset are reliably clean.

If you are generating long text, note that it is split into chunks of ~18
tokens and each chunk is generated fresh, so the first-word position recurs at
every boundary.

**There is no punctuation in this model's vocabulary.** Not one of the 4000
pieces contains a comma or a full stop: the G2P discards punctuation, so the
training text was pure phonemes. You cannot request a pause through the text,
and `--pause-at-punct` has nothing to act on. Phrasing is entirely the model's
own. Keeping the ezafe marks (above) is what stops a chunk boundary landing
inside a noun phrase, which is the one lever you do have.

**Voice prompt choice matters a lot.** Some prompts produce stable output while
others cause the model to continue the prompt's own speech instead of the
requested text. Prompts that end at a natural phrase boundary, around 5 seconds,
work best; `--voice-sec 5` is the default cap for that reason.

**Absolute WER is high on this benchmark.** Whisper's own error floor on the
real Common Voice recordings is 0.273, so a meaningful part of the measured
error is the ASR, not the model. Use these numbers to compare models, not as an
absolute quality score.

**Non-commercial licence**, inherited from the training data.

---

## Model

6-layer FlowLM student, depth-distilled from a 24-layer teacher, plus the Mimi
codec at 24 kHz / 12.5 fps. Identical architecture to v1, so it is a drop-in
replacement *except* for the phoneme input requirement.

Teacher trained 400 k steps; student distilled 200 k steps, checkpoint 152,500
selected on held-out WER across three seeds — not the final step, which scored
worse.
