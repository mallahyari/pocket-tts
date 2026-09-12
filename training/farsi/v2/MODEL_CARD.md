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

G2P = "mehdi-hf/Homo-GE2PE-Persian-HF"
tok = AutoTokenizer.from_pretrained(G2P)
g2p = T5ForConditionalGeneration.from_pretrained(G2P).eval()

TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

def phonemise(text: str) -> str:
    # "؟" shares a symbol with the glottal stop in this notation; drop it first
    text = text.replace("؟", "").replace("?", "")
    enc = tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw.translate(TO_PHONEMES).replace("1", "")

phonemise("سلام، حال شما چطور است؟")
# 'salAm hAle SomA Cetor ?ast'
```

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

---

## Results

Measured on 300 held-out Common Voice pairs — speakers and clips the model has
never seen, no overlap with training. Compared against
[v1](https://huggingface.co/mehdi-hf/pocket-tts-farsi) evaluated identically.

| | v1 | **v2** |
|---|---|---|
| mean WER | 2.048 | **0.582** |
| median per-item WER | 0.333 | 0.333 |
| runaway generations | 25 / 300 | **0–1 / 300** |
| speaker similarity | 0.764 | **0.859** |
| UTMOS (naturalness) | 2.826 | **3.11** |

**A typical utterance is about as accurate as v1's** — the medians are level.
What changed is that the catastrophic failures largely stop: v1 ran past the end
of the text on 25 of 300 items, repeating itself or drifting into the voice
prompt's words. v2 does that on zero to one. That, plus noticeably better voice
cloning, is the difference you hear.

Figures are the mean of three evaluation runs. Single runs of this metric vary
by up to 0.3 WER, because one runaway generation carries an enormous insertion
count.

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

**The first word of each chunk is the weak point.** Utterance-initial
consonants are sometimes softened or dropped — `کریس` can render as `ریس`,
`من` as `این`. Long text is split into chunks of ~18 tokens and each chunk is
generated fresh, so this recurs at every boundary. Words with a glottal-stop
onset (`?emruz`, `?in`) are reliably clean.

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
