# Pocket TTS — Farsi (local Gradio app)

A small browser front-end for the Farsi model: type Persian text, pick a
voice, get speech. Same generation logic as the CLI (`training/farsi/synthesize.py`)
and the standalone `farsi_tts.py` script — chunking, Persian normalization,
voice-prompt trimming — just with a UI instead of flags.

This is meant to be run **locally**; it is not deployed as a HuggingFace
Space (Gradio/Docker Spaces dropped their free CPU tier, so hosting this
would require a paid plan). Running it on your own machine costs nothing and
works fine on CPU.

## Run it

From the repo root:

```bash
pip install -r training/farsi/space/requirements.txt
python training/farsi/space/app.py
```

Then open http://127.0.0.1:7860. The first launch downloads the model from
[mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi)
and caches it; the four example texts are pre-generated on first launch too
(`cache_examples=True`), so they play back instantly after that.

`app.py` imports `training/farsi/normalize_fa.py` and `training/farsi/synthesize.py`
directly rather than duplicating them, so it must be run from inside this
repo checkout (not copied out on its own — for that, use the standalone
`farsi_tts.py` script instead, which carries its own copy of the normalizer).

## Files

| file | what it is |
|---|---|
| `app.py` | the Gradio app |
| `requirements.txt` | its extra dependencies (`pocket-tts`, `sphn`, `soundfile`, `gradio`) — everything else it needs is already in the repo |
