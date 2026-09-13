---
title: Pocket TTS Farsi v2
emoji: 🗣️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
short_description: Persian TTS with voice cloning, on CPU — v2
python_version: "3.12"
startup_duration_timeout: 30m
models:
  - mehdi-hf/pocket-tts-farsi-v2
  - mehdi-hf/Homo-GE2PE-Persian-HF
tags:
  - text-to-speech
  - persian
  - farsi
  - voice-cloning
license: cc-by-nc-4.0
---

# Pocket TTS — Farsi v2

109.5M parameters, CPU-only, voice cloning from a five-second prompt. Trained on
973 hours of Persian speech from 2,978 speakers.

v2 takes **romanised phonemes**, not Persian script, so this app runs
[Homo-GE2PE](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) in front of
it. The phonemes it produces are shown with every generation.

Model card: [mehdi-hf/pocket-tts-farsi-v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)
