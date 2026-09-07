# Examples

Five clips generated with the released model
([mehdi-hf/pocket-tts-farsi](https://huggingface.co/mehdi-hf/pocket-tts-farsi)),
its one bundled voice, default settings, no cherry-picking or re-takes — what
you hear is what `farsi_tts.py` or the [Gradio app](../space/) gives you out
of the box. Click a file to open GitHub's built-in audio player.

| clip | text | notes |
|---|---|---|
| [`01_greeting.wav`](01_greeting.wav) | سلام، حال شما چطور است؟ این صدای مصنوعی فارسی است. | short greeting |
| [`02_weather.wav`](02_weather.wav) | امروز هوای تهران آفتابی است و دمای هوا به بیست و پنج درجه می‌رسد. | numbers spelled out by the normalizer (۲۵ → بیست و پنج) |
| [`03_tech.wav`](03_tech.wav) | این یک نمونه است از تبدیل متن فارسی به گفتار روی پردازنده مرکزی، بدون نیاز به کارت گرافیک. | one sentence, single chunk |
| [`04_literature.wav`](04_literature.wav) | شعر و ادبیات فارسی یکی از غنی‌ترین میراث‌های فرهنگی جهان است. | formal register |
| [`05_paragraph.wav`](05_paragraph.wav) | زبان فارسی یکی از قدیمی‌ترین زبان‌های زنده جهان است. این زبان قرن‌ها تاریخ و فرهنگ غنی را در خود حفظ کرده و امروز میلیون‌ها نفر در ایران، افغانستان و تاجیکستان به آن سخن می‌گویند. | two sentences, so `synthesize.py`'s chunker splits and rejoins with a 0.15s pause — listen for the seam at "می‌گویند" |

Generated with:

```bash
uv run farsi_tts.py --text "<one row from the table above>"
```

(`farsi_tts.py` is the standalone script from the model card, not part of this
repo — see [`../README.md`](../README.md) or the
[model card](https://huggingface.co/mehdi-hf/pocket-tts-farsi) to get it.)
