"""Gradio Space for pocket-tts-farsi.

Same generation logic as farsi_tts.py in the model repo (chunking, Persian
normalization, voice-prompt trimming) -- this is the browser front-end for the
same model, not a separate implementation.
"""

import logging
import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import sphn
from pocket_tts import TTSModel
from pocket_tts.utils.utils import download_if_necessary

# This app lives inside the pocket-tts checkout (unlike the standalone
# farsi_tts.py, which deliberately carries its own copy of normalize_fa.py so
# it can run without the repo). Import the real modules directly instead of
# duplicating them a third time, so a fix in either only has to happen once.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from training.farsi.normalize_fa import normalize  # noqa: E402
from training.farsi.synthesize import generate_chunk, split_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("app")

MODEL_CONFIG = "hf://mehdi-hf/pocket-tts-farsi/farsi.yaml"

VOICES = {"زن — صدای پیش‌فرض (Female — default)": "hf://mehdi-hf/pocket-tts-farsi/example_voice.wav"}
DEFAULT_VOICE = next(iter(VOICES))

TEMPERATURE = 0.3
EOS_THRESHOLD = -2.0
FRAMES_AFTER_EOS = 0
VOICE_PROMPT_MAX_SEC = 5.0
# 18, not 25: on held-out speakers chunks of 21+ tokens ran past EOS
# deterministically, while 9-16 token chunks were clean. Training utterances
# averaged ~11 tokens. generate_chunk() rescues anything that still runs away.
MAX_TOKENS_PER_CHUNK = 18
PAUSE_SEC = 0.15

logger.info("loading model...")
MODEL = TTSModel.load_model(config=MODEL_CONFIG, temp=TEMPERATURE, eos_threshold=EOS_THRESHOLD)
SP = MODEL.flow_lm.conditioner.tokenizer.sp
SAMPLE_RATE = int(MODEL.mimi.sample_rate)
logger.info("model loaded")


def trim_voice(path: str) -> str:
    # sphn.read only understands local files; get_state_for_audio_prompt
    # resolves hf:// itself, but reading a voice here to trim it needs the
    # same resolution done explicitly first.
    local_path = download_if_necessary(path)
    wav, sr = sphn.read(local_path)
    keep = int(VOICE_PROMPT_MAX_SEC * sr)
    if wav.shape[-1] <= keep:
        return path
    trimmed = Path(tempfile.mkdtemp()) / "voice_prompt.wav"
    sphn.write_wav(str(trimmed), wav.mean(axis=0)[:keep].astype("float32"), int(sr))
    return str(trimmed)


def generate(text: str, voice_name: str) -> str:
    if not text or not text.strip():
        raise gr.Error("متنی وارد نشده. / Please enter some text.")

    text = normalize(text)
    voice_path = trim_voice(VOICES[voice_name])
    state = MODEL.get_state_for_audio_prompt(voice_path)

    chunks = split_text(text, lambda s: len(SP.encode(s)), MAX_TOKENS_PER_CHUNK)
    gap = np.zeros(int(PAUSE_SEC * SAMPLE_RATE), dtype=np.float32)
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"[{i}/{len(chunks)}] {chunk}")
        pieces.append(
            generate_chunk(
                MODEL, state, chunk, frames_after_eos=FRAMES_AFTER_EOS, sample_rate=SAMPLE_RATE
            )
        )
        if i < len(chunks):
            pieces.append(gap)

    wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    out_path = str(Path(tempfile.mkdtemp()) / "output.wav")
    sphn.write_wav(out_path, wav, SAMPLE_RATE)
    return out_path


EXAMPLE_TEXTS = [
    "سلام، حال شما چطور است؟ این صدای مصنوعی فارسی است.",
    "امروز هوای تهران آفتابی است و دمای هوا به بیست و پنج درجه می‌رسد.",
    "این یک نمونه است از تبدیل متن فارسی به گفتار روی پردازنده مرکزی، بدون نیاز به کارت گرافیک.",
    "شعر و ادبیات فارسی یکی از غنی‌ترین میراث‌های فرهنگی جهان است.",
]

CSS = """
#text_in textarea { direction: rtl; text-align: right; font-size: 17px; }
"""

with gr.Blocks(title="Pocket TTS — Farsi") as demo:
    gr.Markdown(
        "# 🎙️ Pocket TTS — فارسی (Persian)\n"
        "A 100M-parameter Persian text-to-speech model that runs on a **CPU**. "
        "Built with [pocket-tts](https://github.com/kyutai-labs/pocket-tts), trained "
        "on 497h of public-domain Persian speech. "
        "[Model card](https://huggingface.co/mehdi-hf/pocket-tts-farsi) · "
        "[Training code](https://github.com/mallahyari/pocket-tts)"
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="متن (Text)",
                placeholder="متن فارسی خود را اینجا بنویسید...",
                lines=4,
                elem_id="text_in",
            )
            voice_in = gr.Dropdown(choices=list(VOICES), value=DEFAULT_VOICE, label="صدا (Voice)")
            btn = gr.Button("تولید گفتار (Generate)", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="خروجی (Output)", type="filepath")

    btn.click(generate, inputs=[text_in, voice_in], outputs=audio_out)

    gr.Examples(
        examples=[[t, DEFAULT_VOICE] for t in EXAMPLE_TEXTS],
        inputs=[text_in, voice_in],
        outputs=audio_out,
        fn=generate,
        cache_examples=True,
        label="نمونه‌ها (Examples) — click to hear pre-generated audio instantly",
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
