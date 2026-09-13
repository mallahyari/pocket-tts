"""Pocket TTS — Farsi v2: 109.5M-parameter Persian TTS with voice cloning, on CPU.

v2 is trained on romanised phonemes rather than Persian script, so this app runs
three stages: normalise the Persian (which also spells numbers out), phonemise
with Homo-GE2PE, then synthesise. The phonemes are shown with every generation,
because when something sounds wrong they are usually where it went wrong.
"""

from __future__ import annotations

import logging
import math
import re
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import scipy.io.wavfile
import torch
from pocket_tts import TTSModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

from normalize_fa import normalize_for_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pocket-tts-farsi-v2")

MODEL_ID = "mehdi-hf/pocket-tts-farsi-v2"
MODEL_CONFIG = f"hf://{MODEL_ID}/model.yaml"
G2P_ID = "mehdi-hf/Homo-GE2PE-Persian-HF"

DEFAULT_TEMPERATURE = 0.3
DEFAULT_EOS_THRESHOLD = -2.0
FRAMES_AFTER_EOS = 0
DEFAULT_MAX_TOKENS = 18     # ~11 tokens was the training average; 21+ runs past EOS
DEFAULT_MIN_TOKENS = 8
DEFAULT_VOICE_SEC = 5.0     # training capped voice prompts at 5 s
DEFAULT_JOIN_SEC = 0.0
MAX_CHARS = 2000
SAMPLE_RATE = 24000

EXAMPLE_VOICE = str(Path(__file__).parent / "example_voice.wav")
DEFAULT_TEXT = (
    "شرکت آنتروپیک مدلی اقتصادی منتشر کرده که نشان می‌دهد هوش مصنوعی چگونه "
    "می‌تواند اقتصاد آمریکا را تا سال ۲۰۳۰ تغییر دهد."
)

# GE2PE's own alphabet -> the notation this model was trained on.
TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

# GE2PE marks the ezafe with a trailing "1": "?eqtesAde1 ?AmrikA" is
# eqtesad-E amrika, one phrase. The model never sees the mark -- it is stripped
# before generation, exactly as the training corpus had it stripped -- but the
# chunker needs it, or a chunk boundary lands inside a noun phrase and you hear
# a gap in the middle of it.
EZAFE_MARK = "1"


def strip_ezafe(text: str) -> str:
    return text.replace(EZAFE_MARK, "")


_LOCK = threading.Lock()
logger.info("loading %s ...", MODEL_ID)
model = TTSModel.load_model(
    config=MODEL_CONFIG, temp=DEFAULT_TEMPERATURE, eos_threshold=DEFAULT_EOS_THRESHOLD
)
logger.info("loading %s ...", G2P_ID)
_g2p_tok = AutoTokenizer.from_pretrained(G2P_ID)
_g2p = T5ForConditionalGeneration.from_pretrained(G2P_ID).eval()
_sp = model.flow_lm.conditioner.tokenizer.sp
logger.info("ready")


def _count_tokens(s: str) -> int:
    return len(_sp.encode(strip_ezafe(s)))


def phonemise(persian: str) -> str:
    """Persian script -> phonemes, ezafe marks kept for the chunker.

    Normalisation runs first on purpose: G2P has no reading for digits and drops
    them silently, so "تا سال ۲۰۳۰" would come back as "tA sAle" with the year
    simply gone.
    """
    text = normalize_for_model(persian)
    # "؟" and the glottal stop share a symbol in this notation, so the mark has
    # to go before G2P rather than after.
    text = text.replace("؟", "").replace("?", "")
    if not text.strip():
        raise gr.Error("Nothing left after Persian normalisation — check the input script.")
    enc = _g2p_tok([text], add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        out = _g2p.generate(**enc, num_beams=5, max_length=512, early_stopping=True)
    raw = _g2p_tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    return raw.translate(TO_PHONEMES)


def split_phonemes(text: str, max_tokens: int, min_tokens: int = DEFAULT_MIN_TOKENS):
    """Chunk the phonemes, keeping ezafe-bound words together.

    There is no punctuation to break on -- not one of the 4000 vocabulary pieces
    contains any -- so chunk count is chosen first and tokens are spread evenly
    across it. Packing greedily instead leaves a scrap at the end, and a chunk
    that short runs past end-of-speech and invents words.
    """
    words = text.split()
    if not words:
        return []
    total = _count_tokens(text)
    n_chunks = max(1, math.ceil(total / max_tokens))
    for _ in range(4):
        target = math.ceil(total / n_chunks)
        out, cur = [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            owed = len(out) < n_chunks - 1
            bound = bool(cur) and cur.split()[-1].endswith(EZAFE_MARK)
            over = _count_tokens(trial) > max_tokens
            if cur and not bound and (over or (owed and _count_tokens(trial) > target)):
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
        if len(out) <= n_chunks:
            break
        n_chunks = len(out)
    # Pull words back from the neighbour rather than leave a starved tail.
    for i in range(len(out) - 1, 0, -1):
        while _count_tokens(out[i]) < min_tokens:
            prev = out[i - 1].split()
            if len(prev) < 2 or _count_tokens(" ".join(prev[:-1])) < min_tokens:
                break
            out[i - 1] = " ".join(prev[:-1])
            out[i] = f"{prev[-1]} {out[i]}"
    return out


def _prepare_voice_prompt(voice_audio, voice_sec: float) -> str:
    if voice_audio is None:
        return EXAMPLE_VOICE
    sr, data = voice_audio
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float32)
    if data.size == 0:
        raise gr.Error("The voice prompt is empty — upload or record 2–5 seconds of speech.")
    if voice_sec > 0:
        data = data[: int(voice_sec * sr)]
    peak = float(np.max(np.abs(data)))
    if peak > 1.0:
        data = data / peak
    path = Path(tempfile.mkdtemp()) / "voice_prompt.wav"
    scipy.io.wavfile.write(str(path), int(sr), (data * 32767.0).astype(np.int16))
    return str(path)


def synthesize(
    text: str,
    voice_audio=None,
    temperature: float = DEFAULT_TEMPERATURE,
    eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    voice_sec: float = DEFAULT_VOICE_SEC,
    join_sec: float = DEFAULT_JOIN_SEC,
):
    """Speak Persian text in the voice of the audio prompt.

    Args:
        text: Persian (Farsi) text, in Persian script.
        voice_audio: 2-5 seconds of one clean speaker to clone.
        temperature: Sampling temperature; 0.3 is what the model was tuned at.
        eos_threshold: Higher keeps the model talking longer.
        max_tokens: Token budget per chunk. 21+ is where stability degrades.
        voice_sec: Seconds of the prompt to use; training capped this at 5.
        join_sec: Silence inserted at a chunk seam.

    Returns:
        (sample_rate, waveform), and a report showing the phonemes used.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter some Persian text.")
    phonemes = phonemise(text.strip()[:MAX_CHARS])
    chunks = split_phonemes(phonemes, int(max_tokens))
    if not chunks:
        raise gr.Error("Nothing to synthesize.")
    voice_path = _prepare_voice_prompt(voice_audio, float(voice_sec))
    join = np.zeros(int(float(join_sec) * SAMPLE_RATE), dtype=np.float32)

    with _LOCK:
        model.temp = float(temperature)
        model.eos_threshold = float(eos_threshold)
        state = model.get_state_for_audio_prompt(voice_path)
        pieces = []
        for i, chunk in enumerate(chunks, 1):
            spoken = strip_ezafe(chunk)
            logger.info("[%d/%d] %d tokens: %s", i, len(chunks), _count_tokens(chunk), spoken)
            audio = model.generate_audio(state, spoken, frames_after_eos=FRAMES_AFTER_EOS)
            if torch.is_tensor(audio):
                audio = audio.detach().float().cpu().numpy()
            pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            if i < len(chunks) and join.size:
                pieces.append(join)

    wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    report = (
        f"**{len(chunks)} chunk(s)** · {wav.shape[-1] / SAMPLE_RATE:.1f} s @ 24 kHz\n\n"
        f"**Phonemes fed to the model:**\n\n`{strip_ezafe(phonemes)}`"
    )
    return (SAMPLE_RATE, wav), report


CSS = "#col-container { max-width: 1000px; margin: 0 auto; }"

with gr.Blocks(title="Pocket TTS — Farsi v2", css=CSS) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            "# Pocket TTS — Farsi v2\n"
            "Persian text-to-speech with voice cloning, 109.5M parameters, running on a CPU.\n\n"
            "Type Persian text and give it 2–5 seconds of any speaker to clone. "
            "The model reads romanised phonemes, so "
            f"[Homo-GE2PE]( https://huggingface.co/{G2P_ID} ) converts the script first — "
            "the phonemes it produced are shown under each result.\n\n"
            f"Model card: [{MODEL_ID}](https://huggingface.co/{MODEL_ID})"
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Persian text", value=DEFAULT_TEXT, lines=4, rtl=True, text_align="right"
                )
                voice = gr.Audio(label="Voice to clone (2–5 s)", type="numpy", sources=["upload", "microphone"])
                go = gr.Button("Generate", variant="primary")
                with gr.Accordion("Settings", open=False):
                    temperature = gr.Slider(0.05, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                    eos = gr.Slider(-6.0, 0.0, value=DEFAULT_EOS_THRESHOLD, step=0.5, label="EOS threshold")
                    maxtok = gr.Slider(8, 24, value=DEFAULT_MAX_TOKENS, step=1, label="Max tokens per chunk")
                    vsec = gr.Slider(1.0, 10.0, value=DEFAULT_VOICE_SEC, step=0.5, label="Voice prompt seconds")
                    jsec = gr.Slider(0.0, 0.5, value=DEFAULT_JOIN_SEC, step=0.05, label="Silence at a chunk seam")
            with gr.Column():
                out_audio = gr.Audio(label="Generated speech", type="numpy", autoplay=False)
                out_report = gr.Markdown()
        go.click(
            synthesize,
            inputs=[text, voice, temperature, eos, maxtok, vsec, jsec],
            outputs=[out_audio, out_report],
        )

if __name__ == "__main__":
    demo.queue().launch()
