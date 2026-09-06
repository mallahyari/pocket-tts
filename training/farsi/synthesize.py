"""Synthesize arbitrarily long Farsi text by splitting it into safe chunks.

pocket-tts splits text at sentence boundaries and warns when a chunk exceeds
`MAX_TOKEN_PER_CHUNK` ("generation may skip words"). A long Persian sentence
joined by commas has no sentence boundary to split on, so it arrives whole and
the model runs to its length cap without emitting EOS -- the second half comes
out as repetition.

There is a corpus reason to keep chunks short as well as a mechanical one: a
model trained on subtitle-derived data has seen mostly 2-6 second utterances,
so a 20-second target is out of distribution no matter what the token limit
says. Splitting at punctuation and re-prompting per chunk keeps every request
inside the distribution the model was trained on.

Each chunk is generated from the same voice prompt, so the speaker stays
consistent, and the pieces are joined with a short pause.

    python -m training.farsi.synthesize --config model.yaml --voice prompt.wav \
        --text-file article.txt --out article.wav
"""

import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import sphn
import typer
from typing_extensions import Annotated

logger = logging.getLogger("synthesize")
app = typer.Typer(pretty_exceptions_show_locals=False)

# Persian sentence enders, then softer breaks. Split points are kept with the
# preceding text so the model still sees the punctuation it was trained on.
HARD_BREAK = re.compile(r"(?<=[.!؟])\s+")
SOFT_BREAK = re.compile(r"(?<=[،؛:])\s+")
BREAK_CHARS = (".", "!", "؟", "،", "؛", ":")


def split_text(
    text: str,
    count_tokens,
    max_tokens: int = 40,
    keep_punct_boundaries: bool = False,
    min_tokens: int = 8,
) -> list[str]:
    """Split `text` into chunks of at most `max_tokens` tokens.

    Sentence boundaries first, then clause punctuation, then whitespace as a
    last resort. `max_tokens` defaults below pocket-tts's 50-token chunk limit
    so a chunk never trips the "may skip words" path.

    With `keep_punct_boundaries`, every sentence and clause mark becomes a chunk
    boundary even when the text would have fitted in one chunk. Only boundaries
    get an inserted pause, so this is what makes a comma audible when the model
    itself rushes it -- which a model trained on clipped subtitle segments does.
    """
    text = " ".join(text.split())
    if not text:
        return []

    def fits(s: str) -> bool:
        return count_tokens(s) <= max_tokens

    def split_by(pattern, piece: str) -> list[str]:
        parts = [p.strip() for p in pattern.split(piece) if p.strip()]
        return parts if len(parts) > 1 else []

    def recurse(piece: str) -> list[str]:
        if fits(piece):
            return [piece]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            parts = split_by(pattern, piece)
            if parts:
                return [c for p in parts for c in recurse(p)]
        # No punctuation left: pack words greedily.
        out, cur = [], ""
        for word in piece.split():
            trial = f"{cur} {word}".strip()
            if cur and not fits(trial):
                out.append(cur)
                cur = word
            else:
                cur = trial
        if cur:
            out.append(cur)
        return out

    if keep_punct_boundaries:
        pieces = [text]
        for pattern in (HARD_BREAK, SOFT_BREAK):
            pieces = [q for piece in pieces for q in (split_by(pattern, piece) or [piece])]
        parts = [c for piece in pieces for c in recurse(piece)]
    else:
        parts = recurse(text)

    # Merge neighbours back together while they still fit: fewer, fuller chunks
    # sound better than many tiny ones. A chunk ending at punctuation is never
    # merged away when its boundary is what earns the pause.
    merged: list[str] = []
    for chunk in parts:
        # A punctuation boundary is only worth keeping if the chunk before it is
        # long enough to synthesize well. "چهارم شهریور،" on its own is two
        # words, and a model trained on ~4s utterances renders such a fragment
        # badly -- especially as the first chunk after the voice prompt.
        keep_apart = (
            keep_punct_boundaries
            and merged
            and merged[-1].rstrip().endswith(BREAK_CHARS)
            and count_tokens(merged[-1]) >= min_tokens
        )
        if merged and not keep_apart and fits(f"{merged[-1]} {chunk}"):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


@app.command()
def main(
    config: Annotated[str, typer.Option(help="model config yaml")],
    voice: Annotated[str, typer.Option(help="voice prompt wav")],
    out: Annotated[str, typer.Option(help="output wav")] = "out.wav",
    text: Annotated[str | None, typer.Option(help="text to speak")] = None,
    text_file: Annotated[str | None, typer.Option(help="file to read the text from")] = None,
    max_tokens: Annotated[
        int,
        typer.Option(
            help="token budget per chunk (pocket-tts's own limit is 50). Lower it for "
            "voices the model has not seen: stability degrades with generation length, "
            "and 40 can run past EOS on an unfamiliar speaker where 25 does not."
        ),
    ] = 25,
    temperature: Annotated[float, typer.Option()] = 0.3,
    eos_threshold: Annotated[float, typer.Option()] = -2.0,
    frames_after_eos: Annotated[
        int, typer.Option(help="0 trims the trailing breath the model learned")
    ] = 0,
    voice_sec: Annotated[
        float,
        typer.Option(
            help="seconds of the voice prompt to use. Training capped prompts at "
            "max_voice_prompt_sec (5.0 by default), so a longer one is out of "
            "distribution and the model tends to run past EOS. 0 uses the whole file."
        ),
    ] = 5.0,
    pause_sec: Annotated[
        float, typer.Option(help="silence after a chunk that ends at punctuation")
    ] = 0.15,
    join_sec: Annotated[
        float,
        typer.Option(
            help="silence at a mid-phrase split (no punctuation). 0 is linguistically "
            "right -- it preserves the ezafe -- but chunks are generated independently, "
            "so a hard butt-join exposes the seam. 0.15 sounded better in practice."
        ),
    ] = 0.15,
    pause_at_punct: Annotated[
        bool,
        typer.Option(
            help="make every . ! ؟ ، ؛ : a chunk boundary, so each one gets --pause-sec. "
            "Tested and NOT recommended for a subtitle-trained model: it leaves chunks "
            "ending mid-clause at a comma, which every training utterance never did (they "
            "end at the last aligned word), and the whole chunk degrades -- reproducibly, "
            "across repeated draws. Off by default. Try it only if commas sound rushed and "
            "your corpus has long, well-punctuated utterances."
        ),
    ] = False,
    min_tokens: Annotated[
        int,
        typer.Option(
            help="never leave a chunk shorter than this; short fragments synthesize badly "
            "because the model was trained on ~4 second utterances"
        ),
    ] = 8,
    normalize_text: Annotated[
        bool, typer.Option(help="apply the training-time Persian normalization first")
    ] = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    from pocket_tts import TTSModel

    from training.farsi.normalize_fa import normalize

    raw = Path(text_file).read_text(encoding="utf-8") if text_file else text
    if not raw or not raw.strip():
        raise typer.BadParameter("pass --text or --text-file")
    if normalize_text:
        raw = normalize(raw)

    # frames_after_eos is a generate_audio argument, not a load_model one.
    model = TTSModel.load_model(config=config, temp=temperature, eos_threshold=eos_threshold)
    sp = model.flow_lm.conditioner.tokenizer.sp
    chunks = split_text(raw, lambda s: len(sp.encode(s)), max_tokens, pause_at_punct, min_tokens)
    logger.info(f"{len(chunks)} chunk(s)")

    sample_rate = int(model.mimi.sample_rate)
    gap = np.zeros(int(pause_sec * sample_rate), dtype=np.float32)
    join = np.zeros(int(join_sec * sample_rate), dtype=np.float32)
    prompt = voice
    if voice_sec > 0:
        wav, sr = sphn.read(voice)
        keep = int(voice_sec * sr)
        if wav.shape[-1] > keep:
            trimmed = Path(tempfile.mkdtemp()) / "voice_prompt.wav"
            sphn.write_wav(str(trimmed), wav.mean(axis=0)[:keep], int(sr))
            logger.info(f"voice prompt trimmed {wav.shape[-1] / sr:.1f}s -> {voice_sec:.1f}s")
            prompt = str(trimmed)

    # generate_audio defaults to copy_state=True, so one prompt state serves
    # every chunk and the speaker stays identical across them.
    state = model.get_state_for_audio_prompt(prompt)
    pieces = []
    for i, chunk in enumerate(chunks, 1):
        # Full chunk text -- truncating this preview once made it look like
        # words were being dropped from generation when they were only being
        # dropped from the printed preview.
        logger.info(f"[{i}/{len(chunks)}] {len(sp.encode(chunk))} tokens: {chunk}")
        audio = model.generate_audio(state, chunk, frames_after_eos=frames_after_eos)
        pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if i < len(chunks):
            # A chunk ending at punctuation gets a real pause; one split
            # mid-phrase to fit the token budget gets --join-sec instead.
            pieces.append(gap if chunk.rstrip().endswith(BREAK_CHARS) else join)

    wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sphn.write_wav(out, wav, sample_rate)
    logger.info(f"wrote {out}  ({len(wav) / sample_rate:.1f}s from {len(chunks)} chunk(s))")


if __name__ == "__main__":
    app()
