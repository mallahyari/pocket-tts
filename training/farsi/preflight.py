"""Ten-second check that a training config can actually encode audio.

Mimi's encoder turns audio into the latents the model is trained to predict.
`kyutai/pocket-tts-without-voice-cloning` ships that encoder **zeroed out** --
that is how voice cloning was removed from it -- while keeping every tensor
name, so `load_state_dict(strict=True)` succeeds and nothing raises. A config
pointing there produces all-zero latents, and training happily optimizes
against silence: the loss even looks excellent, because predicting zero is
easy. The samples are a constant drone that ignores text, voice and noise.

That failure cost a 47k-step run before this check existed. Run it before
every training run, and after any change to `weights_path`.

    python -m training.farsi.preflight training/farsi/configs/model_farsi.yaml \
        --manifest data/farsi_600h/train_aligned.jsonl
"""

import json
import logging
from pathlib import Path

import numpy as np
import safetensors.torch
import sphn
import torch
import typer
from typing_extensions import Annotated

from pocket_tts.models.mimi import build_mimi
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary

logger = logging.getLogger("preflight")
app = typer.Typer(pretty_exceptions_show_locals=False)

# Everything encode_to_latent depends on. A zeroed tensor here is silent death.
ENCODE_PATH = ("encoder", "encoder_transformer", "downsample")


def check_weights(weights_path: str) -> list[str]:
    """Names of encode-path submodules whose tensors are entirely zero."""
    state = safetensors.torch.load_file(download_if_necessary(weights_path))
    dead = []
    for part in ENCODE_PATH:
        prefix = f"mimi.{part}."
        tensors = [v for k, v in state.items() if k.startswith(prefix)]
        if tensors and all(float(t.float().abs().max()) == 0.0 for t in tensors):
            dead.append(f"{part} ({len(tensors)} tensors)")
    return dead


@app.command()
def main(
    model_config: Annotated[str, typer.Argument(help="the config training will use")],
    manifest: Annotated[
        str | None, typer.Option(help="manifest to draw a real utterance from")
    ] = None,
    min_std: Annotated[
        float, typer.Option(help="minimum acceptable std of the encoded latents")
    ] = 1e-3,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(Path(model_config))
    weights = str(config.weights_path)
    logger.info(f"weights_path: {weights}")

    dead = check_weights(weights)
    if dead:
        raise SystemExit(
            "FAIL: Mimi's encode path is all zeros in this checkpoint: "
            + ", ".join(dead)
            + "\n\nThis is the 'without-voice-cloning' checkpoint, whose encoder was "
            "deliberately zeroed. Training against it optimizes toward silence while "
            "reporting an excellent loss.\n"
            "Point weights_path at hf://kyutai/pocket-tts/... (accept the licence on "
            "the model page, then `hf auth login`)."
        )
    logger.info("weights: Mimi encode path is non-zero")

    if not manifest:
        logger.info("no --manifest given; skipping the end-to-end encode check")
        return

    mimi = build_mimi(config.mimi)
    state = safetensors.torch.load_file(download_if_necessary(weights))
    mimi.load_state_dict(
        {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}, strict=True
    )
    mimi.eval()

    with open(manifest) as f:
        row = json.loads(f.readline())
    start = float(row.get("start", 0.0))
    wav, sr = sphn.read(
        row["path"],
        start_sec=start if start > 0 else None,
        duration_sec=float(row["duration"]) if start > 0 else None,
    )
    wav = wav.mean(axis=0)
    if sr != mimi.sample_rate:
        wav = sphn.resample(wav, src_sample_rate=int(sr), dst_sample_rate=int(mimi.sample_rate))
    audio_absmean = float(np.abs(wav).mean())
    logger.info(f"audio: {row['path']}  absmean {audio_absmean:.6f}")
    if audio_absmean == 0.0:
        raise SystemExit("FAIL: the source audio itself is silent.")

    with torch.no_grad():
        latents = mimi.encode_to_latent(torch.from_numpy(wav).float()[None, None])
    std = float(latents.std())
    logger.info(f"latents: {tuple(latents.shape)}  std {std:.6f}")
    if std < min_std:
        raise SystemExit(
            f"FAIL: encoded latents are ~constant (std {std:.2e} < {min_std:.0e}).\n"
            "Training would optimize against a degenerate target."
        )
    logger.info("PASS: audio encodes to non-degenerate latents; safe to train")


if __name__ == "__main__":
    app()
