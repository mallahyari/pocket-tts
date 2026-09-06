"""Export a chosen training checkpoint as an inference-ready model.safetensors.

`train.py` refreshes `model.safetensors` at every checkpoint, so the copy in a
run directory is the EMA of the *last* step. When evaluation says an earlier
checkpoint is better -- which it often does on a small or noisy corpus, where
the loss keeps falling long after quality has plateaued -- that file is the one
you do not want. This writes the same format from any checkpoint you name.

The output carries `flow_lm.*` (EMA weights when the checkpoint has them) and
`mimi.*`, exactly what `weights_path` in a pocket-tts config expects.

    python -m training.farsi.export_model \
        --run-dir /mnt/data/runs/lsd_distill_fa \
        --checkpoint checkpoint_00100000.pt \
        --out /mnt/data/exports/farsi_student_100k.safetensors
"""

import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

logger = logging.getLogger("export_model")
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    run_dir: Annotated[str, typer.Option(help="training run directory (holds args.yaml)")],
    checkpoint: Annotated[str, typer.Option(help="checkpoint file, name or full path")],
    out: Annotated[str, typer.Option(help="destination .safetensors")],
    use_ema: Annotated[
        bool, typer.Option(help="export the EMA shadow rather than the raw weights")
    ] = True,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    import torch

    from training.args import load_args
    from training.checkpointing import EMA, export_pocket_safetensors, load_checkpoint
    from training.modules.builders import build_models

    run = Path(run_dir)
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = run / checkpoint
    if not ckpt.exists():
        raise SystemExit(f"no such checkpoint: {ckpt}")

    args = load_args(run / "args.yaml")
    # Inference never needs the distillation teacher, and building it would
    # require the teacher checkpoint to still sit where training left it.
    args.distill_cfg_coef = 0.0
    model, mimi, _ = build_models(args)

    ema = EMA(model, 1.0) if use_ema else None
    step = load_checkpoint(ckpt, model, ema=ema)
    if ema is not None:
        if not ema.shadow:
            raise SystemExit(f"{ckpt.name} carries no EMA shadow; re-run with --no-use-ema")
        model.load_state_dict(ema.shadow, strict=False)

    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export_pocket_safetensors(dest, model.flow_lm, mimi, ema)
    size_mb = dest.stat().st_size / 1e6
    logger.info(f"step {step}, ema={use_ema} -> {dest} ({size_mb:.0f} MB)")
    logger.info("point weights_path at this file in your model config")


if __name__ == "__main__":
    app()
