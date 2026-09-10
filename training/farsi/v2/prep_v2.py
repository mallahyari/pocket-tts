#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["typer"]
# ///
"""Run the whole v2 data-prep chain on one GPU box, with gates between steps.

Turns the ingested corpus into something trainable: validated, aligned, merged,
phonemised, tokenised, and encoded to latents. Roughly 5 h on 8xH100 Spot
(~$160). Every step is **resumable** -- rerunning skips whatever already exists,
so a preemption costs one step, not the session.

    uv run prep_v2.py --plan          # what it would do, and what is already done
    uv run prep_v2.py                 # run it
    uv run prep_v2.py --from align    # resume from a named step

Steps, in order:

  0 preflight    mount, disk headroom, GPUs, ffmpeg, HF auth, inputs present
  1 validate     do the ingested time windows contain the speech they claim?  [GATE]
  2 align        word timings for the 144k new rows
  3 merge        v1 + new into one corpus manifest
  4 phonemize    transcripts (and `words`) into phonemes
  5 tokenizer    sentencepiece over the phoneme alphabet, vocab 4000
  6 latents      precompute Mimi latents for the merged corpus
  7 snapshot     reminder, with the command

**Step 1 is a real gate.** `ingest_farsi_asr_yt.py` trusts subtitle timings to
locate speech inside hour-long recordings, and if they are wrong the failure is
silent -- training simply learns from mismatched audio and text. Everything
below step 1 is expensive, so it stops here rather than finding out later.

**Why alignment is not optional (step 2).** `../RESULTS.md` found 12% of v1
utterances carried >1 s of trailing silence, and training on it "teaches the
model to emit silence instead of EOS, so generations never terminate". v1
already has a 9% no-EOS rate; adding 608 h of unaligned subtitle windows without
that trim risks making the most audible failure worse.

**Why the tokenizer keeps vocab 4000 (step 5).** Every tensor shape then stays
identical (`lookup_table.n_bins` unchanged, embedding still 4001 rows), so the
existing checkpoints load and only the text embedding has to be relearned --
turning a from-scratch retrain into a warm start. See `configs/finetune_fa.yaml`.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import typer

REPO = Path(__file__).resolve().parents[3]
V2 = REPO / "training" / "farsi" / "v2"
app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)

# Mimi encoding is per-utterance and index-keyed to the manifest, so a merged
# manifest cannot reuse v1's latents -- budget a fresh pass over the whole corpus.
STEPS = ["preflight", "validate", "align", "merge", "phonemize", "tokenizer", "latents", "snapshot"]


@dataclass
class Paths:
    data: Path

    @property
    def v1_aligned(self) -> Path:
        return self.data / "train_aligned.jsonl"

    @property
    def v1_valid(self) -> Path:
        return self.data / "valid_aligned.jsonl"

    @property
    def yt_raw(self) -> Path:
        return self.data / "farsi_asr_yt.jsonl"

    @property
    def yt_aligned(self) -> Path:
        return self.data / "farsi_asr_yt_aligned.jsonl"

    @property
    def merged(self) -> Path:
        return self.data / "v2_train.jsonl"

    @property
    def merged_ph(self) -> Path:
        return self.data / "v2_train_ph.jsonl"

    @property
    def valid_ph(self) -> Path:
        return self.data / "v2_valid_ph.jsonl"

    @property
    def tokenizer(self) -> Path:
        return self.data / "tokenizer_ph.model"

    @property
    def latents(self) -> Path:
        # the `_latents` suffix is what makes train.py read precomputed latents
        return self.data / "v2_train_ph_latents.jsonl"


# This file runs under its own PEP 723 environment, which deliberately carries
# nothing but typer. Anything touching torch, the repo modules, or a sibling
# script therefore has to be launched in the environment that owns it -- calling
# them with sys.executable would fail on the first import.
def run_repo(args: list[str]) -> None:
    """Run a repo module in the repo's uv environment (has torch, pocket_tts...)."""
    cmd = ["uv", "run", "python", *args]
    typer.echo(f"    $ {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=REPO).returncode != 0:
        raise typer.Exit(1)


def run_script(script: Path, args: list[str]) -> int:
    """Run a sibling PEP 723 script, which builds its own environment."""
    cmd = ["uv", "run", str(script), *args]
    typer.echo(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def head(msg: str, n: int) -> None:
    typer.echo(f"\n{'=' * 72}\n[{n}/{len(STEPS) - 1}] {msg}\n{'=' * 72}")


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def step_preflight(p: Paths, gpus: int) -> None:
    """Everything that has bitten a previous session, checked in ten seconds."""
    problems: list[str] = []

    if not p.data.is_dir():
        problems.append(f"{p.data} not found — is the disk mounted? (see --data)")
    else:
        for f in (p.v1_aligned, p.v1_valid, p.yt_raw):
            if not f.exists():
                problems.append(f"missing input: {f}")

    if shutil.which("ffmpeg") is None:
        problems.append("ffmpeg missing — sudo apt-get install -y ffmpeg")

    if p.data.is_dir():
        free_gb = shutil.disk_usage(p.data).free / 1e9
        # latents for ~1,100 h plus manifests; measured ~0.2 GB/h for audio and
        # latents are smaller, but leave real headroom rather than dying at 90%.
        if free_gb < 150:
            problems.append(f"only {free_gb:.0f} GB free on the data disk; want >=150 GB")
        else:
            typer.echo(f"    disk headroom: {free_gb:.0f} GB")

    # nvidia-smi rather than torch: this script's environment has no torch, and
    # shelling into the repo's just to count GPUs is slower and no more reliable.
    if shutil.which("nvidia-smi") is None:
        problems.append("nvidia-smi missing — alignment and latents need a GPU")
    else:
        listed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        n = len([x for x in listed.stdout.splitlines() if x.strip()])
        typer.echo(f"    GPUs visible: {n}")
        if n == 0:
            problems.append("no GPU visible — the driver may still be installing; retry in a minute")
        elif n < gpus:
            problems.append(f"--gpus {gpus} requested but only {n} visible")

    hf = Path.home() / ".cache" / "huggingface" / "token"
    if not hf.exists():
        problems.append("not logged in to HF — `uv run hf auth login` (Mimi weights are gated)")

    if problems:
        typer.echo("\n  PREFLIGHT FAILED:")
        for x in problems:
            typer.echo(f"    - {x}")
        raise typer.Exit(1)
    typer.echo("    preflight OK")


def step_validate(p: Paths, n: int) -> None:
    """Gate: do the ingested windows actually contain the speech they claim?"""
    code = run_script(
        V2 / "validate_ingest.py",
        ["--manifest", str(p.yt_raw), "--n", str(n), "--offsets", " -1,-0.5,0,0.5,1"],
    )
    if code == 0:
        typer.echo("    validation passed")
        return
    typer.echo(
        "\n  STOPPING. Everything below this step is expensive, and training on\n"
        "  mis-timed windows fails silently rather than loudly.\n"
        + (
            "  A systematic offset was found — it is CORRECTABLE: shift `start`\n"
            "  across the manifest by the reported amount and re-run.\n"
            if code == 2
            else "  The transcripts themselves look wrong, not the timings.\n"
            if code == 3
            else "  No usable windows — check the audio paths in the manifest.\n"
        )
    )
    raise typer.Exit(code)


def step_align(p: Paths, gpus: int, model: str) -> None:
    if p.yt_aligned.exists():
        typer.echo(f"    {p.yt_aligned.name} exists — skipping")
        return
    # The repo's own helper: shards across GPUs, streams to .partial, and resumes
    # after an interrupt. Reused rather than reimplemented so alignment behaves
    # identically to the way v1's corpus was built.
    run_repo([
        "-c",
        "import sys; from pathlib import Path;"
        " from training.scripts.prepare_data import align;"
        " align(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])",
        str(p.yt_raw), str(p.yt_aligned), str(gpus), model, "farsi-asr yt manifest",
    ])


def step_merge(p: Paths) -> None:
    if p.merged.exists():
        typer.echo(f"    {p.merged.name} exists ({count_lines(p.merged):,} rows) — skipping")
        return
    total = 0
    with open(p.merged, "w") as out:
        for src in (p.v1_aligned, p.yt_aligned):
            n = 0
            with open(src) as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
            typer.echo(f"    + {n:,} rows from {src.name}")
            total += n
    typer.echo(f"    wrote {p.merged.name}: {total:,} rows")


# Below this, splitting and concatenating costs more than it saves.
SHARD_MIN_ROWS = 10_000


def split_manifest(src: Path, dst: Path, shards: int) -> list[tuple[Path, Path]]:
    """Cut `src` into `shards` contiguous chunks, paired with their output paths.

    Contiguous and in order, so concatenating the outputs in this same order
    reproduces exactly what one pass over `src` would have written.
    """
    lines = src.open().readlines()
    per = (len(lines) + shards - 1) // shards
    pairs = []
    for i in range(shards):
        chunk = src.with_suffix(f".phchunk{i}")
        chunk.write_text("".join(lines[i * per : (i + 1) * per]))
        pairs.append((chunk, dst.with_suffix(f".phpart{i}")))
    return pairs


def concat_parts(pairs: list[tuple[Path, Path]], dst: Path) -> None:
    """Join the shard outputs back into `dst` and clean up the pieces."""
    with dst.open("w") as f:
        for chunk, part in pairs:
            f.write(part.read_text())
            chunk.unlink()
            part.unlink()


def phonemize_sharded(src: Path, dst: Path, batch: int, shards: int) -> None:
    """One G2P process per GPU over a contiguous slice of `src`.

    G2P here is five-beam search over a byte-level T5. On a single card the
    merged corpus needs about seven hours while every other card sits idle,
    which is most of a prep session spent on the cheapest step in it. Each
    slice resumes independently, since phonemize_manifest.py appends and skips
    whatever its own output already holds.
    """
    pairs = split_manifest(src, dst, shards)
    procs = [
        subprocess.Popen(
            [
                "env", f"CUDA_VISIBLE_DEVICES={i}",
                "uv", "run", str(V2 / "phonemize_manifest.py"),
                "--manifest", str(chunk), "--out", str(part), "--batch-size", str(batch),
            ],
            cwd=REPO,
        )
        for i, (chunk, part) in enumerate(pairs)
    ]
    if [i for i, proc in enumerate(procs) if proc.wait() != 0]:
        typer.echo("    a phonemize shard failed — rerun to resume the rest")
        raise typer.Exit(1)
    concat_parts(pairs, dst)


def step_phonemize(p: Paths, batch: int, gpus: int) -> None:
    for src, dst in ((p.merged, p.merged_ph), (p.v1_valid, p.valid_ph)):
        if dst.exists() and count_lines(dst) >= count_lines(src):
            typer.echo(f"    {dst.name} complete — skipping")
            continue
        rows = count_lines(src)
        if gpus > 1 and rows >= SHARD_MIN_ROWS:
            typer.echo(f"    {src.name}: {rows:,} rows across {gpus} GPUs")
            phonemize_sharded(src, dst, batch, gpus)
            continue
        if run_script(
            V2 / "phonemize_manifest.py",
            ["--manifest", str(src), "--out", str(dst), "--batch-size", str(batch)],
        ):
            raise typer.Exit(1)


def step_tokenizer(p: Paths) -> None:
    if p.tokenizer.exists():
        typer.echo(f"    {p.tokenizer.name} exists — skipping")
        return
    # vocab 4000 on purpose: keeps every tensor shape identical so the existing
    # checkpoints still load and only the text embedding is relearned.
    run_repo([
        "-m", "training.scripts.train_tokenizer",
        str(p.tokenizer.with_suffix("")), str(p.merged_ph), "--vocab-size", "4000",
    ])


def step_latents(p: Paths, train_config: str) -> None:
    if p.latents.exists():
        typer.echo(f"    {p.latents.name} exists — skipping")
        return
    # precompute_latents takes a TRAINING config and reads data.train_jsonl from
    # it -- it has no manifest argument. So the config must already point at the
    # phonemised manifest, which is why configs are checked before this runs.
    cfg = REPO / train_config
    if not cfg.exists():
        typer.echo(f"    missing {train_config}")
        raise typer.Exit(1)
    want = str(p.merged_ph.name)
    if want not in cfg.read_text(encoding="utf-8"):
        typer.echo(
            f"    {train_config} does not reference {want}.\n"
            "    precompute_latents reads the manifest from the config, so it would\n"
            "    silently encode the wrong corpus. Fix data.train_jsonl first."
        )
        raise typer.Exit(1)
    typer.echo(
        "    Mimi latents are index-keyed to their manifest, so the merged corpus\n"
        "    cannot reuse v1's — this encodes ~1,100 h fresh. Expect ~1-2 h."
    )
    run_repo(["-m", "training.scripts.precompute_latents", train_config])


def step_snapshot(p: Paths) -> None:
    typer.echo(
        "    Prep is done. Snapshot before releasing the VM — this is hours of work\n"
        "    that exists nowhere else:\n\n"
        "      gcloud compute snapshots create fa-data-v2-prep \\\n"
        "          --source-disk=fa-data --source-disk-zone=$ZONE\n\n"
        "    Then delete the disk (an idle 1 TB volume is ~$100/month):\n\n"
        "      gcloud compute disks delete fa-data --zone=$ZONE --quiet\n"
    )


# --------------------------------------------------------------------------


@app.command()
def main(
    data: Path = typer.Option(Path("/mnt/data/farsi_600h"), help="manifests directory"),
    gpus: int = typer.Option(8, help="alignment shards; one process per GPU"),
    validate_n: int = typer.Option(150, help="windows to check in the validation gate"),
    phonemize_batch: int = typer.Option(64),
    align_model: str = typer.Option("m3hrdadfi/wav2vec2-large-xlsr-persian-v3"),
    train_config: str = typer.Option(
        "training/farsi/configs/lsd_scratch_v2.yaml",
        help="training config whose data.train_jsonl the latents pass reads",
    ),
    from_step: str = typer.Option("preflight", "--from", help=f"start here: {', '.join(STEPS)}"),
    plan: bool = typer.Option(False, "--plan", help="show what would run, and what is already done"),
) -> None:
    p = Paths(data=data)
    if from_step not in STEPS:
        typer.echo(f"unknown step {from_step!r}; expected one of {', '.join(STEPS)}")
        raise typer.Exit(1)

    if plan:
        typer.echo(f"data dir: {p.data}\n")
        outputs = {
            "align": p.yt_aligned, "merge": p.merged, "phonemize": p.merged_ph,
            "tokenizer": p.tokenizer, "latents": p.latents,
        }
        for s in STEPS:
            out = outputs.get(s)
            if out is None:
                state = "always runs"
            elif out.exists():
                state = f"DONE  ({out.name}, {count_lines(out):,} rows)" if out.suffix == ".jsonl" \
                    else f"DONE  ({out.name})"
            else:
                state = f"todo  -> {out.name}"
            typer.echo(f"  {s:<10} {state}")
        return

    start = STEPS.index(from_step)
    t0 = time.monotonic()
    for i, name in enumerate(STEPS):
        if i < start:
            continue
        head(name, i)
        step_t = time.monotonic()
        if name == "preflight":
            step_preflight(p, gpus)
        elif name == "validate":
            step_validate(p, validate_n)
        elif name == "align":
            step_align(p, gpus, align_model)
        elif name == "merge":
            step_merge(p)
        elif name == "phonemize":
            step_phonemize(p, phonemize_batch, gpus)
        elif name == "tokenizer":
            step_tokenizer(p)
        elif name == "latents":
            step_latents(p, train_config)
        elif name == "snapshot":
            step_snapshot(p)
        typer.echo(f"    [{name} took {(time.monotonic() - step_t) / 60:.1f} min]")

    typer.echo(f"\nall steps done in {(time.monotonic() - t0) / 60:.1f} min")


if __name__ == "__main__":
    app()
