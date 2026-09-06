"""One-shot Farsi data preparation: download public Persian speech corpora,
normalize the transcripts, build manifests, and attach word alignments,
leaving `train_aligned.jsonl` + `valid_aligned.jsonl` in `data/farsi_<hours>h/`,
ready for `training/train.py`.

    python -m training.farsi.prepare_data_fa --hours 600 --align-shards 8

This is the Farsi counterpart of training/scripts/prepare_data.py, and writes
the same manifest format: one json object per line with `path`, `start`,
`duration`, `transcript` (plus `speaker`/`source` for bookkeeping).

Sources (--sources, in priority order; each contributes until --hours is met):

    manatts      114h  CC0     MahtaFetrat/Mana-TTS. One narrator, 44.1 kHz,
                               hand-verified transcripts. The cleanest Persian
                               speech data that exists publicly -- put it first.
    filimo       245h  CC0     PerSets/filimo-persian-asr. Iranian VOD movies
                               and series, subtitle-derived transcripts, many
                               speakers, some background music.
    youtube      385h  CC0     PerSets/youtube-persian-asr. Podcasts, shows and
                               interviews; subtitle-derived, noisiest of the three.
    commonvoice   56h  CC0     fsicoli/common_voice_22_0, fa. Crowd-sourced read
                               speech, ~4s per clip, 70% male voices. Only the
                               official train/dev/test splits are mirrored with
                               audio (the other ~300h of validated clips need a
                               manual download from commonvoice.mozilla.org).

Audio is downloaded once into --audio-out and left alone: Mana-TTS arrives as
raw float arrays inside parquet and is written out as 24 kHz FLAC, everything
else is extracted as the mp3 it ships as, and manifest rows point straight at
those files. The script is resumable -- extracted audio, manifests and
alignment shards are all skipped on a re-run.
"""

import csv
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import huggingface_hub
import numpy as np
import sphn
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from training.farsi.normalize_fa import normalize, reject_reason
from training.scripts.prepare_data import align

logger = logging.getLogger("prepare_data_fa")

app = typer.Typer(pretty_exceptions_show_locals=False)

MANATTS_REPO = "MahtaFetrat/Mana-TTS"
FILIMO_REPO = "PerSets/filimo-persian-asr"
YOUTUBE_REPO = "PerSets/youtube-persian-asr"
CV_REPO = "fsicoli/common_voice_22_0"
# Char-level Persian wav2vec2 CTC; its alphabet is exactly what
# training.farsi.normalize_fa emits (Persian letters + ZWNJ).
DEFAULT_ALIGN_MODEL = "m3hrdadfi/wav2vec2-large-xlsr-persian-v3"

AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a")
TARGET_SR = 24000  # Mimi's rate: transcoding Mana-TTS once saves 40% of the disk


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def sharded_path(root: Path, name: str) -> Path:
    """`root/<last 3 chars of the stem>/<name>`: 400k files in one directory
    is a bad time for every filesystem and every `ls` that follows."""
    return root / Path(name).stem[-3:] / name


def probe_durations(paths: list[Path], workers: int = 16) -> list[float | None]:
    """Duration of every file, in seconds (None when unreadable)."""
    chunks = [paths[i : i + 256] for i in range(0, len(paths), 256)]
    out: list[float | None] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        bar = tqdm(total=len(paths), unit="file", desc="probe durations")
        for got in pool.map(lambda c: sphn.durations([str(p) for p in c]), chunks):
            out.extend(got)
            bar.update(len(got))
        bar.close()
    return out


def download_archive(repo: str, filename: str, keep: bool) -> tuple[Path, bool]:
    """Fetch one file from a HF dataset repo. Returns (path, delete_after)."""
    path = Path(huggingface_hub.hf_hub_download(repo, filename, repo_type="dataset"))
    return path, not keep


def drop_from_cache(path: Path) -> None:
    """Remove a downloaded archive (and the blob it points at) to save disk."""
    try:
        blob = path.resolve()
        path.unlink(missing_ok=True)
        if blob != path:
            blob.unlink(missing_ok=True)
    except OSError as exc:  # noqa: BLE001 -- best effort, never fatal
        logger.debug(f"could not drop {path} from the cache: {exc}")


def extract_audio(tar_path: Path, out_root: Path) -> int:
    """Extract every audio member of `tar_path` into the sharded `out_root`."""
    n = 0
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if not name.lower().endswith(AUDIO_EXTS):
                continue
            dest = sharded_path(out_root, name)
            if dest.exists():
                n += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            with src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.rename(dest)
            n += 1
    return n


def read_delimited(path: Path) -> list[dict]:
    """Read a .tsv/.csv whose delimiter we do not want to hardcode.

    Mana-TTS, Filimo and YouTube each ship a different metadata dialect (tab
    with a leading index column, plain comma, tab without), so the delimiter
    is sniffed from the header rather than assumed.
    """
    with open(path, encoding="utf-8", newline="") as f:
        head = f.readline()
        delim = "\t" if head.count("\t") >= head.count(",") else ","
        f.seek(0)
        # Common Voice's tsv is unquoted and contains bare double quotes;
        # the comma-separated ones use ordinary quoting.
        quoting = csv.QUOTE_NONE if delim == "\t" else csv.QUOTE_MINIMAL
        return list(csv.DictReader(f, delimiter=delim, quoting=quoting))


class SourceCheckpoint:
    """One source's collected rows and finished archives, persisted as it goes.

    Archives are deleted after processing (that is the point -- 60 GB of tars
    should not sit on the disk), so without this a preempted run would
    re-download everything it had already handled just to rebuild the row
    metadata that lived inside them. Rows are appended after each archive, so a
    resume picks up at the next one.
    """

    def __init__(self, out_dir: Path, source: str) -> None:
        self.rows_path = out_dir / f"raw_{source}.jsonl"
        self.done_path = out_dir / f"done_{source}.txt"
        self.rows: list[dict] = []
        if self.rows_path.exists():
            self.rows = [json.loads(line) for line in self.rows_path.open() if line.strip()]
        self.done: set[str] = set()
        if self.done_path.exists():
            self.done = {
                line.strip() for line in self.done_path.read_text().splitlines() if line.strip()
            }
        if self.rows:
            logger.info(
                f"{source}: resuming with {len(self.rows)} utterances (~{self.hours:.1f}h) "
                f"from {len(self.done)} archive(s) already processed"
            )

    @property
    def hours(self) -> float:
        return sum(r["duration"] for r in self.rows) / 3600

    def add(self, archive: str, rows: list[dict]) -> None:
        """Record one archive's rows, then mark the archive finished.

        Rows are flushed before the archive is marked done: a kill between the
        two costs a repeat of one archive, never a silently missing one.
        """
        with self.rows_path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
        with self.done_path.open("a") as f:
            f.write(archive + "\n")
            f.flush()
        self.rows.extend(rows)
        self.done.add(archive)


def make_row(path: Path, duration: float, transcript: str, speaker: str, source: str) -> dict:
    return {
        "path": str(path),
        "start": 0.0,
        "duration": round(float(duration), 3),
        "transcript": transcript,
        "speaker": speaker,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def source_manatts(
    audio_out: Path, out_dir: Path, budget_h: float, keep: bool, min_quality: str
) -> list[dict]:
    """Mana-TTS: parquet of raw float audio + verified transcripts -> 24 kHz FLAC."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "the manatts source needs pyarrow (it ships as parquet): uv sync, or "
            "run with --sources filimo,youtube,commonvoice to skip it"
        ) from exc
    import soundfile

    out_root = audio_out / "manatts"
    parts = sorted(
        f
        for f in huggingface_hub.list_repo_files(MANATTS_REPO, repo_type="dataset")
        if f.endswith(".parquet")
    )
    logger.info(f"manatts: {len(parts)} parquet parts, target {budget_h:.1f}h")
    ckpt = SourceCheckpoint(out_dir, "manatts")
    kept_h = ckpt.hours
    bar = tqdm(
        total=budget_h,
        initial=min(kept_h, budget_h),
        unit="h",
        desc="manatts",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}h",
    )
    for part in parts:
        if kept_h >= budget_h:
            break
        if part in ckpt.done:
            continue
        rows: list[dict] = []
        path, delete_after = download_archive(MANATTS_REPO, part, keep)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=8,
            columns=["file name", "transcript", "duration", "match quality", "audio", "samplerate"],
        ):
            names = batch.column("file name").to_pylist()
            texts = batch.column("transcript").to_pylist()
            durs = batch.column("duration").to_pylist()
            quals = batch.column("match quality").to_pylist()
            srs = batch.column("samplerate").to_pylist()
            audio = batch.column("audio")
            for i, name in enumerate(names):
                if kept_h >= budget_h:
                    break
                if min_quality == "HIGH" and (quals[i] or "").upper() != "HIGH":
                    continue
                stem = Path(str(name)).stem
                dest = sharded_path(out_root, stem + ".flac")
                if not dest.exists():
                    wav = np.asarray(
                        audio[i].values.to_numpy(zero_copy_only=False), dtype=np.float32
                    )
                    sr = int(srs[i] or 44100)
                    if sr != TARGET_SR:
                        wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=TARGET_SR)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".flac.part")
                    soundfile.write(str(tmp), wav, TARGET_SR, format="FLAC", subtype="PCM_16")
                    tmp.rename(dest)
                duration = float(durs[i] or 0.0)
                rows.append(make_row(dest, duration, texts[i] or "", "manatts_narrator", "manatts"))
                kept_h += duration / 3600
                bar.update(min(duration / 3600, max(0.0, bar.total - bar.n)))
        if delete_after:
            drop_from_cache(path)
        ckpt.add(part, rows)
    bar.close()
    logger.info(f"manatts: {len(ckpt.rows)} utterances (~{ckpt.hours:.1f}h)")
    return ckpt.rows


def source_persets(
    repo: str, tag: str, audio_out: Path, out_dir: Path, budget_h: float, keep: bool
) -> list[dict]:
    """Filimo / YouTube: `data/unvalidated_*.tar` of mp3 + one metadata csv."""
    out_root = audio_out / tag
    files = huggingface_hub.list_repo_files(repo, repo_type="dataset")
    tars = sorted(
        (f for f in files if f.startswith("data/") and f.endswith(".tar")),
        key=lambda f: int("".join(c for c in Path(f).stem if c.isdigit()) or 0),
    )
    meta_name = next(f for f in files if f.endswith(".csv") and "unvalidated" in f)
    meta_path = Path(huggingface_hub.hf_hub_download(repo, meta_name, repo_type="dataset"))
    logger.info(f"{tag}: {len(tars)} archives, reading {meta_name}")
    text_of = {}
    for row in read_delimited(meta_path):
        name = row.get("file_name") or row.get("file name")
        text = row.get("sentence") or row.get("text")
        if name and text:
            text_of[Path(name).stem] = text

    ckpt = SourceCheckpoint(out_dir, tag)
    kept_h = ckpt.hours
    bar = tqdm(
        total=budget_h,
        initial=min(kept_h, budget_h),
        unit="h",
        desc=tag,
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}h",
    )
    for tar_name in tars:
        if kept_h >= budget_h:
            break
        if tar_name in ckpt.done:
            continue
        rows: list[dict] = []
        tar_path, delete_after = download_archive(repo, tar_name, keep)
        n = extract_audio(tar_path, out_root / Path(tar_name).stem)
        if delete_after:
            drop_from_cache(tar_path)
        shard_root = out_root / Path(tar_name).stem
        paths = sorted(p for p in shard_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        logger.info(f"{tag}: {tar_name} -> {n} clips")
        for path, duration in zip(paths, probe_durations(paths)):
            if duration is None:
                continue
            text = text_of.get(path.stem)
            if not text:
                continue
            # "0053700001.mp3": the first five digits identify the source video,
            # which is the closest thing to a speaker id these corpora carry.
            rows.append(make_row(path, duration, text, f"{tag}_{path.stem[:5]}", tag))
            kept_h += duration / 3600
            bar.update(min(duration / 3600, max(0.0, bar.total - bar.n)))
        ckpt.add(tar_name, rows)
    bar.close()
    logger.info(f"{tag}: {len(ckpt.rows)} utterances (~{ckpt.hours:.1f}h)")
    return ckpt.rows


def source_commonvoice(
    audio_out: Path, out_dir: Path, budget_h: float, keep: bool, splits: list[str]
) -> list[dict]:
    """Common Voice fa: per-split tar of mp3 + tsv, durations from clip_durations."""
    out_root = audio_out / "commonvoice"
    durations_tsv = Path(
        huggingface_hub.hf_hub_download(
            CV_REPO, "transcript/fa/clip_durations.tsv", repo_type="dataset"
        )
    )
    clip_ms = {}
    for row in read_delimited(durations_tsv):
        try:
            clip_ms[Path(row["clip"]).stem] = float(row["duration[ms]"]) / 1000
        except (KeyError, TypeError, ValueError):
            continue

    ckpt = SourceCheckpoint(out_dir, "commonvoice")
    kept_h = ckpt.hours
    bar = tqdm(
        total=budget_h,
        initial=min(kept_h, budget_h),
        unit="h",
        desc="commonvoice",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f}h",
    )
    for split in splits:
        if kept_h >= budget_h:
            break
        if split in ckpt.done:
            continue
        rows: list[dict] = []
        tar_path, delete_after = download_archive(
            CV_REPO, f"audio/fa/{split}/fa_{split}_0.tar", keep
        )
        extract_audio(tar_path, out_root / split)
        if delete_after:
            drop_from_cache(tar_path)
        tsv = Path(
            huggingface_hub.hf_hub_download(
                CV_REPO, f"transcript/fa/{split}.tsv", repo_type="dataset"
            )
        )
        for row in read_delimited(tsv):
            if kept_h >= budget_h:
                break
            stem = Path(row["path"]).stem
            path = sharded_path(out_root / split, stem + ".mp3")
            if not path.exists():
                continue
            duration = clip_ms.get(stem)
            if duration is None:
                continue
            speaker = "cv_" + (row.get("client_id") or "")[:16]
            rows.append(make_row(path, duration, row.get("sentence") or "", speaker, "commonvoice"))
            kept_h += duration / 3600
            bar.update(min(duration / 3600, max(0.0, bar.total - bar.n)))
        ckpt.add(split, rows)
    bar.close()
    logger.info(f"commonvoice: {len(ckpt.rows)} utterances (~{ckpt.hours:.1f}h)")
    return ckpt.rows


SOURCES = ("manatts", "filimo", "youtube", "commonvoice")


# ---------------------------------------------------------------------------
# Filtering and splitting
# ---------------------------------------------------------------------------


def clean_rows(
    rows: list[dict], min_sec: float, max_sec: float, max_latin: int = 0
) -> tuple[list[dict], dict]:
    """Normalize transcripts and drop what should not be trained on."""
    stats: defaultdict = defaultdict(int)
    seen: set[tuple[str, str]] = set()
    kept = []
    for row in rows:
        if not (min_sec <= row["duration"] <= max_sec):
            stats["bad_duration"] += 1
            continue
        text = normalize(row["transcript"])
        reason = reject_reason(row["transcript"], text, max_latin=max_latin)
        if reason:
            stats[reason] += 1
            continue
        key = (row["speaker"], text)
        if key in seen:
            # The same line read twice inside one video is a subtitle artifact.
            stats["duplicate"] += 1
            continue
        seen.add(key)
        row["transcript"] = text
        kept.append(row)
        stats["kept"] += 1
    return kept, dict(stats)


def split_rows(
    rows: list[dict], valid_hours: float, max_valid: int, max_fraction: float = 0.05
) -> tuple[list[dict], list[dict]]:
    """Hold out whole speakers, so no valid speaker is also a training speaker.

    The held-out set is capped at `max_fraction` of the corpus as well as at
    `valid_hours`/`max_valid`: on a small pilot corpus an uncapped 1-hour
    target swallows every speaker and leaves nothing to train on.
    """
    by_speaker: defaultdict = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    total_s = sum(r["duration"] for r in rows)
    budget_s = min(valid_hours * 3600, max_fraction * total_s)

    valid: list[dict] = []
    held: set[str] = set()
    if len(by_speaker) > 1:
        # Prefer small speakers: holding out a 100h narrator would cost more
        # training data than the eval is worth.
        order = sorted(by_speaker, key=lambda s: sum(r["duration"] for r in by_speaker[s]))
        valid_s = 0.0
        for speaker in order:
            group = by_speaker[speaker]
            if valid_s >= budget_s or len(valid) + len(group) > max_valid:
                break
            valid.extend(group)
            held.add(speaker)
            valid_s += sum(r["duration"] for r in group)
    train = [r for r in rows if r["speaker"] not in held]
    if not valid or not train:
        # One narrator (Mana-TTS on its own): a speaker-disjoint split is not
        # possible, so hold out utterances instead and say so.
        rng = random.Random(0)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        valid, valid_s = [], 0.0
        for row in shuffled:
            if valid_s >= budget_s or len(valid) >= max_valid:
                break
            valid.append(row)
            valid_s += row["duration"]
        held_paths = {(r["path"], r["start"]) for r in valid}
        train = [r for r in rows if (r["path"], r["start"]) not in held_paths]
        logger.warning(
            f"{len(by_speaker)} speaker(s) in the corpus: the valid split holds out "
            "utterances, not speakers, so it measures reconstruction rather than "
            "generalization to a new voice"
        )
    return train, valid


def write_manifest(rows: list[dict], path: Path) -> float:
    hours = sum(r["duration"] for r in rows) / 3600
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"{path.name}: {len(rows)} utterances (~{hours:.1f}h)")
    return hours


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    hours: Annotated[
        float, typer.Option(help="total training hours to collect, across --sources in order")
    ] = 600.0,
    sources: Annotated[
        str, typer.Option(help=f"comma-separated, in priority order: {','.join(SOURCES)}")
    ] = "manatts,filimo,youtube",
    manifests_out: Annotated[
        str | None,
        typer.Option(help="where the manifests are written (default: data/farsi_<hours>h)"),
    ] = None,
    audio_out: Annotated[
        str, typer.Option(help="where the decoded audio lives; manifests point into it")
    ] = "data/farsi_audio",
    min_sec: Annotated[float, typer.Option(help="drop utterances shorter than this")] = 2.0,
    max_sec: Annotated[float, typer.Option(help="drop utterances longer than this")] = 30.0,
    max_latin: Annotated[
        int,
        typer.Option(
            help="Latin characters tolerated in a transcript before the utterance is dropped. "
            "0 is right for most corpora: normalization deletes Latin script, and training "
            "audio that says an English word against a transcript without it teaches the "
            "model to hallucinate. Raise it only for heavily code-switched data."
        ),
    ] = 0,
    valid_hours: Annotated[float, typer.Option(help="hours held out for validation")] = 1.0,
    max_valid: Annotated[int, typer.Option(help="cap on held-out utterances")] = 1000,
    manatts_quality: Annotated[
        str, typer.Option(help="Mana-TTS match quality to keep: HIGH or ANY")
    ] = "HIGH",
    cv_splits: Annotated[
        str, typer.Option(help="Common Voice splits to pull (audio exists for these only)")
    ] = "train,dev,test",
    keep_archives: Annotated[
        bool,
        typer.Option(help="keep downloaded tars/parquet in the HF cache (needs ~70 GB more disk)"),
    ] = False,
    align_shards: Annotated[
        int, typer.Option(help="parallel alignment processes (one GPU each)")
    ] = 1,
    align_model: Annotated[str, typer.Option()] = DEFAULT_ALIGN_MODEL,
    skip_align: Annotated[bool, typer.Option(help="stop after building raw manifests")] = False,
    train_tokenizer: Annotated[
        bool, typer.Option(help="also train a Persian sentencepiece tokenizer on the transcripts")
    ] = True,
    vocab_size: Annotated[
        int, typer.Option(help="tokenizer vocab; the model config's n_bins must match exactly")
    ] = 4000,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%d-%m %H:%M:%S",
    )
    # One INFO line per HTTP redirect buries this script's own progress in a
    # log that is thousands of downloads long.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    wanted = [s.strip() for s in sources.split(",") if s.strip()]
    unknown = set(wanted) - set(SOURCES)
    if unknown:
        raise typer.BadParameter(f"unknown source(s): {sorted(unknown)}; known: {SOURCES}")

    out_dir = Path(manifests_out or f"data/farsi_{hours:g}h")
    audio_root = Path(audio_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)

    raw_manifest = out_dir / "all_raw.jsonl"
    rows: list[dict] = []
    if raw_manifest.exists():
        logger.info(f"reusing {raw_manifest} (delete it to re-collect)")
        rows = [json.loads(line) for line in raw_manifest.open()]
    else:
        remaining = hours
        for name in wanted:
            if remaining <= 0:
                break
            logger.info(f"=== {name}: up to {remaining:.1f}h ===")
            if name == "manatts":
                got = source_manatts(
                    audio_root, out_dir, remaining, keep_archives, manatts_quality.upper()
                )
            elif name == "commonvoice":
                got = source_commonvoice(
                    audio_root,
                    out_dir,
                    remaining,
                    keep_archives,
                    [s.strip() for s in cv_splits.split(",")],
                )
            else:
                repo = FILIMO_REPO if name == "filimo" else YOUTUBE_REPO
                got = source_persets(repo, name, audio_root, out_dir, remaining, keep_archives)
            rows.extend(got)
            remaining -= sum(r["duration"] for r in got) / 3600
        with open(raw_manifest, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept, stats = clean_rows(rows, min_sec, max_sec, max_latin)
    logger.info(f"text/duration filtering: {stats}")
    if not kept:
        raise SystemExit("every utterance was filtered out -- check the stats above")
    train, valid = split_rows(kept, valid_hours, max_valid)
    train_m, valid_m = out_dir / "train.jsonl", out_dir / "valid.jsonl"
    train_h = write_manifest(train, train_m)
    write_manifest(valid, valid_m)

    if train_tokenizer:
        prefix = out_dir / "tokenizer"
        if Path(str(prefix) + ".model").exists():
            logger.info(f"{prefix}.model exists, skipping tokenizer training")
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.scripts.train_tokenizer",
                    str(prefix),
                    str(train_m),
                    "--vocab-size",
                    str(vocab_size),
                ],
                check=True,
            )
        logger.info(
            f"tokenizer at {prefix}.model -- set lookup_table.tokenizer_path to it and "
            f"lookup_table.n_bins to {vocab_size} in your model config"
        )

    if skip_align:
        logger.info(f"Done (unaligned). Manifests in {out_dir.resolve()}")
        return
    train_a, valid_a = out_dir / "train_aligned.jsonl", out_dir / "valid_aligned.jsonl"
    align(train_m, train_a, align_shards, align_model, "training manifest")
    align(valid_m, valid_a, 1, align_model, "valid manifest")
    logger.info(
        f"Done. ~{train_h:.0f}h of training audio. Point data.train_jsonl at "
        f"{train_a.resolve()} and data.valid_jsonl at {valid_a.resolve()}"
    )


if __name__ == "__main__":
    app()
