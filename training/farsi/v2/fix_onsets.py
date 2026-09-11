#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sphn", "tqdm", "typer"]
# ///
"""Move subtitle-derived windows back to the silence before their first word.

    uv run fix_onsets.py --manifest /mnt/data/farsi_600h/farsi_asr_yt_aligned.jsonl \
        --out /mnt/data/farsi_600h/farsi_asr_yt_onset.jsonl
    uv run fix_onsets.py --manifest ... --dry-run     # measure, change nothing

Subtitle cues are timed to be readable, not to bound speech, so a window
routinely opens partway through a word. Measured on 296 sampled utterances of
the farsi-asr YouTube half, **46% begin already at more than half their own
average loudness** -- speech is underway at t=0. The v1 studio corpus scores 7%
on the same test, so this arrived with the YouTube data.

The model learns what it is shown. Trained on a corpus where nearly half of all
utterances begin mid-sound, it learns that utterances can begin abruptly at full
volume, and the casualty is every quiet onset: a /b/ is a burst and survives, an
/m/ is a low-energy nasal murmur and is swallowed. In listening tests on the v2
teacher, "man" (I) came out as "in" (this) and "mAdar" (mother) as "Adar", while
the same words mid-sentence were clean and an initial /b/ was fine.

**Backing up is the repair, not trimming.** When a window opens mid-word, the
first word's audio is incomplete while the transcript still names it in full, so
extending the start to the preceding silence makes audio and text agree. Of the
clipped windows, 62% have a quiet gap within 750 ms (median 90 ms back, p90
330 ms). The other 38% sit inside continuous speech with nowhere to back up to;
those are dropped, because a window whose audio does not match its transcript
is worse than no window at all.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sphn
import typer
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)

# A window is "clipped" when its first 50 ms already reaches half the loudness of
# the whole utterance. Silence or a soft onset sits far below that; speech already
# in progress sits above it. 0.5 separates the v1 studio corpus (7%) from the
# YouTube half (46%) cleanly, so it is measuring the thing it claims to.
ONSET_WINDOW_S = 0.05
ONSET_RATIO = 0.5

# How far back to look, and what counts as quiet. 750 ms covers the p90 of 330 ms
# with room to spare; past that we are no longer near this utterance.
MAX_BACKUP_S = 0.75
QUIET_WINDOW_S = 0.03
QUIET_RATIO = 0.15
HOP_S = 0.01


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x.astype(np.float64) ** 2).mean())) if x.size else 0.0


def onset_is_clipped(body: np.ndarray, sr: int) -> bool:
    """True when speech is already under way at the start of `body`."""
    level = rms(body)
    if level <= 0:
        return False
    head = body[: max(1, int(ONSET_WINDOW_S * sr))]
    return rms(head) / level > ONSET_RATIO


def find_backup(lead: np.ndarray, sr: int, level: float) -> float | None:
    """Seconds to extend backwards to reach quiet, or None if there is none.

    `lead` is the audio immediately before the window, oldest sample first.
    Searches from the window edge backwards so the result is the *nearest*
    silence: backing up further than necessary drags in the previous word.
    """
    if level <= 0 or lead.size == 0:
        return None
    win = max(1, int(QUIET_WINDOW_S * sr))
    hop = max(1, int(HOP_S * sr))
    for end in range(lead.size, win - 1, -hop):
        if rms(lead[end - win : end]) / level < QUIET_RATIO:
            return (lead.size - end + win) / sr
    return None


def previous_end(ordered: list[dict], pos: int) -> float:
    """End time of the preceding window in the same recording, else 0.

    Backing up past it would duplicate audio another utterance already owns,
    training the model on the same speech under two different transcripts.
    """
    if pos == 0:
        return 0.0
    prev = ordered[pos - 1]
    return float(prev.get("start", 0.0)) + float(prev.get("duration", 0.0))


@app.command()
def main(
    manifest: Path = typer.Option(..., help="aligned manifest to repair"),
    out: Path = typer.Option(None, help="where to write the repaired manifest"),
    dry_run: bool = typer.Option(False, "--dry-run", help="measure only, write nothing"),
    limit: int = typer.Option(0, help="stop after N rows (for a quick look)"),
) -> None:
    if not dry_run and out is None:
        raise typer.BadParameter("pass --out, or --dry-run to measure only")

    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    if limit:
        rows = rows[:limit]
    typer.echo(f"{len(rows):,} rows from {manifest.name}")

    # Group by recording so a window can be bounded by its predecessor.
    by_path: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_path[row["path"]].append(i)
    for idxs in by_path.values():
        idxs.sort(key=lambda i: float(rows[i].get("start", 0.0)))

    clean = clipped = repaired = dropped = unreadable = 0
    backups: list[float] = []
    keep = [True] * len(rows)

    for idxs in tqdm(list(by_path.values()), unit="recording"):
        ordered = [rows[i] for i in idxs]
        for pos, i in enumerate(idxs):
            row = rows[i]
            start = float(row.get("start", 0.0))
            duration = float(row["duration"])
            room = min(MAX_BACKUP_S, max(0.0, start - previous_end(ordered, pos)))
            try:
                wav, sr = sphn.read(
                    row["path"], start_sec=max(0.0, start - room), duration_sec=duration + room
                )
            except Exception:  # noqa: BLE001 -- a bad file is a dropped row, not a crash
                unreadable += 1
                keep[i] = False
                continue
            wav = wav.mean(axis=0)
            pad = int(room * sr)
            body = wav[pad:]
            if body.size < sr // 10:
                unreadable += 1
                keep[i] = False
                continue
            if not onset_is_clipped(body, sr):
                clean += 1
                continue
            clipped += 1
            backup = find_backup(wav[:pad], sr, rms(body))
            if backup is None:
                dropped += 1
                keep[i] = False
                continue
            repaired += 1
            backups.append(backup)
            row["start"] = round(max(0.0, start - backup), 3)
            row["duration"] = round(duration + min(backup, start), 3)

    kept = sum(keep)
    typer.echo(f"\n  clean onsets      {clean:>8,}")
    typer.echo(f"  clipped onsets    {clipped:>8,}  ({clipped / max(len(rows), 1):.0%})")
    typer.echo(f"    repaired        {repaired:>8,}")
    typer.echo(f"    dropped         {dropped:>8,}")
    typer.echo(f"  unreadable        {unreadable:>8,}")
    if backups:
        arr = np.array(backups)
        typer.echo(
            f"  backup: median {np.median(arr) * 1000:.0f} ms, "
            f"p90 {np.percentile(arr, 90) * 1000:.0f} ms, max {arr.max() * 1000:.0f} ms"
        )
    typer.echo(f"\n  keeping {kept:,} of {len(rows):,} rows ({kept / max(len(rows), 1):.1%})")

    if dry_run:
        typer.echo("\n  --dry-run: nothing written")
        return
    with out.open("w") as f:
        for row, ok in zip(rows, keep):
            if ok:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"  wrote {out}")
    typer.echo(
        "\n  Latents are index-keyed to their manifest, so this manifest needs its\n"
        "  own precompute pass -- it cannot reuse the previous one."
    )


if __name__ == "__main__":
    app()
