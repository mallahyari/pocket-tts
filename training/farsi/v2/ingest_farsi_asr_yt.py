#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub", "tqdm", "typer"]  # typer: normalize_fa imports it
# ///
"""Ingest farsi-asr/farsi-asr-dataset's youtube half as long-form training data.

**Why this dataset, specifically.** Every corpus behind v1 ships *pre-cut clips*
-- Filimo and YouTube-ASR extract at subtitle timings, Mana-TTS ships
per-utterance FLAC -- so training utterances average 3.8 s and long-form is
permanently out of distribution. There is no way to lengthen them: the source
recordings were never distributed. See `PLAN.md` / the v2 notes on why merging
the existing manifests is impossible.

This dataset ships **full recordings** (`.opus`, 48 kHz) alongside the
**complete subtitle file** (`.vtt`), so segmentation is ours to choose. That is
what makes 10-20 s utterances possible at all. MIT licensed, ~915 h.

    uv run ingest_farsi_asr_yt.py --out-dir /mnt/data/farsi_asr_yt \\
        --manifest-out /mnt/data/farsi_600h/farsi_asr_yt.jsonl

Resumable per shard. Run with `--shards 2` first to sanity-check the output.

**Audio is transcoded, not copied.** `sphn` -- which the dataloader uses -- cannot
read opus ("unsupported codec"), so each recording becomes 24 kHz mono FLAC:
Mimi's own rate, lossless from the opus decode rather than stacking a second
lossy generation, and ~3.6x the source size (47 GB -> ~169 GB).

**The manifest points *into* the recordings.** Rows carry `start`/`duration`
windows rather than cut files, which is what the format was built for and what
lets the segmentation be changed later without re-ingesting anything.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

REPO = "farsi-asr/farsi-asr-dataset"
SAMPLE_RATE = 24000  # Mimi's rate; storing more would be discarded downstream

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from training.farsi.normalize_fa import normalize, reject_reason  # noqa: E402

TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
TAGS = re.compile(r"<[^>]+>")
SENTENCE_END = (".", "!", "؟", "?", "،", "؛", ":")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def parse_vtt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    pending: tuple[float, float] | None = None
    buf: list[str] = []

    def flush() -> None:
        if pending and buf:
            text = " ".join(buf).strip()
            if text:
                cues.append(Cue(pending[0], pending[1], text))

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = TAGS.sub("", raw).strip()
        m = TS.search(line)
        if m:
            flush()
            buf = []
            g = [int(x) for x in m.groups()]
            pending = (
                g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
            )
        elif line and not line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) and pending:
            buf.append(line)
    flush()
    return cues


def merge_cues(cues: list[Cue], min_sec: float, max_sec: float, max_gap: float) -> list[Cue]:
    """Greedily join consecutive cues into utterances of `min_sec`..`max_sec`.

    Subtitle cues are display-length, not utterance-length -- often a few words.
    Joining them is the whole point of using this corpus, but two limits keep the
    result honest: never span a gap longer than `max_gap` (that is music, silence
    or a scene change, not continuous speech), and prefer to close a segment at
    sentence-ending punctuation so utterances end where a speaker would stop.
    """
    out: list[Cue] = []
    cur: Cue | None = None
    for cue in cues:
        if cur is None:
            cur = Cue(cue.start, cue.end, cue.text)
            continue
        gap = cue.start - cur.end
        would_be = cue.end - cur.start
        if gap > max_gap or would_be > max_sec:
            out.append(cur)
            cur = Cue(cue.start, cue.end, cue.text)
            continue
        cur = Cue(cur.start, cue.end, f"{cur.text} {cue.text}")
        # Long enough and at a natural boundary: stop here rather than running on.
        if cur.end - cur.start >= min_sec and cur.text.rstrip().endswith(SENTENCE_END):
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return [c for c in out if min_sec <= (c.end - c.start) <= max_sec]


def shard_names(limit: int | None, only: str | None = None) -> list[str]:
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    yt = sorted(f for f in files if f.startswith("youtube/") and f.endswith(".tar.gz"))
    if only:
        yt = [f for f in yt if only in f]
        if not yt:
            raise SystemExit(f"no shard matches {only!r}")
    return yt[:limit] if limit else yt


def process_shard(shard: str, out_dir: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = {"videos": 0, "cues": 0, "merged": 0, "kept": 0, "rejected": 0, "no_vtt": 0}
    tar_path = hf_hub_download(REPO, shard, repo_type="dataset")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp, tarfile.open(tar_path) as tar:
        tar.extractall(tmp)
        for vtt in sorted(Path(tmp).rglob("*.vtt")):
            opus = next((p for p in vtt.parent.glob("*.opus")), None)
            if opus is None:
                stats["no_vtt"] += 1
                continue
            stats["videos"] += 1
            video_id = vtt.parent.name
            flac = audio_dir / f"{video_id}.flac"
            if not flac.exists():
                subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-i", str(opus),
                     "-ar", str(SAMPLE_RATE), "-ac", "1", str(flac)],
                    check=True,
                )
            cues = parse_vtt(vtt)
            stats["cues"] += len(cues)
            merged = merge_cues(cues, args.min_sec, args.max_sec, args.max_gap)
            stats["merged"] += len(merged)
            for c in merged:
                norm = normalize(c.text)
                if reject_reason(c.text, norm) is not None or len(norm.split()) < args.min_words:
                    stats["rejected"] += 1
                    continue
                rows.append(
                    {
                        "path": str(flac.resolve()),
                        "start": round(c.start, 3),
                        "duration": round(c.end - c.start, 3),
                        "transcript": norm,
                        # One speaker per video is a proxy: interviews and
                        # podcasts have several. It is good enough for the
                        # speaker-disjoint split and voice-prompt pairing, but
                        # do not read it as a verified speaker label.
                        "speaker": f"fayt_{video_id}",
                        "source": "farsi_asr_yt",
                    }
                )
                stats["kept"] += 1
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="where transcoded FLAC goes")
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=None, help="only the first N shards (smoke test)")
    ap.add_argument(
        "--only",
        default=None,
        help="substring-match a specific shard. Shards vary from 1.7 MB to 450 MB, and "
        "--shards takes them in sorted order, so this is how you grab a small one to test with.",
    )
    ap.add_argument("--min-sec", type=float, default=6.0, help="v1 averaged 3.8s; this is the point")
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--max-gap", type=float, default=1.5, help="never merge across a longer silence")
    ap.add_argument("--min-words", type=int, default=4)
    args = ap.parse_args()

    shards = shard_names(args.shards, args.only)
    done_file = args.manifest_out.with_suffix(".done.txt")
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = [s for s in shards if s not in done]
    print(f"{len(shards)} shards, {len(done)} already done, {len(todo)} to process")

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"videos": 0, "cues": 0, "merged": 0, "kept": 0, "rejected": 0, "no_vtt": 0}
    secs = 0.0
    for shard in tqdm(todo, unit="shard"):
        rows, stats = process_shard(shard, args.out_dir, args)
        with open(args.manifest_out, "a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(done_file, "a") as f:
            f.write(shard + "\n")
        for k, v in stats.items():
            totals[k] += v
        secs += sum(r["duration"] for r in rows)
        tqdm.write(f"  {shard}: {stats['videos']} videos -> {stats['kept']:,} utterances")

    print(f"\nwrote {args.manifest_out}")
    print(f"  videos: {totals['videos']:,}")
    print(f"  subtitle cues: {totals['cues']:,} -> merged into {totals['merged']:,} utterances")
    print(f"  kept {totals['kept']:,}, rejected {totals['rejected']:,} (normalizer / too few words)")
    if totals["kept"]:
        print(f"  {secs / 3600:.1f} h, mean {secs / totals['kept']:.1f}s per utterance "
              f"(v1 corpus averaged 3.8s)")


if __name__ == "__main__":
    main()
