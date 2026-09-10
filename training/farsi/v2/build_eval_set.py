#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "huggingface-hub",
#     "typer",
# ]
# ///
"""Build a clean held-out eval set for the Farsi TTS model from Common Voice.

Why: the v1 held-out set uses subtitle transcripts that whisper-large-v3
disagrees with **48% of the time** (see ../RESULTS.md). That is not an eval set,
it is noise -- and every v2 experiment needs something trustworthy to be judged
against.

Common Voice `fa` is close to ideal for this and was never touched in training:
the v1 corpus was manatts + filimo + youtube (~601 h collected), so there is
zero contamination. Its transcripts are community-*validated* rather than
ASR-derived, and it has thousands of speakers, so a speaker-disjoint set that
measures generalization to unseen voices is easy to build.

    uv run build_eval_set.py                      # ~300 speakers, 2 clips each
    uv run build_eval_set.py --speakers 150       # smaller/faster
    uv run build_eval_set.py --out-dir /data/cv_eval

Output is a manifest + wavs in the format ../eval_fa.py expects:

    uv run python -m training.farsi.eval_fa runs/<run> \\
        --manifest <out-dir>/eval.jsonl --use-ema --reference-floor

**Two clips per speaker is required, not a preference.** eval_fa.py pairs
utterances *within* a speaker (prompt = one, target = another) and skips any
speaker with fewer than two, so one-clip-per-speaker would yield zero items.
Pairs are non-overlapping, so N clips per speaker gives N//2 eval items --
2 clips/speaker maximizes speaker diversity per item, which is the point.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "fsicoli/common_voice_22_0"
SPLITS = ["test", "dev", "train"]  # the splits whose audio ships on the mirror

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from training.farsi.normalize_fa import normalize, reject_reason  # noqa: E402


def load_metadata(splits: list[str]) -> pd.DataFrame:
    """Metadata for splits whose audio actually ships on the HF mirror.

    Deliberately NOT `validated.tsv`: it indexes 317k clips, but the mirror only
    carries audio for train/dev/test (~50k). Selecting from validated silently
    picks speakers whose clips cannot be extracted -- it found 195 of 600 before
    this was pinned down. Each row is tagged with its split so extraction knows
    which tar to open.
    """
    frames = []
    for split in splits:
        tsv = hf_hub_download(REPO, f"transcript/fa/{split}.tsv", repo_type="dataset")
        # Some CV TSVs contain unescaped quotes that trip the default parser.
        d = pd.read_csv(tsv, sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip", low_memory=False)
        d["split"] = split
        frames.append(d)
        print(f"  {split}.tsv: {len(d):,} clips, {d['client_id'].nunique():,} speakers")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="path")

    dur = pd.read_csv(
        hf_hub_download(REPO, "transcript/fa/clip_durations.tsv", repo_type="dataset"), sep="\t"
    )
    dur.columns = ["path", "duration_ms"]
    dur["duration_ms"] = pd.to_numeric(dur["duration_ms"], errors="coerce")
    return df.merge(dur, on="path", how="left")


def filter_rows(df: pd.DataFrame, min_sec: float, max_sec: float, min_words: int) -> pd.DataFrame:
    n0 = len(df)
    reasons: dict[str, int] = {}

    def drop(mask: pd.Series, label: str) -> pd.DataFrame:
        reasons[label] = int((~mask).sum())
        return df[mask]

    # Community validation signal: at least two up-votes and no down-votes.
    df = drop(df["up_votes"].fillna(0) >= 2, "up_votes<2")
    df = drop(df["down_votes"].fillna(0) == 0, "has down_votes")

    secs = df["duration_ms"] / 1000.0
    df = drop(secs.between(min_sec, max_sec), f"duration outside [{min_sec},{max_sec}]s")

    # Same normalizer the model is trained through, so the eval text matches the
    # training distribution exactly and WER is not measuring a formatting gap.
    norm, keep = [], []
    for s in df["sentence"].astype(str):
        n = normalize(s)
        norm.append(n)
        keep.append(reject_reason(s, n) is None and len(n.split()) >= min_words)
    df = df.assign(transcript=norm)
    df = drop(pd.Series(keep, index=df.index), f"rejected by normalize_fa / <{min_words} words")

    print(f"filtered {n0:,} -> {len(df):,}")
    for k, v in reasons.items():
        print(f"    -{v:,}  {k}")
    return df


def pick_speakers(df: pd.DataFrame, n_speakers: int, per_speaker: int, seed: int) -> pd.DataFrame:
    counts = df["client_id"].value_counts()
    eligible = counts[counts >= per_speaker].index.tolist()
    print(f"\nspeakers with >={per_speaker} usable clips: {len(eligible):,}")

    rng = random.Random(seed)
    rng.shuffle(eligible)

    # Common Voice fa skews ~70% male, so take gendered speakers round-robin to
    # avoid handing the eval set that same skew. Unknown-gender speakers backfill.
    first_gender = df.drop_duplicates("client_id").set_index("client_id")["gender"]
    gender_of: dict[str, str] = {}
    by_gender: dict[str, list[str]] = {}
    for spk in eligible:
        g = str(first_gender.get(spk, "")).lower()
        g = "male" if "male" in g and "female" not in g else "female" if "female" in g else "unknown"
        gender_of[spk] = g
        by_gender.setdefault(g, []).append(spk)
    print("  eligible by gender: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_gender.items())))

    chosen: list[str] = []
    pools = [by_gender.get("female", []), by_gender.get("male", [])]
    i = 0
    while len(chosen) < n_speakers and any(pools):
        pool = pools[i % 2]
        if pool:
            chosen.append(pool.pop())
        elif not any(pools):
            break
        i += 1
    for spk in by_gender.get("unknown", []):
        if len(chosen) >= n_speakers:
            break
        chosen.append(spk)

    rows = []
    for spk in chosen[:n_speakers]:
        got = df[df["client_id"] == spk].head(per_speaker)
        rows.append(got)
    out = pd.concat(rows).reset_index(drop=True)

    # Report what was actually selected, not just what was available -- the
    # female pool is small enough that a balanced request can silently degrade.
    # Uses gender_of, not by_gender: the pools are mutated by pop() above, so
    # counting against them reports zero for everything.
    picked: dict[str, int] = {}
    for spk in chosen[:n_speakers]:
        picked[gender_of[spk]] = picked.get(gender_of[spk], 0) + 1
    print(f"selected {out['client_id'].nunique():,} speakers, {len(out):,} clips")
    print("  selected by gender: " + ", ".join(f"{k}={v}" for k, v in sorted(picked.items())))
    if picked.get("female", 0) < picked.get("male", 0) / 2:
        print("  NOTE: female speakers under-represented -- the eligible pool ran out.")
    return out


def extract_audio(sel: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Pull only the selected clips out of their tars and decode to wav.

    Each row carries the split it came from, so only tars that actually hold a
    selected clip get downloaded -- picking few speakers may touch just one.
    """
    # Checked before any tar is fetched: ffmpeg is absent from GCP's Deep
    # Learning images, and discovering that after a 331MB download is a waste.
    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found -- needed to decode Common Voice mp3 to wav.\n"
            "  Debian/Ubuntu:  sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            "  macOS:          brew install ffmpeg"
        )

    tars = sorted(sel["split"].unique())
    wanted = set(sel["path"])
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    found: dict[str, Path] = {}
    for tar_split in tars:
        if len(found) == len(wanted):
            break
        print(f"  scanning {tar_split} tar ({len(wanted) - len(found):,} clips still needed) ...")
        tar_path = hf_hub_download(REPO, f"audio/fa/{tar_split}/fa_{tar_split}_0.tar", repo_type="dataset")
        with tarfile.open(tar_path) as tar:
            for member in tar:
                base = Path(member.name).name
                if base not in wanted or base in found:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                wav = audio_dir / (Path(base).stem + ".wav")
                if not wav.exists():
                    # 24 kHz mono to match Mimi; decoding once here keeps eval
                    # free of mp3-decoder variance.
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-v", "error", "-i", "pipe:0",
                            "-ar", "24000", "-ac", "1", str(wav),
                        ],
                        input=f.read(),
                        check=True,
                    )
                found[base] = wav
                if len(found) == len(wanted):
                    break

    print(f"extracted {len(found):,}/{len(wanted):,} clips")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("cv_eval"))
    ap.add_argument("--speakers", type=int, default=300, help="unseen speakers to include")
    ap.add_argument("--per-speaker", type=int, default=2, help="clips each; 2 -> 1 eval item per speaker")
    ap.add_argument("--min-sec", type=float, default=4.0, help="below ~4s makes a poor voice prompt")
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--splits", default=",".join(SPLITS), help="comma-separated CV splits to draw from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print("loading metadata ...")
    df = load_metadata(splits)
    print(f"pooled: {len(df):,} clips, {df['client_id'].nunique():,} speakers\n")

    df = filter_rows(df, args.min_sec, args.max_sec, args.min_words)
    sel = pick_speakers(df, args.speakers, args.per_speaker, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    found = extract_audio(sel, args.out_dir)

    manifest = args.out_dir / "eval.jsonl"
    n = 0
    with open(manifest, "w") as f:
        for _, r in sel.iterrows():
            wav = found.get(r["path"])
            if wav is None:
                continue
            f.write(
                json.dumps(
                    {
                        "path": str(wav.resolve()),
                        "duration": round(float(r["duration_ms"]) / 1000.0, 3),
                        "transcript": r["transcript"],
                        "speaker": r["client_id"],
                        "sentence_raw": r["sentence"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1

    secs = sel["duration_ms"].sum() / 1000.0
    print(f"\nwrote {manifest}  ({n:,} utterances, {secs / 60:.1f} min audio)")
    print(f"  -> ~{n // 2:,} eval items after eval_fa.py pairs them within speakers")
    print("\nrun it with:")
    print("  uv run python -m training.farsi.eval_fa runs/<run> \\")
    print(f"      --manifest {manifest} --use-ema --reference-floor")


if __name__ == "__main__":
    main()
