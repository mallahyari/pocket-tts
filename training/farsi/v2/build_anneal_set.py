#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Build a clean-but-still-diverse manifest for the final annealing stage.

The idea behind annealing is borrowed from LLM training: after the main run,
spend the last few thousand steps on your cleanest data so the model finishes by
learning what *good* sounds like. It only works if "clean" is also
representative.

**Mana-TTS alone is not.** All 49,746 of its utterances are one narrator
(`manatts_narrator`), so annealing on it would specialise the model to that
voice and put the 0.794 held-out speaker similarity at risk -- the opposite of
what we want. So this keeps every Mana-TTS utterance *and* adds the Filimo and
YouTube utterances whose forced alignment looks as healthy as Mana-TTS's.

The filter uses only what is already in `train_aligned.jsonl` -- no audio, no
GPU. For each utterance the word timings give:

* **coverage** -- what fraction of the clip the aligned words actually occupy.
  A transcript that does not match its audio forces the CTC aligner into a
  path with implausibly short words.
* **words/sec** -- speech rate. Far outside the normal band means the aligner
  crammed or stretched the transcript to fit.
* **max gap** -- the longest silence between consecutive words. Large gaps mean
  audio the transcript does not account for (music, another speaker, noise).

Thresholds are taken from Mana-TTS's own distribution rather than invented, so
"clean" means "as well-aligned as our hand-verified corpus".

    uv run build_anneal_set.py --manifest data/farsi_600h/train_aligned.jsonl \\
        --out data/farsi_600h/anneal_train.jsonl

**This is a proxy, not a verdict.** It measures whether an alignment is
*plausible*, not whether the transcript is *correct* -- a confidently wrong
alignment can still look tidy. Validate before trusting it: transcribe a sample
of passing vs failing clips with whisper and check the WER gap is real. That is
Tier 1 C's job and it needs a GPU; this is the free approximation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

CLEAN_SOURCE = "manatts"  # hand-verified; supplies the reference distribution


def alignment_stats(row: dict) -> tuple[float, float, float] | None:
    """(coverage, words_per_sec, max_gap) from word timings, or None if unusable."""
    words = row.get("words") or []
    duration = float(row.get("duration") or 0.0)
    if not words or duration <= 0:
        return None
    voiced = sum(w["end"] - w["start"] for w in words)
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])] or [0.0]
    return voiced / duration, len(words) / duration, max(gaps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/farsi_600h/train_aligned.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/farsi_600h/anneal_train.jsonl"))
    ap.add_argument(
        "--coverage-pct",
        type=float,
        default=10.0,
        help="keep rows above this percentile of the clean source's coverage (lower = more permissive)",
    )
    ap.add_argument("--wps-pct", type=float, default=5.0, help="words/sec band, as a two-sided percentile")
    ap.add_argument("--gap-pct", type=float, default=90.0, help="max-gap ceiling, as a percentile")
    args = ap.parse_args()

    rows: list[dict] = []
    stats: list[tuple[float, float, float] | None] = []
    with open(args.manifest) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            stats.append(alignment_stats(row))
    print(f"read {len(rows):,} utterances from {args.manifest}")

    ref = np.array([s for r, s in zip(rows, stats) if s and r.get("source") == CLEAN_SOURCE])
    if not len(ref):
        raise SystemExit(f"no `{CLEAN_SOURCE}` rows found -- cannot derive a reference band")
    lo_cov = float(np.percentile(ref[:, 0], args.coverage_pct))
    lo_wps = float(np.percentile(ref[:, 1], args.wps_pct))
    hi_wps = float(np.percentile(ref[:, 1], 100 - args.wps_pct))
    hi_gap = float(np.percentile(ref[:, 2], args.gap_pct))
    print(f"reference band from {len(ref):,} {CLEAN_SOURCE} rows:")
    print(f"  coverage >= {lo_cov:.2f}   words/sec in [{lo_wps:.2f}, {hi_wps:.2f}]   max gap <= {hi_gap:.2f}s")

    kept: list[dict] = []
    seen, dropped = Counter(), Counter()
    for row, s in zip(rows, stats):
        src = row.get("source", "?")
        seen[src] += 1
        # The clean source is kept whole -- it defines the target, and its
        # hand-verified transcripts are the reason this stage exists.
        if src == CLEAN_SOURCE:
            kept.append(row)
            continue
        if s is None:
            dropped[src] += 1
            continue
        cov, wps, gap = s
        if cov >= lo_cov and lo_wps <= wps <= hi_wps and gap <= hi_gap:
            kept.append(row)
        else:
            dropped[src] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}  ({len(kept):,} utterances)")

    # If the input carried precomputed latents, keep them. Each row points at its
    # OWN latents/<tag>/<name>_<idx>.safetensors file, so dropping rows leaves the
    # survivors' files valid -- filtering costs nothing and saves re-encoding the
    # whole subset through Mimi. train.py only reuses them when the sibling
    # .meta.json exists and its mimi_hash matches, so copy that across too, and
    # write the audio-only manifest it derives the name from.
    if kept and "latents_file" in kept[0]:
        src_meta = args.manifest.with_suffix(".meta.json")
        if args.out.name.endswith("_latents.jsonl"):
            audio_out = args.out.with_name(args.out.name.replace("_latents.jsonl", ".jsonl"))
            with open(audio_out, "w") as f:
                for row in kept:
                    f.write(json.dumps({k: v for k, v in row.items() if k != "latents_file"},
                                       ensure_ascii=False) + "\n")
            print(f"wrote {audio_out}  (audio-only sibling)")
            if src_meta.exists():
                shutil.copy(src_meta, args.out.with_suffix(".meta.json"))
                print(f"copied {src_meta.name} -> {args.out.with_suffix('.meta.json').name}"
                      "  (so training reuses the existing latents)")
            else:
                print(f"WARNING: {src_meta} not found -- training will re-encode from audio")
        else:
            print("NOTE: input had latents but --out does not end in '_latents.jsonl';")
            print("      name it that way to have training reuse them.")

    hours = sum(float(r.get("duration") or 0) for r in kept) / 3600
    print(f"  {hours:.1f} h, {len({r.get('speaker') for r in kept}):,} distinct speakers")
    print(f"  {'source':10s} {'kept':>9s} {'of':>9s}   share")
    for src in sorted(seen):
        k = seen[src] - dropped[src]
        print(f"  {src:10s} {k:>9,} {seen[src]:>9,}   {k / seen[src]:5.1%}")

    n_clean = sum(1 for r in kept if r.get("source") == CLEAN_SOURCE)
    print(f"\nspeaker diversity check: {n_clean:,} rows are the single {CLEAN_SOURCE} narrator "
          f"({n_clean / len(kept):.0%} of the set)")
    if n_clean / len(kept) > 0.5:
        print("  WARNING: majority single-speaker -- loosen the filter or annealing will")
        print("           specialise the model to that one voice.")


if __name__ == "__main__":
    main()
