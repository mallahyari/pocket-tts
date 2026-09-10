#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "sphn", "numpy", "tqdm", "typer", "accelerate"]
# ///
"""Check that a manifest's time windows actually contain the speech they claim.

Run this before spending GPU time aligning or training on freshly ingested data.
`ingest_farsi_asr_yt.py` trusts subtitle timings to locate speech inside hour-long
recordings; if those timings are wrong, every downstream step is built on sand and
the failure is silent -- training simply learns from mismatched audio and text.

    uv run validate_ingest.py --manifest /mnt/data/farsi_600h/farsi_asr_yt.jsonl
    uv run validate_ingest.py --manifest ... --offsets " -1,-0.5,0,0.5,1"

Reads each sampled window out of its recording exactly as the dataloader will,
transcribes it, and compares against the manifest transcript -- both through
`normalize_fa` so the comparison is not measuring formatting.

**The offset sweep is the point.** A single WER number cannot distinguish "the
transcripts are noisy" from "every window is shifted half a second". Re-scoring
the same clips at several time shifts separates them: if WER is lowest at 0, the
timings are right and whatever error remains is transcript noise. If it bottoms
at some other shift, there is a systematic offset -- which is *correctable*, by
adjusting `start` across the manifest, rather than a reason to discard the data.

Expect roughly 25-35% WER at offset 0 for this corpus. Persian ASR on
conversational YouTube audio is simply not better than that (`../RESULTS.md`
records a 13.4% floor on *studio* audio), so treat the shape of the curve as the
signal, not the absolute value.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import sphn
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from training.farsi.normalize_fa import normalize  # noqa: E402

ASR = "openai/whisper-large-v3"
ASR_SR = 16000


def collapse(text: str) -> str:
    text = normalize(text).replace("‌", " ")
    return re.sub(r"\s+", " ", text).strip()


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m], n


def read_window(row: dict, offset: float) -> np.ndarray | None:
    """The dataloader's own read path, shifted by `offset` seconds."""
    start = float(row.get("start", 0.0)) + offset
    if start < 0:
        return None
    try:
        wav, sr = sphn.read(row["path"], start_sec=start, duration_sec=float(row["duration"]))
    except Exception:
        return None
    wav = wav.mean(axis=0)
    if sr != ASR_SR:  # whisper wants 16k
        idx = np.linspace(0, len(wav) - 1, int(len(wav) * ASR_SR / sr))
        wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
    return wav if len(wav) > ASR_SR // 2 else None


# Below this margin, a difference between offsets is sampling noise rather than a
# real shift. Set from observation: a tie at 23.8% once got reported as a
# "systematic offset ... 0.0% better", which would send someone chasing nothing.
MIN_GAIN = 0.03


def verdict(results: dict[float, tuple[int, int, int]], offsets: list[float]) -> list[str]:
    """Read the offset sweep. Pure arithmetic, so it can be tested without an ASR."""
    return _verdict(results, offsets)[0]


def _verdict(
    results: dict[float, tuple[int, int, int]], offsets: list[float]
) -> tuple[list[str], int]:
    """(report lines, exit code). 0 = usable, 2 = offset, 3 = bad transcripts, 4 = no data."""
    if 0.0 not in results or not results[0.0][1]:
        return ["(no usable windows at offset 0 — cannot judge)"], 4
    scored = {o: (e / n if n else 9e9) for o, (e, n, _) in results.items()}
    base = scored[0.0]
    # sorted by |offset| first so that ties resolve toward 0 rather than toward
    # whichever offset happens to come first in the dict.
    best = min(sorted(scored, key=lambda o: abs(o)), key=lambda o: scored[o])
    gain = base - scored[best]

    if len(offsets) > 1 and best != 0.0 and gain >= MIN_GAIN:
        return [
            f"⚠️  WER is lowest at {best:+.1f}s, not 0 — {gain:.1%} better than the manifest's own",
            "    timings. That is a SYSTEMATIC OFFSET. Correct `start` across the manifest",
            "    rather than discarding the data, then re-run this.",
        ], 2
    if len(offsets) > 1 and best != 0.0:
        return [
            f"ℹ️  {best:+.1f}s scores {gain:.1%} better than 0 — under the {MIN_GAIN:.0%} noise",
            "    floor, so treat the timings as correct rather than shifting them.",
            f"    WER at offset 0 is {base:.1%}.",
        ], 0
    if base <= 0.40:
        return [
            f"✅ WER {base:.1%} at offset 0, and no shift beats it — windows contain the right",
            "   speech. Remaining error is transcript noise, normal for this corpus.",
        ], 0
    return [
        f"⚠️  WER {base:.1%} at offset 0 is high and no shift improves it. Not a timing",
        "    problem — inspect the transcripts themselves before training on this.",
    ], 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="windows to sample")
    ap.add_argument(
        "--offsets",
        default="0",
        help='comma-separated seconds to test, e.g. " -1,-0.5,0,0.5,1" (lead with a space)',
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    offsets = [float(x) for x in args.offsets.replace(" ", "").split(",") if x]

    rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    print(f"manifest: {len(rows):,} rows")
    rng = random.Random(args.seed)
    # Sample across distinct recordings: consecutive rows come from one video, so
    # a contiguous slice would test one file and tell you nothing about the rest.
    by_path: dict[str, list[dict]] = {}
    for r in rows:
        by_path.setdefault(r["path"], []).append(r)
    paths = sorted(by_path)
    rng.shuffle(paths)
    # One row per recording first, for maximum spread; only then take seconds and
    # thirds. Stopping after one pass would silently under-sample any manifest
    # with fewer recordings than --n.
    sample: list[dict] = []
    pools = {p: rng.sample(by_path[p], len(by_path[p])) for p in paths}
    while len(sample) < args.n and any(pools.values()):
        for p in paths:
            if pools[p]:
                sample.append(pools[p].pop())
                if len(sample) >= args.n:
                    break
    n_recordings = len({r["path"] for r in sample})
    print(f"sampled {len(sample)} windows from {n_recordings} distinct recordings\n")

    from transformers import pipeline  # noqa: PLC0415

    print(f"loading {ASR} on {args.device} ...")
    asr = pipeline(
        "automatic-speech-recognition",
        model=ASR,
        device=torch.device(args.device),
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    gen = {"language": "fa", "task": "transcribe"}

    results: dict[float, tuple[int, int, int]] = {}
    for off in offsets:
        errs = words = skipped = 0
        for i in tqdm(range(0, len(sample), args.batch_size), desc=f"offset {off:+.1f}s", unit="batch"):
            batch = sample[i : i + args.batch_size]
            wavs, refs = [], []
            for row in batch:
                wav = read_window(row, off)
                if wav is None:
                    skipped += 1
                    continue
                wavs.append(wav)
                refs.append(collapse(row["transcript"]))
            if not wavs:
                continue
            out = asr(wavs, generate_kwargs=gen, batch_size=len(wavs))
            for ref, o in zip(refs, out):
                e, n = wer(ref.split(), collapse(o["text"]).split())
                errs += e
                words += n
        results[off] = (errs, words, skipped)

    print("\n" + "=" * 56)
    print(f"{'offset':>8}  {'WER':>8}  {'words':>9}  {'unreadable':>11}")
    print("-" * 56)
    for off in offsets:
        e, n, sk = results[off]
        print(f"{off:>+7.1f}s  {e / n if n else float('nan'):>7.1%}  {n:>9,}  {sk:>11}")
    print("=" * 56)

    lines, code = _verdict(results, offsets)
    for line in lines:
        print(line)
    # Non-zero so an orchestrator can gate on this rather than scraping stdout.
    raise SystemExit(code)

if __name__ == "__main__":
    main()
