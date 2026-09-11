#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sphn", "sentencepiece", "typer"]
# ///
"""Check a prepared corpus before spending a day of GPU time training on it.

    uv run verify_corpus.py --data /mnt/data/farsi_600h

Every check here exists because its absence already cost something on this
project: a corpus whose latents were keyed to a different manifest, phoneme
transcripts silently normalised to nothing, word timings left behind when their
windows moved, a stale `meta.json` outliving the run that wrote it. None of
those fail loudly on their own -- they surface hours later as a model that will
not learn, and by then the cause is a day behind you.

Exit code is 0 when every check passes, 1 otherwise, so it can gate a launch.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import sphn
import typer

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)

PERSIAN = re.compile(r"[؀-ۿ]")
# Matches fix_onsets: a window whose first 50 ms already reaches half the
# loudness of the whole utterance has speech under way at t=0.
ONSET_WINDOW_S = 0.05
ONSET_RATIO = 0.5


class Report:
    def __init__(self) -> None:
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        typer.echo(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failed += 1
        return ok

    def note(self, label: str, detail: str) -> None:
        typer.echo(f"  [ -- ] {label} — {detail}")


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def clipped_rate(rows: list[dict], sample: int, seed: int) -> tuple[int, int]:
    """(clipped, usable) over a random sample — the outcome fix_onsets targets."""
    rng = random.Random(seed)
    picked = rng.sample(rows, min(sample, len(rows)))
    clipped = usable = 0
    for row in picked:
        start = float(row.get("start", 0.0))
        try:
            wav, sr = sphn.read(
                row["path"], start_sec=start if start > 0 else None,
                duration_sec=float(row["duration"]),
            )
        except Exception:  # noqa: BLE001 -- unreadable rows are counted elsewhere
            continue
        wav = wav.mean(axis=0)
        if wav.size < sr // 10:
            continue
        level = float(np.sqrt((wav.astype(np.float64) ** 2).mean()))
        if level <= 0:
            continue
        usable += 1
        head = wav[: int(ONSET_WINDOW_S * sr)]
        if float(np.sqrt((head.astype(np.float64) ** 2).mean())) / level > ONSET_RATIO:
            clipped += 1
    return clipped, usable


@app.command()
def main(
    data: Path = typer.Option(Path("/mnt/data/farsi_600h"), help="prepared corpus directory"),
    corpus: str = typer.Option("v2_train_ph.jsonl"),
    latents: str = typer.Option("v2_train_ph_latents.jsonl"),
    valid: str = typer.Option("v2_valid_ph.jsonl"),
    tokenizer: str = typer.Option("tokenizer_ph.model"),
    vocab_size: int = typer.Option(4000, help="must equal the model's lookup_table.n_bins"),
    sample: int = typer.Option(300, help="rows to open for the audio-level checks"),
    seed: int = typer.Option(0),
) -> None:
    r = Report()
    corpus_p, latents_p, valid_p = data / corpus, data / latents, data / valid
    meta_p = latents_p.with_name(latents_p.stem + ".meta.json")

    typer.echo("\nfiles")
    for path in (corpus_p, latents_p, valid_p, data / tokenizer, meta_p):
        if not r.check(path.exists(), f"{path.name} exists"):
            typer.echo("\n  cannot continue without it")
            raise typer.Exit(1)

    typer.echo("\nrow counts")
    n_corpus, n_latents = count_lines(corpus_p), count_lines(latents_p)
    r.check(
        n_corpus == n_latents,
        "latents manifest matches the corpus",
        f"{n_corpus:,} vs {n_latents:,}",
    )

    typer.echo("\nlatents metadata")
    meta = json.loads(meta_p.read_text())
    r.check("mimi_hash" in meta, "mimi_hash recorded", str(meta.get("mimi_hash", ""))[:16])
    r.check(meta.get("frame_rate") == 12.5, "frame_rate is 12.5", str(meta.get("frame_rate")))
    # A meta older than the manifest is a leftover from an earlier run, and the
    # trainer trusts it to decide whether the latents are current.
    r.check(
        meta_p.stat().st_mtime >= latents_p.stat().st_mtime - 60,
        "meta.json is not stale",
        f"meta {int(meta_p.stat().st_mtime)} vs manifest {int(latents_p.stat().st_mtime)}",
    )

    typer.echo("\nleftover temporary files")
    strays = sorted(
        p.name for p in data.glob("*")
        if re.search(r"\.(phchunk|phpart|shard)\d+$|\.partial$|\.tmp$", p.name)
    )
    r.check(not strays, "no shard/chunk leftovers", ", ".join(strays[:4]) or "clean")

    typer.echo("\ncorpus content")
    rows = [json.loads(line) for line in corpus_p.open() if line.strip()]
    lat_rows = [json.loads(line) for line in latents_p.open() if line.strip()]
    persian = [i for i, x in enumerate(rows) if PERSIAN.search(x["transcript"])]
    r.check(not persian, "transcripts are phonemes, not Persian", f"{len(persian):,} Persian rows")
    no_graph = sum(1 for x in rows if not x.get("transcript_graphemes"))
    r.check(no_graph == 0, "transcript_graphemes kept for scoring", f"{no_graph:,} missing")
    empty = sum(1 for x in rows if not x["transcript"].strip())
    r.check(empty == 0, "no empty transcripts", f"{empty:,} empty")
    # Reported, not judged. Question marks cannot leak in as glottal stops: the
    # mark is removed before G2P sees it, and a unit test pins that. Checking it
    # from the corpus cannot work, because Persian routinely drops the hamza in
    # spelling -- مبدا for مبدأ, رای for رأی, ارتقای for ارتقاء -- so a word
    # ending in a real glottal stop often carries no glottal letter at all.
    # Judged as a failure, this flagged 557 rows that were every one correct.
    ending_glottal = sum(1 for x in rows if x["transcript"].rstrip().endswith("?"))
    r.note(
        "transcripts ending in a glottal stop",
        f"{ending_glottal:,} of {len(rows):,} — expected: ع and dropped-hamza spellings",
    )

    typer.echo("\ndurations and duplicates")
    durations = [float(x["duration"]) for x in rows]
    too_short = sum(1 for d in durations if d < 1.0)
    too_long = sum(1 for d in durations if d > 30.0)
    r.check(too_short == 0, "no utterance under 1 s", f"{too_short:,} rows")
    # Mimi encodes per utterance and the loader pads to the longest in a batch,
    # so one runaway row inflates memory for everything batched beside it.
    r.check(too_long == 0, "no utterance over 30 s", f"{too_long:,} rows")
    hours = sum(durations) / 3600
    r.note("corpus size", f"{hours:,.0f} h across {len(rows):,} utterances")
    # An identical (file, offset) twice is the same audio under two rows: it
    # trains on that clip twice as often and inflates any eval drawn from it.
    seen = {(x["path"], round(float(x.get("start", 0.0)), 3)) for x in rows}
    dupes = len(rows) - len(seen)
    r.check(dupes == 0, "no duplicate (file, start) rows", f"{dupes:,} duplicates")

    typer.echo("\nword timings")
    bad_span = bad_order = 0
    for x in rows:
        words = [w for w in (x.get("words") or []) if w.get("start") is not None]
        duration = float(x["duration"])
        for w in words:
            if float(w["start"]) < -0.001 or float(w["end"]) > duration + 0.5:
                bad_span += 1
                break
        if any(float(b["start"]) < float(a["start"]) for a, b in zip(words, words[1:])):
            bad_order += 1
    r.check(bad_span == 0, "word times lie inside their window", f"{bad_span:,} rows outside")
    r.check(bad_order == 0, "word times are ordered", f"{bad_order:,} rows out of order")
    with_words = sum(1 for x in rows if x.get("words"))
    r.note("rows carrying alignment", f"{with_words:,} of {len(rows):,} ({with_words / len(rows):.0%})")

    typer.echo("\nlatents files")
    rng = random.Random(seed)
    missing = sum(
        1 for x in rng.sample(lat_rows, min(sample, len(lat_rows)))
        if not (data / x["latents_file"]).exists()
    )
    r.check(missing == 0, f"sampled {sample} latents files present", f"{missing} missing")

    typer.echo("\naudio")
    missing_audio = sum(
        1 for x in rng.sample(rows, min(sample, len(rows))) if not Path(x["path"]).exists()
    )
    r.check(missing_audio == 0, f"sampled {sample} audio files present", f"{missing_audio} missing")

    typer.echo("\ntokenizer")
    import sentencepiece as spm  # noqa: PLC0415 -- only needed here

    sp = spm.SentencePieceProcessor(model_file=str(data / tokenizer))
    r.check(
        sp.get_piece_size() == vocab_size,
        "vocab matches the model's n_bins",
        f"{sp.get_piece_size()} vs {vocab_size}",
    )
    unk = sum(1 for i in sp.encode(rows[0]["transcript"]) if i == sp.unk_id())
    r.check(unk == 0, "corpus text tokenizes without <unk>", f"{unk} unknown pieces")

    typer.echo("\nonsets (the defect fix_onsets targets)")
    # Report by source. Only the subtitle-derived half can be repaired: the v1
    # clips are one per file starting at 0, so there is no audio in front of
    # them to back into. Mixing the two hides whether the repair worked -- the
    # first run read 16% overall and looked like a partial failure, when the
    # repaired half was actually at 0% and the v1 half accounted for all of it.
    repairable = [x for x in rows if "farsi_asr_yt" in x["path"]]
    fixed_src = [x for x in rows if "farsi_asr_yt" not in x["path"]]
    if repairable:
        clipped, usable = clipped_rate(repairable, sample, seed)
        rate = clipped / max(usable, 1)
        r.check(rate <= 0.05, "repaired half has clean onsets",
                f"{clipped}/{usable} = {rate:.0%} (was 41% unrepaired)")
    if fixed_src:
        clipped2, usable2 = clipped_rate(fixed_src, sample, seed)
        r.note("v1 half (cannot be repaired: one clip per file, start 0)",
               f"{clipped2}/{usable2} = {clipped2 / max(usable2, 1):.0%}")

    typer.echo("\nvalidation set")
    n_valid = count_lines(valid_p)
    r.check(n_valid > 0, "validation manifest is non-empty", f"{n_valid:,} rows")
    r.check(
        not (valid_p.with_name(valid_p.stem + "_latents.jsonl")).exists(),
        "validation has no precomputed latents (by design)",
        "train.py refuses a latents manifest for validation",
    )

    typer.echo("")
    if r.failed:
        typer.echo(f"{r.failed} check(s) FAILED — do not start training\n")
        raise typer.Exit(1)
    typer.echo("all checks passed — corpus is ready to train\n")


if __name__ == "__main__":
    app()
