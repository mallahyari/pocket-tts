#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "sentencepiece", "protobuf", "tqdm"]
# ///
"""Rewrite a training manifest's transcripts from Persian script into phonemes.

This is step one of the phoneme-frontend retrain (Tier 1 A). Persian omits short
vowels and never writes the ezafe, so a grapheme-input model has to infer
pronunciation from context and often does not -- `RESULTS.md` calls a phoneme
front-end "the single highest-value improvement available".

Uses [mehdi-hf/Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF),
chosen by benchmark over the alternatives (PER 3.22%, homograph 70.28%,
ezafe F1 85.04% on SentenceBench).

    uv run phonemize_manifest.py --manifest data/farsi_600h/train_aligned.jsonl \\
        --out data/farsi_600h/train_aligned_ph.jsonl

**The `words` list matters as much as the transcript.** The dataloader's cut
augmentation builds its text from `words[i:]` -- phonemising only `transcript`
would feed *grapheme* words to a phoneme-trained tokenizer on every cut sample.
So each word is rewritten too, positionally, whenever G2P preserves the word
count. When it does not (G2P sometimes merges or splits), the `words` field is
**dropped** for that utterance: `loader._sample` then falls back to the whole
transcript with a random-window voice prompt, which is a valid training sample.
Silently keeping mismatched words would not be.

Resumable: rows already present in `--out` are skipped, so a preempted run
continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

G2P_REPO = "mehdi-hf/Homo-GE2PE-Persian-HF"

# GE2PE's own romanisation -> the conventional one. `1` is its internal ezafe
# flag, not a phoneme, and it survives inconsistently (its rules() only strips it
# when word counts line up), so removing it unconditionally is what makes the
# output uniform enough to train a tokenizer on.
_TO_REF = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


# GE2PE renders "؟" as "@", the same symbol it uses for the glottal stop, so a
# question mark arrives as a glottal stop the speaker never utters -- 6.6% of
# utterances, measured on the corpus.
#
# It cannot be undone afterwards. The number of "@" a question mark adds is not
# fixed: "زده؟" gains two, while "موقع؟" and "جمع؟" gain one because their ع
# already contributes its own. Counting trailing symbols therefore cannot tell
# a question mark from a real glottal stop, and guessing wrong corrupts the
# word. Removing the mark before G2P sees it leaves only genuine glottal stops.
_QUESTION_MARKS = "؟?"


def strip_question_marks(text: str) -> str:
    """Drop "؟" before phonemisation. Every other punctuation mark is dropped by
    G2P itself; this one is not, because it shares a symbol with a phoneme."""
    return text.translate({ord(c): None for c in _QUESTION_MARKS})


def to_phonemes(text: str) -> str:
    return text.translate(_TO_REF).replace("1", "")


class G2P:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(G2P_REPO)
        self.model = T5ForConditionalGeneration.from_pretrained(G2P_REPO).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> list[str]:
        # add_special_tokens=False and 5 beams match how the model was trained;
        # changing either measurably degrades it.
        texts = [strip_question_marks(t) for t in texts]
        enc = self.tok(
            texts, padding=True, add_special_tokens=False, return_attention_mask=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=5,
            min_length=1,
            max_length=512,
            early_stopping=True,
        )
        return [to_phonemes(d.strip()) for d in self.tok.batch_decode(out, skip_special_tokens=True)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (for a smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"read {len(rows):,} utterances")

    done = 0
    if args.out.exists():
        with open(args.out) as f:
            done = sum(1 for line in f if line.strip())
        if done:
            print(f"resuming: {done:,} already written")
    rows = rows[done:]
    if not rows:
        print("nothing to do")
        return

    print(f"loading G2P on {args.device} ...")
    g2p = G2P(args.device)

    kept_words = dropped_words = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as out:
        for i in tqdm(range(0, len(rows), args.batch_size), unit="batch"):
            batch = rows[i : i + args.batch_size]
            phon = g2p([r["transcript"] for r in batch])
            for row, ph in zip(batch, phon):
                row["transcript_graphemes"] = row["transcript"]  # keep for debugging
                row["transcript"] = ph
                words = row.get("words")
                if words:
                    parts = ph.split()
                    if len(parts) == len(words):
                        for w, p in zip(words, parts):
                            w["word"] = p
                        kept_words += 1
                    else:
                        # Timings no longer map onto the text; the loader's
                        # no-alignment path is correct, mismatched words are not.
                        row.pop("words", None)
                        dropped_words += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    total = kept_words + dropped_words
    print(f"\nwrote {args.out}")
    if total:
        print(f"  word alignment kept:    {kept_words:,} / {total:,} = {kept_words / total:.1%}")
        print(f"  word alignment dropped: {dropped_words:,} / {total:,} = {dropped_words / total:.1%}")
        print("  (dropped rows still train, via the whole-transcript + random-prompt path)")


if __name__ == "__main__":
    main()
