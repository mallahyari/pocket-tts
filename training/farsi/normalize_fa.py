"""Persian (Farsi) text normalization for TTS training manifests.

Persian transcripts from public corpora mix Arabic and Persian codepoints for
the same letter, carry optional diacritics no reader pronounces separately,
and spell numbers in three different digit systems. The text tokenizer, the
CTC aligner and the model all see the transcript verbatim, so the same string
written two ways becomes two unrelated token sequences and the model has to
learn both. This module folds those variants onto one spelling.

What it does, in order:
  - strip bidi/zero-width control marks (but keep U+200C, the ZWNJ, which is
    phonemically meaningful in Persian: "می‌رود" is one word, "می رود" is two)
  - fold Arabic letter forms onto Persian ones (ي->ی, ك->ک, ة->ه, أإٱ->ا, ؤ->و)
  - drop harakat/tashdid/tatweel
  - convert Persian and Arabic-Indic digits to words ("۱۴۰۲" -> "هزار و چهارصد و دو")
  - fold punctuation onto the small set that survives into the tokenizer
  - collapse whitespace and stray ZWNJ

`ALIGNER_ALPHABET` is the vocabulary of the Persian wav2vec2 CTC checkpoints
used for forced alignment (m3hrdadfi/SLPL/masoudmzb all share it). Anything
outside it is silently dropped by the aligner, so `reject_reason` refuses
utterances that would arrive there mangled instead of letting them through as
mis-timed training rows.

Usage as a library:
    from training.farsi.normalize_fa import normalize, reject_reason

Usage as a CLI (normalize the transcripts of a manifest in place-ish):
    python -m training.farsi.normalize_fa data/farsi/train.jsonl data/farsi/train_norm.jsonl
    python -m training.farsi.normalize_fa data/farsi/train.jsonl --stats-only
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(pretty_exceptions_show_locals=False)

ZWNJ = "‌"

# The 32 letters of the Persian alphabet, in the spelling the aligner expects.
PERSIAN_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی" + "ئ"
# Exactly the alphabet of m3hrdadfi/wav2vec2-large-xlsr-persian-v3 (and of
# SLPL/Sharif-wav2vec2 and masoudmzb/wav2vec2-xlsr-multilingual-53-fa).
ALIGNER_ALPHABET = set(PERSIAN_LETTERS + ZWNJ)

# Punctuation kept in the transcript: it carries prosody (pauses, question
# intonation) and the tokenizer learns it. Everything else is dropped.
KEPT_PUNCT = ".،؛؟!:"

ALLOWED = ALIGNER_ALPHABET | set(KEPT_PUNCT) | {" "}

# Letter-form folding. Arabic codepoints that look identical to Persian ones
# but are a different character to every downstream model.
CHAR_MAP = {
    "ي": "ی",  # ARABIC YEH
    "ى": "ی",  # ALEF MAKSURA
    "ۍ": "ی",  # YEH WITH TAIL
    "ې": "ی",  # YEH WITH TWO DOTS BELOW
    "ك": "ک",  # ARABIC KAF
    "ڪ": "ک",  # SWASH KAF
    "ة": "ه",  # TEH MARBUTA
    "ۀ": "ه",  # HEH WITH YEH ABOVE
    "أ": "ا",  # ALEF WITH HAMZA ABOVE
    "إ": "ا",  # ALEF WITH HAMZA BELOW
    "ٱ": "ا",  # ALEF WASLA
    "ٲ": "ا",
    "ٳ": "ا",
    "ؤ": "و",  # WAW WITH HAMZA
    "ۂ": "ه",
    "ۃ": "ه",
    "ء": "",  # bare hamza: not spoken on its own
    "ـ": "",  # tatweel
}
# Harakat, tashdid, sukun, superscript alef, hamza-above/below combining marks.
DIACRITICS = re.compile(r"[ً-ٰٕۖ-ۭٖ-ٟ]")
# Bidi controls, BOM, and every zero-width mark EXCEPT U+200C.
INVISIBLES = re.compile(r"[​‍‎‏‪-‮⁦-⁩﻿­]")

DIGIT_MAP = {chr(0x06F0 + i): str(i) for i in range(10)}  # ۰-۹
DIGIT_MAP.update({chr(0x0660 + i): str(i) for i in range(10)})  # ٠-٩

PUNCT_MAP = {
    ",": "،",
    "?": "؟",
    ";": "؛",
    "٬": "",  # Arabic thousands separator
    "٫": ".",  # Arabic decimal separator
    "…": ".",  # ellipsis
    "»": " ",
    "«": " ",
    "“": " ",
    "”": " ",
    "‘": " ",
    "’": " ",
    '"': " ",
    "'": " ",
    "`": " ",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    "–": " ",  # en dash
    "—": " ",  # em dash
    "−": " ",
    "-": " ",
    "‐": " ",
    "/": " ",
    "\\": " ",
    "*": " ",
    "_": " ",
    "|": " ",
    "×": " ",
    "=": " ",
    "+": " ",
    "،": "،",
    "؛": "؛",
    "؟": "؟",
}

LATIN = re.compile(r"[A-Za-z]")

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = [(10**12, "تریلیون"), (10**9, "میلیارد"), (10**6, "میلیون"), (10**3, "هزار")]
_FRACTION_UNITS = {1: "دهم", 2: "صدم", 3: "هزارم"}
_JOIN = " و "
# Above this many digits a run is read one digit at a time (phone numbers,
# account numbers, timestamps), which is what a speaker actually does.
MAX_NUMBER_DIGITS = 12


def _under_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    if 10 <= n < 20:
        parts.append(_TEENS[n - 10])
        n = 0
    if n > 0:
        parts.append(_ONES[n])
    return _JOIN.join(parts)


def number_to_words(n: int) -> str:
    """A non-negative integer as Persian words ("1402" -> "هزار و چهارصد و دو")."""
    if n == 0:
        return "صفر"
    parts = []
    for value, name in _SCALES:
        if n >= value:
            count = n // value
            n %= value
            # "هزار" rather than "یک هزار", but "یک میلیون" is the normal form.
            head = "" if (count == 1 and value == 10**3) else _under_thousand(count) + " "
            parts.append(f"{head}{name}".strip())
    if n > 0:
        parts.append(_under_thousand(n))
    return _JOIN.join(parts)


def _digits_one_by_one(digits: str) -> str:
    return " ".join("صفر" if d == "0" else _ONES[int(d)] for d in digits)


def _number_token_to_words(match: re.Match) -> str:
    whole, frac = match.group(1), match.group(2)
    # A leading zero marks a digit string that is spoken, not counted: phone
    # numbers, national ids, "۰۹۱۲..." — read those one digit at a time.
    if len(whole) > MAX_NUMBER_DIGITS or (whole.startswith("0") and len(whole) > 1):
        return (
            " " + _digits_one_by_one(whole) + (" " + _digits_one_by_one(frac) if frac else "") + " "
        )
    whole = whole.lstrip("0") or "0"
    words = number_to_words(int(whole))
    if frac:
        frac = frac.rstrip("0")
        if not frac:
            return " " + words + " "
        unit = _FRACTION_UNITS.get(len(frac))
        if unit:
            words += " ممیز " + number_to_words(int(frac)) + " " + unit
        else:
            words += " ممیز " + _digits_one_by_one(frac)
    return " " + words + " "


_NUMBER_RE = re.compile(r"(\d+)(?:[.](\d+))?")


def numbers_to_words(text: str) -> str:
    """Every ASCII digit run replaced by its Persian reading."""
    text = re.sub(r"(?<=\d),(?=\d\d\d\b)", "", text)  # 1,234,567 -> 1234567
    return _NUMBER_RE.sub(_number_token_to_words, text)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize(text: str, *, spell_numbers: bool = True) -> str:
    """Fold `text` onto the canonical Persian spelling used for training.

    The result contains only Persian letters, ZWNJ, spaces and the punctuation
    in KEPT_PUNCT — anything else is dropped, so check `reject_reason` on the
    original if a dropped character means the row should not be trained on.
    """
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLES.sub("", text)
    text = DIACRITICS.sub("", text)
    text = text.translate(str.maketrans(CHAR_MAP))
    text = text.translate(str.maketrans(DIGIT_MAP))
    # Number-internal separators have to go before the digit runs are read:
    # "۱٬۲۵۰٬۰۰۰" is one number, "۳٫۵" is one decimal.
    text = text.replace("٬", "").replace("٫", ".")
    text = re.sub(r"[٪%]", " درصد ", text)
    if spell_numbers:
        text = numbers_to_words(text)
    text = text.translate(str.maketrans(PUNCT_MAP))
    # Anything still outside the allowed set (Latin, emoji, CJK, leftover
    # symbols) becomes a space rather than being glued onto its neighbours.
    text = "".join(c if c in ALLOWED else " " for c in text)
    # ZWNJ only means something between two letters.
    text = re.sub(rf"{ZWNJ}+", ZWNJ, text)
    text = re.sub(rf"\s*{ZWNJ}\s*", lambda m: ZWNJ if " " not in m.group(0) else " ", text)
    text = re.sub(rf"(?<![{PERSIAN_LETTERS}]){ZWNJ}|{ZWNJ}(?![{PERSIAN_LETTERS}])", "", text)
    text = re.sub(rf"\s+([{re.escape(KEPT_PUNCT)}])", r"\1", text)
    text = re.sub(rf"([{re.escape(KEPT_PUNCT)}])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Right-to-left text routinely arrives with the sentence-final period as
    # the FIRST character in logical order (it renders on the left). Leading
    # punctuation says nothing about how the utterance is spoken.
    return text.lstrip(KEPT_PUNCT + " ")


def letter_ratio(text: str) -> float:
    """Share of characters that are Persian letters (spaces excluded)."""
    body = [c for c in text if not c.isspace()]
    if not body:
        return 0.0
    return sum(c in PERSIAN_LETTERS for c in body) / len(body)


def reject_reason(
    original: str,
    normalized: str | None = None,
    *,
    min_chars: int = 3,
    max_chars: int = 400,
    min_letter_ratio: float = 0.75,
    max_latin: int = 0,
) -> str | None:
    """Why this utterance should be dropped, or None to keep it.

    Judged on the ORIGINAL as well as the normalized form: normalization
    silently deletes foreign scripts, and an utterance whose audio says an
    English brand name must not be trained against a transcript with that name
    removed.
    """
    if normalized is None:
        normalized = normalize(original)
    # Checked first: an all-Latin transcript normalizes to the empty string,
    # and "too_short" would hide why the corpus is shrinking.
    if len(LATIN.findall(original)) > max_latin:
        return "latin_script"
    if len(normalized) < min_chars:
        return "too_short"
    if len(normalized) > max_chars:
        return "too_long"
    if letter_ratio(normalized) < min_letter_ratio:
        return "low_letter_ratio"
    if not any(c in PERSIAN_LETTERS for c in normalized):
        return "no_persian_letters"
    return None


def char_histogram(texts) -> Counter:
    counts: Counter = Counter()
    for t in texts:
        counts.update(t)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_jsonl: Annotated[Path, typer.Argument(help="manifest with a 'transcript' field")],
    output_jsonl: Annotated[
        Path | None, typer.Argument(help="where the normalized manifest is written")
    ] = None,
    stats_only: Annotated[
        bool, typer.Option(help="report the character histogram and rejections, write nothing")
    ] = False,
    keep_rejected: Annotated[
        bool, typer.Option(help="normalize but keep rows reject_reason would drop")
    ] = False,
) -> None:
    if output_jsonl is None and not stats_only:
        raise typer.BadParameter("pass an output path or --stats-only")
    reasons: Counter = Counter()
    hist: Counter = Counter()
    kept = total = 0
    out = None if stats_only else open(output_jsonl, "w")
    with open(input_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            text = normalize(row["transcript"])
            reason = reject_reason(row["transcript"], text)
            reasons[reason or "kept"] += 1
            hist.update(text)
            if reason and not keep_rejected:
                continue
            kept += 1
            if out is not None:
                row["transcript"] = text
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out is not None:
        out.close()
    print(f"{kept}/{total} rows kept")
    for reason, n in reasons.most_common():
        print(f"  {reason}: {n}")
    print("characters seen (count, codepoint, char):")
    for char, n in hist.most_common():
        print(
            f"  {n:>10}  U+{ord(char):04X}  {char!r}{'' if char in ALLOWED else '  <-- UNEXPECTED'}"
        )


if __name__ == "__main__":
    app()
