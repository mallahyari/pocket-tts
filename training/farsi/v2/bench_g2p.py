#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "sentencepiece",
#     "pandas",
#     "huggingface-hub",
#     "protobuf",
#     "unidecode",
#     "nltk",
# ]
# ///
"""Benchmark Persian G2P models on SentenceBench, with an ezafe breakdown.

Why this exists: the Homo-GE2PE paper (arXiv 2505.12973) reports PER and
homograph accuracy but *no ezafe metric*, and ezafe is the single failure this
project's TTS model is worst at (see ../RESULTS.md "Known limitations"). The
ezafe number is computable from SentenceBench -- its `phoneme` column marks
ezafe explicitly -- the paper just didn't report it.

    uv run bench_g2p.py                  # both models, all 400 rows
    uv run bench_g2p.py --limit 40       # quick smoke run
    uv run bench_g2p.py --models negara  # just one

SentenceBench is GPL-licensed. It is *downloaded and read* here for evaluation,
never redistributed or trained on -- keep it that way, and never train on it, or
the benchmark stops meaning anything.

The two candidate models use different phoneme conventions:

    reference / GE2PE   man qadr-e to rA mi-dAnam    ezafe explicit, ? = glottal stop
    Negara              man qadre to rA midAnam      ezafe fused, no glottal stop

so everything is compared on a normalized "fused" form (hyphens removed). The
glottal stop is stripped by default for the same reason -- Negara never emits
one, and leaving it in would score that as a phoneme error on every word
starting with ع/ء. Pass --keep-glottal to measure it instead.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, snapshot_download

SENTENCEBENCH = ("MahtaFetrat/SentenceBench", "SentenceBench.csv")
NEGARA = "Reza2kn/negara-g2p-clean-v7.1"
HOMO_GE2PE = "MahtaFetrat/Homo-GE2PE-Persian"

# A trailing -e / -ye is the ezafe. Other hyphens in this notation are different
# clitics and must not be counted: `mi-dAnam` is a verbal prefix, `dast-o` is the
# conjunction "and". Anchoring to the end and to e|ye excludes both.
EZAFE_SUFFIX = re.compile(r"-(ye|e)$")


# GE2PE emits a different romanization than SentenceBench's `phoneme` column,
# even though both come from the same authors:
#     GE2PE  m/n q/dre to ra midan/m      / = short a, a = long ā, @ = glottal
#     ref    man qadr-e to rA mi-dAnam    a = short a, A = long ā, ? = glottal
# Comparing raw strings scores every word wrong (PER ~30%). str.translate remaps
# in a single pass, so a->A cannot clobber the /->a just produced.
#
# The full inventory was derived by diffing character frequencies between model
# output and reference over all 400 rows; the counts match closely enough
# ($ 328 vs S 327, c 66 vs C 63) to confirm notation rather than error.
_GE2PE_TO_REF = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})


def ge2pe_to_ref(text: str) -> str:
    """Map any GE2PE-family output into SentenceBench's convention."""
    # '1' is GE2PE's internal ezafe flag, not a phoneme. Its own rules() drops it
    # with .replace('1',''), but only on sentences whose word counts line up --
    # so strip it unconditionally or the leftovers score as phoneme errors.
    return text.translate(_GE2PE_TO_REF).replace("1", "")


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    """Temporarily run inside `path` (needed by Parsivar's relative data paths)."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------


def normalize(text: str, keep_glottal: bool = False) -> str:
    """Canonical form for cross-convention comparison.

    Case is phonemic in this romanization (A = ā, S = š, C = č), so it is
    deliberately NOT lowercased.
    """
    text = text.strip()
    if not keep_glottal:
        text = text.replace("?", "")
    text = text.replace("-", "")  # fuse clitics: qadr-e -> qadre
    text = re.sub(r"[.,!?؟،؛:«»\"'()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


@dataclass
class EzafeCounts:
    """Ezafe scored per word slot, against the reference's own ezafe marks."""

    correct: int = 0  # ref has ezafe, model produced it
    missing: int = 0  # ref has ezafe, model produced the bare stem
    spurious: int = 0  # ref has no ezafe, model added one
    unaligned: int = 0  # word counts differed; sentence skipped
    other: int = 0  # ref has ezafe but model's word matches neither form

    @property
    def recall(self) -> float:
        """Strict: a wrong stem counts against ezafe even if an ezafe was emitted."""
        d = self.correct + self.missing + self.other
        return self.correct / d if d else float("nan")

    @property
    def recall_lexical(self) -> float:
        """Ezafe decision quality *given* the model got the word right.

        Excludes `other` (the model produced a different word entirely, usually a
        homograph error), which is a lexical failure, not an ezafe failure. This
        is the truer answer to "does it know where ezafe belongs".
        """
        d = self.correct + self.missing
        return self.correct / d if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.correct + self.spurious
        return self.correct / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else float("nan")


def score_ezafe(ref: str, hyp: str, counts: EzafeCounts, keep_glottal: bool) -> None:
    """Compare ezafe slot by slot.

    Ezafe is a suffix in both conventions, so it never changes the word count --
    which makes positional alignment safe once the counts match.
    """
    ref_tokens = normalize(ref, keep_glottal).split()
    hyp_tokens = normalize(hyp, keep_glottal).split()

    # Ezafe marks survive normalization only in the raw string, so re-derive
    # them from the un-normalized reference, token-aligned to the same split.
    raw_tokens = re.sub(r"\s+", " ", ref.strip()).split()
    if not (len(ref_tokens) == len(hyp_tokens) == len(raw_tokens)):
        counts.unaligned += 1
        return

    for raw, hyp_tok in zip(raw_tokens, hyp_tokens):
        m = EZAFE_SUFFIX.search(raw)
        if m:
            stem = normalize(raw[: m.start()], keep_glottal)
            with_ezafe = stem + m.group(1)
            if hyp_tok == with_ezafe:
                counts.correct += 1
            elif hyp_tok == stem:
                counts.missing += 1
            else:
                counts.other += 1
        else:
            bare = normalize(raw, keep_glottal)
            if hyp_tok in (bare + "e", bare + "ye"):
                counts.spurious += 1


@dataclass
class Results:
    name: str
    per_num: int = 0  # char edit distance, summed
    per_den: int = 0
    wer_num: int = 0  # word edit distance, summed
    wer_den: int = 0
    homograph_hit: int = 0
    homograph_total: int = 0
    ezafe: EzafeCounts = field(default_factory=EzafeCounts)
    failures: int = 0
    samples: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def per(self) -> float:
        return self.per_num / self.per_den if self.per_den else float("nan")

    @property
    def wer(self) -> float:
        return self.wer_num / self.wer_den if self.wer_den else float("nan")

    @property
    def homograph_acc(self) -> float:
        return self.homograph_hit / self.homograph_total if self.homograph_total else float("nan")


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class Negara:
    """Plain seq2seq -- loads straight from the Hub with transformers."""

    name = "Negara v7.1"

    @staticmethod
    def to_ref(text: str) -> str:
        """Already emits SentenceBench's convention (long a = `A`, no glottal)."""
        return text

    def __init__(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(NEGARA)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NEGARA).eval()

    def __call__(self, sentences: list[str]) -> list[str]:
        out: list[str] = []
        for s in sentences:
            ids = self.tok(s, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                gen = self.model.generate(**ids, max_length=512)
            out.append(self.tok.decode(gen[0], skip_special_tokens=True))
        return out


class HomoGE2PE:
    """Ships as zipped weights + a vendored GE2PE.py + a bundled Parsivar.

    Not a `from_pretrained()` model, so this unpacks the repo into a cache dir
    and imports the vendored module.
    """

    name = "Homo-GE2PE"

    to_ref = staticmethod(ge2pe_to_ref)

    def __init__(self, workdir: Path) -> None:
        repo = Path(snapshot_download(HOMO_GE2PE))
        workdir.mkdir(parents=True, exist_ok=True)

        # Parsivar.zip holds `Parsivar/` (a package) plus `g2p_resources/` at its
        # root, so it unpacks into workdir and workdir goes on sys.path.
        parsivar = workdir / "Parsivar"
        if not parsivar.exists():
            with zipfile.ZipFile(repo / "assets" / "Parsivar.zip") as z:
                z.extractall(workdir)

        # Python 3.10 moved Iterable out of `collections`; the vendored Parsivar
        # predates that and crashes on import. Upstream's own notebook sed-patches
        # this exact line, so do the same rather than pinning an ancient Python.
        merger = parsivar / "token_merger.py"
        if merger.exists():
            src = merger.read_text(encoding="utf-8")
            if "from collections import Iterable" in src:
                merger.write_text(
                    src.replace("from collections import Iterable", "from collections.abc import Iterable"),
                    encoding="utf-8",
                )

        # The weights zip has config.json etc. at its root, so it must unpack
        # *into* a directory of its own rather than alongside everything else.
        weights = workdir / "homo-ge2pe"
        if not (weights / "config.json").exists():
            with zipfile.ZipFile(repo / "model-weights" / "homo-ge2pe.zip") as z:
                z.extractall(weights)

        sys.path.insert(0, str(workdir))  # for `import Parsivar`
        sys.path.insert(0, str(repo / "assets"))  # for `import GE2PE`

        from GE2PE import GE2PE  # noqa: PLC0415  (vendored; importable only after unpack)

        # Parsivar's Normalizer resolves some of its data files relative to the
        # *working directory* (e.g. './Parsivar/resource/normalizer/N_cctt.txt'),
        # not to __file__, so it only imports cleanly from inside workdir. Scope
        # the chdir to construction and restore it, so --csv-out and friends
        # still resolve against wherever the user actually ran this.
        self.workdir = workdir
        with _chdir(workdir):
            self.model = GE2PE(model_path=str(weights))

    def __call__(self, sentences: list[str]) -> list[str]:
        # generate() batches internally and beam-searches; far faster than one call each.
        # Held inside workdir too, in case normalization lazy-loads more data files.
        with _chdir(self.workdir):
            return [str(x) for x in self.model.generate(sentences, batch_size=16, use_rules=True)]


class HomoGE2PELite:
    """Homo-GE2PE's T5 with Parsivar removed.

    Parsivar is used by GE2PE for exactly one thing -- normalizing the Persian
    input before it reaches the T5 -- and this repo already has a normalizer for
    that (`normalize_fa.py`). Dropping Parsivar removes a 49MB vendored blob, a
    CWD-relative data path, and a Python-3.10 compatibility patch from anything
    we would have to ship to users.

    The risk this measures: GE2PE was *trained* on Parsivar-normalized text, so
    substituting a different normalizer shifts the input distribution. If PER and
    homograph accuracy hold, the T5 can be re-saved standalone and shipped behind
    a plain `from_pretrained()`.

    --models homo-lite      our normalize_fa.py
    --models homo-raw       no normalization at all (isolates how much any of it matters)
    """

    def __init__(self, workdir: Path, normalizer: str = "farsi") -> None:
        from transformers import AutoTokenizer, T5ForConditionalGeneration  # noqa: PLC0415

        self.name = f"Homo lite ({normalizer})"
        repo = Path(snapshot_download(HOMO_GE2PE))
        weights = workdir / "homo-ge2pe"
        if not (weights / "config.json").exists():
            workdir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(repo / "model-weights" / "homo-ge2pe.zip") as z:
                z.extractall(weights)

        self.tok = AutoTokenizer.from_pretrained(weights)
        self.model = T5ForConditionalGeneration.from_pretrained(weights).eval()

        if normalizer == "farsi":
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from training.farsi.normalize_fa import normalize as fa_normalize  # noqa: PLC0415

            self.normalize_input = fa_normalize
        else:
            self.normalize_input = lambda t: t

    to_ref = staticmethod(ge2pe_to_ref)

    def __call__(self, sentences: list[str], batch_size: int = 16) -> list[str]:
        # Mirrors GE2PE.generate() exactly (add_special_tokens=False, 5 beams,
        # harakat stripped) so the only variable under test is the normalizer.
        texts = [self.normalize_input(s).replace("ك", "ک") for s in sentences]
        texts = [t.replace("ِ", "").replace("ُ", "").replace("َ", "") for t in texts]
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[i : i + batch_size],
                padding=True,
                add_special_tokens=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                gen = self.model.generate(
                    enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    num_beams=5,
                    min_length=1,
                    max_length=512,
                    early_stopping=True,
                )
            out += [s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True)]
        return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def evaluate(name: str, hyps: list[str], df: pd.DataFrame, keep_glottal: bool) -> Results:
    res = Results(name=name)
    for hyp, (_, row) in zip(hyps, df.iterrows()):
        ref = str(row["phoneme"])
        if not hyp.strip():
            res.failures += 1
            continue

        ref_n, hyp_n = normalize(ref, keep_glottal), normalize(hyp, keep_glottal)
        # PER is over the phoneme *sequence*, so word boundaries are dropped:
        # the models disagree on where words split (GE2PE emits "beqadre" where
        # the reference has "be qadar-e") and that is a tokenization convention,
        # not a pronunciation error. WER below still measures it.
        res.per_num += edit_distance(ref_n.replace(" ", ""), hyp_n.replace(" ", ""))
        res.per_den += len(ref_n.replace(" ", ""))
        res.wer_num += edit_distance(ref_n.split(), hyp_n.split())
        res.wer_den += len(ref_n.split())

        score_ezafe(ref, hyp, res.ezafe, keep_glottal)

        pron = row.get("pronunciation")
        if isinstance(pron, str) and pron.strip():
            res.homograph_total += 1
            target = normalize(pron, keep_glottal)
            # The homograph often carries an ezafe, which fuses onto it after
            # normalization (qadr + ezafe -> "qadre"). Matching only the bare
            # form would score those correct readings as wrong.
            if any(t in (target, target + "e", target + "ye") for t in hyp_n.split()):
                res.homograph_hit += 1

        if len(res.samples) < 6:
            res.samples.append((str(row["grapheme"]), ref, hyp))
    return res


def report(all_results: list[Results]) -> None:
    print("\n" + "=" * 78)
    print("RESULTS".center(78))
    print("=" * 78)
    head = (
        f"{'model':<16}{'PER':>8}{'WER':>8}{'homogr':>9}"
        f"{'ez-F1':>8}{'ez-rec':>8}{'ez-prec':>9}{'ez-lex':>8}"
    )
    print(head)
    print("-" * 78)
    for r in all_results:
        e = r.ezafe
        print(
            f"{r.name:<16}{r.per:>7.2%}{r.wer:>8.2%}{r.homograph_acc:>9.2%}"
            f"{e.f1:>8.2%}{e.recall:>8.2%}{e.precision:>9.2%}{e.recall_lexical:>8.2%}"
        )
    print("-" * 78)
    print("PER/WER lower is better; the rest higher. Published Homo-GE2PE: PER 3.98%,")
    print("homograph 76.89% (their normalization may differ slightly from ours).")
    print("ez-rec is strict (a wrong stem counts against it); ez-lex isolates the")
    print("ezafe decision by ignoring slots where the model picked a different word.\n")

    for r in all_results:
        e = r.ezafe
        print(f"--- {r.name}: ezafe detail ---")
        print(
            f"  correct {e.correct}  missing {e.missing}  spurious {e.spurious}  "
            f"other {e.other}  | sentences skipped (word-count mismatch): {e.unaligned}"
        )
        print("  'missing' = ezafe genuinely dropped; 'other' = different word chosen.")
        if r.failures:
            print(f"  empty outputs: {r.failures}")
        print()

    for r in all_results:
        print(f"--- {r.name}: samples ---")
        for graph, ref, hyp in r.samples[:4]:
            print(f"  fa   {graph}")
            print(f"  ref  {ref}")
            print(f"  hyp  {hyp}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        default="negara,homo-ge2pe",
        help="comma-separated: negara, homo-ge2pe, homo-lite (no Parsivar), homo-raw (no normalizer)",
    )
    ap.add_argument("--limit", type=int, default=None, help="sample ~N rows, stratified by source (smoke test)")
    ap.add_argument("--keep-glottal", action="store_true", help="score the ? glottal stop instead of stripping it")
    ap.add_argument("--workdir", type=Path, default=Path.home() / ".cache" / "g2p_bench")
    ap.add_argument("--csv-out", type=Path, default=None, help="write per-sentence outputs here")
    args = ap.parse_args()

    csv = hf_hub_download(SENTENCEBENCH[0], SENTENCEBENCH[1], repo_type="dataset")
    df = pd.read_csv(csv)
    if args.limit:
        # The CSV is grouped by source, so head() would return homograph rows
        # only and hide how the models do on ordinary sentences. Stratify.
        frac = args.limit / len(df)
        parts = [g.head(max(1, round(len(g) * frac))) for _, g in df.groupby("dataset", sort=False)]
        df = pd.concat(parts).reset_index(drop=True)
    print(f"SentenceBench: {len(df)} rows")
    print(f"  by source: {df['dataset'].value_counts().to_dict()}")

    sentences = [str(s) for s in df["grapheme"]]
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results: list[Results] = []
    outputs: dict[str, list[str]] = {}

    for key in wanted:
        print(f"\nloading {key} ...")
        try:
            if key == "negara":
                runner: Negara | HomoGE2PE | HomoGE2PELite = Negara()
            elif key == "homo-lite":
                runner = HomoGE2PELite(args.workdir / "homo", normalizer="farsi")
            elif key == "homo-raw":
                runner = HomoGE2PELite(args.workdir / "homo", normalizer="none")
            else:
                runner = HomoGE2PE(args.workdir / "homo")
        except Exception as exc:  # noqa: BLE001 -- one model failing must not sink the run
            print(f"  !! could not load {key}: {type(exc).__name__}: {exc}")
            print("     (skipping it; the other model still runs)")
            continue

        print(f"  running {len(sentences)} sentences ...")
        try:
            hyps = runner(sentences)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! inference failed for {key}: {type(exc).__name__}: {exc}")
            continue

        hyps = [runner.to_ref(h) for h in hyps]  # into SentenceBench's convention
        outputs[runner.name] = hyps
        all_results.append(evaluate(runner.name, hyps, df, args.keep_glottal))

    if not all_results:
        print("\nNo model produced results.")
        raise SystemExit(1)

    report(all_results)

    if args.csv_out:
        out = df[["dataset", "grapheme", "phoneme"]].copy()
        for name, hyps in outputs.items():
            out[name] = hyps
        out.to_csv(args.csv_out, index=False)
        print(f"per-sentence outputs -> {args.csv_out}")


if __name__ == "__main__":
    main()
