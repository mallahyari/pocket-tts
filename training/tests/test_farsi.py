"""The Farsi pipeline's pure functions: text normalization, manifest
filtering, speaker-disjoint splitting and eval pairing.

Nothing here touches the network, a GPU or a model -- the point is that the
transcripts reaching the tokenizer and the CTC aligner are spelled one way,
and that the held-out split is actually held out.
"""

import json
from pathlib import Path
from types import ModuleType

import pytest

from training.farsi import eval_fa, prepare_data_fa, synthesize
from training.farsi.normalize_fa import (
    ALIGNER_ALPHABET,
    ALLOWED,
    ZWNJ,
    is_phonemic,
    normalize,
    number_to_words,
    reject_reason,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Arabic yeh/kaf are a different codepoint from the Persian ones and
        # are absent from every Persian CTC vocabulary.
        ("كتاب ايران", "کتاب ایران"),
        # Harakat are optional in Persian orthography and never transcribed
        # consistently, so they cannot be allowed to split the vocabulary.
        ("کتابِ خوبی بود", "کتاب خوبی بود"),
        ("مدرسهٔ ما", "مدرسه ما"),
        # Teh marbuta and hamza forms.
        ("نتيجة أول", "نتیجه اول"),
    ],
)
def test_letter_forms_are_folded(raw, expected):
    assert normalize(raw) == expected


def test_leading_punctuation_is_stripped():
    # Common Voice fa is full of rows whose sentence-final period ended up
    # first in logical order.
    assert normalize(".این چیزی نیست") == "این چیزی نیست"


def test_zwnj_between_letters_survives():
    assert normalize("می‌رود") == "می" + ZWNJ + "رود"


def test_stray_zwnj_is_dropped():
    # A ZWNJ next to a space or at a word edge is a typing artifact, not a
    # word boundary, and the aligner would try to align it as a character.
    assert normalize("می ‌رود") == "می رود"
    assert normalize(ZWNJ + "خانه" + ZWNJ) == "خانه"


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "صفر"),
        (17, "هفده"),
        (21, "بیست و یک"),
        (1000, "هزار"),
        (1402, "هزار و چهارصد و دو"),
        (1_000_000, "یک میلیون"),
    ],
)
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


def test_digits_are_spelled_out():
    # Persian and Arabic-Indic digits both appear in scraped transcripts.
    assert normalize("۲۵٪") == "بیست و پنج درصد"
    assert normalize("١٢") == "دوازده"
    assert normalize("۱٬۲۵۰") == "هزار و دویست و پنجاه"
    assert normalize("۳٫۵") == "سه ممیز پنج دهم"


def test_leading_zero_runs_are_read_digit_by_digit():
    # "۰۹۱۲..." is a phone number; nobody says "nine billion, one hundred...".
    assert normalize("۰۹۱۲") == "صفر نه یک دو"


def test_output_stays_inside_the_aligner_alphabet():
    text = normalize("سلام! این تست ۱۲۳ است، (با) «نقل‌قول» و emoji 🎉")
    assert set(text) <= ALLOWED
    assert set(text) - set(" .،؛؟!:") <= ALIGNER_ALPHABET


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("Hello دنیا", "latin_script"),  # the audio says a word we would delete
        ("!!", "too_short"),
        ("سلام دنیا", None),
    ],
)
def test_reject_reason(raw, reason):
    assert reject_reason(raw) == reason


def test_sharded_path_shards_on_the_stem_not_the_extension():
    root = prepare_data_fa.Path("/data/fa")
    assert prepare_data_fa.sharded_path(root, "0053700001.mp3") == root / "001" / "0053700001.mp3"


def test_read_delimited_handles_both_dialects(tmp_path):
    tsv = tmp_path / "meta.tsv"
    tsv.write_text("\tfile_name\tsentence\n0\ta.mp3\tسلام\n", encoding="utf-8")
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("file_name,sentence\na.mp3,سلام\n", encoding="utf-8")
    for path in (tsv, csv_path):
        rows = prepare_data_fa.read_delimited(path)
        assert rows[0]["file_name"] == "a.mp3"
        assert rows[0]["sentence"] == "سلام"


def test_clean_rows_normalizes_and_filters():
    rows = [
        prepare_data_fa.make_row("/a.mp3", 5.0, "كتاب ايران", "spk", "src"),
        prepare_data_fa.make_row("/b.mp3", 0.4, "سلام دنیا", "spk", "src"),  # too short
        prepare_data_fa.make_row("/c.mp3", 5.0, "Hello world", "spk", "src"),  # latin
        prepare_data_fa.make_row("/d.mp3", 5.0, "كتاب ايران", "spk", "src"),  # duplicate
    ]
    kept, stats = prepare_data_fa.clean_rows(rows, min_sec=2.0, max_sec=30.0)
    assert [r["transcript"] for r in kept] == ["کتاب ایران"]
    assert stats["bad_duration"] == 1 and stats["latin_script"] == 1 and stats["duplicate"] == 1


def test_split_rows_holds_out_whole_speakers():
    rows = [
        prepare_data_fa.make_row(f"/{s}{i}.mp3", 10.0, "سلام دنیا", s, "src")
        for s in ("big", "small")
        for i in range(20 if s == "big" else 2)
    ]
    train, valid = prepare_data_fa.split_rows(rows, valid_hours=0.005, max_valid=100)
    assert {r["speaker"] for r in valid} == {"small"}  # the cheap speaker is held out
    assert {r["speaker"] for r in train} == {"big"}
    assert not {r["path"] for r in train} & {r["path"] for r in valid}


def test_eval_pairs_come_from_one_speaker_and_are_distinct(tmp_path):
    manifest = tmp_path / "valid.jsonl"
    rows = [
        prepare_data_fa.make_row(f"/{s}{i}.mp3", 5.0, f"جمله {i}", s, "src")
        for s in ("a", "b")
        for i in range(4)
    ]
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    items = eval_fa.build_items(manifest, num_items=None, seed=0)
    assert items
    for item in items:
        assert item["prompt"]["speaker"] == item["target"]["speaker"]
        assert item["prompt"]["path"] != item["target"]["path"]
        assert item["text"] == item["target"]["transcript"]


def test_wer_text_ignores_punctuation_and_zwnj():
    assert eval_fa.wer_text("می‌رود، خانه؟") == eval_fa.wer_text("می رود خانه")


def test_split_rows_never_holds_out_the_whole_corpus():
    # Many tiny speakers and a 1-hour target: an uncapped split would hold out
    # everything and leave no training data.
    rows = [
        prepare_data_fa.make_row(f"/{i}.mp3", 5.0, "سلام دنیا", f"spk{i}", "src") for i in range(20)
    ]
    train, valid = prepare_data_fa.split_rows(rows, valid_hours=1.0, max_valid=1000)
    assert train and valid
    assert len(valid) < len(rows) * 0.2


def test_split_rows_falls_back_to_utterances_for_one_speaker():
    rows = [
        prepare_data_fa.make_row(f"/{i}.mp3", 60.0, "سلام دنیا", "narrator", "manatts")
        for i in range(100)
    ]
    train, valid = prepare_data_fa.split_rows(rows, valid_hours=1.0, max_valid=1000)
    assert train and valid
    assert not {r["path"] for r in train} & {r["path"] for r in valid}


def test_source_checkpoint_resumes_without_reprocessing(tmp_path):
    # Archives are deleted after processing, so a preempted run must not need
    # them again: the rows it already collected are replayed from disk.
    rows = [
        prepare_data_fa.make_row(f"/{i}.mp3", 5.0, "سلام دنیا", "spk", "filimo") for i in range(3)
    ]
    first = prepare_data_fa.SourceCheckpoint(tmp_path, "filimo")
    assert first.rows == [] and first.done == set()
    first.add("data/unvalidated_001.tar", rows)

    resumed = prepare_data_fa.SourceCheckpoint(tmp_path, "filimo")
    assert len(resumed.rows) == 3
    assert resumed.done == {"data/unvalidated_001.tar"}
    assert resumed.hours == pytest.approx(15.0 / 3600)


def test_source_checkpoints_are_per_source(tmp_path):
    prepare_data_fa.SourceCheckpoint(tmp_path, "filimo").add(
        "a.tar", [prepare_data_fa.make_row("/a.mp3", 5.0, "سلام", "s", "filimo")]
    )
    assert prepare_data_fa.SourceCheckpoint(tmp_path, "youtube").rows == []


def test_preflight_rejects_a_zeroed_encoder(tmp_path, monkeypatch):
    """The check that would have saved a 47k-step run.

    `kyutai/pocket-tts-without-voice-cloning` keeps every Mimi tensor name but
    zeroes the encode path, so `load_state_dict(strict=True)` succeeds and
    `encode_to_latent` silently returns zeros.
    """
    import safetensors.torch
    import torch

    from training.farsi import preflight

    good = tmp_path / "good.safetensors"
    bad = tmp_path / "bad.safetensors"
    live = {f"mimi.encoder.{i}.weight": torch.randn(4, 4) for i in range(3)}
    live["mimi.decoder.0.weight"] = torch.randn(4, 4)
    safetensors.torch.save_file(live, str(good))
    dead = dict(live)
    for k in list(dead):
        if k.startswith("mimi.encoder."):
            dead[k] = torch.zeros(4, 4)
    safetensors.torch.save_file(dead, str(bad))

    monkeypatch.setattr(preflight, "download_if_necessary", lambda p: p)
    assert preflight.check_weights(str(good)) == []
    assert preflight.check_weights(str(bad)) == ["encoder (3 tensors)"]


def test_plot_progress_renders_from_a_torn_log(tmp_path):
    """A preempted run leaves a half-written last line; the plot must survive it."""
    import json

    from training.farsi import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    lines = [
        json.dumps(
            {
                "type": "train",
                "step": s,
                "lr": 2e-4,
                "grad_norm": 0.5,
                "metrics": {"flow_diag": 30.0 - s / 100, "flow_loss": 0.4, "eos_loss": 0.05},
            }
        )
        for s in range(0, 2000, 50)
    ]
    lines.append(json.dumps({"type": "valid", "step": 1000, "metrics": {"flow_diag": 20.0}}))
    lines.append('{"type": "train", "step": 20')  # torn by a preemption
    (run / "progress.jsonl").write_text("\n".join(lines))

    plot_progress.main(str(run), out=None, window=5)
    html = (run / "progress.html").read_text()
    # This synthetic log has flow_diag/flow_loss/eos_loss in metrics plus a
    # top-level grad_norm -- one panel per key actually present, not a fixed
    # constant (different objectives log different metrics; see PANEL_ORDER).
    assert "<svg" in html and html.count("<h2>") == 4


def test_plot_progress_dedupes_resumed_steps(tmp_path):
    # A resume replays steps already in the file; each step must appear once.
    import json

    from training.farsi import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    rows = [{"type": "train", "step": s, "metrics": {"flow_diag": 1.0}} for s in (0, 50, 100)]
    rows += [{"type": "train", "step": s, "metrics": {"flow_diag": 2.0}} for s in (50, 100, 150)]
    (run / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    train, _ = plot_progress.read_progress(run / "progress.jsonl")
    assert [r["step"] for r in train] == [0, 50, 100, 150]
    assert train[1]["metrics"]["flow_diag"] == 2.0  # the later run wins


def _fake_count(text: str) -> int:
    """Stand-in for the sentencepiece tokenizer: one token per word."""
    return len(text.split())


def test_split_text_keeps_every_chunk_within_budget():
    from training.farsi.synthesize import split_text

    text = (
        "یک مقام شورای امنیت ملی اسرائیل به ایران‌اینترنشنال گفت جمهوری اسلامی "
        "تلاش‌های خود را برای هدف قرار دادن مخالفان حکومت، روزنامه‌نگاران، یهودیان "
        "و شهروندان اسرائیلی افزایش داده و واحدهای عملیاتی آن برای اجرای حملات "
        "برون‌مرزی به‌شدت فعال شده‌اند."
    )
    chunks = split_text(text, _fake_count, max_tokens=10)
    assert chunks
    assert all(_fake_count(c) <= 10 for c in chunks)
    # nothing dropped: every word survives, in order
    assert " ".join(chunks).split() == text.split()


def test_split_text_splits_a_comma_only_sentence():
    # The real failure: one sentence, no sentence-ending punctuation, so
    # pocket-tts cannot split it and hands the model an oversized chunk.
    from training.farsi.synthesize import split_text

    text = "الف ب پ، ت ث ج، چ ح خ، د ذ ر"
    chunks = split_text(text, _fake_count, max_tokens=4)
    assert len(chunks) > 1
    assert all(_fake_count(c) <= 4 for c in chunks)


def test_split_text_handles_no_punctuation_at_all():
    from training.farsi.synthesize import split_text

    text = " ".join(["کلمه"] * 25)
    chunks = split_text(text, _fake_count, max_tokens=6)
    assert all(_fake_count(c) <= 6 for c in chunks)
    assert sum(_fake_count(c) for c in chunks) == 25


def test_split_text_merges_short_neighbours():
    from training.farsi.synthesize import split_text

    text = "یک. دو. سه. چهار."
    assert split_text(text, _fake_count, max_tokens=10) == ["یک. دو. سه. چهار."]


def test_split_text_empty_input():
    from training.farsi.synthesize import split_text

    assert split_text("   ", _fake_count) == []


def test_pause_only_follows_punctuation():
    """A chunk split mid-phrase must not get a silence after it.

    Persian's ezafe (the unwritten linking -e in "حملات برون‌مرزی") is broken
    by any pause, so gaps belong only where the text itself pauses.
    """
    from training.farsi.synthesize import BREAK_CHARS

    assert "حملات".rstrip().endswith(BREAK_CHARS) is False
    assert "حکومت،".rstrip().endswith(BREAK_CHARS) is True
    assert "شده‌اند.".rstrip().endswith(BREAK_CHARS) is True


def test_pause_at_punct_makes_every_comma_a_boundary():
    from training.farsi.synthesize import split_text

    text = "الف ب، پ ت، ث ج."
    assert split_text(text, _fake_count, max_tokens=40) == ["الف ب، پ ت، ث ج."]
    # min_tokens=1 so the boundaries are not merged back for being too short --
    # the default of 8 deliberately does merge them (see the test below).
    assert split_text(
        text, _fake_count, max_tokens=40, keep_punct_boundaries=True, min_tokens=1
    ) == ["الف ب،", "پ ت،", "ث ج."]


def test_pause_at_punct_does_not_leave_tiny_first_chunk():
    """A two-word opening fragment renders badly; merge it forward."""
    from training.farsi.synthesize import split_text

    text = "چهارم شهریور، ابوالفضل شکارچی، سخنگوی ارشد نیروهای مسلح جمهوری اسلامی، تهدید کرد"
    chunks = split_text(text, _fake_count, max_tokens=40, keep_punct_boundaries=True, min_tokens=5)
    assert all(_fake_count(c) >= 5 for c in chunks[:-1]), chunks
    assert " ".join(chunks).split() == text.split()


def test_plot_progress_detects_distillation_metrics(tmp_path):
    """A depth-distillation run logs distill_mse, not flow_diag/flow_loss/
    eos_loss. Rendering the from-scratch panel list against it used to produce
    three "no data" placeholders and hide the one metric the run actually has.
    """
    import json

    from training.farsi import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    lines = [
        json.dumps(
            {
                "type": "train",
                "step": s,
                "lr": 4e-4,
                "grad_norm": 0.01,
                # "loss" duplicates distill_mse exactly on this objective
                # (see training/modules/model.py) and must not get its own panel.
                "metrics": {"distill_mse": 0.05, "loss": 0.05},
            }
        )
        for s in range(0, 1000, 50)
    ]
    (run / "progress.jsonl").write_text("\n".join(lines))

    plot_progress.main(str(run), out=None, window=5)
    html = (run / "progress.html").read_text()
    assert "no data" not in html
    assert html.count("<h2>") == 2  # distill_mse, grad_norm -- not "loss" too
    assert "depth-distillation" in html


def test_plot_progress_svg_has_no_unbounded_overflow(tmp_path):
    """A chart's raw trace once painted across the whole page instead of
    clipping to its own box: overflow:visible on an svg sized only by
    height:auto, with no width/height attributes to guarantee its box. Both
    must hold for every panel, on any run.
    """
    import json

    from training.farsi import plot_progress

    run = tmp_path / "run"
    run.mkdir()
    lines = [
        json.dumps({"type": "train", "step": s, "grad_norm": 0.5, "metrics": {"flow_diag": 1.0}})
        for s in range(0, 500, 50)
    ]
    (run / "progress.jsonl").write_text("\n".join(lines))

    plot_progress.main(str(run), out=None, window=3)
    html = (run / "progress.html").read_text()

    import re

    # Strip CSS/HTML comments before searching: the fix is explained in one,
    # and that explanation names the very string it forbids.
    without_comments = re.sub(r"/\*.*?\*/", "", html, flags=re.S)
    assert "overflow:visible" not in without_comments.replace(" ", "")
    assert "overflow: visible" not in without_comments

    svg_tags = re.findall(r"<svg[^>]*>", without_comments)
    assert svg_tags, "expected at least one rendered chart"
    for tag in svg_tags:
        assert 'width="900"' in tag and 'height="220"' in tag


# --------------------------------------------------------------------------
# generate_chunk: recovery from runaway (no-EOS) generations
# --------------------------------------------------------------------------


class _FakeConditioner:
    """prepare() only needs to report a token count for the cap estimate."""

    def prepare(self, text: str) -> "torch.Tensor":
        import torch

        return torch.zeros(1, len(text.split()))


class _FakeModel:
    """Returns audio of a length dictated by `plan`, so cap-hits are scriptable.

    `plan` maps a chunk's word count to the list of durations (as a fraction of
    that chunk's cap) it returns on successive calls.
    """

    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0

    def __init__(self, plan: dict[int, list[float]], sample_rate: int = 100) -> None:
        self.plan = plan
        self.sample_rate = sample_rate
        self.calls: list[str] = []
        self.flow_lm = type("F", (), {"conditioner": _FakeConditioner()})()
        self.config = type("C", (), {"mimi": type("M", (), {"frame_rate": 12.5})()})()
        self.mimi = type("Mi", (), {"sample_rate": sample_rate})()

    def generate_audio(self, state: object, text: str, frames_after_eos: int | None = None) -> "np.ndarray":
        import numpy as np

        self.calls.append(text)
        n_words = len(text.split())
        ratios = self.plan.get(n_words, [0.5])
        ratio = ratios[min(len([c for c in self.calls if c == text]) - 1, len(ratios) - 1)]
        cap = synthesize._cap_seconds(self, text)
        return np.zeros(int(ratio * cap * self.sample_rate), dtype="float32")


def test_generate_chunk_returns_a_healthy_generation_untouched() -> None:
    model = _FakeModel({4: [0.5]})
    out = synthesize.generate_chunk(model, None, "a b c d", sample_rate=100)
    assert len(model.calls) == 1, "a healthy chunk must not be retried"
    assert len(out) > 0


def test_generate_chunk_retries_a_runaway_before_splitting() -> None:
    # first attempt hits the cap, second is fine -- the stochastic case
    model = _FakeModel({4: [1.05, 0.5]})
    synthesize.generate_chunk(model, None, "a b c d", sample_rate=100)
    assert model.calls == ["a b c d", "a b c d"], "should retry the same text, not split yet"


def test_generate_chunk_splits_when_retries_keep_hitting_the_cap() -> None:
    # 4-word chunk always caps; the 2-word halves are healthy
    model = _FakeModel({4: [1.05], 2: [0.5]})
    synthesize.generate_chunk(model, None, "a b c d", sample_rate=100)
    assert model.calls[:2] == ["a b c d", "a b c d"], "retries come first"
    assert "a b" in model.calls and "c d" in model.calls, "then it splits at the midpoint"


def test_generate_chunk_gives_up_rather_than_recursing_forever() -> None:
    # everything caps at every depth; must terminate and still return audio
    model = _FakeModel({4: [1.05], 2: [1.05], 1: [1.05]})
    out = synthesize.generate_chunk(model, None, "a b c d", sample_rate=100)
    assert len(out) > 0, "must still return the least-bad attempt"
    assert len(model.calls) < 40, "recursion must be bounded by MAX_SPLIT_DEPTH"


def test_generate_chunk_does_not_split_text_too_short_to_halve() -> None:
    model = _FakeModel({3: [1.05]})
    synthesize.generate_chunk(model, None, "a b c", sample_rate=100)
    assert all(c == "a b c" for c in model.calls), "a 3-word chunk has no useful split"


# --------------------------------------------------------------------------
# validate_ingest.verdict: reading the offset sweep
# --------------------------------------------------------------------------


def _verdict(results: dict, offsets: list) -> str:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "farsi" / "v2" / "validate_ingest.py"
    spec = importlib.util.spec_from_file_location("validate_ingest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return " ".join(mod.verdict(results, offsets))


OFFSETS = [-1.0, 0.0, 1.0]


def test_a_tie_is_not_reported_as_an_offset() -> None:
    # Regression: -1.0 and 0.0 both scored 23.8%, min() picked the first, and the
    # run announced a "SYSTEMATIC OFFSET ... 0.0% better" -- chasing nothing.
    out = _verdict({-1.0: (5, 21, 0), 0.0: (5, 21, 0), 1.0: (12, 21, 0)}, OFFSETS)
    assert "SYSTEMATIC OFFSET" not in out
    assert "no shift beats it" in out


def test_a_real_offset_is_reported() -> None:
    out = _verdict({-1.0: (2, 21, 0), 0.0: (9, 21, 0), 1.0: (12, 21, 0)}, OFFSETS)
    assert "SYSTEMATIC OFFSET" in out and "-1.0s" in out


def test_a_gain_under_the_noise_floor_is_not_an_offset() -> None:
    out = _verdict({-1.0: (9, 100, 0), 0.0: (11, 100, 0), 1.0: (30, 100, 0)}, OFFSETS)
    assert "SYSTEMATIC OFFSET" not in out
    assert "noise" in out


def test_high_wer_with_no_helpful_shift_blames_the_transcripts() -> None:
    out = _verdict({-1.0: (19, 21, 0), 0.0: (18, 21, 0), 1.0: (20, 21, 0)}, OFFSETS)
    assert "inspect the transcripts" in out


def test_no_usable_windows_is_reported_rather_than_crashing() -> None:
    assert "cannot judge" in _verdict({0.0: (0, 0, 5)}, [0.0])


# --------------------------------------------------------------------------
# prep_v2 shard split/merge: phonemizing across GPUs must not reorder the corpus
# --------------------------------------------------------------------------


def _prep_v2() -> ModuleType:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "farsi" / "v2" / "prep_v2.py"
    spec = importlib.util.spec_from_file_location("prep_v2", path)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves its own module out of sys.modules; without this the
    # decorator sees None there and dies before the file finishes loading.
    sys.modules["prep_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_sharded_phonemize_reproduces_a_single_pass(tmp_path: Path) -> None:
    """Split, phonemize each slice, concatenate: byte-identical to one pass.

    Latents are index-keyed to their manifest, so a reordered corpus would pair
    every utterance with the wrong audio -- silently, and only visible as a
    model that never learns.
    """
    prep = _prep_v2()
    src = tmp_path / "corpus.jsonl"
    src.write_text("".join(f'{{"i": {i}}}\n' for i in range(97)))
    dst = tmp_path / "corpus_ph.jsonl"

    pairs = prep.split_manifest(src, dst, 8)
    for chunk, part in pairs:  # stand in for the G2P process
        part.write_text(chunk.read_text())
    prep.concat_parts(pairs, dst)

    assert dst.read_text() == src.read_text()


def test_sharding_covers_every_row_exactly_once(tmp_path: Path) -> None:
    prep = _prep_v2()
    src = tmp_path / "corpus.jsonl"
    src.write_text("".join(f"row{i}\n" for i in range(1000)))
    pairs = prep.split_manifest(src, tmp_path / "out.jsonl", 7)
    seen = [line for chunk, _ in pairs for line in chunk.read_text().splitlines()]
    assert sorted(seen, key=lambda r: int(r[3:])) == [f"row{i}" for i in range(1000)]


def test_shards_are_balanced_not_contiguous(tmp_path: Path) -> None:
    """The corpus is ordered by source and cost per row follows it.

    Contiguous blocks gave one shard every long subtitle line: 24 minutes on
    the fastest, 6.5 hours on the slowest, and the slowest is what the step
    costs. Dealing rows round-robin gives every shard the same mixture.
    """
    prep = _prep_v2()
    src = tmp_path / "corpus.jsonl"
    # First half cheap, second half expensive -- the real corpus's shape.
    src.write_text("".join(f"{'cheap' if i < 500 else 'costly'}{i}\n" for i in range(1000)))
    pairs = prep.split_manifest(src, tmp_path / "out.jsonl", 8)
    per_shard = [chunk.read_text().count("costly") for chunk, _ in pairs]
    assert max(per_shard) - min(per_shard) <= 1
    assert sum(per_shard) == 500


def test_shard_pieces_are_cleaned_up(tmp_path: Path) -> None:
    prep = _prep_v2()
    src = tmp_path / "corpus.jsonl"
    src.write_text("a\nb\nc\nd\n")
    dst = tmp_path / "out.jsonl"
    pairs = prep.split_manifest(src, dst, 2)
    for chunk, part in pairs:
        part.write_text(chunk.read_text())
    prep.concat_parts(pairs, dst)
    assert not any(chunk.exists() or part.exists() for chunk, part in pairs)


def test_fewer_rows_than_shards_still_round_trips(tmp_path: Path) -> None:
    prep = _prep_v2()
    src = tmp_path / "corpus.jsonl"
    src.write_text("only\n")
    dst = tmp_path / "out.jsonl"
    pairs = prep.split_manifest(src, dst, 8)
    for chunk, part in pairs:
        part.write_text(chunk.read_text())
    prep.concat_parts(pairs, dst)
    assert dst.read_text() == "only\n"


def test_localize_configs_repoints_manifests_at_the_data_disk(tmp_path: Path) -> None:
    """Committed configs are repo-relative; on a GPU box the corpus is elsewhere.

    Left unrewritten, precompute_latents opens 'data/farsi_600h/...', finds
    nothing, and the step dies after every expensive step before it has run.
    """
    prep = _prep_v2()
    # Mirrors the real /mnt/data/farsi_600h: the absolute path itself contains
    # "data/farsi_600h", so a second rewrite pass would corrupt it.
    data = tmp_path / "mnt" / "data" / "farsi_600h"
    data.mkdir(parents=True)
    (data / "v2_train_ph.jsonl").write_text("{}\n")

    repo = tmp_path / "repo"
    (repo / "training" / "farsi" / "configs").mkdir(parents=True)
    model_rel = "training/farsi/configs/model_farsi_ph.yaml"
    (repo / model_rel).write_text("tokenizer_path: data/farsi_600h/tokenizer_ph.model\n")
    train_rel = "training/farsi/configs/lsd_scratch_v2.yaml"
    (repo / train_rel).write_text(
        f"model_config: {model_rel}\n"
        "data:\n"
        "  train_jsonl: data/farsi_600h/v2_train_ph.jsonl\n"
        "  valid_jsonl: data/farsi_600h/v2_valid_ph.jsonl\n"
    )

    prep.REPO = repo
    out = prep.localize_configs(prep.Paths(data=data), train_rel)

    text = out.read_text()
    assert f"train_jsonl: {data}/v2_train_ph.jsonl" in text
    assert f"valid_jsonl: {data}/v2_valid_ph.jsonl" in text
    assert "data/farsi_600h" not in text.replace(str(data), "")
    # the model config is localized too, and pointed at
    model_out = out.parent / "model_farsi_ph.yaml"
    assert f"model_config: {model_out}" in text
    assert "/mnt//mnt" not in text and str(data) + "/" + str(data).lstrip("/") not in text
    assert f"tokenizer_path: {data}/tokenizer_ph.model" in model_out.read_text()
    # the committed originals are untouched
    assert "data/farsi_600h" in (repo / train_rel).read_text()


def test_localize_configs_rejects_a_config_for_the_wrong_corpus(tmp_path: Path) -> None:
    """precompute_latents takes no manifest argument, so a stale config
    silently encodes the wrong corpus rather than failing."""
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "cfg" / "train.yaml").write_text("  train_jsonl: data/farsi_600h/old_corpus.jsonl\n")
    prep.REPO = repo
    with pytest.raises(prep.typer.Exit):
        prep.localize_configs(prep.Paths(data=data), "cfg/train.yaml")


def test_training_config_points_at_the_latents_manifest(tmp_path: Path) -> None:
    """The loader uses precomputed latents only for a `_latents` manifest.

    Handed the plain one it decodes audio every epoch instead, silently, with
    nothing in the logs to say the encode pass was wasted.
    """
    prep = _prep_v2()
    data = tmp_path / "mnt" / "data" / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    localized = data / "lsd_scratch_v2.yaml"
    localized.write_text(
        f"data:\n  train_jsonl: {p.merged_ph}\n  valid_jsonl: {p.valid_ph}\n"
        "run_dir: runs/lsd_scratch_v2\n"
    )

    out = prep.write_training_config(p, localized)
    text = out.read_text()
    assert f"train_jsonl: {p.latents}" in text
    # validation has no latents -- precompute_latents only encodes train_jsonl
    assert f"valid_jsonl: {p.valid_ph}" in text
    # checkpoints go to the data disk, not the boot disk the repo sits on
    assert "run_dir: " + str(data.parent / "runs" / "lsd_scratch_v2") in text
    # the prep-time config is left alone, so a rerun still encodes the right thing
    assert f"train_jsonl: {p.merged_ph}\n" in localized.read_text()


# --------------------------------------------------------------------------
# phoneme text must survive the Persian-only paths
#
# The v2 model is trained on romanised phonemes. Persian normalisation deletes
# every character of those, and the damage only shows three steps later as a
# model that generates silence -- which is how the same mistake reached the
# sampler, the synthesiser and the evaluator before anyone noticed.
# --------------------------------------------------------------------------


def test_phoneme_text_is_recognised_and_left_alone() -> None:
    from training.farsi.normalize_fa import is_phonemic, normalize_for_model

    ph = "salAm hAle SomA Cetor ?ast"
    assert is_phonemic(ph)
    assert normalize_for_model(ph) == ph


def test_persian_text_is_still_normalised() -> None:
    from training.farsi.normalize_fa import is_phonemic, normalize_for_model

    fa = "سلام حال شما چطور است"
    assert not is_phonemic(fa)
    assert normalize_for_model(fa) == normalize(fa)
    # digits still spell out, i.e. we did not bypass normalisation wholesale
    assert normalize_for_model("۱۲۳") == normalize("۱۲۳")


def test_normalisation_that_empties_the_text_raises() -> None:
    """The silent version of this produced a model that appeared to have
    forgotten how to speak."""
    from training.farsi.normalize_fa import normalize_for_model

    # Not phonemic (no ascii letters) and not Persian: normalisation kills it.
    with pytest.raises(ValueError, match="emptied the text"):
        normalize_for_model("《》〈〉")


def _phonemize_mod() -> ModuleType:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "farsi" / "v2" / "phonemize_manifest.py"
    spec = importlib.util.spec_from_file_location("phonemize_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phonemize_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_question_marks_are_removed_before_g2p_sees_them() -> None:
    """GE2PE renders "؟" as "@", the symbol it also uses for the glottal stop.

    It cannot be undone afterwards: "زده؟" gains two "@" while "موقع؟" gains one,
    because موقع's ع already contributes its own. Counting trailing symbols
    cannot tell them apart, so the mark has to go in before phonemisation.
    """
    mod = _phonemize_mod()
    assert mod.strip_question_marks("چقدر زده؟") == "چقدر زده"
    assert mod.strip_question_marks("از اون موقع؟") == "از اون موقع"
    assert mod.strip_question_marks("no marks here") == "no marks here"
    # the ASCII form too, for manifests that were normalised differently
    assert mod.strip_question_marks("what?") == "what"


def test_real_glottal_stops_survive_phonemisation() -> None:
    """جمع and موقع end in ع. Stripping their "@" would corrupt the word, which
    is what a post-hoc rule on the source text did."""
    mod = _phonemize_mod()
    # GE2PE notation: "@" glottal stop, "/" short a, "a" long aa
    assert mod.to_phonemes("@in j/m@") == "?in jam?"
    assert mod.to_phonemes("moqe@") == "moqe?"


def test_to_phonemes_maps_the_full_notation() -> None:
    mod = _phonemize_mod()
    assert mod.to_phonemes("$om1a") == "SomA"      # $ -> S, the ezafe flag dropped
    assert mod.to_phonemes("cera") == "CerA"       # c -> C, a -> A
    assert mod.to_phonemes("@/st") == "?ast"       # @ -> ?, / -> a


def test_eval_scores_against_graphemes_when_the_model_eats_phonemes(tmp_path: Path) -> None:
    """The model is fed phonemes; the ASR returns Persian, so WER needs graphemes.

    Scoring the phoneme string against an ASR transcript compares two different
    alphabets and reports near-100% error regardless of how good the audio is.
    """
    manifest = tmp_path / "eval.jsonl"
    rows = [
        {"path": "/a.wav", "speaker": "s1", "transcript": "salAm donyA",
         "transcript_graphemes": "سلام دنیا"},
        {"path": "/b.wav", "speaker": "s1", "transcript": "xub ?ast",
         "transcript_graphemes": "خوب است"},
    ]
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    items = eval_fa.build_items(manifest, None, seed=0)
    assert len(items) == 1
    item = items[0]
    assert is_phonemic(item["text"])          # what the model is given
    assert not is_phonemic(item["ref_text"])  # what WER is scored against


def test_eval_falls_back_to_the_transcript_for_grapheme_manifests(tmp_path: Path) -> None:
    """v1 manifests have no `transcript_graphemes`; behaviour must not change."""
    manifest = tmp_path / "eval.jsonl"
    rows = [
        {"path": "/a.wav", "speaker": "s1", "transcript": "سلام دنیا"},
        {"path": "/b.wav", "speaker": "s1", "transcript": "خوب است"},
    ]
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    item = eval_fa.build_items(manifest, None, seed=0)[0]
    assert item["ref_text"] == item["text"]


def test_merge_refuses_the_unrepaired_youtube_manifest(tmp_path: Path) -> None:
    """Merging farsi_asr_yt_aligned.jsonl rebuilds the defective corpus.

    41% of those windows open mid-sound. The first v2 teacher trained on them
    and learned to begin utterances abruptly at full volume: every
    continuant-initial first word was damaged (`man` heard as `in`, `salAm` as
    `shlaam`) while the same words mid-sentence were clean.
    """
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    p.v1_aligned.write_text('{"path": "/a.wav", "start": 0.0}\n')
    p.yt_aligned.write_text('{"path": "/b.wav", "start": 0.0}\n')
    # yt_onset deliberately absent
    with pytest.raises(prep.typer.Exit):
        prep.step_merge(p)
    assert not p.merged.exists()


def test_merge_uses_the_repaired_manifest(tmp_path: Path) -> None:
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    p.v1_aligned.write_text('{"id": "v1-unrepaired", "path": "/v1.flac", "start": 0.0}\n')
    p.v1_onset.write_text('{"id": "v1", "path": "/v1.flac", "start": 0.0}\n')
    p.yt_aligned.write_text('{"id": "unrepaired", "path": "/y.flac", "start": 2.0}\n')
    p.yt_onset.write_text('{"id": "repaired", "path": "/y.flac", "start": 1.6}\n')

    prep.step_merge(p)
    merged = p.merged.read_text()
    assert '"repaired"' in merged
    assert "unrepaired" not in merged   # neither half's pre-repair manifest
    assert '"v1"' in merged


def test_onsets_comes_between_align_and_merge() -> None:
    """Order matters: repairing after the merge would miss the v1 half's
    neighbours and leave the merged corpus stale."""
    prep = _prep_v2()
    steps = prep.STEPS
    assert steps.index("align") < steps.index("onsets") < steps.index("merge")


def test_merge_drops_duplicate_windows(tmp_path: Path) -> None:
    """The youtube ingest emitted 710 (file, offset) pairs twice over.

    Left in, that audio trains twice as often as its neighbours for no reason,
    and any eval drawn from the corpus inherits the same skew.
    """
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    p.v1_onset.write_text('{"path": "/a.flac", "start": 0.0}\n')
    p.yt_onset.write_text(
        '{"path": "/b.flac", "start": 1.5}\n'
        '{"path": "/b.flac", "start": 1.5}\n'   # exact duplicate
        '{"path": "/b.flac", "start": 9.0}\n'   # same file, different window: keep
    )
    prep.step_merge(p)
    kept = [json.loads(x) for x in p.merged.read_text().splitlines() if x.strip()]
    assert len(kept) == 3
    assert sorted((r["path"], r["start"]) for r in kept) == [
        ("/a.flac", 0.0), ("/b.flac", 1.5), ("/b.flac", 9.0)
    ]


def test_onsets_repairs_both_halves_of_the_corpus(tmp_path: Path) -> None:
    """v1 clips are clipped too, and at a higher rate than anyone assumed.

    Measured per source: manatts 4%, filimo 17%, v1's own youtube subset 30%.
    They start at 0 with no audio in front, so they cannot be backed up --
    fix_onsets drops them instead, costing 109 h of 1,092.
    """
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    p.v1_aligned.write_text('{"path": "/a.flac", "start": 0.0, "duration": 3.0}\n')
    p.yt_aligned.write_text('{"path": "/b.flac", "start": 9.0, "duration": 8.0}\n')

    calls: list[list[str]] = []
    prep.run_script = lambda script, args: (calls.append(args), Path(args[-1]).write_text("{}\n"), 0)[-1]
    prep.step_onsets(p)

    manifests = [args[1] for args in calls]
    assert str(p.yt_aligned) in manifests
    assert str(p.v1_aligned) in manifests, "the v1 half must be repaired too"


def test_merge_refuses_when_either_half_is_unrepaired(tmp_path: Path) -> None:
    prep = _prep_v2()
    data = tmp_path / "farsi_600h"
    data.mkdir(parents=True)
    p = prep.Paths(data=data)
    p.yt_onset.write_text('{"path": "/b.flac", "start": 1.0}\n')
    # v1_onset absent
    with pytest.raises(prep.typer.Exit):
        prep.step_merge(p)


def test_split_text_leaves_no_starved_tail():
    """Greedy packing left a runt the budget could not merge back.

    36 tokens at a budget of 18 came out 16/18/2: the 2-token tail could not
    rejoin because 18+2 exceeds the budget, and `min_tokens` never enforced a
    floor -- it was only consulted at punctuation boundaries. That tail is far
    outside the ~11-token distribution the model was trained on; in practice it
    ran past end-of-speech and appended invented words.
    """
    from training.farsi.synthesize import split_text

    text = " ".join(["کلمه"] * 36)
    chunks = split_text(text, _fake_count, max_tokens=18, min_tokens=8)
    assert all(_fake_count(c) >= 8 for c in chunks), [_fake_count(c) for c in chunks]
    assert " ".join(chunks).split() == text.split()


def test_split_text_spreads_tokens_evenly():
    """Chunk sizes should be close together, not full-full-scrap."""
    from training.farsi.synthesize import split_text

    chunks = split_text(" ".join(["کلمه"] * 36), _fake_count, max_tokens=18, min_tokens=8)
    sizes = [_fake_count(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 2, sizes


def test_split_text_never_breaks_an_ezafe_pair():
    """`1` marks the ezafe: "?eqtesAde1 ?AmrikA" is one bound phrase.

    Splitting between them is audible as a gap in the middle of a noun phrase.
    The mark is a splitter hint only -- strip_ezafe removes it before the model
    sees the text, exactly as the training corpus had it removed.
    """
    from training.farsi.synthesize import split_text, strip_ezafe

    words = ["yek", "do", "se", "CahAr", "panj", "?eqtesAde1", "?AmrikA", "haft", "haSt"]
    chunks = split_text(" ".join(words), _fake_count, max_tokens=3, min_tokens=1)
    for c in chunks:
        assert not c.rstrip().endswith("1"), f"chunk ends on an ezafe: {c!r}"
    joined = " ".join(chunks)
    assert "?eqtesAde1 ?AmrikA" in joined
    assert strip_ezafe(joined).split() == [w.replace("1", "") for w in words]


def test_strip_ezafe_is_a_noop_without_marks():
    from training.farsi.synthesize import split_text, strip_ezafe

    text = " ".join(["kalame"] * 20)
    assert strip_ezafe(text) == text
    assert split_text(text, _fake_count, max_tokens=6) == split_text(
        text, _fake_count, max_tokens=6
    )


def test_trim_silence_keeps_the_speech():
    """It must remove silence and nothing else.

    A first version used a 0.15 RMS threshold with 40 ms of margin. It removed
    the gap, and on a quiet onset it would lock onto the following vowel and eat
    the consonant -- recreating the missing-first-consonant bug by another route.
    """
    import numpy as np

    from training.farsi.synthesize import trim_silence

    sr = 24000
    quiet = np.zeros(sr, dtype=np.float32)                       # 1 s of silence
    onset = np.full(int(0.03 * sr), 0.02, dtype=np.float32)      # a brief, quiet stop
    vowel = np.full(int(0.40 * sr), 0.5, dtype=np.float32)       # the loud part
    a = np.concatenate([quiet, onset, vowel, quiet])

    out = trim_silence(a, sr)
    assert out.size < a.size, "no silence removed"
    # the quiet onset survives: peak-below-threshold audio is still speech
    head = out[: int(0.20 * sr)]
    assert float(np.abs(head).max()) >= 0.02, "the quiet onset was clipped"
    # and the loud part is untouched
    assert np.isclose(float(np.abs(out).max()), 0.5)


def test_trim_silence_passes_through_degenerate_input():
    import numpy as np

    from training.farsi.synthesize import trim_silence

    assert trim_silence(np.zeros(0, dtype=np.float32), 24000).size == 0
    silent = np.zeros(2400, dtype=np.float32)
    assert trim_silence(silent, 24000).size == silent.size  # all-silence is returned as-is
