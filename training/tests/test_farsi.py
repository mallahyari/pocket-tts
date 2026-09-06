"""The Farsi pipeline's pure functions: text normalization, manifest
filtering, speaker-disjoint splitting and eval pairing.

Nothing here touches the network, a GPU or a model -- the point is that the
transcripts reaching the tokenizer and the CTC aligner are spelled one way,
and that the held-out split is actually held out.
"""

import json

import pytest

from training.farsi import eval_fa, prepare_data_fa
from training.farsi.normalize_fa import (
    ALIGNER_ALPHABET,
    ALLOWED,
    ZWNJ,
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
