"""The shipped configs must be runnable as-is.

Every defect these catch has actually shipped: a scratch config that stopped
before the quality transition, a batch size a quarter of the floor it needs,
and a teacher path pointing at an architecture the distill step cannot load.
"""

from pathlib import Path
from typing import Any

import pytest

from training.args import TrainArgs, _from_dict, load_args
from training.modules.builders import load_model_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRATCH = CONFIGS / "scratch.yaml"
DISTILL = CONFIGS / "depth_distill.yaml"

# Below 64 rows per optimizer step the acoustic-quality transition arrives late
# or not at all, and 400k steps is where expressivity settles (see README).
MIN_EFFECTIVE_BATCH = 64
MIN_SCRATCH_STEPS = 400_000


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_parses(path: Path):
    load_args(path)


def test_scratch_reaches_the_effective_batch_floor():
    args = load_args(SCRATCH)
    assert args.batch_size * args.grad_accum_steps >= MIN_EFFECTIVE_BATCH, (
        "scratch must reach 64 rows per step on a single GPU: "
        f"{args.batch_size} x {args.grad_accum_steps}"
    )


def test_scratch_runs_past_the_quality_transition():
    assert load_args(SCRATCH).max_steps >= MIN_SCRATCH_STEPS


def test_scratch_builds_the_reference_teacher_depth():
    args = load_args(SCRATCH)
    config = load_model_config(args.model_config, args.model_overrides)
    assert config.flow_lm.transformer.num_layers == 24


def test_distill_teacher_is_deeper_than_its_student():
    args = load_args(DISTILL)
    student = load_model_config(args.model_config, args.model_overrides)
    teacher = load_model_config(args.distill_teacher_config, args.distill_teacher_overrides)
    assert teacher.flow_lm.transformer.num_layers > student.flow_lm.transformer.num_layers
    # Depth distillation copies every non-backbone tensor, so the rest must match.
    assert teacher.flow_lm.transformer.d_model == student.flow_lm.transformer.d_model


def test_distill_teacher_weights_point_at_the_scratch_run():
    args = load_args(DISTILL)
    assert args.distill_teacher_weights, "the distill config must name a teacher checkpoint"
    assert str(load_args(SCRATCH).run_dir) in args.distill_teacher_weights, (
        "the documented path is scratch -> distill; the teacher checkpoint should come from "
        f"{load_args(SCRATCH).run_dir}"
    )


class TestArgValidation:
    """Misconfigurations that used to run and quietly do the wrong thing."""

    def test_num_ckpt_keep_zero_is_rejected(self):
        with pytest.raises(ValueError, match="num_ckpt_keep"):
            TrainArgs(num_ckpt_keep=0)

    def test_zero_frequencies_are_rejected(self):
        for field in ("valid_freq", "ckpt_freq", "log_freq"):
            kwargs: dict[str, Any] = {field: 0}
            with pytest.raises(ValueError, match=field):
                TrainArgs(**kwargs)

    def test_distillation_without_a_teacher_is_rejected(self):
        with pytest.raises(ValueError, match="teacher"):
            TrainArgs(distill_cfg_coef=1.5, start_from_pretrained=False)

    def test_teacher_config_without_weights_is_rejected(self):
        with pytest.raises(ValueError, match="distill_teacher_weights"):
            TrainArgs(distill_teacher_config="x.yaml")

    def test_unknown_keys_are_rejected(self):
        """A key the parser doesn't recognize is a setting the user thinks is applied."""
        with pytest.raises(ValueError, match="distill_seed_layers"):
            _from_dict(TrainArgs, {"distill_seed_layers": "first"})


def test_phoneme_configs_have_phoneme_sample_sentences():
    """A phoneme model's sample sentences must not be Persian script.

    This shipped: lsd_scratch_v2.yaml kept v1's Persian sentences while its
    tokenizer moved to phonemes, so every word tokenized to <unk>, the model
    was conditioned on nothing, and every sample from step 110k on came out
    empty. The log still said "wrote 3 samples", so nothing looked wrong.
    """
    import yaml

    from training.farsi.normalize_fa import is_phonemic

    root = Path(__file__).resolve().parents[1] / "farsi" / "configs"
    for path in sorted(root.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        if "model_config" not in raw:  # a model config, not a training one
            continue
        args = load_args(str(path))
        model_cfg = load_model_config(args.model_config, args.model_overrides)
        tokenizer = str(model_cfg.flow_lm.lookup_table.tokenizer_path)
        if "_ph" not in tokenizer:  # grapheme run: Persian sentences are right
            continue
        for sentence in args.sample_sentences:
            assert is_phonemic(sentence), (
                f"{path.name}: sample sentence is not phonemes, so it will tokenize "
                f"to <unk> and generate silence: {sentence!r}"
            )


def test_teacher_inference_config_matches_the_trained_depth():
    """Loading the v2 teacher needs a config of the same depth it was trained at.

    lsd_scratch_v2.yaml deepens the model through `model_overrides`, which only
    exists on the training side. Inference reads a model config directly, so
    without a matching one every attention tensor fails to load -- the error is
    a wall of missing key names that does not mention depth at all.
    """
    root = Path(__file__).resolve().parents[1] / "farsi" / "configs"
    train = load_args(str(root / "lsd_scratch_v2.yaml"))
    trained_depth = train.model_overrides["flow_lm.transformer.num_layers"]

    teacher = load_model_config(str(root / "model_farsi_ph_teacher.yaml"), {})
    assert teacher.flow_lm.transformer.num_layers == trained_depth

    # and it must still be the phoneme tokenizer, not v1's
    assert "_ph" in str(teacher.flow_lm.lookup_table.tokenizer_path)


def test_phoneme_configs_disable_the_orthographic_text_frontend():
    """A phoneme model must not have its text capitalised before generation.

    `prepare_text_prompt` upper-cases the first letter of every chunk, which is
    right for the Latin-script languages pocket-tts shipped with and wrong here:
    the Latin letters are phonemes. `man` becomes `Man`, and `M` is not in the
    4000-entry phoneme vocabulary, so the first word tokenizes to the unknown
    token and is simply not spoken. Worse, where a capital *is* a phoneme it
    changes the sound silently: `salAm` -> `SalAm` is /salaam/ -> /shalaam/,
    because `S` is the symbol for sh.

    This shipped. Every generation lost or mangled its first word, and the bug
    was invisible to the evaluation, which tokenizes the text itself and never
    goes through this frontend. Words beginning with a glottal stop hid it
    further, since "?".upper() is "?".
    """
    from pocket_tts.utils.config import load_config

    root = Path(__file__).resolve().parents[1] / "farsi" / "configs"
    for name in ("model_farsi_ph.yaml", "model_farsi_ph_teacher.yaml"):
        cfg = load_config(str(root / name))
        assert not cfg.capitalize_first_letter, (
            f"{name}: capitalize_first_letter must be false for a phoneme model"
        )
        assert not cfg.append_terminal_punctuation, (
            f"{name}: the phoneme vocabulary has no terminal punctuation to append"
        )
        assert not cfg.pad_with_spaces_for_short_inputs, (
            f"{name}: leading spaces are out of distribution for this model"
        )


def test_capitalising_a_phoneme_prompt_destroys_the_first_word():
    """Pin the mechanism itself, not just the flag that disables it."""
    from pocket_tts.models.text_chunking import prepare_text_prompt

    on, _ = prepare_text_prompt("man be bAzAr raftam", False, False, False, True)
    off, _ = prepare_text_prompt("man be bAzAr raftam", False, False, False, False)
    assert on.startswith("Man"), "upstream behaviour changed; this test is stale"
    assert off == "man be bAzAr raftam"

    # "S" is sh, so capitalising silently swaps the phoneme rather than
    # producing an unknown one -- the failure that sounded like a lisp.
    assert prepare_text_prompt("salAm", False, False, False, True)[0] == "SalAm"
    assert prepare_text_prompt("salAm", False, False, False, False)[0] == "salAm"

    # A glottal-initial word is unchanged either way, which is why the defect
    # looked like it only hit some words.
    assert prepare_text_prompt("?emruz man", False, False, False, True)[0] == "?emruz man"
