"""The pre-flight checks that gate a training launch.

Each check here exists because its absence cost something: latents keyed to a
different manifest, phoneme transcripts normalised to nothing, word timings
left behind when their windows moved, a stale meta.json outliving its run.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "farsi" / "v2" / "verify_corpus.py"
    spec = importlib.util.spec_from_file_location("verify_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_persian_text_is_detected_as_not_phonemes() -> None:
    mod = _mod()
    assert mod.PERSIAN.search("سلام دنیا")
    assert not mod.PERSIAN.search("salAm donyA")
    # the glottal-stop and long-vowel symbols must not trip the Persian test
    assert not mod.PERSIAN.search("?ejtemA?i mo?allem SomA")


def test_report_counts_failures_and_passes() -> None:
    mod = _mod()
    r = mod.Report()
    r.check(True, "fine")
    assert r.failed == 0
    r.check(False, "broken")
    assert r.failed == 1
    r.note("informational", "not a pass or fail")
    assert r.failed == 1


def test_count_lines_ignores_blank_lines(tmp_path: Path) -> None:
    mod = _mod()
    p = tmp_path / "m.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n\n')
    assert mod.count_lines(p) == 2


SR = 16000


def _write(path: Path, wav: np.ndarray) -> None:
    import sphn

    sphn.write_wav(str(path), wav.astype(np.float32), SR)


def test_clipped_rate_flags_speech_already_running(tmp_path: Path) -> None:
    mod = _mod()
    rng = np.random.default_rng(0)
    loud = tmp_path / "loud.wav"
    _write(loud, rng.standard_normal(SR * 2) * 0.2)
    rows = [{"path": str(loud), "start": 0.0, "duration": 2.0}] * 5
    clipped, usable = mod.clipped_rate(rows, sample=5, seed=0)
    assert usable == 5
    assert clipped == 5


def test_clipped_rate_accepts_a_soft_onset(tmp_path: Path) -> None:
    mod = _mod()
    rng = np.random.default_rng(0)
    soft = tmp_path / "soft.wav"
    _write(soft, np.concatenate([np.zeros(int(0.3 * SR)), rng.standard_normal(SR * 2) * 0.2]))
    rows = [{"path": str(soft), "start": 0.0, "duration": 2.3}] * 5
    clipped, usable = mod.clipped_rate(rows, sample=5, seed=0)
    assert usable == 5
    assert clipped == 0


def test_clipped_rate_skips_unreadable_rows(tmp_path: Path) -> None:
    """A missing file must not be counted as a clean onset."""
    mod = _mod()
    rows = [{"path": str(tmp_path / "nope.wav"), "start": 0.0, "duration": 2.0}]
    clipped, usable = mod.clipped_rate(rows, sample=1, seed=0)
    assert (clipped, usable) == (0, 0)
