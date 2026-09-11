"""Subtitle windows that open mid-word teach the model to swallow quiet onsets.

Measured on the farsi-asr YouTube half: 46% of windows are already at more than
half their own loudness in their first 50 ms, against 7% for the v1 studio
corpus. On the v2 teacher this was audible -- "man" (I) generated as "in"
(this), "mAdar" as "Adar" -- while the same words mid-sentence were clean.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _mod() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "farsi" / "v2" / "fix_onsets.py"
    spec = importlib.util.spec_from_file_location("fix_onsets", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fix_onsets"] = mod
    spec.loader.exec_module(mod)
    return mod


SR = 16000


def _speech(seconds: float, level: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_speech_already_running_at_t0_is_clipped() -> None:
    mod = _mod()
    assert mod.onset_is_clipped(_speech(2.0), SR)


def test_a_soft_onset_is_not_clipped() -> None:
    mod = _mod()
    body = np.concatenate([_silence(0.2), _speech(1.8)])
    assert not mod.onset_is_clipped(body, SR)


def test_silence_is_not_reported_as_clipped() -> None:
    mod = _mod()
    assert not mod.onset_is_clipped(_silence(1.0), SR)


def test_backup_finds_the_nearest_silence() -> None:
    """Nearest, not furthest: backing up further drags in the previous word."""
    mod = _mod()
    # [older speech][0.3 s gap][more speech] then the window begins
    lead = np.concatenate([_speech(0.2), _silence(0.3), _speech(0.1)])
    backup = mod.find_backup(lead, SR, level=mod.rms(_speech(1.0)))
    assert backup is not None
    # the gap ends 0.1 s before the window, so we back up a little past that
    assert 0.1 <= backup <= 0.45


def test_no_backup_inside_continuous_speech() -> None:
    """38% of clipped windows sit in continuous speech; those get dropped."""
    mod = _mod()
    assert mod.find_backup(_speech(0.75), SR, level=mod.rms(_speech(1.0))) is None


def test_backup_is_none_for_an_empty_lead() -> None:
    mod = _mod()
    assert mod.find_backup(np.zeros(0, dtype=np.float32), SR, level=0.2) is None


def test_previous_end_bounds_the_backup() -> None:
    mod = _mod()
    ordered = [
        {"start": 10.0, "duration": 3.0},
        {"start": 14.0, "duration": 2.0},
    ]
    assert mod.previous_end(ordered, 0) == 0.0
    assert mod.previous_end(ordered, 1) == pytest.approx(13.0)


def test_rms_of_empty_is_zero() -> None:
    mod = _mod()
    assert mod.rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_recut_moves_the_shared_boundary_rather_than_overlapping() -> None:
    """Consecutive subtitle windows are back-to-back.

    The silence a clipped window backs into lies inside the previous window's
    tail, and the word after it belongs to the later window. Moving the shared
    boundary fixes both: one gains its onset, the other sheds a fragment its
    transcript never claimed. Bounding the backup by the neighbour's end
    instead left 92% of clipped rows unrepairable on real data.
    """
    mod = _mod()
    prev = {"start": 10.0, "duration": 5.0}   # ends at 15.0
    row = {"start": 15.0, "duration": 8.0}
    assert mod.recut(prev, row, 0.3)
    assert row["start"] == pytest.approx(14.7)
    assert row["duration"] == pytest.approx(8.3)
    assert prev["duration"] == pytest.approx(4.7)   # now ends at 14.7, no overlap
    assert prev["start"] + prev["duration"] == pytest.approx(row["start"])


def test_recut_refuses_to_shrink_a_neighbour_below_the_floor() -> None:
    mod = _mod()
    prev = {"start": 10.0, "duration": 2.1}   # ends at 12.1
    row = {"start": 12.1, "duration": 8.0}
    assert not mod.recut(prev, row, 0.5)      # would leave prev at 1.6 s
    assert prev["duration"] == pytest.approx(2.1)   # untouched
    assert row["start"] == pytest.approx(12.1)


def test_recut_handles_the_first_window_of_a_recording() -> None:
    mod = _mod()
    row = {"start": 0.4, "duration": 6.0}
    assert mod.recut(None, row, 0.75)
    assert row["start"] == pytest.approx(0.0)       # clamped at the file start
    assert row["duration"] == pytest.approx(6.4)    # grew by what it actually moved


def test_recut_at_the_very_start_of_a_file_is_a_no_op() -> None:
    mod = _mod()
    row = {"start": 0.0, "duration": 6.0}
    assert not mod.recut(None, row, 0.5)
    assert row["duration"] == pytest.approx(6.0)


def test_recut_leaves_a_distant_neighbour_alone() -> None:
    """A gap already exists; only the clipped window moves."""
    mod = _mod()
    prev = {"start": 10.0, "duration": 2.0}   # ends at 12.0
    row = {"start": 15.0, "duration": 6.0}
    assert mod.recut(prev, row, 0.3)
    assert prev["duration"] == pytest.approx(2.0)
    assert row["start"] == pytest.approx(14.7)
