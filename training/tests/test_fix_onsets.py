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
