"""align_data reads exactly the utterance's window out of its file.

The long-form Farsi manifests point thousands of utterances at one hour-long
recording, so a row's `start` is meaningful even when it is 0.0 -- reading the
whole file for those rows blew up the alignment trellis (a 346 GiB allocation)
and would have mis-timed the words on any row small enough not to crash.
"""

from typing import Any

import numpy as np
import pytest

from training.scripts import align_data


class _RecordingSphn:
    """Stands in for sphn, capturing the window arguments it was asked for."""

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.calls: list[dict[str, Any]] = []

    def read(
        self, path: str, start_sec: float | None = None, duration_sec: float | None = None
    ) -> tuple[np.ndarray, int]:
        self.calls.append({"path": path, "start_sec": start_sec, "duration_sec": duration_sec})
        seconds = duration_sec if duration_sec is not None else 3600.0
        return np.zeros((1, int(seconds * self.sr)), dtype=np.float32), self.sr


@pytest.fixture
def sphn_stub(monkeypatch: pytest.MonkeyPatch) -> _RecordingSphn:
    stub = _RecordingSphn()
    monkeypatch.setattr(align_data, "sphn", stub)
    return stub


def test_first_utterance_of_a_recording_reads_only_its_own_window(sphn_stub: _RecordingSphn):
    """start == 0.0 must still bound the read by duration, not read the file."""
    wav = align_data.read_utterance(
        {"path": "/audio/long.flac", "start": 0.0, "duration": 18.06}, sphn_stub.sr
    )
    assert sphn_stub.calls[0]["duration_sec"] == 18.06
    # 18 s, not the hour the recording actually holds.
    assert len(wav) == pytest.approx(18.06 * sphn_stub.sr, rel=1e-6)


def test_later_utterance_reads_its_offset_window(sphn_stub: _RecordingSphn):
    align_data.read_utterance(
        {"path": "/audio/long.flac", "start": 42.5, "duration": 12.0}, sphn_stub.sr
    )
    assert sphn_stub.calls[0]["start_sec"] == 42.5
    assert sphn_stub.calls[0]["duration_sec"] == 12.0


def test_row_without_duration_reads_the_whole_file(sphn_stub: _RecordingSphn):
    """One-clip-per-file manifests carry no window; reading it all is correct."""
    align_data.read_utterance({"path": "/audio/utt.flac"}, sphn_stub.sr)
    assert sphn_stub.calls[0] == {
        "path": "/audio/utt.flac",
        "start_sec": None,
        "duration_sec": None,
    }


def test_channels_are_mixed_to_mono(monkeypatch: pytest.MonkeyPatch):
    class _Stereo:
        def read(
            self, path: str, start_sec: float | None = None, duration_sec: float | None = None
        ) -> tuple[np.ndarray, int]:
            return np.stack([np.full(1600, 1.0), np.full(1600, 3.0)]).astype(np.float32), 16000

    monkeypatch.setattr(align_data, "sphn", _Stereo())
    wav = align_data.read_utterance({"path": "/a.flac", "duration": 0.1}, 16000)
    assert wav.ndim == 1
    assert wav[0] == pytest.approx(2.0)
