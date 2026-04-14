"""
test_windowing.py

Tests for the windowing and labelling logic in 03_window_and_label.py.

Tests are self-contained — no real audio files or parsed_segments.csv needed.
Synthetic wavs are written to a temp directory and cleaned up after each test.

Run with:  pytest test_windowing.py -v
"""

import sys
import importlib.util
import pytest
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# Module name starts with a digit so standard import won't work — load directly
_mod_path = Path(__file__).resolve().parent / 'preprocessing' / '03_window_and_label.py'
_spec = importlib.util.spec_from_file_location('window_and_label', _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

segment_overlap         = _mod.segment_overlap
total_overlap           = _mod.total_overlap
window_recording        = _mod.window_recording
WINDOW_SAMPLES          = _mod.WINDOW_SAMPLES
HOP_SAMPLES             = _mod.HOP_SAMPLES
SAMPLE_RATE             = _mod.SAMPLE_RATE
LABEL_OVERLAP_THRESHOLD = _mod.LABEL_OVERLAP_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav(path: Path, duration_s: float, sr: int = SAMPLE_RATE) -> None:
    """Write a silent wav of the given duration."""
    n = int(duration_s * sr)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr, subtype='PCM_16')


def make_events(*segments, das=0, cas=0) -> pd.DataFrame:
    """
    Build a minimal events DataFrame from (event_start, event_end) tuples.
    das / cas flags apply uniformly to all rows (override per-row after if needed).
    """
    rows = [{'event_start': s, 'event_end': e, 'das': das, 'cas': cas}
            for s, e in segments]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# segment_overlap
# ---------------------------------------------------------------------------

class TestSegmentOverlap:

    def test_no_overlap_before(self):
        # event ends before window starts
        assert segment_overlap(2.0, 3.0, 0.0, 1.5) == 0.0

    def test_no_overlap_after(self):
        # event starts after window ends
        assert segment_overlap(0.0, 1.0, 1.5, 2.5) == 0.0

    def test_event_touches_window_edge(self):
        # event ends exactly at window start — zero overlap
        assert segment_overlap(1.0, 2.0, 0.0, 1.0) == 0.0

    def test_partial_overlap_left(self):
        # event partially overlaps left side of window
        assert segment_overlap(1.0, 2.0, 0.5, 1.5) == pytest.approx(0.5)

    def test_partial_overlap_right(self):
        # event partially overlaps right side of window
        assert segment_overlap(1.0, 2.0, 1.5, 2.5) == pytest.approx(0.5)

    def test_event_fully_inside_window(self):
        assert segment_overlap(0.0, 1.0, 0.2, 0.6) == pytest.approx(0.4)

    def test_event_spans_full_window(self):
        assert segment_overlap(0.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_event_larger_than_window(self):
        assert segment_overlap(0.5, 1.5, 0.0, 2.0) == pytest.approx(1.0)

    def test_result_never_negative(self):
        assert segment_overlap(5.0, 6.0, 0.0, 1.0) >= 0.0


# ---------------------------------------------------------------------------
# total_overlap
# ---------------------------------------------------------------------------

class TestTotalOverlap:

    def test_no_labelled_events(self):
        events = make_events((0.0, 1.0), das=0)
        assert total_overlap(0.0, 1.0, events, 'das') == 0.0

    def test_empty_dataframe(self):
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        assert total_overlap(0.0, 1.0, events, 'das') == 0.0

    def test_single_das_event_inside_window(self):
        events = make_events((0.2, 0.5), das=1)
        assert total_overlap(0.0, 1.0, events, 'das') == pytest.approx(0.3)

    def test_two_das_events_summed(self):
        # two separate crackle events, each contributing 0.1s
        events = make_events((0.1, 0.2), (0.5, 0.6), das=1)
        assert total_overlap(0.0, 1.0, events, 'das') == pytest.approx(0.2)

    def test_cas_event_does_not_count_for_das(self):
        # one CAS event, no DAS — das overlap should be 0
        rows = [{'event_start': 0.0, 'event_end': 0.5, 'das': 0, 'cas': 1}]
        events = pd.DataFrame(rows)
        assert total_overlap(0.0, 1.0, events, 'das') == 0.0
        assert total_overlap(0.0, 1.0, events, 'cas') == pytest.approx(0.5)

    def test_mixed_das_and_cas_events(self):
        rows = [
            {'event_start': 0.1, 'event_end': 0.3, 'das': 1, 'cas': 0},
            {'event_start': 0.6, 'event_end': 0.9, 'das': 0, 'cas': 1},
        ]
        events = pd.DataFrame(rows)
        assert total_overlap(0.0, 1.0, events, 'das') == pytest.approx(0.2)
        assert total_overlap(0.0, 1.0, events, 'cas') == pytest.approx(0.3)

    def test_event_partially_outside_window(self):
        # event spans 0.8–1.2 s, window is 0–1 s → only 0.2 s counts
        events = make_events((0.8, 1.2), das=1)
        assert total_overlap(0.0, 1.0, events, 'das') == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# window_recording — window count and duration
# ---------------------------------------------------------------------------

class TestWindowCounts:

    def test_exact_multiple_of_hop(self, tmp_path):
        # 3s recording, 1s window, 0.5s hop → windows at 0,0.5,1,1.5,2 = 5 windows
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert len(rows) == 5

    def test_recording_shorter_than_one_window(self, tmp_path):
        # 0.8s recording — no full 1s window possible, should return nothing
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 0.8)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows == []

    def test_partial_tail_is_dropped(self, tmp_path):
        # 2.3s recording → windows at 0, 0.5, 1, 1.5 start positions
        # window at 1.5 ends at 2.5 > 2.3 → dropped, so 3 windows
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 2.3)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert len(rows) == 3

    def test_window_audio_is_exactly_one_second(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 4.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        for row in rows:
            saved_audio, sr = sf.read(row['wav_path'])
            assert len(saved_audio) == WINDOW_SAMPLES
            assert sr == SAMPLE_RATE

    def test_windows_start_times_correct(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        starts = [r['window_start'] for r in rows]
        expected = [0.0, 0.5, 1.0, 1.5, 2.0]
        assert starts == pytest.approx(expected)


# ---------------------------------------------------------------------------
# window_recording — labelling correctness
# ---------------------------------------------------------------------------

class TestWindowLabelling:

    def test_no_events_all_negative(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        for row in rows:
            assert row['das'] == 0
            assert row['cas'] == 0

    def test_das_above_threshold_labelled_1(self, tmp_path):
        # DAS event from 0.0–0.5s → first window [0,1] gets 0.5s overlap → das=1
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = make_events((0.0, 0.5), das=1)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows[0]['das'] == 1

    def test_das_below_threshold_labelled_0(self, tmp_path):
        # DAS event 0.0–0.05s → first window gets 0.05s < 0.1s threshold → das=0
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = make_events((0.0, 0.05), das=1)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows[0]['das'] == 0

    def test_das_exactly_at_threshold_labelled_1(self, tmp_path):
        # Overlap exactly equal to threshold → das=1 (>= not >)
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = make_events((0.0, LABEL_OVERLAP_THRESHOLD), das=1)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows[0]['das'] == 1

    def test_cas_labelled_independently_of_das(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        # DAS event in first window, CAS event in second window only
        rows_data = [
            {'event_start': 0.1, 'event_end': 0.4, 'das': 1, 'cas': 0},  # 0.3s DAS in w0
            {'event_start': 0.6, 'event_end': 0.9, 'das': 0, 'cas': 1},  # 0.3s CAS in w1 [0.5,1.5]
        ]
        events = pd.DataFrame(rows_data)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        # Window 0: [0.0, 1.0] — DAS 0.3s → das=1; CAS event is in [0.6,0.9] also in w0 → cas=1
        assert rows[0]['das'] == 1
        assert rows[0]['cas'] == 1
        # Window 1: [0.5, 1.5] — DAS event [0.1,0.4] overlaps 0s; CAS [0.6,0.9] overlaps 0.3s → cas=1
        assert rows[1]['das'] == 0
        assert rows[1]['cas'] == 1

    def test_event_straddling_two_windows(self, tmp_path):
        # DAS event 0.4–0.8s straddles window 0 [0,1] and window 1 [0.5,1.5]
        # w0 overlap = 0.4s → das=1; w1 overlap = 0.3s → das=1
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        events = make_events((0.4, 0.8), das=1)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows[0]['das'] == 1
        assert rows[1]['das'] == 1

    def test_event_only_in_middle_window(self, tmp_path):
        # DAS event 1.1–1.3s — only overlaps window 2 [1.0,2.0] significantly
        # w0 [0,1]: 0s; w1 [0.5,1.5]: 0.2s → das=1; w2 [1,2]: 0.2s → das=1; w3 [1.5,2.5]: 0s
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 4.0)
        events = make_events((1.1, 1.3), das=1)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        labelled = [(r['window_start'], r['das']) for r in rows]
        assert labelled[0] == (0.0, 0)    # [0.0, 1.0]: no overlap
        assert labelled[1] == (0.5, 1)    # [0.5, 1.5]: 0.2s overlap → das=1
        assert labelled[2] == (1.0, 1)    # [1.0, 2.0]: 0.2s overlap → das=1
        assert labelled[3] == (1.5, 0)    # [1.5, 2.5]: no overlap

    def test_two_das_events_combined_cross_threshold(self, tmp_path):
        # Each DAS event contributes 0.06s — individually below threshold,
        # combined they total 0.12s which is above threshold → das=1
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 3.0)
        rows_data = [
            {'event_start': 0.1, 'event_end': 0.16, 'das': 1, 'cas': 0},
            {'event_start': 0.5, 'event_end': 0.56, 'das': 1, 'cas': 0},
        ]
        events = pd.DataFrame(rows_data)
        rows = window_recording(wav, events, tmp_path, split='train', device='steth')
        assert rows[0]['das'] == 1
        assert pytest.approx(rows[0]['das_overlap'], abs=1e-3) == 0.12

    def test_manifest_row_fields_present(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 2.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='test', device='trunc')
        required = {'wav_path', 'recording_id', 'device', 'split',
                    'window_start', 'window_end', 'das', 'cas',
                    'das_overlap', 'cas_overlap'}
        for row in rows:
            assert required.issubset(row.keys())

    def test_split_and_device_propagated(self, tmp_path):
        wav = tmp_path / 'rec.wav'
        make_wav(wav, 2.0)
        events = pd.DataFrame(columns=['event_start', 'event_end', 'das', 'cas'])
        rows = window_recording(wav, events, tmp_path, split='test', device='trunc')
        for row in rows:
            assert row['split'] == 'test'
            assert row['device'] == 'trunc'
