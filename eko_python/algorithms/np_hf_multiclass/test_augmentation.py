"""
test_augmentation.py

Tests for the augmentation logic in 04_augment.py.

Covers:
  - apply_volume   : scaling correctness, shape preserved
  - apply_noise    : SNR roughly achieved, shape preserved, silent input unchanged
  - apply_speed    : length changes in correct direction, identity rate
  - fix_length     : truncation, reflect-padding, exact target length
  - make_augmented_copy : output is exactly TARGET_SAMPLES, differs from input
  - run_augmentation    : correct file count, manifest rows, labels preserved,
                          aug_index values, test windows untouched

Run with:  pytest test_augmentation.py -v
"""

import sys
import importlib.util
import pytest
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

_mod_path = Path(__file__).resolve().parent / 'preprocessing' / '04_augment.py'
_spec = importlib.util.spec_from_file_location('augment', _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

apply_volume        = _mod.apply_volume
apply_noise         = _mod.apply_noise
apply_speed         = _mod.apply_speed
fix_length          = _mod.fix_length
make_augmented_copy = _mod.make_augmented_copy
run_augmentation    = _mod.run_augmentation
TARGET_SAMPLES      = _mod.TARGET_SAMPLES
SAMPLE_RATE         = _mod.SAMPLE_RATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sine_wave(duration_s: float = 1.0, freq: float = 440.0,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), audio, sr, subtype='PCM_16')


def make_manifest_row(wav_path: Path, split: str = 'train',
                      das: int = 0, cas: int = 0) -> dict:
    return {
        'wav_path':    str(wav_path),
        'recording_id': wav_path.stem,
        'device':      'steth',
        'split':       split,
        'window_start': 0.0,
        'window_end':   1.0,
        'das':         das,
        'cas':         cas,
        'das_overlap': 0.0,
        'cas_overlap': 0.0,
    }


# ---------------------------------------------------------------------------
# apply_volume
# ---------------------------------------------------------------------------

class TestApplyVolume:

    def test_gain_scales_amplitude(self):
        audio = sine_wave()
        out = apply_volume(audio, 2.0)
        np.testing.assert_allclose(out, audio * 2.0)

    def test_gain_below_one_reduces_amplitude(self):
        audio = sine_wave()
        out = apply_volume(audio, 0.5)
        assert np.max(np.abs(out)) < np.max(np.abs(audio))

    def test_shape_preserved(self):
        audio = sine_wave()
        assert apply_volume(audio, 1.3).shape == audio.shape

    def test_identity_gain(self):
        audio = sine_wave()
        np.testing.assert_array_equal(apply_volume(audio, 1.0), audio)


# ---------------------------------------------------------------------------
# apply_noise
# ---------------------------------------------------------------------------

class TestApplyNoise:

    def test_output_shape_preserved(self):
        audio = sine_wave()
        assert apply_noise(audio, snr_db=30.0).shape == audio.shape

    def test_noise_increases_rms(self):
        audio = sine_wave()
        noisy = apply_noise(audio, snr_db=30.0)
        rms_in  = np.sqrt(np.mean(audio ** 2))
        rms_out = np.sqrt(np.mean(noisy ** 2))
        assert rms_out > rms_in

    def test_higher_snr_closer_to_original(self):
        audio = sine_wave()
        low_snr  = apply_noise(audio, snr_db=10.0)
        high_snr = apply_noise(audio, snr_db=40.0)
        diff_low  = np.mean((low_snr  - audio) ** 2)
        diff_high = np.mean((high_snr - audio) ** 2)
        assert diff_high < diff_low

    def test_silent_input_returned_unchanged(self):
        silent = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        out = apply_noise(silent, snr_db=30.0)
        np.testing.assert_array_equal(out, silent)

    def test_output_dtype_float32(self):
        audio = sine_wave()
        assert apply_noise(audio, 30.0).dtype == np.float32


# ---------------------------------------------------------------------------
# apply_speed
# ---------------------------------------------------------------------------

class TestApplySpeed:

    def test_rate_above_one_gives_fewer_samples(self):
        # rate > 1 → sped up → fewer output samples
        audio = sine_wave()
        out = apply_speed(audio, rate=1.1)
        assert len(out) < len(audio)

    def test_rate_below_one_gives_more_samples(self):
        # rate < 1 → slowed down → more output samples
        audio = sine_wave()
        out = apply_speed(audio, rate=0.9)
        assert len(out) > len(audio)

    def test_identity_rate_preserves_length(self):
        audio = sine_wave()
        out = apply_speed(audio, rate=1.0)
        assert len(out) == len(audio)


# ---------------------------------------------------------------------------
# fix_length
# ---------------------------------------------------------------------------

class TestFixLength:

    def test_truncation(self):
        audio = np.ones(TARGET_SAMPLES + 500, dtype=np.float32)
        out = fix_length(audio, TARGET_SAMPLES)
        assert len(out) == TARGET_SAMPLES

    def test_padding_reaches_target(self):
        audio = np.ones(TARGET_SAMPLES - 200, dtype=np.float32)
        out = fix_length(audio, TARGET_SAMPLES)
        assert len(out) == TARGET_SAMPLES

    def test_exact_length_unchanged(self):
        audio = np.ones(TARGET_SAMPLES, dtype=np.float32)
        out = fix_length(audio, TARGET_SAMPLES)
        assert len(out) == TARGET_SAMPLES

    def test_reflect_pad_does_not_introduce_zeros(self):
        # Reflect padding should mirror existing content — no silent gaps
        audio = np.ones(TARGET_SAMPLES // 2, dtype=np.float32) * 0.5
        out = fix_length(audio, TARGET_SAMPLES)
        assert np.all(out != 0.0)

    def test_truncation_keeps_start(self):
        audio = np.arange(TARGET_SAMPLES + 100, dtype=np.float32)
        out = fix_length(audio, TARGET_SAMPLES)
        np.testing.assert_array_equal(out, audio[:TARGET_SAMPLES])


# ---------------------------------------------------------------------------
# make_augmented_copy
# ---------------------------------------------------------------------------

class TestMakeAugmentedCopy:

    def test_output_is_exactly_target_samples(self):
        audio = sine_wave()
        out = make_augmented_copy(audio, volume=1.0, noise_snr=35.0, speed=1.05)
        assert len(out) == TARGET_SAMPLES

    def test_output_differs_from_input(self):
        audio = sine_wave()
        out = make_augmented_copy(audio, volume=1.2, noise_snr=35.0, speed=0.97)
        assert not np.allclose(out, audio)

    def test_output_dtype_float32(self):
        audio = sine_wave()
        out = make_augmented_copy(audio, volume=1.0, noise_snr=35.0, speed=1.0)
        assert out.dtype == np.float32

    def test_volume_only_still_target_length(self):
        # speed=1.0, no resampling — still should come out at TARGET_SAMPLES
        audio = sine_wave()
        out = make_augmented_copy(audio, volume=0.8, noise_snr=40.0, speed=1.0)
        assert len(out) == TARGET_SAMPLES


# ---------------------------------------------------------------------------
# run_augmentation
# ---------------------------------------------------------------------------

class TestRunAugmentation:

    def _make_manifest(self, tmp_path, n_train=2, n_test=1,
                       das=0, cas=1) -> pd.DataFrame:
        rows = []
        for i in range(n_train):
            audio = sine_wave()
            wav   = tmp_path / f'train_w{i:04d}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='train', das=das, cas=cas))
        for i in range(n_test):
            audio = sine_wave()
            wav   = tmp_path / f'test_w{i:04d}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='test'))
        return pd.DataFrame(rows)

    def test_correct_number_of_augmented_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 3)
        manifest = self._make_manifest(tmp_path, n_train=2)
        run_augmentation(manifest)
        aug_wavs = list((tmp_path / 'augmented').glob('*.wav'))
        assert len(aug_wavs) == 2 * 3   # 2 train windows × 3 copies

    def test_manifest_row_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 3)
        manifest = self._make_manifest(tmp_path, n_train=2, n_test=1)
        result = run_augmentation(manifest)
        # 3 originals (2 train + 1 test) + 6 augmented = 9
        assert len(result) == 3 + 2 * 3

    def test_aug_index_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 3)
        manifest = self._make_manifest(tmp_path, n_train=2)
        result = run_augmentation(manifest)
        assert set(result['aug_index'].unique()) == {0, 1, 2, 3}

    def test_labels_inherited_by_augmented_copies(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 2)
        manifest = self._make_manifest(tmp_path, n_train=2, das=1, cas=0)
        result = run_augmentation(manifest)
        aug_rows = result[result['aug_index'] > 0]
        assert (aug_rows['das'] == 1).all()
        assert (aug_rows['cas'] == 0).all()

    def test_test_windows_not_augmented(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 3)
        manifest = self._make_manifest(tmp_path, n_train=1, n_test=2)
        result = run_augmentation(manifest)
        test_rows = result[result['split'] == 'test']
        assert (test_rows['aug_index'] == 0).all()

    def test_augmented_wavs_are_correct_length(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 2)
        manifest = self._make_manifest(tmp_path, n_train=1)
        result = run_augmentation(manifest)
        aug_rows = result[result['aug_index'] > 0]
        for _, row in aug_rows.iterrows():
            audio, sr = sf.read(row['wav_path'])
            assert len(audio) == TARGET_SAMPLES
            assert sr == SAMPLE_RATE

    def test_rerunning_does_not_compound_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'AUGMENTED_DIR', tmp_path / 'augmented')
        monkeypatch.setattr(_mod, 'N_AUGMENTATIONS', 2)
        manifest = self._make_manifest(tmp_path, n_train=2)
        result_first  = run_augmentation(manifest)
        result_second = run_augmentation(result_first)
        # Second run should produce same row count as first
        assert len(result_second) == len(result_first)
