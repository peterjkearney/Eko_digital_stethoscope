"""
test_spectrograms.py

Tests for the spectrogram logic in 05_make_spectrograms.py.

Covers:
  - get_vtlp_filterbank : identity warp, shape, non-negative values,
                          alpha<1 shifts energy lower, alpha>1 shifts higher
  - compute_logmel      : output shape, log scale (all finite), mono input
  - to_png_array        : output shape, dtype uint8, value range [0, 255],
                          constant input → all zeros, resize to target
  - run_make_spectrograms : file count, manifest spec_path populated,
                            vtlp applied only to augmented train copies,
                            identity filterbank on originals/test,
                            skips rows already processed

Run with:  pytest test_spectrograms.py -v
"""

import sys
import importlib.util
import pytest
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from PIL import Image

_mod_path = Path(__file__).resolve().parent / 'preprocessing' / '05_make_spectrograms.py'
_spec = importlib.util.spec_from_file_location('make_spectrograms', _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_vtlp_filterbank   = _mod.get_vtlp_filterbank
compute_logmel        = _mod.compute_logmel
to_png_array          = _mod.to_png_array
run_make_spectrograms = _mod.run_make_spectrograms

import librosa
from config import (
    SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH,
    N_MELS, FMIN, FMAX, MODEL_INPUT_SIZE,
)

TARGET_H, TARGET_W = MODEL_INPUT_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_filterbank():
    return librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )


def _freq_bins():
    return librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)


def sine_wave(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio, SAMPLE_RATE, subtype='PCM_16')


def make_manifest_row(wav_path: Path, split: str = 'train',
                      aug_index: int = 0, das: int = 0, cas: int = 0) -> dict:
    return {
        'wav_path':     str(wav_path),
        'recording_id': wav_path.stem,
        'device':       'steth',
        'split':        split,
        'window_start': 0.0,
        'window_end':   1.0,
        'das':          das,
        'cas':          cas,
        'das_overlap':  0.0,
        'cas_overlap':  0.0,
        'aug_index':    aug_index,
    }


# ---------------------------------------------------------------------------
# get_vtlp_filterbank
# ---------------------------------------------------------------------------

class TestGetVtlpFilterbank:

    def setup_method(self):
        self.base_fb   = _base_filterbank()
        self.freq_bins = _freq_bins()

    def _warp(self, alpha, fhi=None):
        if fhi is None:
            fhi = FMAX
        return get_vtlp_filterbank(
            n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE,
            fmin=FMIN, fmax=FMAX,
            alpha=alpha, fhi=fhi,
            base_filterbank=self.base_fb,
            freq_bins=self.freq_bins,
        )

    def test_identity_alpha_equals_base(self):
        """alpha=1.0, fhi=FMAX should reproduce the base filterbank exactly."""
        warped = self._warp(alpha=1.0, fhi=FMAX)
        np.testing.assert_allclose(warped, self.base_fb, atol=1e-5)

    def test_output_shape(self):
        warped = self._warp(alpha=1.0)
        assert warped.shape == self.base_fb.shape

    def test_non_negative(self):
        for alpha in (0.9, 1.0, 1.1):
            warped = self._warp(alpha=alpha)
            assert np.all(warped >= 0.0), f"Negative values with alpha={alpha}"

    def test_alpha_lt_one_shifts_energy_lower(self):
        """alpha<1 warps frequencies down: centre-of-mass along freq axis shifts lower."""
        warped = self._warp(alpha=0.85, fhi=FMAX)
        mel_idx = np.arange(N_MELS)
        com_base   = (self.base_fb.sum(axis=1) * mel_idx).sum() / (self.base_fb.sum() + 1e-10)
        com_warped = (warped.sum(axis=1) * mel_idx).sum()       / (warped.sum()        + 1e-10)
        assert com_warped < com_base

    def test_alpha_gt_one_shifts_energy_higher(self):
        """alpha>1 warps frequencies up: centre-of-mass along freq axis shifts higher."""
        warped = self._warp(alpha=1.15, fhi=FMAX)
        mel_idx = np.arange(N_MELS)
        com_base   = (self.base_fb.sum(axis=1) * mel_idx).sum() / (self.base_fb.sum() + 1e-10)
        com_warped = (warped.sum(axis=1) * mel_idx).sum()       / (warped.sum()        + 1e-10)
        assert com_warped > com_base

    def test_different_fhi_produces_different_filterbank(self):
        fb_low  = self._warp(alpha=0.95, fhi=1200.0)
        fb_high = self._warp(alpha=0.95, fhi=1800.0)
        assert not np.allclose(fb_low, fb_high)


# ---------------------------------------------------------------------------
# compute_logmel
# ---------------------------------------------------------------------------

class TestComputeLogmel:

    def setup_method(self):
        self.filterbank = _base_filterbank()

    def test_output_shape(self):
        audio = sine_wave()
        out = compute_logmel(audio, self.filterbank)
        # librosa.stft uses center=True by default, padding by n_fft//2 each side
        # → n_frames = 1 + n_samples // hop_length
        n_frames = 1 + len(audio) // HOP_LENGTH
        assert out.shape == (N_MELS, n_frames)

    def test_output_is_finite(self):
        audio = sine_wave()
        out = compute_logmel(audio, self.filterbank)
        assert np.all(np.isfinite(out))

    def test_values_are_log_scale(self):
        """Log-mel values should be negative or small — log of values ≤ 1 is ≤ 0."""
        audio = sine_wave()
        out = compute_logmel(audio, self.filterbank)
        # log(1e-10) ≈ -23; log(large) can be positive for loud signals
        # Just confirm the range is consistent with log scale (not linear magnitude)
        assert out.min() < 0

    def test_silent_input_gives_floor_values(self):
        """Silent audio → filterbank output near zero → log clamps to log(1e-10)."""
        silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
        out = compute_logmel(silent, self.filterbank)
        expected_floor = np.log(1e-10)
        np.testing.assert_allclose(out, expected_floor, atol=1e-4)

    def test_louder_signal_gives_higher_log_mel(self):
        quiet = sine_wave() * 0.01
        loud  = sine_wave() * 0.9
        out_quiet = compute_logmel(quiet, self.filterbank)
        out_loud  = compute_logmel(loud,  self.filterbank)
        assert out_loud.mean() > out_quiet.mean()


# ---------------------------------------------------------------------------
# to_png_array
# ---------------------------------------------------------------------------

class TestToPngArray:

    def _dummy_logmel(self, rows=N_MELS, cols=247):
        return np.random.randn(rows, cols).astype(np.float32)

    def test_output_shape(self):
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        assert out.shape == (TARGET_H, TARGET_W)

    def test_dtype_uint8(self):
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        assert out.dtype == np.uint8

    def test_values_in_range(self):
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_constant_input_gives_zeros(self):
        """A flat log-mel (lo == hi) should produce all-zero output."""
        logmel = np.full((N_MELS, 247), -5.0, dtype=np.float32)
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        assert np.all(out == 0)

    def test_max_value_is_255_for_varied_input(self):
        """Normalised output should reach 255 when there is contrast."""
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        assert out.max() == 255

    def test_arbitrary_target_size(self):
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, 64, 64)
        assert out.shape == (64, 64)

    def test_can_be_saved_as_png(self, tmp_path):
        logmel = self._dummy_logmel()
        out = to_png_array(logmel, TARGET_H, TARGET_W)
        png_path = tmp_path / 'test.png'
        Image.fromarray(out).save(str(png_path))
        assert png_path.exists()
        loaded = np.array(Image.open(str(png_path)))
        assert loaded.shape == (TARGET_H, TARGET_W)


# ---------------------------------------------------------------------------
# run_make_spectrograms
# ---------------------------------------------------------------------------

class TestRunMakeSpectrograms:

    def _make_manifest(self, tmp_path, n_train=2, n_test=1, n_aug=0):
        rows = []
        for i in range(n_train):
            audio = sine_wave()
            wav   = tmp_path / f'train_w{i:04d}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='train', aug_index=0))
        for i in range(n_aug):
            audio = sine_wave()
            wav   = tmp_path / f'aug_w{i:04d}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='train', aug_index=i + 1))
        for i in range(n_test):
            audio = sine_wave()
            wav   = tmp_path / f'test_w{i:04d}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='test', aug_index=0))
        return pd.DataFrame(rows)

    def test_spec_path_populated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=2, n_test=1)
        result = run_make_spectrograms(manifest)
        assert result['spec_path'].notna().all()
        assert (result['spec_path'] != '').all()

    def test_png_files_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=2, n_test=1)
        result = run_make_spectrograms(manifest)
        for _, row in result.iterrows():
            assert Path(row['spec_path']).exists()

    def test_png_size_is_target(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=1)
        result = run_make_spectrograms(manifest)
        for _, row in result.iterrows():
            img = Image.open(row['spec_path'])
            assert img.size == (TARGET_W, TARGET_H)

    def test_png_is_greyscale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=1)
        result = run_make_spectrograms(manifest)
        for _, row in result.iterrows():
            img = Image.open(row['spec_path'])
            assert img.mode == 'L'

    def test_vtlp_applied_to_augmented_train(self, tmp_path, monkeypatch):
        """Augmented train copies should have vtlp_alpha != 1.0 (or at least recorded)."""
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        # Use alpha range that guarantees non-identity
        monkeypatch.setattr(_mod, 'VTLP_ALPHA_MIN', 0.85)
        monkeypatch.setattr(_mod, 'VTLP_ALPHA_MAX', 0.90)
        monkeypatch.setattr(_mod, 'RNG', np.random.default_rng(0))
        manifest = self._make_manifest(tmp_path, n_train=1, n_aug=3)
        result = run_make_spectrograms(manifest)
        aug_rows = result[result['aug_index'] >= 1]
        assert (aug_rows['vtlp_alpha'] != 1.0).all()

    def test_originals_get_identity_vtlp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=2, n_aug=0)
        result = run_make_spectrograms(manifest)
        orig_rows = result[result['aug_index'] == 0]
        assert (orig_rows['vtlp_alpha'] == 1.0).all()

    def test_test_windows_get_identity_vtlp(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=0, n_test=2)
        result = run_make_spectrograms(manifest)
        test_rows = result[result['split'] == 'test']
        assert (test_rows['vtlp_alpha'] == 1.0).all()

    def test_missing_wav_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=1)
        # Point one row to a nonexistent file
        manifest.at[0, 'wav_path'] = str(tmp_path / 'does_not_exist.wav')
        result = run_make_spectrograms(manifest)
        # Row with missing wav should have empty spec_path
        assert result.at[0, 'spec_path'] == ''

    def test_rerun_skips_already_processed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        manifest = self._make_manifest(tmp_path, n_train=2)
        result_first  = run_make_spectrograms(manifest)
        result_second = run_make_spectrograms(result_first)
        # Paths should be identical — no new files written
        pd.testing.assert_series_equal(
            result_first['spec_path'].reset_index(drop=True),
            result_second['spec_path'].reset_index(drop=True),
        )

    def test_labels_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_mod, 'SPECTROGRAMS_DIR', tmp_path / 'spectrograms')
        rows = []
        for das, cas in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            audio = sine_wave()
            wav   = tmp_path / f'w_d{das}_c{cas}.wav'
            make_wav(wav, audio)
            rows.append(make_manifest_row(wav, split='train', das=das, cas=cas))
        manifest = pd.DataFrame(rows)
        result = run_make_spectrograms(manifest)
        for (das, cas), row in zip([(0,0),(1,0),(0,1),(1,1)], result.itertuples()):
            assert row.das == das
            assert row.cas == cas
