"""
test_spectrum_correction.py

Tests for the spectrum correction logic in 02_spectrum_correction.py.

Covers:
  - compute_mean_log_power_spectrum : shape, mono/stereo, resampling, bad file
  - compute_correction_filters      : reference identity, filter values, missing reference
  - apply_and_save                  : output created, correct sr, identity filter preserves RMS

Run with:  pytest test_spectrum_correction.py -v
"""

import sys
import importlib.util
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

# Module name starts with a digit — load directly
_mod_path = Path(__file__).resolve().parent / 'preprocessing' / '02_spectrum_correction.py'
_spec = importlib.util.spec_from_file_location('spectrum_correction', _mod_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_mean_log_power_spectrum = _mod.compute_mean_log_power_spectrum
compute_correction_filters      = _mod.compute_correction_filters
apply_and_save                  = _mod.apply_and_save
SAMPLE_RATE                     = _mod.SAMPLE_RATE
REFERENCE_DEVICE                = _mod.REFERENCE_DEVICE
_CORR_N_FFT                     = _mod._CORR_N_FFT

N_BINS = _CORR_N_FFT // 2 + 1   # expected spectrum length (129 at N_FFT=256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav(path: Path, duration_s: float, sr: int = SAMPLE_RATE,
             channels: int = 1, frequency: float = 440.0) -> None:
    """Write a sine-tone wav at the given sample rate."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    if channels == 2:
        audio = np.stack([audio, audio * 0.8], axis=1)
    sf.write(str(path), audio, sr, subtype='PCM_16')


def flat_profile(value: float = 0.0) -> np.ndarray:
    return np.full(N_BINS, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# compute_mean_log_power_spectrum
# ---------------------------------------------------------------------------

class TestComputeMeanLogPowerSpectrum:

    def test_output_shape(self, tmp_path):
        wav = tmp_path / 'test.wav'
        make_wav(wav, 2.0)
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert spectrum is not None
        assert spectrum.shape == (N_BINS,)

    def test_mono_audio(self, tmp_path):
        wav = tmp_path / 'mono.wav'
        make_wav(wav, 1.0, channels=1)
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert spectrum is not None

    def test_stereo_averaged_to_mono(self, tmp_path):
        wav = tmp_path / 'stereo.wav'
        make_wav(wav, 1.0, channels=2)
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert spectrum is not None
        assert spectrum.shape == (N_BINS,)

    def test_resampling_from_higher_sr(self, tmp_path):
        # Write at 8 kHz, request 4 kHz — should resample successfully
        wav = tmp_path / 'hires.wav'
        make_wav(wav, 1.0, sr=8000)
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert spectrum is not None
        assert spectrum.shape == (N_BINS,)

    def test_values_are_finite(self, tmp_path):
        wav = tmp_path / 'test.wav'
        make_wav(wav, 2.0)
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert np.all(np.isfinite(spectrum))

    def test_nonexistent_file_returns_none(self, tmp_path):
        wav = tmp_path / 'does_not_exist.wav'
        spectrum = compute_mean_log_power_spectrum(wav, sr=SAMPLE_RATE)
        assert spectrum is None

    def test_louder_signal_has_higher_power(self, tmp_path):
        quiet = tmp_path / 'quiet.wav'
        loud  = tmp_path / 'loud.wav'
        n = SAMPLE_RATE * 2
        t = np.linspace(0, 2.0, n, endpoint=False)
        sf.write(str(quiet), (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
                 SAMPLE_RATE, subtype='PCM_16')
        sf.write(str(loud),  (0.9 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
                 SAMPLE_RATE, subtype='PCM_16')
        s_quiet = compute_mean_log_power_spectrum(quiet, sr=SAMPLE_RATE)
        s_loud  = compute_mean_log_power_spectrum(loud,  sr=SAMPLE_RATE)
        assert s_loud.mean() > s_quiet.mean()


# ---------------------------------------------------------------------------
# compute_correction_filters
# ---------------------------------------------------------------------------

class TestComputeCorrectionFilters:

    def test_reference_device_gets_identity_filter(self):
        profiles = {
            REFERENCE_DEVICE: flat_profile(1.0),
            'trunc':           flat_profile(0.5),
        }
        filters = compute_correction_filters(profiles)
        assert filters[REFERENCE_DEVICE] == pytest.approx(
            np.zeros(N_BINS, dtype=np.float32))

    def test_non_reference_filter_is_ref_minus_device(self):
        ref_val    = 2.0
        trunc_val  = 0.5
        profiles   = {
            REFERENCE_DEVICE: flat_profile(ref_val),
            'trunc':           flat_profile(trunc_val),
        }
        filters = compute_correction_filters(profiles)
        expected = flat_profile(ref_val - trunc_val).astype(np.float32)
        assert filters['trunc'] == pytest.approx(expected)

    def test_none_profile_gets_identity_filter(self):
        profiles = {
            REFERENCE_DEVICE: flat_profile(1.0),
            'trunc':           None,
        }
        filters = compute_correction_filters(profiles)
        assert filters['trunc'] == pytest.approx(np.zeros(N_BINS, dtype=np.float32))

    def test_missing_reference_raises(self):
        profiles = {'trunc': flat_profile(0.5)}
        with pytest.raises(ValueError, match=REFERENCE_DEVICE):
            compute_correction_filters(profiles)

    def test_filter_dtype_is_float32(self):
        profiles = {
            REFERENCE_DEVICE: flat_profile(1.0),
            'trunc':           flat_profile(0.5),
        }
        filters = compute_correction_filters(profiles)
        for filt in filters.values():
            assert filt.dtype == np.float32

    def test_filter_shape_matches_n_bins(self):
        profiles = {REFERENCE_DEVICE: flat_profile(0.0)}
        filters = compute_correction_filters(profiles)
        assert filters[REFERENCE_DEVICE].shape == (N_BINS,)


# ---------------------------------------------------------------------------
# apply_and_save
# ---------------------------------------------------------------------------

class TestApplyAndSave:

    def test_output_file_created(self, tmp_path):
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 2.0)
        identity = np.zeros(N_BINS, dtype=np.float32)
        result = apply_and_save(wav_in, wav_out, identity)
        assert result is True
        assert wav_out.exists()

    def test_output_sample_rate_is_target(self, tmp_path):
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 2.0)
        apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        _, sr = sf.read(str(wav_out))
        assert sr == SAMPLE_RATE

    def test_output_length_approximately_matches_input(self, tmp_path):
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 3.0)
        apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        audio_in,  _ = sf.read(str(wav_in))
        audio_out, _ = sf.read(str(wav_out))
        # iSTFT may differ by up to one window length (256 samples) at boundaries
        assert abs(len(audio_out) - len(audio_in)) <= _CORR_N_FFT

    def test_identity_filter_preserves_rms(self, tmp_path):
        # An all-zero correction filter (exp(0)=1) should leave amplitude unchanged.
        # RMS before and after should match within a small tolerance.
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 2.0)
        apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        audio_in,  _ = sf.read(str(wav_in))
        audio_out, _ = sf.read(str(wav_out))
        rms_in  = np.sqrt(np.mean(audio_in.astype(np.float32) ** 2))
        rms_out = np.sqrt(np.mean(audio_out.astype(np.float32) ** 2))
        assert rms_out == pytest.approx(rms_in, rel=0.05)

    def test_positive_filter_boosts_amplitude(self, tmp_path):
        # A uniformly positive filter (exp(filter) > 1) should increase RMS.
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 2.0)
        boost_filter = np.full(N_BINS, 1.0, dtype=np.float32)
        apply_and_save(wav_in, wav_out, boost_filter)
        audio_in,  _ = sf.read(str(wav_in))
        audio_out, _ = sf.read(str(wav_out))
        rms_in  = np.sqrt(np.mean(audio_in.astype(np.float32) ** 2))
        rms_out = np.sqrt(np.mean(audio_out.astype(np.float32) ** 2))
        assert rms_out > rms_in

    def test_input_resampled_from_higher_sr(self, tmp_path):
        # Input recorded at 8 kHz — should be resampled to 4 kHz before saving
        wav_in  = tmp_path / 'hires.wav'
        wav_out = tmp_path / 'output.wav'
        make_wav(wav_in, 2.0, sr=8000)
        apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        _, sr = sf.read(str(wav_out))
        assert sr == SAMPLE_RATE

    def test_nonexistent_input_returns_false(self, tmp_path):
        wav_in  = tmp_path / 'missing.wav'
        wav_out = tmp_path / 'output.wav'
        result = apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        assert result is False

    def test_output_parent_dir_created(self, tmp_path):
        wav_in  = tmp_path / 'input.wav'
        wav_out = tmp_path / 'subdir' / 'nested' / 'output.wav'
        make_wav(wav_in, 1.0)
        apply_and_save(wav_in, wav_out, np.zeros(N_BINS, dtype=np.float32))
        assert wav_out.exists()
