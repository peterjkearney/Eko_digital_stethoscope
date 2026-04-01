"""
analyse_spectrum.py

Estimates the spectral profile of the Eko device recordings and computes
the correction filter needed to match the Meditron reference device used
in ICBHI training.

Method (mirrors spectrum_correction.py):
    1. Load existing ICBHI device profiles (Meditron reference + others).
    2. Compute mean log power spectrum across all Eko recordings.
    3. Correction filter = Meditron profile − Eko profile.
    4. Plot: per-recording spectra, device average, ICBHI device profiles,
       and the correction filter.
    5. Save the Eko profile to device_profiles.json for use in the pipeline.

The Eko device records natively at 4 kHz so no upsampling step is required.
Files are read directly from data/pf_samples/raw/.

Usage:
    python pf_tests/analyse_spectrum.py
"""

import json
import sys
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SAMPLE_RATE,
    FMAX,
    REFERENCE_DEVICE,
    DEVICE_PROFILES_PATH,
    EKO_PROJECT_ROOT,
)

# Must match the values used in 02_spectrum_correction.py to produce profiles
# with the same number of frequency bins
_CORR_N_FFT      = 256
_CORR_WIN_LENGTH = 256
_CORR_HOP_LENGTH = 64

INPUT_DIR       = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples'
RESULTS_DIR     = Path(__file__).parent / 'results'
EKO_DEVICE_NAME = 'Eko'


# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def compute_mean_log_power_spectrum(
    wav_path: Path,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> np.ndarray | None:
    try:
        audio, file_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=file_sr, target_sr=sr)
        stft  = librosa.stft(audio, n_fft=n_fft,
                             win_length=win_length, hop_length=hop_length)
        power = np.maximum(np.abs(stft) ** 2, 1e-10)
        return np.mean(np.log(power), axis=1)
    except Exception as e:
        print(f"  Warning: could not process '{wav_path.name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wav_files = sorted(INPUT_DIR.glob('*.wav'))

    if not wav_files:
        print(f"No wav files found in '{INPUT_DIR}'.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EKO DEVICE SPECTRUM ANALYSIS  (4 kHz)")
    print("=" * 60)
    print(f"Input:      {INPUT_DIR}")
    print(f"Files:      {len(wav_files)}")
    print(f"Reference:  {REFERENCE_DEVICE}")
    print(f"N_FFT:      {_CORR_N_FFT}  |  WIN: {_CORR_WIN_LENGTH}  |  HOP: {_CORR_HOP_LENGTH}")
    print("=" * 60)

    # ── Compute per-recording spectra ─────────────────────────────────────
    freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=_CORR_N_FFT)  # Hz
    spectra   = []

    for wav_path in wav_files:
        spectrum = compute_mean_log_power_spectrum(
            wav_path, sr=SAMPLE_RATE,
            n_fft=_CORR_N_FFT, win_length=_CORR_WIN_LENGTH, hop_length=_CORR_HOP_LENGTH,
        )
        if spectrum is not None:
            spectra.append((wav_path.name, spectrum))
            print(f"  Profiled: {wav_path.name}")

    if not spectra:
        print("No spectra could be computed. Exiting.")
        return

    eko_spectra_array = np.stack([s for _, s in spectra], axis=0)
    eko_profile       = eko_spectra_array.mean(axis=0)
    print(f"\nEko profile estimated from {len(spectra)} recording(s).")

    # ── Load ICBHI device profiles ────────────────────────────────────────
    with open(DEVICE_PROFILES_PATH) as f:
        icbhi_profiles_raw = json.load(f)

    icbhi_profiles = {
        device: np.array(profile)
        for device, profile in icbhi_profiles_raw.items()
        if profile is not None
    }

    if REFERENCE_DEVICE not in icbhi_profiles:
        print(f"Error: reference device '{REFERENCE_DEVICE}' not in '{DEVICE_PROFILES_PATH}'.")
        return

    ref_profile       = icbhi_profiles[REFERENCE_DEVICE]
    correction_filter = ref_profile - eko_profile

    print(f"\nCorrection filter (Eko → {REFERENCE_DEVICE}):")
    print(f"  Mean:  {correction_filter.mean():.4f} log units")
    print(f"  Std:   {correction_filter.std():.4f} log units")
    print(f"  Range: [{correction_filter.min():.4f}, {correction_filter.max():.4f}]")

    # ── Save Eko profile to device_profiles.json ──────────────────────────
    icbhi_profiles_raw[EKO_DEVICE_NAME] = eko_profile.tolist()
    with open(DEVICE_PROFILES_PATH, 'w') as f:
        json.dump(icbhi_profiles_raw, f, indent=2)
    print(f"\nEko profile saved to '{DEVICE_PROFILES_PATH}'.")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle('Eko Device Spectrum Analysis  (4 kHz pipeline)',
                 fontsize=14, fontweight='bold')

    # Panel 1: per-recording spectra + Eko mean
    ax = axes[0]
    for name, spectrum in spectra:
        ax.plot(freq_bins, spectrum, alpha=0.3, linewidth=0.8, color='steelblue')
    ax.plot(freq_bins, eko_profile, color='steelblue', linewidth=2,
            label=f'Eko mean (n={len(spectra)})')
    ax.set_title('Eko Recordings — Individual and Mean Log Power Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mean log power')
    ax.legend()
    ax.set_xlim(0, FMAX)
    ax.grid(True, alpha=0.3)

    # Panel 2: Eko vs ICBHI device profiles
    ax = axes[1]
    colors = {'Meditron': 'red', 'LittC2SE': 'green',
              'Litt3200': 'orange', 'AKGC417L': 'purple'}
    for device, profile in icbhi_profiles.items():
        color = colors.get(device, 'grey')
        lw    = 2.5 if device == REFERENCE_DEVICE else 1.5
        ls    = '-'  if device == REFERENCE_DEVICE else '--'
        ax.plot(freq_bins, profile, color=color, linewidth=lw,
                linestyle=ls, label=device)
    ax.plot(freq_bins, eko_profile, color='steelblue', linewidth=2,
            label='Eko (this device)')
    ax.set_title('Eko vs ICBHI Device Spectral Profiles')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mean log power')
    ax.legend()
    ax.set_xlim(0, FMAX)
    ax.grid(True, alpha=0.3)

    # Panel 3: correction filter
    ax = axes[2]
    ax.plot(freq_bins, correction_filter, color='darkred', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(freq_bins, correction_filter, 0,
                    where=correction_filter > 0, alpha=0.2, color='green',
                    label=f'Boost (Eko weaker than {REFERENCE_DEVICE})')
    ax.fill_between(freq_bins, correction_filter, 0,
                    where=correction_filter < 0, alpha=0.2, color='red',
                    label=f'Cut (Eko stronger than {REFERENCE_DEVICE})')
    ax.set_title(f'Correction Filter: Eko → {REFERENCE_DEVICE}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Correction (log units)')
    ax.legend()
    ax.set_xlim(0, FMAX)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / 'eko_spectrum_analysis.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to '{plot_path}'.")


if __name__ == '__main__':
    main()
