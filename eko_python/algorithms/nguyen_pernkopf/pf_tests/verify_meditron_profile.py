"""
verify_meditron_profile.py

Sanity-checks the spectrum analysis pipeline by recomputing the Meditron
device profile from scratch and comparing it against the stored profile in
device_profiles.json.

If the two profiles are nearly identical, we can trust that:
  - compute_mean_log_power_spectrum() is consistent between the original
    spectrum_correction.py run and the pf_tests/analyse_spectrum.py run.
  - The Eko correction filter (Meditron − Eko) is valid.

Method:
  1. Load the manifest and filter to train-split Meditron recordings.
  2. Recompute the mean log power spectrum for each recording.
  3. Average them → recomputed Meditron profile.
  4. Load the stored Meditron profile from device_profiles.json.
  5. Plot both overlaid and report MAE / max deviation.

Usage:
    python pf_tests/verify_meditron_profile.py            # all train Meditron files
    python pf_tests/verify_meditron_profile.py --max 10   # quick check with 10 files
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    REFERENCE_DEVICE,
    MANIFEST_PATH,
    RAW_DATA_PATH,
    SPECTRUM_CORRECTION_PROFILES_PATH,
)

RESULTS_DIR = Path(__file__).parent / 'results'


def compute_mean_log_power_spectrum(wav_path: Path, sr: int, n_fft: int, hop_length: int):
    try:
        audio, orig_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=sr,
                                     res_type='soxr_hq')
        stft  = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        power = np.maximum(np.abs(stft) ** 2, 1e-10)
        return np.mean(np.log(power), axis=1)
    except Exception as e:
        print(f"  Warning: could not process '{wav_path.name}': {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None,
                        help='Max number of files to process (default: all)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MEDITRON PROFILE VERIFICATION")
    print("=" * 60)

    # ── Load manifest, filter to train Meditron recordings ───────────────
    manifest = pd.read_csv(MANIFEST_PATH)
    train_meditron = (
        manifest[
            (manifest['split'] == 'train') &
            (manifest['device'] == REFERENCE_DEVICE)
        ]
        [['recording_id']]
        .drop_duplicates('recording_id')
    )

    if args.max is not None:
        train_meditron = train_meditron.head(args.max)

    print(f"Train Meditron recordings: {len(train_meditron)}"
          + (f" (capped at {args.max})" if args.max else ""))
    print(f"Raw data path:             {RAW_DATA_PATH}")
    print(f"Stored profiles:           {SPECTRUM_CORRECTION_PROFILES_PATH}")
    print("=" * 60)

    # ── Recompute profile ─────────────────────────────────────────────────
    freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    spectra   = []

    for i, (_, row) in enumerate(train_meditron.iterrows(), 1):
        wav_path = Path(RAW_DATA_PATH) / f"{row['recording_id']}.wav"
        if not wav_path.exists():
            print(f"  [{i}/{len(train_meditron)}] Missing: {wav_path.name}")
            continue
        print(f"  [{i}/{len(train_meditron)}] {wav_path.name}", flush=True)
        spectrum = compute_mean_log_power_spectrum(wav_path, SAMPLE_RATE, N_FFT, HOP_LENGTH)
        if spectrum is not None:
            spectra.append(spectrum)

    if not spectra:
        print("No spectra computed — check RAW_DATA_PATH in config.py.")
        return

    recomputed_profile = np.stack(spectra).mean(axis=0)
    print(f"\nRecomputed profile from {len(spectra)} recordings.")

    # ── Load stored profile ───────────────────────────────────────────────
    with open(SPECTRUM_CORRECTION_PROFILES_PATH) as f:
        profiles_raw = json.load(f)

    stored_profile = np.array(profiles_raw[REFERENCE_DEVICE])

    # ── Compare ───────────────────────────────────────────────────────────
    diff = recomputed_profile - stored_profile
    mae  = np.mean(np.abs(diff))
    max_dev = np.max(np.abs(diff))

    print(f"\nProfile comparison (recomputed vs stored):")
    print(f"  MAE:         {mae:.6f} log units")
    print(f"  Max dev:     {max_dev:.6f} log units")
    print(f"  Mean stored: {stored_profile.mean():.4f}")
    print(f"  Mean recomp: {recomputed_profile.mean():.4f}")

    if mae < 1e-6:
        verdict = "EXACT MATCH — profiles are identical."
    elif mae < 0.01:
        verdict = "NEAR-PERFECT MATCH — negligible numerical difference."
    elif mae < 0.1:
        verdict = "CLOSE MATCH — small deviation, likely acceptable."
    else:
        verdict = "MISMATCH — profiles differ substantially. Investigate."

    print(f"\nVerdict: {verdict}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Meditron Profile Verification', fontsize=14, fontweight='bold')

    # Panel 1: individual spectra + both profiles
    ax = axes[0]
    for spectrum in spectra:
        ax.plot(freq_bins, spectrum, alpha=0.15, linewidth=0.7, color='steelblue')
    ax.plot(freq_bins, recomputed_profile, color='steelblue', linewidth=2.5,
            label=f'Recomputed mean (n={len(spectra)})')
    ax.plot(freq_bins, stored_profile, color='red', linewidth=2,
            linestyle='--', label='Stored profile (device_profiles.json)')
    ax.set_title('Meditron: Individual Recordings, Recomputed Mean, and Stored Profile')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mean log power')
    ax.legend()
    ax.set_xlim(0, SAMPLE_RATE // 2)
    ax.grid(True, alpha=0.3)

    # Panel 2: difference between profiles
    ax = axes[1]
    ax.plot(freq_bins, diff, color='darkred', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(freq_bins, diff, 0, alpha=0.2, color='red')
    ax.set_title(f'Difference: Recomputed − Stored  (MAE = {mae:.6f})')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Difference (log units)')
    ax.set_xlim(0, SAMPLE_RATE // 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / 'meditron_profile_verification.png'
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to '{plot_path}'.")


if __name__ == '__main__':
    main()
