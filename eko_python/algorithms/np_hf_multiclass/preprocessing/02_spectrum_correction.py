"""
02_spectrum_correction.py

Normalises the frequency response of the two HF_Lung_V1 recording devices so
that all audio entering the training pipeline looks spectrally similar.

HF_Lung_V1 contains recordings from two devices:
    steth  — Littmann 3200 electronic stethoscope (reference)
    trunc  — proprietary truncated-bandwidth device

The 'trunc' device has a different frequency response to the 'steth' device.
This step estimates the average spectral profile of each device from training
recordings only, then applies a correction filter to every recording (train
and test) so that 'trunc' recordings match the 'steth' reference profile.
Position metadata is ignored at this stage — all recordings of the same
device type receive the same correction.

All audio is resampled to SAMPLE_RATE (4 kHz) before profiling and
correction, so profiles capture the frequency response in the 0–2 kHz band
only — the meaningful range for lung-sound analysis.

Pipeline:
    1. Load each training recording, resample to 4 kHz, compute its mean
       log power spectrum by averaging across all STFT frames.
    2. Average these per-recording spectra across all training recordings
       for each device → one spectral profile per device.
    3. For each device compute a correction filter:
           filter = log_profile(steth) − log_profile(device)
       The reference device (steth) gets an identity filter (all zeros).
    4. Apply the correction to every recording (train + test):
           corrected_magnitude = magnitude × exp(filter)
       Reconstruct the time-domain signal via iSTFT (original phase retained)
       and save to CORRECTED_DIR as a 4 kHz wav.

Input:  parsed_segments.csv (from step 01), wav files referenced within it
Output: corrected 4 kHz wav files in CORRECTED_DIR/<original_filename>.wav
        device_profiles.json in CORRECTED_DIR
"""

import json
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    PARSED_SEGMENTS_PATH,
    SAMPLE_RATE,
    REFERENCE_DEVICE,
    CORRECTED_DIR,
    DEVICE_PROFILES_PATH,
)

# STFT parameters used only within this step for profiling and iSTFT
# reconstruction. No zero-padding (n_fft = win_length) and 75% overlap
# so phase reconstruction via iSTFT is clean.
# At 4 kHz: 64 ms window, 16 ms hop, 129 frequency bins (0–2 kHz, 15.6 Hz/bin)
_CORR_N_FFT      = 256
_CORR_WIN_LENGTH = 256
_CORR_HOP_LENGTH = 64


# ---------------------------------------------------------------------------
# Step 1: mean log power spectrum for one recording
# ---------------------------------------------------------------------------

def compute_mean_log_power_spectrum(wav_path: Path, sr: int) -> np.ndarray | None:
    # load the wav file, resample to target sr, compute mean log power across
    # all STFT frames → shape (N_FFT//2 + 1,)
    try:
        audio, orig_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=orig_sr, target_sr=sr)
        power = np.abs(librosa.stft(audio, n_fft=_CORR_N_FFT,
                                    win_length=_CORR_WIN_LENGTH,
                                    hop_length=_CORR_HOP_LENGTH)) ** 2
        power = np.maximum(power, 1e-10)  # avoid log(0)
        return np.mean(np.log(power), axis=1)
    except Exception as e:
        print(f"  Warning: could not process '{wav_path.name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Step 2: estimate one spectral profile per device (training data only)
# ---------------------------------------------------------------------------

def estimate_device_profiles(parsed_segments: pd.DataFrame) -> dict[str, np.ndarray | None]:
    # use training recordings only so test recordings never influence correction
    train = (
        parsed_segments[parsed_segments['split'] == 'train']
        [['wav_path', 'device']]
        .drop_duplicates('wav_path')
    )
    print(f"  Estimating profiles from {len(train)} training recordings...")

    spectra_by_device: dict[str, list[np.ndarray]] = {}

    for _, row in tqdm(train.iterrows(), total=len(train), desc="  Profiling"):
        wav_path = Path(row['wav_path'])
        if not wav_path.exists():
            print(f"  Warning: '{wav_path.name}' not found. Skipping.")
            continue
        spectrum = compute_mean_log_power_spectrum(wav_path, sr=SAMPLE_RATE)
        if spectrum is not None:
            spectra_by_device.setdefault(row['device'], []).append(spectrum)

    profiles = {}
    for device, spectra in spectra_by_device.items():
        # median rather than mean to avoid being pulled by outlier recordings
        profiles[device] = np.median(np.stack(spectra), axis=0)
        print(f"  {device}: profile from {len(spectra)} recordings.")

    return profiles


# ---------------------------------------------------------------------------
# Step 3: correction filter per device
# ---------------------------------------------------------------------------

def compute_correction_filters(profiles: dict[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    # correction = log_profile(reference) − log_profile(device)
    # applying exp(correction) to linear magnitude shifts the device spectrum
    # to match the reference. the reference device gets zeros (identity).
    if REFERENCE_DEVICE not in profiles:
        raise ValueError(
            f"Reference device '{REFERENCE_DEVICE}' has no profile. "
            f"Check that it has training recordings in parsed_segments.csv."
        )
    ref = profiles[REFERENCE_DEVICE]

    filters = {}
    for device, profile in profiles.items():
        if profile is None:
            print(f"  Warning: no profile for '{device}'. Using identity filter.")
            filters[device] = np.zeros(_CORR_N_FFT // 2 + 1, dtype=np.float32)
        else:
            filters[device] = (ref - profile).astype(np.float32)

    # Reference device always gets identity correction
    filters[REFERENCE_DEVICE] = np.zeros(_CORR_N_FFT // 2 + 1, dtype=np.float32)
    return filters


# ---------------------------------------------------------------------------
# Step 4: apply correction to one recording and save
# ---------------------------------------------------------------------------

def apply_and_save(
    wav_path: Path,
    out_path: Path,
    correction_filter: np.ndarray,
) -> bool:
    # load → resample to 4 kHz → apply correction in STFT magnitude domain
    # → reconstruct via iSTFT (original phase retained) → save
    try:
        audio, orig_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=orig_sr, target_sr=SAMPLE_RATE)

        stft      = librosa.stft(audio, n_fft=_CORR_N_FFT,
                                 win_length=_CORR_WIN_LENGTH,
                                 hop_length=_CORR_HOP_LENGTH)
        magnitude = np.abs(stft) * np.exp(correction_filter)[:, np.newaxis]
        # Retain original phase — preserves transient structure which
        # Griffin-Lim would smear by discarding phase information
        corrected_stft = magnitude * np.exp(1j * np.angle(stft))
        audio_out = librosa.istft(corrected_stft,
                                  hop_length=_CORR_HOP_LENGTH,
                                  win_length=_CORR_WIN_LENGTH)

        # Only normalise if output would clip PCM_16 (float range ±1.0)
        peak = np.max(np.abs(audio_out))
        if peak > 1.0:
            audio_out = audio_out / peak * 0.95

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_out, SAMPLE_RATE, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"  Error processing '{wav_path.name}': {e}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_spectrum_correction(parsed_segments: pd.DataFrame) -> None:
    # ── Step 1 + 2: estimate device profiles ─────────────────────────────
    print("\nEstimating device spectral profiles...")
    profiles = estimate_device_profiles(parsed_segments)

    # Save profiles to JSON for reference and future inference use
    DEVICE_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEVICE_PROFILES_PATH, 'w') as f:
        json.dump({k: v.tolist() for k, v in profiles.items()}, f, indent=2)
    print(f"  Profiles saved to '{DEVICE_PROFILES_PATH}'.")

    # ── Step 3: compute correction filters ───────────────────────────────
    print("\nComputing correction filters...")
    filters = compute_correction_filters(profiles)
    for device, filt in filters.items():
        print(f"  {device}: mean correction = {filt.mean():.4f} log units")

    # ── Step 4: apply corrections to all recordings ───────────────────────
    print("\nApplying corrections...")
    recordings = (
        parsed_segments[['wav_path', 'device']]
        .drop_duplicates('wav_path')
    )

    n_ok = n_fail = 0
    for _, row in tqdm(recordings.iterrows(), total=len(recordings),
                       desc="  Correcting"):
        wav_path = Path(row['wav_path'])
        out_path = Path(CORRECTED_DIR) / wav_path.name

        if not wav_path.exists():
            print(f"  Warning: '{wav_path.name}' not found. Skipping.")
            n_fail += 1
            continue

        # Skip if already done (allows resuming interrupted runs)
        if out_path.exists():
            n_ok += 1
            continue

        device = row['device']
        if device not in filters:
            print(f"  Warning: no filter for device '{device}'. Using identity.")
            filt = np.zeros(_CORR_N_FFT // 2 + 1, dtype=np.float32)
        else:
            filt = filters[device]

        if apply_and_save(wav_path, out_path, filt):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n  Done. {n_ok} corrected, {n_fail} failed.")
    print(f"  Output: '{CORRECTED_DIR}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 2: SPECTRUM CORRECTION")
    print("=" * 60)
    print(f"Parsed segments:  {PARSED_SEGMENTS_PATH}")
    print(f"Reference device: {REFERENCE_DEVICE}")
    print(f"Output dir:       {CORRECTED_DIR}")
    print(f"Sample rate:      {SAMPLE_RATE} Hz")
    print("=" * 60)

    if not Path(PARSED_SEGMENTS_PATH).exists():
        raise FileNotFoundError(
            f"Parsed segments not found: '{PARSED_SEGMENTS_PATH}'. Run step 01 first."
        )

    parsed_segments = pd.read_csv(PARSED_SEGMENTS_PATH)
    n_recordings = parsed_segments['wav_path'].nunique()
    print(f"\n  {n_recordings} unique recordings in parsed_segments.csv.")
    print(f"  Devices: {sorted(parsed_segments['device'].unique())}")

    run_spectrum_correction(parsed_segments)


if __name__ == '__main__':
    main()
