"""
02_spectrum_correction.py

Normalises the frequency response of each recording device so that all
recordings entering the training pipeline look spectrally similar.

Each ICBHI device (Meditron, AKGC417L, LittC2SE, Litt3200) has a different
frequency response — the same lung sound recorded through different
stethoscopes will have a different spectral shape. This step estimates the
average spectral profile of each device from training recordings only, then
applies a correction filter to every recording (train and test) so that they
all match the Meditron reference profile.

All audio is resampled to SAMPLE_RATE (4 kHz) before profiling and
correction, so the profiles capture frequency response in the 0–2 kHz band
only — the meaningful range for this pipeline.

Pipeline:
    1. Load each training recording, resample to 4 kHz, compute its mean
       log power spectrum by averaging across all STFT frames.
    2. Average these per-recording spectra across all training recordings
       for each device → one spectral profile per device.
    3. For each device compute a correction filter:
           filter = log_profile(Meditron) − log_profile(device)
       The reference device (Meditron) gets an identity filter (all zeros).
    4. Apply the correction to every recording (train + test):
           corrected_magnitude = magnitude × exp(filter)
       Reconstruct the time-domain signal via Griffin-Lim and save to
       CORRECTED_DIR as a 4 kHz wav.

Saves device_profiles.json to CORRECTED_DIR for reference and for use
during inference on new devices (e.g. Eko).

Input:  manifest.csv (from step 01), raw wav files in RAW_DATA_PATH
Output: corrected 4 kHz wav files in CORRECTED_DIR/{recording_id}.wav
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
    RAW_DATA_PATH,
    MANIFEST_PATH,
    SAMPLE_RATE,
    REFERENCE_DEVICE,
    CORRECTED_DIR,
    DEVICE_PROFILES_PATH,
)

# STFT parameters used only within this step for profiling and Griffin-Lim
# reconstruction. Kept separate from the spectrogram STFT params in config
# because reconstruction quality requires different constraints:
#   - no zero-padding (n_fft = win_length) so Griffin-Lim converges cleanly
#   - 75% overlap (hop = win_length // 4) for smooth phase reconstruction
# At 4 kHz: 64 ms window, 16 ms hop, 129 frequency bins (0–2 kHz, 15.6 Hz/bin)
_CORR_N_FFT      = 256
_CORR_WIN_LENGTH = 256
_CORR_HOP_LENGTH = 64


# ---------------------------------------------------------------------------
# Step 1: mean log power spectrum for one recording
# ---------------------------------------------------------------------------

def compute_mean_log_power_spectrum(wav_path: Path, sr: int) -> np.ndarray | None:
    #load the wav file, resample to target sr, compute mean log power across
    #all STFT frames → shape (N_FFT//2 + 1,)
    try:
        audio, orig_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=orig_sr, target_sr=sr)
        power = np.abs(librosa.stft(audio, n_fft=_CORR_N_FFT, win_length=_CORR_WIN_LENGTH, hop_length=_CORR_HOP_LENGTH)) ** 2
        power = np.maximum(power, 1e-10)  # avoid log(0)
        return np.mean(np.log(power), axis=1)
    except Exception as e:
        print(f"  Warning: could not process '{wav_path.name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Step 2: estimate one spectral profile per device (training data only)
# ---------------------------------------------------------------------------

def estimate_device_profiles(manifest: pd.DataFrame) -> dict[str, np.ndarray | None]:
    #use training recordings only so test recordings never influence the correction
    train = (
        manifest[manifest['split'] == 'train']
        [['recording_id', 'device']]
        .drop_duplicates('recording_id')
    )
    print(f"  Estimating profiles from {len(train)} training recordings...")

    spectra_by_device: dict[str, list[np.ndarray]] = {}

    for _, row in tqdm(train.iterrows(), total=len(train), desc="  Profiling"):
        wav_path = Path(RAW_DATA_PATH) / f"{row['recording_id']}.wav"
        if not wav_path.exists():
            print(f"  Warning: '{wav_path.name}' not found. Skipping.")
            continue
        spectrum = compute_mean_log_power_spectrum(wav_path, sr=SAMPLE_RATE)
        if spectrum is not None:
            spectra_by_device.setdefault(row['device'], []).append(spectrum)

    profiles = {}
    for device, spectra in spectra_by_device.items():
        profiles[device] = np.median(np.stack(spectra), axis=0) #taking median of power values to avoid pulling
        #towards bad recordings (i.e. the loud hum in some AKGC417L recordings)
        print(f"  {device}: profile from {len(spectra)} recordings.")

    return profiles


# ---------------------------------------------------------------------------
# Step 3: correction filter per device
# ---------------------------------------------------------------------------

def compute_correction_filters(profiles: dict[str, np.ndarray | None],) -> dict[str, np.ndarray]:
    #correction = log_profile(reference) − log_profile(device)
    #applying exp(correction) to linear magnitude shifts the device spectrum
    #to match the reference. the reference device gets zeros (identity).
    if REFERENCE_DEVICE not in profiles:
        raise ValueError(
            f"Reference device '{REFERENCE_DEVICE}' has no profile. "
            f"Check that it has training recordings."
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
    n_iter: int = 32,
) -> bool:
    #load → resample to 4kHz → apply correction in STFT magnitude domain
    #→ reconstruct via Griffin-Lim → save
    try:
        audio, orig_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != SAMPLE_RATE:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=orig_sr, target_sr=SAMPLE_RATE)

        stft      = librosa.stft(audio, n_fft=_CORR_N_FFT, win_length=_CORR_WIN_LENGTH, hop_length=_CORR_HOP_LENGTH)
        magnitude = np.abs(stft) * np.exp(correction_filter)[:, np.newaxis]
        # Reconstruct using the original phase — preserves transient structure
        # (crackle onsets etc.) which Griffin-Lim would smear by discarding phase
        corrected_stft = magnitude * np.exp(1j * np.angle(stft))
        audio_out = librosa.istft(corrected_stft, hop_length=_CORR_HOP_LENGTH, win_length=_CORR_WIN_LENGTH)

        # Only normalise if Griffin-Lim output would clip PCM_16 (float range ±1.0)
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

def run_spectrum_correction(manifest: pd.DataFrame) -> None:
    # ── Step 1 + 2: estimate device profiles ─────────────────────────────
    print("\nEstimating device spectral profiles...")
    profiles = estimate_device_profiles(manifest)

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
        manifest[['recording_id', 'device']]
        .drop_duplicates('recording_id')
    )

    n_ok = n_fail = 0
    for _, row in tqdm(recordings.iterrows(), total=len(recordings),
                       desc="  Correcting"):
        wav_path = Path(RAW_DATA_PATH) / f"{row['recording_id']}.wav"
        out_path = Path(CORRECTED_DIR) / f"{row['recording_id']}.wav"

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
    print(f"Raw data:         {RAW_DATA_PATH}")
    print(f"Reference device: {REFERENCE_DEVICE}")
    print(f"Output dir:       {CORRECTED_DIR}")
    print(f"Sample rate:      {SAMPLE_RATE} Hz")
    print("=" * 60)

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(f"Manifest not found: '{MANIFEST_PATH}'. Run step 01 first.")
    if not Path(RAW_DATA_PATH).exists():
        raise FileNotFoundError(f"Raw data not found: '{RAW_DATA_PATH}'.")

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\n  {manifest['recording_id'].nunique()} unique recordings in manifest.")

    run_spectrum_correction(manifest)


if __name__ == '__main__':
    main()
