"""
spectrum_correction.py

Applies spectrum correction to normalise recordings across the four ICBHI
recording devices (AKGC417L, Meditron, LittC2SE, Litt3200).

Each device has a different frequency response curve, meaning the same lung
sound recorded through different stethoscopes will have different spectral
shapes. Spectrum correction estimates the average spectral profile of each
device from the training recordings only, then applies a correction filter
to each recording so that all devices appear to have the same response as
the reference device (Meditron, which has the most recordings).

This is a deterministic, offline step. It is applied to all recordings
(both train and test) using correction filters derived from train only,
and the corrected wav files are saved to disk. Everything downstream
(cycle splitting, padding, augmentation) operates on corrected audio.

Method:
    1. For each training recording, compute the mean log power spectrum
       by averaging across all STFT frames.
    2. Average these per-recording spectra across all training recordings
       for each device → device spectral profile.
    3. For each device, compute a correction filter:
           correction = profile_reference - profile_device
       (in log domain, so this is a multiplicative correction in linear)
    4. Apply the correction filter to every recording of that device by
       multiplying its STFT magnitude by the filter, then reconstruct
       the time-domain signal via Griffin-Lim.

References:
    Nguyen & Pernkopf (2022) "Lung Sound Classification Using Co-Tuning
    and Stochastic Normalization", IEEE TBME 69(9):2872-2882.

    Nguyen, Pernkopf & Kosmider (2020) "Acoustic Scene Classification
    for Mismatched Recording Devices Using Heated-Up Softmax and Spectrum
    Correction", ICASSP 2020.
"""

import json
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_DATA_PATH,
    MANIFEST_PATH,
    PREPARED_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    REFERENCE_DEVICE,
    SPECTRUM_CORRECTION_PROFILES_PATH,
    SPECTRUM_CORRECTED_PATH
)


# ---------------------------------------------------------------------------
# Step 1: Compute mean log power spectrum for a single recording
# ---------------------------------------------------------------------------

def compute_mean_log_power_spectrum(
    wav_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray | None:
    """
    Load a wav file and compute its mean log power spectrum, averaged
    across all STFT time frames.

    Parameters
    ----------
    wav_path    : path to wav file
    sr          : target sample rate for resampling
    n_fft       : FFT window size
    hop_length  : STFT hop length

    Returns
    -------
    1D numpy array of shape (n_fft // 2 + 1,) representing the mean
    log power at each frequency bin, or None if the file cannot be loaded.
    """
    try:
        audio, orig_sr = sf.read(wav_path, always_2d=False)

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        power = np.abs(stft) ** 2  # shape: (n_fft//2+1, T)

        # Avoid log(0) by flooring
        power = np.maximum(power, 1e-10)

        # Mean log power across time frames
        mean_log_power = np.mean(np.log(power), axis=1)  # shape: (n_fft//2+1,)

        return mean_log_power

    except Exception as e:
        print(f"  Warning: could not process '{wav_path}': {e}")
        return None


# ---------------------------------------------------------------------------
# Step 2: Estimate device spectral profiles from training data only
# ---------------------------------------------------------------------------

def estimate_device_profiles(
    manifest: pd.DataFrame,
    raw_data_path: str,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> dict[str, np.ndarray | None]:
    """
    Estimate the mean log power spectrum for each device, computed from
    training recordings only.

    Parameters
    ----------
    manifest      : master manifest DataFrame (output of parse_annotations.py)
    raw_data_path : path to raw ICBHI wav files
    sr            : target sample rate
    n_fft         : FFT window size
    hop_length    : STFT hop length

    Returns
    -------
    Dict mapping device name → mean log power spectrum (1D array),
    or None if no recordings were available for that device.
    """
    raw_data_path = Path(raw_data_path)

    train_recordings = (
        manifest[manifest['split'] == 'train']
        [['recording_id', 'device', 'native_sr']]
        .drop_duplicates('recording_id')
    )

    print(f"Reference device: {REFERENCE_DEVICE}")
    print(f"Estimating profiles from {len(train_recordings)} training recordings...\n")

    device_spectra: dict[str, list[np.ndarray]] = {}

    for _, row in tqdm(train_recordings.iterrows(),
                       total=len(train_recordings),
                       desc="Profiling devices"):

        recording_id = row['recording_id']
        device_key   = f"{row['device']}_{int(row['native_sr'])}"
        wav_path     = raw_data_path / f"{recording_id}.wav"

        if not wav_path.exists():
            print(f"  Warning: wav file not found: '{wav_path}'. Skipping.")
            continue

        if device_key not in device_spectra:
            device_spectra[device_key] = []

        spectrum = compute_mean_log_power_spectrum(
            str(wav_path), sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        if spectrum is not None:
            device_spectra[device_key].append(spectrum)

    # Average across recordings per device+SR group
    profiles = {}
    for device_key, spectra_list in device_spectra.items():
        if len(spectra_list) == 0:
            print(f"  Warning: no spectra collected for '{device_key}'. "
                  f"Correction will be identity (no change).")
            profiles[device_key] = None
        else:
            profiles[device_key] = np.mean(np.stack(spectra_list, axis=0), axis=0)
            print(f"  {device_key}: profile estimated from {len(spectra_list)} recordings.")

    return profiles


# ---------------------------------------------------------------------------
# Step 3: Compute per-device correction filters
# ---------------------------------------------------------------------------

def compute_correction_filters(
    profiles: dict[str, np.ndarray | None],
    reference_device: str,
) -> dict[str, np.ndarray | None]:
    """
    Compute a correction filter for each device relative to the reference.

    In the log domain:
        correction[device] = profile[reference] - profile[device]

    Applying this correction to a recording's log power spectrum shifts
    it to match the reference device's average spectral shape.

    Parameters
    ----------
    profiles         : dict of device → mean log power spectrum
    reference_device : device to treat as the target response

    Returns
    -------
    Dict mapping device → correction filter (1D array in log domain),
    or None if no profile was available for that device.
    """
    # Find the highest-SR variant of the reference device as the canonical target
    ref_candidates = {
        k: v for k, v in profiles.items()
        if k.startswith(f"{reference_device}_") and v is not None
    }
    if not ref_candidates:
        raise ValueError(
            f"Reference device '{reference_device}' has no estimated profile. "
            f"Check that the reference device has training recordings."
        )
    reference_key = max(ref_candidates, key=lambda k: int(k.split('_')[-1]))
    ref_profile   = profiles[reference_key]
    print(f"  Reference profile: '{reference_key}'")

    filters = {}

    for device_key, profile in profiles.items():
        if profile is None:
            print(f"  Warning: no profile for '{device_key}', using identity filter.")
            filters[device_key] = None
        else:
            filters[device_key] = ref_profile - profile

    # All variants of the reference device get identity correction —
    # they share the same physical transfer function regardless of SR
    for k in list(filters.keys()):
        if k.startswith(f"{reference_device}_"):
            filters[k] = np.zeros_like(ref_profile)

    return filters


# ---------------------------------------------------------------------------
# Step 4: Apply correction to a single recording and save
# ---------------------------------------------------------------------------

def apply_spectrum_correction(
    wav_path: str,
    out_path: str,
    correction_filter: np.ndarray | None,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_iter: int = 32,
) -> bool:
    """
    Apply a spectrum correction filter to a wav file and save the result.

    The correction is applied in the STFT magnitude domain:
        corrected_magnitude = magnitude * exp(correction_filter)
    (since correction_filter is in log domain, exp converts to linear scale)

    Phase is reconstructed using Griffin-Lim.

    Parameters
    ----------
    wav_path          : input wav file path
    out_path          : output path for corrected wav
    correction_filter : 1D log-domain correction array, or None for identity
    sr                : target sample rate
    n_fft             : FFT window size
    hop_length        : STFT hop length
    n_iter            : number of Griffin-Lim iterations for phase reconstruction

    Returns
    -------
    True on success, False on failure.
    """
    try:
        audio, orig_sr = sf.read(wav_path, always_2d=False)

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to target sample rate
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

        # If no correction (identity), just save the resampled audio
        if correction_filter is None:
            sf.write(out_path, audio, sr, subtype='PCM_16')
            return True

        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)  # shape: (n_fft//2+1, T)

        # Apply correction: multiply magnitude by exp(filter)
        # correction_filter shape: (n_fft//2+1,) → reshape for broadcasting
        linear_correction = np.exp(correction_filter).copy()
        # Cap correction at the device's own Nyquist: bins above orig_sr/2
        # contain no real signal, so applying a large gain there only amplifies
        # noise/artefacts from the resampling process.
        freq_res   = sr / n_fft                      # Hz per STFT bin
        cutoff_bin = int((orig_sr / 2) / freq_res)   # last valid bin for device
        if cutoff_bin < len(linear_correction):
            linear_correction[cutoff_bin:] = 1.0
        corrected_magnitude = magnitude * linear_correction[:, np.newaxis]

        # Reconstruct time-domain signal via Griffin-Lim
        corrected_audio = librosa.griffinlim(
            corrected_magnitude,
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=n_fft,
        )

        # Normalise to prevent clipping
        peak = np.max(np.abs(corrected_audio))
        if peak > 0:
            corrected_audio = corrected_audio / peak * 0.95

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, corrected_audio, sr, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"  Error processing '{wav_path}': {e}")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_spectrum_correction(
    manifest_path: str,
    raw_data_path: str,
    output_dir: str,
    sr: int,
    n_fft: int,
    hop_length: int,
    reference_device: str,
    profiles_save_path: str,
) -> None:
    """
    Full spectrum correction pipeline:
        1. Load manifest
        2. Estimate device profiles from training recordings only
        3. Compute correction filters
        4. Apply corrections to all recordings (train + test)
        5. Save corrected wav files to output_dir
        6. Save device profiles to disk for reproducibility

    Parameters
    ----------
    manifest_path       : path to manifest CSV
    raw_data_path       : path to raw ICBHI wav files
    output_dir          : directory to save corrected wav files
    sr                  : target sample rate
    n_fft               : FFT window size
    hop_length          : STFT hop length
    reference_device    : device to use as correction target
    profiles_save_path  : path to save estimated profiles as JSON
    """
    raw_data_path = Path(raw_data_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    manifest = pd.read_csv(manifest_path)
    print(f"  {len(manifest)} cycles across "
          f"{manifest['recording_id'].nunique()} recordings.\n")

    # ── Step 1: Estimate device profiles ────────────────────────────────
    print("="*60)
    print("STEP 1: Estimating device spectral profiles")
    print("="*60)
    profiles = estimate_device_profiles(
        manifest=manifest,
        raw_data_path=str(raw_data_path),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    profiles_serialisable = {
        k: v.tolist() if v is not None else None
        for k, v in profiles.items()
    }
    Path(profiles_save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(profiles_save_path, 'w') as f:
        json.dump(profiles_serialisable, f, indent=2)
    print(f"\nDevice profiles saved to '{profiles_save_path}'.")

    # ── Step 2: Compute correction filters ──────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Computing correction filters")
    print("="*60)
    filters = compute_correction_filters(profiles, reference_device)
    for device, filt in filters.items():
        if filt is not None:
            print(f"  {device}: filter computed "
                  f"(mean correction = {filt.mean():.4f} log units)")
        else:
            print(f"  {device}: identity filter (no profile available)")

    # ── Step 3: Apply corrections to all recordings ──────────────────────
    print("\n" + "="*60)
    print("STEP 3: Applying corrections to all recordings")
    print("="*60)

    recordings = (
        manifest[['recording_id', 'device', 'native_sr']]
        .drop_duplicates('recording_id')
    )

    success_count = 0
    fail_count    = 0

    for _, row in tqdm(recordings.iterrows(),
                       total=len(recordings),
                       desc="Correcting recordings"):

        recording_id = row['recording_id']
        device_key   = f"{row['device']}_{int(row['native_sr'])}"
        wav_path     = raw_data_path / f"{recording_id}.wav"
        out_path     = output_dir / f"{recording_id}.wav"

        if not wav_path.exists():
            print(f"  Warning: source file not found: '{wav_path}'. Skipping.")
            fail_count += 1
            continue

        # Skip if already processed (allows resuming interrupted runs)
        if out_path.exists():
            success_count += 1
            continue

        ok = apply_spectrum_correction(
            wav_path=str(wav_path),
            out_path=str(out_path),
            correction_filter=filters.get(device_key),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SPECTRUM CORRECTION COMPLETE")
    print("="*60)
    print(f"  Successfully corrected: {success_count} recordings")
    print(f"  Failed:                 {fail_count} recordings")
    print(f"  Output directory:       {output_dir}")
    print("="*60)

    if fail_count > 0:
        print(f"\nWarning: {fail_count} recordings could not be processed. "
              f"Check warnings above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("ICBHI SPECTRUM CORRECTION")
    print("="*60)
    print(f"Raw data:          {RAW_DATA_PATH}")
    print(f"Manifest:          {MANIFEST_PATH}")
    print(f"Output directory:  {SPECTRUM_CORRECTED_PATH}")
    print(f"Reference device:  {REFERENCE_DEVICE}")
    print(f"Sample rate:       {SAMPLE_RATE} Hz")
    print(f"N_FFT:             {N_FFT}")
    print(f"Hop length:        {HOP_LENGTH}")
    print("="*60 + "\n")

    for label, path in [
        ("Raw data directory", RAW_DATA_PATH),
        ("Manifest file",      MANIFEST_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: '{path}'")

    run_spectrum_correction(
        manifest_path=MANIFEST_PATH,
        raw_data_path=RAW_DATA_PATH,
        output_dir=SPECTRUM_CORRECTED_PATH,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        reference_device=REFERENCE_DEVICE,
        profiles_save_path=SPECTRUM_CORRECTION_PROFILES_PATH,
    )


if __name__ == '__main__':
    main()