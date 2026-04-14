"""
04_augment.py

Generates N augmented wav copies of every training window.

For each training window in windows_manifest.csv, N_AUGMENTATIONS new wav
files are produced by applying three audio-domain augmentations in sequence:
    1. Volume scaling  — multiplicative gain drawn from AUG_VOLUME_RANGE
    2. Gaussian noise  — additive noise at SNR drawn from AUG_NOISE_SNR_RANGE
    3. Speed change    — resampling rate drawn from AUG_SPEED_RANGE,
                         followed by truncation/padding back to WINDOW_SAMPLES

Pitch shift is not used: the phase vocoder smears transient detail (crackles).
Speed change via resampling achieves similar spectral variation without it.

DAS/CAS labels are inherited from the source window — augmentation does not
alter the label, only the audio content.

Test windows are not augmented — they keep their original wav from step 03.
Original training windows are retained as aug_index=0 in the manifest.
Augmented copies are aug_index=1..N_AUGMENTATIONS.

Input:  windows_manifest.csv (from step 03), window wavs in WINDOWS_DIR
Output: augmented wavs in AUGMENTED_DIR/
        windows_manifest.csv expanded with one row per augmented copy
"""

import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    WINDOWS_MANIFEST_PATH,
    SAMPLE_RATE,
    WINDOW_DURATION,
    AUGMENTED_DIR,
    N_AUGMENTATIONS,
    AUG_VOLUME_RANGE,
    AUG_NOISE_SNR_RANGE,
    AUG_SPEED_RANGE,
)

TARGET_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
RNG = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def apply_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    return audio * gain


def apply_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise       = RNG.normal(0.0, np.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def apply_speed(audio: np.ndarray, rate: float) -> np.ndarray:
    # Simulate speed change via a single resample — no phase vocoder.
    # rate > 1 → sped-up (fewer output samples, higher pitch)
    # rate < 1 → slowed-down (more output samples, lower pitch)
    return librosa.resample(audio, orig_sr=int(SAMPLE_RATE * rate),
                            target_sr=SAMPLE_RATE)


def fix_length(audio: np.ndarray, target: int) -> np.ndarray:
    # Truncate if too long; reflect-pad if too short (speed change causes
    # small length deviations — typically only a few samples)
    if len(audio) >= target:
        return audio[:target]
    while len(audio) < target:
        pad   = min(target - len(audio), len(audio) - 1)
        audio = np.pad(audio, (0, pad), mode='reflect')
    return audio[:target]


# ---------------------------------------------------------------------------
# Single augmented copy
# ---------------------------------------------------------------------------

def make_augmented_copy(
    audio: np.ndarray,
    volume: float,
    noise_snr: float,
    speed: float,
) -> np.ndarray:
    audio = apply_volume(audio, volume)
    audio = apply_noise(audio, noise_snr)
    audio = apply_speed(audio, speed)
    audio = fix_length(audio.astype(np.float32), TARGET_SAMPLES)
    return audio


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_augmentation(manifest: pd.DataFrame) -> pd.DataFrame:
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    # Drop rows from a previous run so re-running does not compound copies
    if 'aug_index' in manifest.columns:
        n_prev = (manifest['aug_index'] > 0).sum()
        if n_prev:
            print(f"  Dropping {n_prev} augmented rows from a previous run.")
        manifest = manifest[manifest['aug_index'] == 0].copy()

    manifest['aug_index']     = 0
    manifest['aug_volume']    = np.nan
    manifest['aug_noise_snr'] = np.nan
    manifest['aug_speed']     = np.nan

    train_windows = manifest[manifest['split'] == 'train']
    print(f"  {len(train_windows)} training windows → "
          f"{len(train_windows) * N_AUGMENTATIONS} augmented copies to generate.\n")

    new_rows = []
    n_saved = n_failed = 0

    for _, row in tqdm(train_windows.iterrows(), total=len(train_windows),
                       desc="  Augmenting"):
        src_path = Path(row['wav_path'])
        if not src_path.exists():
            print(f"  Warning: '{src_path.name}' not found. Skipping.")
            n_failed += N_AUGMENTATIONS
            continue

        try:
            audio, file_sr = sf.read(str(src_path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if file_sr != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {file_sr} Hz.")
        except Exception as e:
            print(f"  Error loading '{src_path.name}': {e}")
            n_failed += N_AUGMENTATIONS
            continue

        for k in range(1, N_AUGMENTATIONS + 1):
            volume    = float(RNG.uniform(*AUG_VOLUME_RANGE))
            noise_snr = float(RNG.uniform(*AUG_NOISE_SNR_RANGE))
            speed     = float(RNG.uniform(*AUG_SPEED_RANGE))

            try:
                aug_audio = make_augmented_copy(audio, volume, noise_snr, speed)
            except Exception as e:
                print(f"  Error augmenting '{src_path.name}' copy {k}: {e}")
                n_failed += 1
                continue

            peak = np.max(np.abs(aug_audio))
            if peak > 1.0:
                aug_audio = aug_audio / peak * 0.95

            out_name = f"{src_path.stem}_aug{k:02d}.wav"
            out_path = AUGMENTED_DIR / out_name
            sf.write(str(out_path), aug_audio, SAMPLE_RATE, subtype='PCM_16')

            new_row = row.to_dict()
            new_row['wav_path']      = str(out_path)
            new_row['aug_index']     = k
            new_row['aug_volume']    = round(volume,    4)
            new_row['aug_noise_snr'] = round(noise_snr, 4)
            new_row['aug_speed']     = round(speed,     4)
            new_rows.append(new_row)
            n_saved += 1

    manifest = pd.concat(
        [manifest, pd.DataFrame(new_rows)],
        ignore_index=True,
    )

    print(f"\n  Augmented copies saved: {n_saved}")
    print(f"  Failed:                 {n_failed}")
    print(f"  Total manifest rows:    {len(manifest)}")
    print(f"  Output: '{AUGMENTED_DIR}'")

    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 4: AUGMENTATION")
    print("=" * 60)
    print(f"Augmented output:  {AUGMENTED_DIR}")
    print(f"Copies per window: {N_AUGMENTATIONS}")
    print(f"Volume range:      {AUG_VOLUME_RANGE}")
    print(f"Noise SNR range:   {AUG_NOISE_SNR_RANGE} dB")
    print(f"Speed range:       {AUG_SPEED_RANGE}")
    print("=" * 60)

    if not Path(WINDOWS_MANIFEST_PATH).exists():
        raise FileNotFoundError(
            f"Windows manifest not found: '{WINDOWS_MANIFEST_PATH}'. "
            "Run step 03 first."
        )

    manifest = pd.read_csv(WINDOWS_MANIFEST_PATH)

    n_train = (manifest['split'] == 'train').sum()
    n_test  = (manifest['split'] == 'test').sum()
    print(f"\n  Train windows: {n_train}")
    print(f"  Test windows:  {n_test}\n")

    manifest = run_augmentation(manifest)
    manifest.to_csv(WINDOWS_MANIFEST_PATH, index=False)
    print(f"\nManifest updated: '{WINDOWS_MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
