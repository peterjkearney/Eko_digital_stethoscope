"""
04_augment.py

Generates N augmented wav copies of every training cycle.

For each training cycle in the manifest, N_AUGMENTATIONS new wav files are
produced by applying three audio-domain augmentations in sequence:
    1. Volume scaling    — multiplicative gain drawn from AUG_VOLUME_RANGE
    2. Gaussian noise    — additive noise at SNR drawn from AUG_NOISE_SNR_RANGE
    3. Speed change      — resampling rate drawn from AUG_SPEED_RANGE,
                           followed by re-padding/truncation to CYCLE_DURATION

Pitch shift was removed: librosa.effects.pitch_shift uses the phase vocoder
internally, which smears transient detail (crackles). Speed change via
resampling achieves a similar spectral variation without any phase vocoder.

The exact parameters used for each copy are recorded in the manifest so every
augmented spectrogram (step 05) is fully traceable back to its source audio.

Test cycles are not augmented — they keep their original wav from step 03.

Original training cycles are retained as aug_index=0 in the manifest.
Augmented copies are aug_index=1..N_AUGMENTATIONS.

Input:  manifest.csv with wav_path column (from step 03)
Output: augmented wavs in AUGMENTED_DIR/
        manifest.csv expanded with one row per augmented copy
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
    MANIFEST_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    AUGMENTED_DIR,
    N_AUGMENTATIONS,
    AUG_VOLUME_RANGE,
    AUG_NOISE_SNR_RANGE,
    AUG_SPEED_RANGE,
)

TARGET_SAMPLES = int(SAMPLE_RATE * CYCLE_DURATION)
RNG = np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def apply_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    return audio * gain


def apply_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    # Compute signal power, derive noise power from target SNR, add noise
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = RNG.normal(0.0, np.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def apply_speed(audio: np.ndarray, rate: float) -> np.ndarray:
    # Simulate speed change via a single resample — no phase vocoder, transients preserved.
    # Interpret the audio as if it was recorded at sr*rate Hz and resample to sr.
    # rate > 1 → fewer output samples (sped-up, higher pitch);
    # rate < 1 → more output samples (slowed-down, lower pitch).
    # reflect_pad/truncation in make_augmented_copy restores TARGET_SAMPLES length.
    return librosa.resample(audio, orig_sr=int(SAMPLE_RATE * rate),
                            target_sr=SAMPLE_RATE)


def reflect_pad(audio: np.ndarray, target_length: int) -> np.ndarray:
    # Iterative reflect padding — same as step 03
    while len(audio) < target_length:
        pad   = min(target_length - len(audio), len(audio) - 1)
        audio = np.pad(audio, (0, pad), mode='reflect')
    return audio[:target_length]


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
    # Speed change alters length — re-pad or truncate back to CYCLE_DURATION
    audio = reflect_pad(audio.astype(np.float32), TARGET_SAMPLES)
    return audio


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_augmentation(manifest: pd.DataFrame) -> pd.DataFrame:
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    # Drop any rows from a previous run of this step (aug_index > 0) so that
    # re-running does not compound augmented copies on top of augmented copies.
    if 'aug_index' in manifest.columns:
        n_prev = (manifest['aug_index'] > 0).sum()
        if n_prev:
            print(f"  Dropping {n_prev} augmented rows from a previous run.")
        manifest = manifest[manifest['aug_index'] == 0].copy()

    # Mark all remaining rows as originals and initialise parameter columns
    manifest['aug_index']     = 0
    manifest['aug_volume']    = np.nan
    manifest['aug_noise_snr'] = np.nan
    manifest['aug_speed']     = np.nan

    train_cycles = manifest[manifest['split'] == 'train']
    print(f"  {len(train_cycles)} training cycles → "
          f"{len(train_cycles) * N_AUGMENTATIONS} augmented copies to generate.\n")

    new_rows = []
    n_saved = n_failed = 0

    for _, row in tqdm(train_cycles.iterrows(), total=len(train_cycles),
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

            # Only normalise if augmentation pushed audio above ±1.0
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
    print(f"Augmented output: {AUGMENTED_DIR}")
    print(f"Copies per cycle: {N_AUGMENTATIONS}")
    print(f"Volume range:     {AUG_VOLUME_RANGE}")
    print(f"Noise SNR range:  {AUG_NOISE_SNR_RANGE} dB")
    print(f"Speed range:      {AUG_SPEED_RANGE}")
    print("=" * 60)

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(
            f"Manifest not found: '{MANIFEST_PATH}'. Run step 03 first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    if 'wav_path' not in manifest.columns or manifest['wav_path'].eq('').all():
        raise ValueError("manifest.csv has no wav_path entries. Run step 03 first.")

    n_train = (manifest['split'] == 'train').sum()
    n_test  = (manifest['split'] == 'test').sum()
    print(f"\n  Train cycles: {n_train}")
    print(f"  Test cycles:  {n_test}\n")

    manifest = run_augmentation(manifest)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest updated: '{MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
