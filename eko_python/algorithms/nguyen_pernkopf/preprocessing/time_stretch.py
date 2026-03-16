"""
time_stretch.py

Creates offline time-stretched copies of minority class cycles (wheeze and
both) to address class imbalance in the ICBHI ALSC task.

Time stretching changes the duration of a signal without changing its pitch,
using the phase vocoder technique. Each minority class cycle is stretched by
a randomly sampled rate from the configured range, producing one additional
copy per original cycle. The stretched copies are saved alongside the
originals and added to the manifest with is_stretched=True.

Only applied to the training set. Test cycles are never augmented.

The random augmentations (volume, noise, pitch, speed) are NOT applied here —
they are applied online during training to the stretched copies only. The
is_stretched flag in the manifest tells the DataLoader which samples to
apply these augmentations to.

Stretched cycles are saved at their natural post-stretch length. Padding
to the fixed length (CYCLE_DURATION seconds) is handled subsequently by
pad_cycles.py, which pads all cycles uniformly regardless of origin.

Input:
    - manifest.csv with wav_path populated (output of split_cycles.py)

Output:
    - stretched cycle wav files saved to prepared/train/
    - manifest.csv updated with new rows for stretched cycles
      (is_stretched=True)
"""

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MANIFEST_PATH,
    PREPARED_DIR,
    SAMPLE_RATE,
    CYCLE_DURATION,
    TIME_STRETCH_RATES,
    MINORITY_CLASSES,
)


# ---------------------------------------------------------------------------
# Time stretching
# ---------------------------------------------------------------------------

def stretch_cycle(
    audio: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Time-stretch a cycle using the phase vocoder.

    A rate < 1.0 slows the signal down (stretches it).
    A rate > 1.0 speeds the signal up (compresses it).

    Parameters
    ----------
    audio : 1D numpy array containing the cycle audio
    rate  : stretch rate (e.g. 0.9 slows by 10%, 1.1 speeds up by 10%)

    Returns
    -------
    Time-stretched 1D numpy array. Length will differ from input.
    """
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_time_stretch(
    manifest_path: str,
    output_dir: str,
    sr: int,
    stretch_rate_range: tuple[float, float],
    minority_classes: list[str],
    random_seed: int = 42,
) -> None:
    """
    Create time-stretched copies of all minority class training cycles.

    For each minority class cycle in the training set, samples a stretch
    rate uniformly from stretch_rate_range, applies time stretching, and
    saves the result at its natural post-stretch length. Padding to the
    fixed cycle length is handled by pad_cycles.py.

    Parameters
    ----------
    manifest_path      : path to manifest CSV
    output_dir         : root prepared directory (stretched files go in train/)
    sr                 : sample rate
    stretch_rate_range : (min_rate, max_rate) for uniform sampling
    minority_classes   : list of class labels to apply stretching to
    random_seed        : seed for reproducibility
    """
    rng       = np.random.default_rng(random_seed)
    train_dir = Path(output_dir) / 'train'

    print("Loading manifest...")
    manifest = pd.read_csv(manifest_path)

    # Only stretch training cycles from minority classes that were
    # successfully extracted
    to_stretch = manifest[
        (manifest['split'] == 'train') &
        (manifest['label'].isin(minority_classes)) &
        (manifest['wav_path'].notna()) &
        (manifest['wav_path'] != '') &
        (manifest['is_stretched'] == False)  # don't re-stretch already stretched copies
    ]

    print(f"Minority classes to stretch: {minority_classes}")
    print(f"Stretch rate range: {stretch_rate_range}")
    print(f"Cycles to stretch: {len(to_stretch)}\n")
    print("Class breakdown:")
    print(to_stretch.groupby('label').size().to_string())
    print()

    new_rows      = []
    success_count = 0
    fail_count    = 0

    for _, row in tqdm(to_stretch.iterrows(),
                       total=len(to_stretch),
                       desc="Stretching cycles"):

        wav_path     = row['wav_path']
        recording_id = row['recording_id']
        cycle_index  = row['cycle_index']

        if not Path(wav_path).exists():
            print(f"\n  Warning: file not found: '{wav_path}'. Skipping.")
            fail_count += 1
            continue

        # Sample a stretch rate
        rate = rng.uniform(stretch_rate_range[0], stretch_rate_range[1])

        try:
            audio, file_sr = sf.read(wav_path, always_2d=False)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if file_sr != sr:
                print(f"\n  Warning: unexpected sample rate {file_sr} in "
                      f"'{wav_path}' (expected {sr}). Skipping.")
                fail_count += 1
                continue

            # Apply time stretching — save at natural post-stretch length,
            # pad_cycles.py will pad to fixed length in the next step
            stretched = stretch_cycle(audio, rate=rate)

            # Save stretched cycle
            stretched_filename = (
                f"{recording_id}_cycle{cycle_index:04d}_stretched.wav"
            )
            out_path = train_dir / stretched_filename

            sf.write(str(out_path), stretched, sr, subtype='PCM_16')

            # Build new manifest row — copy all metadata from original,
            # update wav_path and is_stretched
            new_row = row.to_dict()
            new_row['wav_path']    = str(out_path)
            new_row['is_stretched'] = True
            new_rows.append(new_row)

            success_count += 1

        except Exception as e:
            print(f"\n  Error processing '{wav_path}': {e}")
            fail_count += 1

    # ── Append new rows to manifest ──────────────────────────────────────
    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        manifest = pd.concat([manifest, new_df], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TIME STRETCHING COMPLETE")
    print("="*60)
    print(f"  Successfully stretched: {success_count} cycles")
    print(f"  Failed:                 {fail_count} cycles")
    print(f"  New manifest rows:      {len(new_rows)}")
    print(f"  Manifest updated:       {manifest_path}")

    print(f"\nUpdated training set class distribution:")
    train_manifest = manifest[manifest['split'] == 'train']
    print(train_manifest.groupby(['label', 'is_stretched']).size().to_string())
    print("="*60)

    if fail_count > 0:
        print(f"\nWarning: {fail_count} cycles could not be stretched. "
              f"Check warnings above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("ICBHI TIME STRETCHING")
    print("="*60)
    print(f"Manifest:           {MANIFEST_PATH}")
    print(f"Output directory:   {PREPARED_DIR}")
    print(f"Sample rate:        {SAMPLE_RATE} Hz")
    print(f"Stretch rate range: {TIME_STRETCH_RATES}")
    print(f"Minority classes:   {MINORITY_CLASSES}")
    print("="*60 + "\n")

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(f"Manifest file not found: '{MANIFEST_PATH}'")

    run_time_stretch(
        manifest_path=MANIFEST_PATH,
        output_dir=PREPARED_DIR,
        sr=SAMPLE_RATE,
        stretch_rate_range=TIME_STRETCH_RATES,
        minority_classes=MINORITY_CLASSES,
    )


if __name__ == '__main__':
    main()