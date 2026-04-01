"""
03_split_and_pad.py

Splits each spectrum-corrected recording into individual respiratory cycles
and reflect-pads each cycle to CYCLE_DURATION seconds.

Reads cycle timings from manifest.csv (produced by step 01). Cycles longer
than CYCLE_DURATION were already removed in step 01, so every cycle here
needs only padding, never truncation.

For each cycle:
    1. Load the corrected 4 kHz wav from CORRECTED_DIR.
    2. Slice the audio between cycle_start and cycle_end (converted to samples).
    3. Reflect-pad to exactly CYCLE_DURATION × SAMPLE_RATE samples.
    4. Save to CYCLES_DIR/{split}/{recording_id}_cycle{index:04d}.wav

Updates manifest.csv with a wav_path column pointing to each saved cycle.

Input:  manifest.csv, corrected wavs in CORRECTED_DIR
Output: cycle wavs in CYCLES_DIR/train/ and CYCLES_DIR/test/
        manifest.csv updated with wav_path column
"""

import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MANIFEST_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    CORRECTED_DIR,
    CYCLES_DIR,
)

TARGET_SAMPLES = int(SAMPLE_RATE * CYCLE_DURATION)


# ---------------------------------------------------------------------------
# Reflect padding
# ---------------------------------------------------------------------------

def reflect_pad(audio: np.ndarray, target_length: int) -> np.ndarray:
    # Iterative reflect padding — numpy's reflect mode requires the pad amount
    # to be less than the signal length, so pad in passes if the cycle is very short
    while len(audio) < target_length:
        pad = min(target_length - len(audio), len(audio) - 1)
        audio = np.pad(audio, (0, pad), mode='reflect')
    return audio[:target_length]


# ---------------------------------------------------------------------------
# Split and pad one recording
# ---------------------------------------------------------------------------

def split_recording(
    corrected_wav: Path,
    cycles: pd.DataFrame,
    out_dir: Path,
) -> dict[int, str]:
    """
    Extract and pad all cycles from one corrected recording.

    Returns a dict mapping cycle_index → saved wav path (as string).
    Cycles that cannot be extracted are omitted from the dict.
    """
    try:
        audio, file_sr = sf.read(str(corrected_wav), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {file_sr} Hz. "
                             f"Run step 02 first.")
    except Exception as e:
        print(f"  Error loading '{corrected_wav.name}': {e}")
        return {}

    total_samples = len(audio)
    saved = {}

    for _, row in cycles.iterrows():
        start = int(row['cycle_start'] * SAMPLE_RATE)
        end   = int(row['cycle_end']   * SAMPLE_RATE)

        # Guard against annotation timings that extend past the recording end
        if start >= total_samples:
            print(f"  Warning: cycle {row['cycle_index']} of "
                  f"'{corrected_wav.stem}' starts beyond recording end. Skipping.")
            continue
        end = min(end, total_samples)

        cycle_audio = audio[start:end].astype(np.float32)

        if len(cycle_audio) < 2:
            print(f"  Warning: cycle {row['cycle_index']} of "
                  f"'{corrected_wav.stem}' is too short to pad. Skipping.")
            continue

        padded = reflect_pad(cycle_audio, TARGET_SAMPLES)

        filename = f"{corrected_wav.stem}_cycle{int(row['cycle_index']):04d}.wav"
        out_path = out_dir / filename
        sf.write(str(out_path), padded, SAMPLE_RATE, subtype='PCM_16')

        saved[int(row['cycle_index'])] = str(out_path)

    return saved


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_split_and_pad(manifest: pd.DataFrame) -> pd.DataFrame:
    train_dir = CYCLES_DIR / 'train'
    test_dir  = CYCLES_DIR / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Add wav_path column if not present
    if 'wav_path' not in manifest.columns:
        manifest['wav_path'] = ''

    recordings = (
        manifest[['recording_id', 'split']]
        .drop_duplicates('recording_id')
    )

    n_saved = n_skipped = 0

    for _, rec in tqdm(recordings.iterrows(), total=len(recordings),
                       desc="  Splitting"):
        corrected_wav = Path(CORRECTED_DIR) / f"{rec['recording_id']}.wav"
        out_dir       = train_dir if rec['split'] == 'train' else test_dir

        if not corrected_wav.exists():
            print(f"  Warning: corrected wav not found for '{rec['recording_id']}'. "
                  f"Skipping. Run step 02 first.")
            n_skipped += len(manifest[manifest['recording_id'] == rec['recording_id']])
            continue

        cycles = manifest[manifest['recording_id'] == rec['recording_id']]
        saved  = split_recording(corrected_wav, cycles, out_dir)

        for cycle_index, wav_path in saved.items():
            mask = (
                (manifest['recording_id'] == rec['recording_id']) &
                (manifest['cycle_index']  == cycle_index)
            )
            manifest.loc[mask, 'wav_path'] = wav_path
            n_saved += 1

        n_skipped += len(cycles) - len(saved)

    print(f"\n  Saved:   {n_saved} cycles")
    print(f"  Skipped: {n_skipped} cycles")
    print(f"  Output:  '{CYCLES_DIR}'")

    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3: SPLIT AND PAD CYCLES")
    print("=" * 60)
    print(f"Corrected wavs: {CORRECTED_DIR}")
    print(f"Cycles output:  {CYCLES_DIR}")
    print(f"Sample rate:    {SAMPLE_RATE} Hz")
    print(f"Cycle duration: {CYCLE_DURATION}s ({TARGET_SAMPLES} samples)")
    print("=" * 60)

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(f"Manifest not found: '{MANIFEST_PATH}'. Run step 01 first.")
    if not Path(CORRECTED_DIR).exists():
        raise FileNotFoundError(f"Corrected dir not found: '{CORRECTED_DIR}'. Run step 02 first.")

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\n  {len(manifest)} cycles across "
          f"{manifest['recording_id'].nunique()} recordings.\n")

    manifest = run_split_and_pad(manifest)

    # Remove cycles that could not be saved (no wav_path)
    n_before = len(manifest)
    manifest = manifest[manifest['wav_path'] != ''].reset_index(drop=True)
    n_dropped = n_before - len(manifest)
    if n_dropped:
        print(f"  Dropped {n_dropped} cycles with no output file.")

    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest updated: '{MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
