"""
split_cycles.py

Splits the spectrum-corrected full recordings into individual respiratory
cycles based on the timing annotations stored in the manifest.

Each cycle is extracted from the corrected wav file using the cycle_start
and cycle_end times from the manifest, then saved as an individual wav file.
The manifest is updated with the path to each saved cycle wav file.

Input:
    - manifest.csv (output of parse_annotations.py)
    - prepared/corrected/ (output of spectrum_correction.py)

Output:
    - prepared/train/ and prepared/test/ directories containing individual
      cycle wav files
    - manifest.csv updated with wav_path column populated

Naming convention for cycle wav files:
    {recording_id}_cycle{cycle_index:04d}.wav
    e.g. 101_1b1_Al_sc_Meditron_cycle0000.wav
"""

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MANIFEST_PATH,
    PREPARED_DIR,
    SAMPLE_RATE,
    SPECTRUM_CORRECTED_PATH
)


# ---------------------------------------------------------------------------
# Cycle extraction
# ---------------------------------------------------------------------------

def extract_cycle(
    audio: np.ndarray,
    sr: int,
    cycle_start: float,
    cycle_end: float,
) -> np.ndarray | None:
    """
    Extract a single respiratory cycle from a full recording.

    Parameters
    ----------
    audio       : full recording as 1D numpy array
    sr          : sample rate of the audio
    cycle_start : start time of the cycle in seconds
    cycle_end   : end time of the cycle in seconds

    Returns
    -------
    1D numpy array containing the cycle audio, or None if the timing
    falls outside the bounds of the recording.
    """
    start_sample = int(round(cycle_start * sr))
    end_sample   = int(round(cycle_end   * sr))

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample   = min(len(audio), end_sample)

    if end_sample <= start_sample:
        return None

    return audio[start_sample:end_sample]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_split_cycles(
    manifest_path: str,
    corrected_dir: str,
    output_dir: str,
    sr: int,
) -> None:
    """
    Split all corrected recordings into individual cycle wav files.

    For each row in the manifest, extracts the corresponding cycle from the
    corrected recording and saves it to the appropriate train or test directory.
    Updates the manifest with the wav_path for each cycle.

    Parameters
    ----------
    manifest_path : path to manifest CSV
    corrected_dir : directory containing spectrum-corrected full recordings
    output_dir    : root prepared directory (train/ and test/ subdirs created here)
    sr            : sample rate (used to convert times to sample indices)
    """
    corrected_dir = Path(corrected_dir)
    output_dir    = Path(output_dir)

    # Create output directories
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)

    print("Loading manifest...")
    manifest = pd.read_csv(manifest_path)
    print(f"  {len(manifest)} cycles across "
          f"{manifest['recording_id'].nunique()} recordings.\n")

    # Cache loaded recordings to avoid re-reading the same file for every cycle
    # (each recording contains multiple cycles)
    current_recording_id = None
    current_audio        = None

    success_count = 0
    fail_count    = 0
    wav_paths     = []

    for _, row in tqdm(manifest.iterrows(),
                       total=len(manifest),
                       desc="Splitting cycles"):

        recording_id = row['recording_id']
        cycle_index  = row['cycle_index']
        cycle_start  = row['cycle_start']
        cycle_end    = row['cycle_end']
        split        = row['split']

        # Build output path
        cycle_filename = f"{recording_id}_cycle{cycle_index:04d}.wav"
        out_path       = output_dir / split / cycle_filename

        # Skip if already processed
        if out_path.exists():
            wav_paths.append(str(out_path))
            success_count += 1
            continue

        # Load recording if not already cached
        if recording_id != current_recording_id:
            corrected_path = corrected_dir / f"{recording_id}.wav"

            if not corrected_path.exists():
                print(f"\n  Warning: corrected file not found: "
                      f"'{corrected_path}'. Skipping all cycles for this recording.")
                # Append empty paths for remaining cycles of this recording
                # (handled below by the None check)
                current_recording_id = recording_id
                current_audio        = None
            else:
                try:
                    audio, file_sr = sf.read(str(corrected_path), always_2d=False)

                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)

                    # Spectrum correction already resampled to target sr,
                    # but verify to be safe
                    if file_sr != sr:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

                    current_recording_id = recording_id
                    current_audio        = audio

                except Exception as e:
                    print(f"\n  Warning: could not load '{corrected_path}': {e}. "
                          f"Skipping.")
                    current_recording_id = recording_id
                    current_audio        = None

        # Skip cycle if recording failed to load
        if current_audio is None:
            wav_paths.append('')
            fail_count += 1
            continue

        # Extract cycle
        cycle_audio = extract_cycle(
            audio=current_audio,
            sr=sr,
            cycle_start=cycle_start,
            cycle_end=cycle_end,
        )

        if cycle_audio is None or len(cycle_audio) == 0:
            print(f"\n  Warning: empty cycle extracted for "
                  f"'{recording_id}' cycle {cycle_index} "
                  f"({cycle_start:.3f}s – {cycle_end:.3f}s). Skipping.")
            wav_paths.append('')
            fail_count += 1
            continue

        # Save cycle
        try:
            sf.write(str(out_path), cycle_audio, sr, subtype='PCM_16')
            wav_paths.append(str(out_path))
            success_count += 1
        except Exception as e:
            print(f"\n  Error saving '{out_path}': {e}")
            wav_paths.append('')
            fail_count += 1

    # ── Update manifest with wav_paths ───────────────────────────────────
    manifest['wav_path'] = wav_paths
    manifest.to_csv(manifest_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CYCLE SPLITTING COMPLETE")
    print("="*60)
    print(f"  Successfully extracted: {success_count} cycles")
    print(f"  Failed:                 {fail_count} cycles")
    print(f"  Output directory:       {output_dir}")
    print(f"  Manifest updated:       {manifest_path}")

    print(f"\nCycles by split:")
    split_counts = manifest[manifest['wav_path'] != ''].groupby('split').size()
    print(split_counts.to_string())

    print(f"\nCycle duration statistics (seconds):")
    print(manifest['duration'].describe().round(3).to_string())
    print("="*60)

    if fail_count > 0:
        print(f"\nWarning: {fail_count} cycles could not be extracted. "
              f"Check warnings above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    
    print("="*60)
    print("ICBHI CYCLE SPLITTING")
    print("="*60)
    print(f"Manifest:         {MANIFEST_PATH}")
    print(f"Corrected audio:  {SPECTRUM_CORRECTED_PATH}")
    print(f"Output directory: {PREPARED_DIR}")
    print(f"Sample rate:      {SAMPLE_RATE} Hz")
    print("="*60 + "\n")

    for label, path in [
        ("Manifest file",          MANIFEST_PATH),
        ("Corrected audio directory", SPECTRUM_CORRECTED_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: '{path}'")

    run_split_cycles(
        manifest_path=MANIFEST_PATH,
        corrected_dir=SPECTRUM_CORRECTED_PATH,
        output_dir=PREPARED_DIR,
        sr=SAMPLE_RATE,
    )


if __name__ == '__main__':
    main()