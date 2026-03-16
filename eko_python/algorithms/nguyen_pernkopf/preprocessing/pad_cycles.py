"""
pad_cycles.py

Pads each extracted respiratory cycle to a fixed length using reflect padding.

Cycles in the ICBHI dataset vary considerably in length (roughly 0.2s to 16s).
To feed them into a neural network we need a fixed-length representation.
Rather than zero-padding (which introduces a discontinuity at the boundary)
we use reflect padding, which mirrors the signal back on itself to produce
a smooth, continuous waveform at the required length.

For cycles longer than the target length, the cycle is truncated.

Reflect padding example for a short signal [A B C D E] padded to length 9:
    [A B C D E D C B A]
    The signal reflects back from the right boundary.

If the cycle is much shorter than the target length, multiple reflections
are applied until the target length is reached (numpy handles this
automatically with mode='reflect').

Input:
    - manifest.csv with wav_path column populated (output of split_cycles.py)

Output:
    - cycle wav files overwritten in-place with padded versions
    - no change to manifest (wav_paths remain the same)
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
    SAMPLE_RATE,
    CYCLE_DURATION,
)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def pad_cycle(
    audio: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Pad or truncate a cycle to a fixed number of samples using reflect padding.

    Parameters
    ----------
    audio         : 1D numpy array containing the cycle audio
    target_length : desired length in samples

    Returns
    -------
    1D numpy array of exactly target_length samples.
    """
    current_length = len(audio)

    if current_length == target_length:
        return audio

    # Truncate if longer than target
    if current_length > target_length:
        return audio[:target_length]

    # Reflect pad to target length
    # numpy reflect mode requires the pad amount to be less than the signal
    # length, so we pad iteratively if the signal is very short
    pad_needed = target_length - current_length

    while len(audio) < target_length:
        # How much we can pad in this iteration without exceeding numpy's
        # reflect mode constraint (pad amount must be < signal length)
        max_pad = len(audio) - 1
        this_pad = min(pad_needed, max_pad)
        audio = np.pad(audio, (0, this_pad), mode='reflect')
        pad_needed = target_length - len(audio)

    return audio[:target_length]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pad_cycles(
    manifest_path: str,
    sr: int,
    cycle_duration: float,
) -> None:
    """
    Pad all cycle wav files in the manifest to a fixed length.

    Files are overwritten in-place — the wav_path in the manifest does not
    change. Cycles with missing or empty wav_path are skipped.

    Parameters
    ----------
    manifest_path  : path to manifest CSV
    sr             : sample rate of the cycle wav files
    cycle_duration : target duration in seconds
    """
    target_length = int(sr * cycle_duration)

    print("Loading manifest...")
    manifest = pd.read_csv(manifest_path)

    # Only process cycles that were successfully extracted
    valid = manifest[manifest['wav_path'].notna() & (manifest['wav_path'] != '')]
    print(f"  {len(valid)} cycles to pad (target length: "
          f"{target_length} samples = {cycle_duration}s).\n")

    success_count  = 0
    fail_count     = 0
    truncated      = 0

    for _, row in tqdm(valid.iterrows(),
                       total=len(valid),
                       desc="Padding cycles"):

        wav_path = row['wav_path']

        if not Path(wav_path).exists():
            print(f"\n  Warning: file not found: '{wav_path}'. Skipping.")
            fail_count += 1
            continue

        try:
            audio, file_sr = sf.read(wav_path, always_2d=False)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if file_sr != sr:
                print(f"\n  Warning: unexpected sample rate {file_sr} in "
                      f"'{wav_path}' (expected {sr}). Skipping.")
                fail_count += 1
                continue

            if len(audio) > target_length:
                truncated += 1

            padded = pad_cycle(audio, target_length)

            sf.write(wav_path, padded, sr, subtype='PCM_16')
            success_count += 1

        except Exception as e:
            print(f"\n  Error processing '{wav_path}': {e}")
            fail_count += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CYCLE PADDING COMPLETE")
    print("="*60)
    print(f"  Successfully padded: {success_count} cycles")
    print(f"  Truncated (too long): {truncated} cycles")
    print(f"  Failed:              {fail_count} cycles")
    print(f"  Target length:       {target_length} samples ({cycle_duration}s)")
    print("="*60)

    if truncated > 0:
        print(f"\nNote: {truncated} cycles exceeded {cycle_duration}s and were "
              f"truncated. Review CYCLE_DURATION in config if this seems high.")
    if fail_count > 0:
        print(f"\nWarning: {fail_count} cycles could not be padded. "
              f"Check warnings above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("ICBHI CYCLE PADDING")
    print("="*60)
    print(f"Manifest:       {MANIFEST_PATH}")
    print(f"Sample rate:    {SAMPLE_RATE} Hz")
    print(f"Target duration: {CYCLE_DURATION}s")
    print("="*60 + "\n")

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(f"Manifest file not found: '{MANIFEST_PATH}'")

    run_pad_cycles(
        manifest_path=MANIFEST_PATH,
        sr=SAMPLE_RATE,
        cycle_duration=CYCLE_DURATION,
    )


if __name__ == '__main__':
    main()