"""
03_window_and_label.py

Slices each spectrum-corrected recording into 1-second windows (50% overlap)
and assigns DAS/CAS labels to each window based on the annotated event
timings in parsed_segments.csv.

Labelling rule:
    das = 1  if the total overlap between the window and all DAS events
               in that recording exceeds LABEL_OVERLAP_THRESHOLD seconds
    cas = 1  if the total overlap between the window and all CAS events
               in that recording exceeds LABEL_OVERLAP_THRESHOLD seconds
    Otherwise das = 0, cas = 0.

Windows that extend beyond the end of the recording are dropped (no padding).

Pipeline:
    1. For each unique recording in parsed_segments.csv, load the corrected
       wav from CORRECTED_DIR.
    2. Generate 1s windows with 0.5s hop across the full recording duration.
    3. For each window compute total DAS and CAS overlap using event timings
       from parsed_segments.csv.
    4. Apply the threshold to assign binary labels.
    5. Save each window as a 4 kHz wav to WINDOWS_DIR/{split}/.
    6. Write windows_manifest.csv with one row per saved window.

Input:  parsed_segments.csv (step 01), corrected wavs in CORRECTED_DIR (step 02)
Output: window wavs in WINDOWS_DIR/train/ and WINDOWS_DIR/test/
        windows_manifest.csv
"""

import sys
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    PARSED_SEGMENTS_PATH,
    SAMPLE_RATE,
    CORRECTED_DIR,
    WINDOW_DURATION,
    WINDOW_HOP,
    LABEL_OVERLAP_THRESHOLD,
    WINDOWS_DIR,
    WINDOWS_MANIFEST_PATH,
)

WINDOW_SAMPLES = int(WINDOW_DURATION * SAMPLE_RATE)
HOP_SAMPLES    = int(WINDOW_HOP * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def segment_overlap(win_start: float, win_end: float,
                    event_start: float, event_end: float) -> float:
    """Overlap in seconds between a window and a single event segment."""
    return max(0.0, min(win_end, event_end) - max(win_start, event_start))


def total_overlap(win_start: float, win_end: float,
                  events: pd.DataFrame, col: str) -> float:
    """
    Total overlap (seconds) between the window and all events where col == 1.
    Events are assumed non-overlapping with each other (separate annotations),
    so summing individual overlaps gives total labelled time in the window.
    """
    labelled = events[events[col] == 1]
    if labelled.empty:
        return 0.0
    overlaps = labelled.apply(
        lambda r: segment_overlap(win_start, win_end,
                                  r['event_start'], r['event_end']),
        axis=1,
    )
    return float(overlaps.sum())


# ---------------------------------------------------------------------------
# Window one recording
# ---------------------------------------------------------------------------

def window_recording(
    corrected_wav: Path,
    events: pd.DataFrame,
    out_dir: Path,
    split: str,
    device: str,
) -> list[dict]:
    """
    Generate all 1s windows for one recording, label them, save wavs.

    Returns a list of row dicts for the windows manifest.
    """
    try:
        audio, file_sr = sf.read(str(corrected_wav), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {file_sr} Hz. "
                             "Run step 02 first.")
    except Exception as e:
        print(f"  Error loading '{corrected_wav.name}': {e}")
        return []

    total_samples = len(audio)
    recording_id  = corrected_wav.stem
    rows = []

    win_idx = 0
    sample_start = 0

    while sample_start + WINDOW_SAMPLES <= total_samples:
        sample_end = sample_start + WINDOW_SAMPLES

        win_start_s = sample_start / SAMPLE_RATE
        win_end_s   = sample_end   / SAMPLE_RATE

        # Assign labels based on total labelled overlap within window
        das_overlap = total_overlap(win_start_s, win_end_s, events, 'das')
        cas_overlap = total_overlap(win_start_s, win_end_s, events, 'cas')

        das_label = int(das_overlap >= LABEL_OVERLAP_THRESHOLD)
        cas_label = int(cas_overlap >= LABEL_OVERLAP_THRESHOLD)

        # Save window wav
        filename = f"{recording_id}_w{win_idx:04d}.wav"
        out_path = out_dir / filename
        window_audio = audio[sample_start:sample_end].astype(np.float32)
        sf.write(str(out_path), window_audio, SAMPLE_RATE, subtype='PCM_16')

        rows.append({
            'wav_path':     str(out_path),
            'recording_id': recording_id,
            'device':       device,
            'split':        split,
            'window_start': round(win_start_s, 4),
            'window_end':   round(win_end_s,   4),
            'das':          das_label,
            'cas':          cas_label,
            'das_overlap':  round(das_overlap, 4),
            'cas_overlap':  round(cas_overlap, 4),
        })

        sample_start += HOP_SAMPLES
        win_idx += 1

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_windowing(parsed_segments: pd.DataFrame) -> pd.DataFrame:
    train_dir = WINDOWS_DIR / 'train'
    test_dir  = WINDOWS_DIR / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # One row per unique recording with its split and device
    recordings = (
        parsed_segments[['wav_path', 'split', 'device']]
        .drop_duplicates('wav_path')
    )

    all_rows = []
    n_recordings_ok = n_recordings_fail = 0

    for _, rec in tqdm(recordings.iterrows(), total=len(recordings),
                       desc="  Windowing"):
        src_wav_path  = Path(rec['wav_path'])
        corrected_wav = Path(CORRECTED_DIR) / src_wav_path.name

        if not corrected_wav.exists():
            print(f"  Warning: corrected wav not found for '{src_wav_path.name}'. "
                  "Skipping. Run step 02 first.")
            n_recordings_fail += 1
            continue

        out_dir = train_dir if rec['split'] == 'train' else test_dir

        # All annotated events for this recording
        events = parsed_segments[parsed_segments['wav_path'] == rec['wav_path']]

        rows = window_recording(corrected_wav, events, out_dir,
                                split=rec['split'], device=rec['device'])
        if rows:
            all_rows.extend(rows)
            n_recordings_ok += 1
        else:
            n_recordings_fail += 1

    print(f"\n  Recordings processed: {n_recordings_ok} ok, "
          f"{n_recordings_fail} failed/skipped.")

    if not all_rows:
        raise RuntimeError("No windows were saved. Check corrected wavs exist.")

    windows_manifest = pd.DataFrame(all_rows)
    return windows_manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3: WINDOW AND LABEL")
    print("=" * 60)
    print(f"Corrected wavs:     {CORRECTED_DIR}")
    print(f"Windows output:     {WINDOWS_DIR}")
    print(f"Window duration:    {WINDOW_DURATION}s  ({WINDOW_SAMPLES} samples)")
    print(f"Hop:                {WINDOW_HOP}s  ({HOP_SAMPLES} samples, 50% overlap)")
    print(f"Label threshold:    {LABEL_OVERLAP_THRESHOLD}s overlap")
    print(f"Sample rate:        {SAMPLE_RATE} Hz")
    print("=" * 60)

    if not Path(PARSED_SEGMENTS_PATH).exists():
        raise FileNotFoundError(
            f"Parsed segments not found: '{PARSED_SEGMENTS_PATH}'. Run step 01 first."
        )
    if not Path(CORRECTED_DIR).exists():
        raise FileNotFoundError(
            f"Corrected dir not found: '{CORRECTED_DIR}'. Run step 02 first."
        )

    parsed_segments = pd.read_csv(PARSED_SEGMENTS_PATH)
    n_recordings = parsed_segments['wav_path'].nunique()
    print(f"\n  {n_recordings} unique recordings in parsed_segments.csv.\n")

    windows_manifest = run_windowing(parsed_segments)

    windows_manifest.to_csv(WINDOWS_MANIFEST_PATH, index=False)
    print(f"\nWindows manifest saved: '{WINDOWS_MANIFEST_PATH}'")

    print("\n" + "=" * 60)
    print("WINDOWING SUMMARY")
    print("=" * 60)
    print(f"Total windows: {len(windows_manifest)}")
    print(f"\nBy split:\n{windows_manifest.groupby('split').size().to_string()}")
    print(f"\nBy device:\n{windows_manifest.groupby('device').size().to_string()}")
    print(f"\nDAS label distribution:\n"
          f"{windows_manifest.groupby('das').size().to_string()}")
    print(f"\nCAS label distribution:\n"
          f"{windows_manifest.groupby('cas').size().to_string()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
