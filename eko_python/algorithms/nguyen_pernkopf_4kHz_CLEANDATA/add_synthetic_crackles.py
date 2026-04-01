"""
add_synthetic_crackles.py

Creates a copy of the spectrogram dataset in data/spectrograms_synthetic/
with synthetic features added per label:

    crackle : 1–5 vertical lines (full height, 1px wide)
              — broadband transients
    wheeze  : 3–5 short horizontal lines (partial width, 1px tall)
              — narrowband sustained tones
    both    : both vertical and horizontal lines
    normal  : copied unchanged

Each line has a fixed random amplitude (intensity added to existing pixel
values, clipped to 255).

The original spectrograms in data/spectrograms/ are not modified.

After running, point SPECTROGRAMS_DIR in config.py to the synthetic folder
and run training/testing to verify the model can detect the added signal.

Usage:
    python add_synthetic_crackles.py
    python add_synthetic_crackles.py --seed 123
"""

import argparse
import shutil
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from config import MANIFEST_PATH, SPECTROGRAMS_DIR

OUTPUT_DIR      = SPECTROGRAMS_DIR.parent / 'spectrograms_synthetic'
MIN_AMPLITUDE   = 60    # intensity units added to pixel values
MAX_AMPLITUDE   = 150

# Vertical lines (crackle — broadband transient)
MIN_V_LINES = 1
MAX_V_LINES = 5

# Horizontal lines (wheeze — narrowband sustained tone)
MIN_H_LINES    = 3
MAX_H_LINES    = 5
MIN_H_LINE_LEN = 20   # pixels (out of 224)
MAX_H_LINE_LEN = 80


def add_vertical_lines(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """1–5 full-height vertical lines, 1px wide."""
    W = arr.shape[1]
    n = rng.integers(MIN_V_LINES, MAX_V_LINES + 1)
    cols = rng.integers(0, W, size=n)
    amps = rng.integers(MIN_AMPLITUDE, MAX_AMPLITUDE + 1, size=n)
    for col, amp in zip(cols, amps):
        arr[:, col] = np.clip(arr[:, col].astype(np.int32) + amp, 0, 255).astype(np.uint8)
    return arr


def add_horizontal_lines(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """3–5 short horizontal lines, 1px tall, random position and length."""
    H, W = arr.shape
    n    = rng.integers(MIN_H_LINES, MAX_H_LINES + 1)
    rows = rng.integers(0, H, size=n)
    amps = rng.integers(MIN_AMPLITUDE, MAX_AMPLITUDE + 1, size=n)
    lens = rng.integers(MIN_H_LINE_LEN, MAX_H_LINE_LEN + 1, size=n)
    for row, amp, length in zip(rows, amps, lens):
        col_start = rng.integers(0, W - length + 1)
        col_end   = col_start + length
        arr[row, col_start:col_end] = np.clip(
            arr[row, col_start:col_end].astype(np.int32) + amp, 0, 255
        ).astype(np.uint8)
    return arr


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest['spec_path'].notna() & (manifest['spec_path'] != '')].copy()

    # Reconstruct spec path the same way dataset.py does
    manifest['_spec_path'] = manifest.apply(
        lambda r: SPECTROGRAMS_DIR / r['split'] / Path(r['spec_path']).name,
        axis=1,
    )

    print("=" * 60)
    print("ADD SYNTHETIC CRACKLES")
    print("=" * 60)
    print(f"  Source:      {SPECTROGRAMS_DIR}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Seed:        {seed}")
    print(f"  Vertical lines (crackle):   {MIN_V_LINES}–{MAX_V_LINES}")
    print(f"  Amplitude:   {MIN_AMPLITUDE}–{MAX_AMPLITUDE} intensity units")
    counts = manifest['label'].value_counts()
    print(f"  crackle (vertical lines):            {counts.get('crackle', 0)}")
    print(f"  wheeze  (horizontal lines):          {counts.get('wheeze', 0)}")
    print(f"  both    (vertical + horizontal):     {counts.get('both', 0)}")
    print(f"  normal  (unchanged):                 {counts.get('normal', 0)}")
    print("=" * 60)

    # Create output subdirs
    for split in ('train', 'test'):
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
        src_path = row['_spec_path']
        dst_path = OUTPUT_DIR / row['split'] / src_path.name

        if not src_path.exists():
            n_skip += 1
            continue

        label = row['label']
        if label in ('crackle', 'wheeze', 'both'):
            img = Image.open(src_path).convert('L')
            arr = np.array(img, dtype=np.uint8)
            if label in ('crackle', 'both'):
                arr = add_vertical_lines(arr, rng)
            if label in ('wheeze', 'both'):
                arr = add_horizontal_lines(arr, rng)
            Image.fromarray(arr, mode='L').save(dst_path)
        else:
            shutil.copy2(src_path, dst_path)

        n_ok += 1

    print(f"\nDone. {n_ok} spectrograms written, {n_skip} skipped (missing source).")
    print(f"Output: '{OUTPUT_DIR}'")
    print(f"\nTo test: set SPECTROGRAMS_DIR = BASE_DIR / 'data' / 'spectrograms_synthetic'")
    print(f"in config.py, then run training or inference as normal.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
