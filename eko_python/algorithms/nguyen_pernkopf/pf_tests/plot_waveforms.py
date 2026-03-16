"""
plot_waveforms.py

Plots the raw waveform of each Eko recording with colour-coded patches
showing the predicted class for each chunk.

Colour scheme (normal predictions are not highlighted):
    crackle → orange
    wheeze  → blue
    both    → red

Reads per-chunk predictions from pf_tests/results/inference_chunk_results.csv.
Run run_inference.py first to generate that file.

Usage:
    python pf_tests/plot_waveforms.py
"""

import sys
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SAMPLE_RATE, EKO_PROJECT_ROOT

INPUT_DIR         = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples' / 'preprocessed'
CHUNK_RESULTS_CSV = Path(__file__).parent / 'results' / 'inference_chunk_results.csv'
PLOTS_DIR         = Path(__file__).parent / 'results' / 'waveform_plots'

PATCH_COLORS = {
    'crackle': '#FF8C00',   # orange
    'wheeze':  '#1E90FF',   # blue
    'both':    '#DC143C',   # red
    # 'normal' → no patch
}
PATCH_ALPHA = 0.35


def load_chunk_results(csv_path: Path) -> dict[str, list[dict]]:
    """Return {filename: [chunk_row, ...]} sorted by chunk index."""
    by_file: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            row['chunk']        = int(row['chunk'])
            row['start_sample'] = int(row['start_sample'])
            row['end_sample']   = int(row['end_sample'])
            by_file[row['file']].append(row)
    for chunks in by_file.values():
        chunks.sort(key=lambda r: r['chunk'])
    return dict(by_file)


def plot_recording(fname: str, chunks: list[dict], audio: np.ndarray,
                   sr: int, out_path: Path) -> None:
    duration = len(audio) / sr
    t        = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(14, 3))

    # Waveform
    ax.plot(t, audio, color='#444444', linewidth=0.4, alpha=0.8)

    # Coloured patches for non-normal chunks
    for chunk in chunks:
        label = chunk['prediction']
        if label == 'normal':
            continue
        color      = PATCH_COLORS[label]
        t_start    = chunk['start_sample'] / sr
        t_end      = chunk['end_sample']   / sr
        ax.axvspan(t_start, t_end, color=color, alpha=PATCH_ALPHA, linewidth=0)

    # Legend
    legend_handles = [
        mpatches.Patch(color=c, alpha=PATCH_ALPHA + 0.2, label=lbl.capitalize())
        for lbl, c in PATCH_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.8)

    ax.set_title(fname, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(0, duration)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    if not CHUNK_RESULTS_CSV.exists():
        print(f"Chunk results not found: '{CHUNK_RESULTS_CSV}'")
        print("Run 'python pf_tests/run_inference.py' first.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_file = load_chunk_results(CHUNK_RESULTS_CSV)
    print(f"Found predictions for {len(by_file)} recording(s).")

    for fname, chunks in sorted(by_file.items()):
        wav_path = INPUT_DIR / fname
        if not wav_path.exists():
            print(f"  Skipping '{fname}' — wav not found.")
            continue

        audio, file_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SAMPLE_RATE:
            print(f"  Warning: '{fname}' is {file_sr} Hz, expected {SAMPLE_RATE} Hz.")

        out_path = PLOTS_DIR / f"{Path(fname).stem}.png"
        plot_recording(fname, chunks, audio.astype(np.float32), file_sr, out_path)

        pred_summary = ', '.join(
            f"chunk{c['chunk']}={c['prediction']}" for c in chunks
        )
        print(f"  {fname}  [{pred_summary}]  → {out_path.name}")

    print(f"\nPlots saved to '{PLOTS_DIR}'.")


if __name__ == '__main__':
    main()
