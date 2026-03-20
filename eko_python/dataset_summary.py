import os
from pathlib import Path
from collections import defaultdict

import wave
import matplotlib.pyplot as plt
import numpy as np

AUDIO_DIR = Path(__file__).parent.parent / "data/ext_databases/ICBHI/audio_and_txt_files"


def get_wav_info(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate) for a wav file."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / rate, rate


def get_cycle_durations(txt_path: Path) -> list[float]:
    """Return list of respiration cycle durations (end - start) from a txt annotation file."""
    cycles = []
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                start, end = float(parts[0]), float(parts[1])
                cycles.append(end - start)
            except ValueError:
                continue
    return cycles


def main():
    wav_files = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")

    durations = []
    device_counts: dict[str, int] = defaultdict(int)
    device_rate_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    cycle_durations = []

    for path in wav_files:
        # Filename format: {patient}_{index}_{location}_{mode}_{device}.wav
        device = path.stem.rsplit("_", 1)[-1]
        duration, rate = get_wav_info(path)

        durations.append(duration)
        device_counts[device] += 1
        device_rate_counts[device][rate] += 1

        txt_path = path.with_suffix(".txt")
        if txt_path.exists():
            cycle_durations.extend(get_cycle_durations(txt_path))

    print(f"Found {len(cycle_durations)} respiration cycles across all recordings")

    # --- Histogram of durations ---
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].hist(durations, bins=30, edgecolor="black")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("WAV File Durations")

    # --- Device counts ---
    devices = sorted(device_counts)
    counts = [device_counts[d] for d in devices]
    axes[1].bar(devices, counts, edgecolor="black")
    axes[1].set_xlabel("Device")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Files per Device")
    for i, v in enumerate(counts):
        axes[1].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=9)

    # --- Device × sample rate counts (grouped bar) ---
    all_rates = sorted({rate for rates in device_rate_counts.values() for rate in rates})
    x = np.arange(len(devices))
    width = 0.8 / len(all_rates)

    for i, rate in enumerate(all_rates):
        bar_counts = [device_rate_counts[d].get(rate, 0) for d in devices]
        offset = (i - len(all_rates) / 2 + 0.5) * width
        bars = axes[2].bar(x + offset, bar_counts, width, label=f"{rate} Hz", edgecolor="black")
        for bar, val in zip(bars, bar_counts):
            if val > 0:
                axes[2].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha="center", va="bottom", fontsize=7,
                )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(devices)
    axes[2].set_xlabel("Device")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Files per Device × Sample Rate")
    axes[2].legend(title="Sample Rate")

    # --- Histogram of respiration cycle durations ---
    axes[3].hist(cycle_durations, bins=40, edgecolor="black")
    axes[3].set_xlabel("Cycle Duration (s)")
    axes[3].set_ylabel("Count")
    axes[3].set_title(f"Respiration Cycle Durations (n={len(cycle_durations)})")

    plt.tight_layout()
    plt.show()

    # Print summary to console
    print(f"\nWAV duration stats (s): min={min(durations):.1f}  max={max(durations):.1f}  "
          f"mean={np.mean(durations):.1f}  median={np.median(durations):.1f}")

    print(f"\nCycle duration stats (s): min={min(cycle_durations):.2f}  max={max(cycle_durations):.2f}  "
          f"mean={np.mean(cycle_durations):.2f}  median={np.median(cycle_durations):.2f}")

    print("\nFiles per device:")
    for d in devices:
        print(f"  {d}: {device_counts[d]}")

    print("\nFiles per device × sample rate:")
    for d in devices:
        for rate in sorted(device_rate_counts[d]):
            print(f"  {d} @ {rate} Hz: {device_rate_counts[d][rate]}")


if __name__ == "__main__":
    main()
