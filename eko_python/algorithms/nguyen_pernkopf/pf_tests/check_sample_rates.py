"""
check_sample_rates.py

Reports the sample rate of every raw ICBHI recording, grouped by device.
Uses sf.info() so no audio is loaded into memory.

Usage:
    python pf_tests/check_sample_rates.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MANIFEST_PATH, RAW_DATA_PATH


def main() -> None:
    # Load manifest — one row per cycle, deduplicate to one row per recording
    seen: set[str] = set()
    recordings: list[dict] = []
    with open(MANIFEST_PATH, newline='') as f:
        for row in csv.DictReader(f):
            rid = row['recording_id']
            if rid not in seen:
                seen.add(rid)
                recordings.append({'recording_id': rid, 'device': row['device']})

    print(f"Unique recordings in manifest: {len(recordings)}")

    # device → {sample_rate: count}
    device_rates: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    skipped = 0

    for rec in recordings:
        wav_path = Path(RAW_DATA_PATH) / f"{rec['recording_id']}.wav"
        try:
            info = sf.info(str(wav_path))
            device_rates[rec['device']][info.samplerate] += 1
        except Exception:
            skipped += 1

    if skipped:
        print(f"Skipped (unreadable/missing): {skipped}\n")

    # Print summary
    all_devices = sorted(device_rates.keys())
    for device in all_devices:
        rates = device_rates[device]
        total = sum(rates.values())
        print(f"{device}  ({total} recordings)")
        for sr in sorted(rates.keys()):
            print(f"    {sr:>6} Hz  —  {rates[sr]} recording(s)")
        print()


if __name__ == '__main__':
    main()
