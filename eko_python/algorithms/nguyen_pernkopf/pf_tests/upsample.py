"""
upsample.py

Resamples all recordings in data/pf_samples/ to 16 kHz and saves them to
data/pf_samples/preprocessed/.

Also verifies each file is mono — warns if not (and mixes down to mono
before saving so downstream steps are unaffected).

Usage:
    python pf_tests/upsample.py
"""

import sys
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SAMPLE_RATE, EKO_PROJECT_ROOT

INPUT_DIR  = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples'
OUTPUT_DIR = INPUT_DIR / 'preprocessed'


def process_file(wav_path: Path, output_dir: Path, target_sr: int) -> dict:
    """
    Load a wav file, verify mono, resample if needed, and save.

    Returns a summary dict for reporting.
    """
    audio, orig_sr = sf.read(str(wav_path), always_2d=False)

    channels = audio.ndim if audio.ndim == 1 else audio.shape[1]
    is_mono  = audio.ndim == 1 or audio.shape[1] == 1

    if not is_mono:
        print(f"  WARNING: {wav_path.name} has {channels} channels — mixing down to mono.")
        audio = audio.mean(axis=1)

    resampled = orig_sr != target_sr
    if resampled:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    out_path = output_dir / wav_path.name
    sf.write(str(out_path), audio, target_sr, subtype='PCM_16')

    return {
        'file':       wav_path.name,
        'orig_sr':    orig_sr,
        'channels':   channels,
        'mono':       is_mono,
        'resampled':  resampled,
        'duration_s': len(audio) / target_sr,
    }


def main() -> None:
    wav_files = sorted(INPUT_DIR.glob('*.wav'))

    if not wav_files:
        print(f"No .wav files found in '{INPUT_DIR}'.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UPSAMPLE PF SAMPLES")
    print("=" * 60)
    print(f"Input:       {INPUT_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Target SR:   {SAMPLE_RATE} Hz")
    print(f"Files found: {len(wav_files)}")
    print("=" * 60)

    results = []
    for wav_path in wav_files:
        result = process_file(wav_path, OUTPUT_DIR, SAMPLE_RATE)
        results.append(result)
        status = []
        if not result['mono']:
            status.append('mixed to mono')
        if result['resampled']:
            status.append(f"{result['orig_sr']} Hz → {SAMPLE_RATE} Hz")
        else:
            status.append(f"already {SAMPLE_RATE} Hz")
        status_str = ', '.join(status)
        print(f"  {result['file']:<45} {result['duration_s']:5.1f}s  {status_str}")

    already_mono    = sum(r['mono']      for r in results)
    needed_resample = sum(r['resampled'] for r in results)

    print("=" * 60)
    print(f"Done. {len(results)} files saved to '{OUTPUT_DIR}'.")
    print(f"  Mono:      {already_mono}/{len(results)} were already mono")
    print(f"  Resampled: {needed_resample}/{len(results)} needed resampling")
    print("=" * 60)


if __name__ == '__main__':
    target_sr = SAMPLE_RATE
    main()
