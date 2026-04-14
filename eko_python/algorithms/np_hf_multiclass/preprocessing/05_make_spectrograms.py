"""
05_make_spectrograms.py

Converts every wav in windows_manifest.csv to a 224×224 greyscale PNG
log-mel spectrogram.

For each row:
    1. Load the wav file from wav_path.
    2. Compute an STFT magnitude spectrogram.
       n_fft=1024 / win=64 (16 ms) / hop=32 (8 ms) → 251 time frames from a 2 s window.
    3. Apply a VTLP-warped mel filterbank (N_MELS=224, FMIN=50, FMAX=2000 Hz).
    4. Take log to get the log-mel spectrogram (224×501).
    5. Resize to 224×224 — slight downsample on time axis, upsample on freq axis.
    6. Scale per-sample to [0, 255] uint8 and save as a greyscale PNG.

VTLP warp:
    Augmented training copies (aug_index >= 1) get a random frequency warp
    sampled from [VTLP_ALPHA_MIN, VTLP_ALPHA_MAX] × [VTLP_FHI_MIN, VTLP_FHI_MAX].
    Originals and all test windows use the identity warp (alpha=1.0), so the
    standard mel filterbank is applied.
    The sampled alpha and fhi are written back into the manifest.

DAS/CAS labels are carried through unchanged from the manifest.

Input:  windows_manifest.csv (from step 04), wavs in WINDOWS_DIR / AUGMENTED_DIR
Output: greyscale PNGs in SPECTROGRAMS_DIR/{split}/{stem}.png
        windows_manifest.csv updated with spec_path, vtlp_alpha, vtlp_fhi columns
"""

import sys
import multiprocessing
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    WINDOWS_MANIFEST_PATH,
    SAMPLE_RATE,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    VTLP_ALPHA_MIN,
    VTLP_ALPHA_MAX,
    VTLP_FHI_MIN,
    VTLP_FHI_MAX,
    MODEL_INPUT_SIZE,
    SPECTROGRAMS_DIR,
)


# ---------------------------------------------------------------------------
# VTLP-warped mel filterbank
# ---------------------------------------------------------------------------

def get_vtlp_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int,
    fmin: float,
    fmax: float,
    alpha: float,
    fhi: float,
    base_filterbank: np.ndarray,
    freq_bins: np.ndarray,
) -> np.ndarray:
    # Piecewise linear frequency warp — continuous at fhi:
    #   f <= fhi : warped = alpha * f
    #   f >  fhi : warped = nyquist - (nyquist - alpha*fhi) * (nyquist - f) / (nyquist - fhi)
    # alpha=1.0 → identity (standard mel filterbank)
    nyquist = sr / 2.0
    warped = np.where(
        freq_bins <= fhi,
        freq_bins * alpha,
        nyquist - (nyquist - alpha * fhi) * (nyquist - freq_bins) / (nyquist - fhi),
    )
    warped = np.clip(warped, 0, nyquist)

    indices = np.searchsorted(freq_bins, warped)
    indices = np.clip(indices, 1, len(freq_bins) - 1)
    lo = freq_bins[indices - 1]
    hi = freq_bins[indices]
    t  = (warped - lo) / (hi - lo + 1e-10)

    return (1 - t) * base_filterbank[:, indices - 1] + t * base_filterbank[:, indices]


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------

def compute_logmel(audio: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    stft      = librosa.stft(audio, n_fft=N_FFT, win_length=WIN_LENGTH,
                             hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)                   # (N_FFT//2+1, T)
    mel       = filterbank @ magnitude          # (N_MELS, T)
    return np.log(np.maximum(mel, 1e-10))      # (N_MELS, T)


# ---------------------------------------------------------------------------
# Resize + quantise to uint8
# ---------------------------------------------------------------------------

def to_png_array(logmel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    img = Image.fromarray(logmel.astype(np.float32))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img)

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Worker (top-level so it can be pickled by multiprocessing on macOS spawn)
# ---------------------------------------------------------------------------

def _process_row(task: tuple) -> tuple:
    """
    Process one manifest row. All config values are passed explicitly so that
    monkeypatched test values (set in the main process before Pool creation)
    are honoured without relying on re-imported module globals in workers.

    Returns (idx, status, spec_path, vtlp_alpha, vtlp_fhi)
    where status is 'ok', 'skip', or 'fail'.
    """
    (idx, wav_path_str, split, aug_index, existing_spec_path,
     base_filterbank, freq_bins, target_h, target_w,
     spectrograms_dir_str,
     n_mels, n_fft, sample_rate, fmin, fmax,
     vtlp_alpha_min, vtlp_alpha_max, vtlp_fhi_min, vtlp_fhi_max) = task

    wav_path = Path(wav_path_str)
    out_path = Path(spectrograms_dir_str) / split / f"{wav_path.stem}.png"

    if out_path.exists() and existing_spec_path == str(out_path):
        return idx, 'skip', None, None, None

    if not wav_path.exists():
        print(f"  Warning: wav not found: '{wav_path.name}'. Skipping.")
        return idx, 'fail', None, None, None

    # VTLP: augmented train copies get a per-row deterministic random warp;
    # originals and test windows use identity (alpha=1.0).
    if int(aug_index) >= 1 and split == 'train':
        rng   = np.random.default_rng(idx)   # seeded by row index → reproducible
        alpha = float(rng.uniform(vtlp_alpha_min, vtlp_alpha_max))
        fhi   = float(rng.uniform(vtlp_fhi_min,   vtlp_fhi_max))
    else:
        alpha = 1.0
        fhi   = float(fmax)

    try:
        audio, file_sr = sf.read(str(wav_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if file_sr != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz, got {file_sr} Hz.")
    except Exception as e:
        print(f"  Error loading '{wav_path.name}': {e}")
        return idx, 'fail', None, None, None

    try:
        filterbank = (base_filterbank if alpha == 1.0 else
                      get_vtlp_filterbank(
                          n_mels=n_mels, n_fft=n_fft, sr=sample_rate,
                          fmin=fmin, fmax=fmax,
                          alpha=alpha, fhi=fhi,
                          base_filterbank=base_filterbank,
                          freq_bins=freq_bins,
                      ))
        logmel  = compute_logmel(audio, filterbank)
        png_arr = to_png_array(logmel, target_h, target_w)
    except Exception as e:
        print(f"  Error computing spectrogram for '{wav_path.name}': {e}")
        return idx, 'fail', None, None, None

    try:
        Image.fromarray(png_arr).save(str(out_path))
    except Exception as e:
        print(f"  Error saving PNG for '{wav_path.name}': {e}")
        return idx, 'fail', None, None, None

    return idx, 'ok', str(out_path), round(alpha, 4), round(fhi, 2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_make_spectrograms(
    manifest: pd.DataFrame,
    n_workers: int | None = None,
) -> pd.DataFrame:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    target_h, target_w = MODEL_INPUT_SIZE

    for col in ('spec_path', 'vtlp_alpha', 'vtlp_fhi'):
        if col not in manifest.columns:
            manifest[col] = np.nan if col != 'spec_path' else ''

    # Pre-compute base (un-warped) mel filterbank once in the main process;
    # passed to every worker to avoid redundant computation.
    base_filterbank = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    (SPECTROGRAMS_DIR / 'train').mkdir(parents=True, exist_ok=True)
    (SPECTROGRAMS_DIR / 'test').mkdir(parents=True, exist_ok=True)

    valid = manifest[manifest['wav_path'].notna() & (manifest['wav_path'] != '')]
    print(f"  {len(valid)} rows to process using {n_workers} workers.")

    # Build task tuples — all config values captured from module globals here
    # in the main process so monkeypatched values are correctly propagated.
    tasks = [
        (
            idx,
            row['wav_path'],
            row['split'],
            row.get('aug_index', 0),
            str(row.get('spec_path', '')),
            base_filterbank,
            freq_bins,
            target_h,
            target_w,
            str(SPECTROGRAMS_DIR),
            N_MELS, N_FFT, SAMPLE_RATE, FMIN, FMAX,
            VTLP_ALPHA_MIN, VTLP_ALPHA_MAX, VTLP_FHI_MIN, VTLP_FHI_MAX,
        )
        for idx, row in valid.iterrows()
    ]

    n_ok = n_fail = n_skip = 0

    with multiprocessing.Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_row, tasks),
            total=len(tasks),
            desc="  Spectrograms",
        ):
            idx, status, spec_path, alpha, fhi = result
            if status == 'ok':
                manifest.at[idx, 'spec_path']  = spec_path
                manifest.at[idx, 'vtlp_alpha'] = alpha
                manifest.at[idx, 'vtlp_fhi']   = fhi
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            else:
                n_fail += 1

    print(f"\n  Done. {n_ok} saved, {n_skip} already existed, {n_fail} failed.")
    print(f"  Output: '{SPECTROGRAMS_DIR}'")
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 5: MAKE SPECTROGRAMS")
    print("=" * 60)
    print(f"Sample rate:    {SAMPLE_RATE} Hz")
    print(f"N_FFT:          {N_FFT}  (win_length={WIN_LENGTH}, hop={HOP_LENGTH})")
    print(f"Time frames:    {(SAMPLE_RATE - N_FFT) // HOP_LENGTH + 1} per 1 s window")
    print(f"Mel bins:       {N_MELS}  ({FMIN}–{FMAX} Hz)")
    print(f"VTLP range:     alpha=[{VTLP_ALPHA_MIN}, {VTLP_ALPHA_MAX}]  "
          f"fhi=[{VTLP_FHI_MIN}, {VTLP_FHI_MAX}] Hz")
    print(f"Output size:    {MODEL_INPUT_SIZE[0]}×{MODEL_INPUT_SIZE[1]}")
    print(f"Output dir:     {SPECTROGRAMS_DIR}")
    print(f"Workers:        {multiprocessing.cpu_count()} (physical cores: use --workers to override)")
    print("=" * 60)

    if not Path(WINDOWS_MANIFEST_PATH).exists():
        raise FileNotFoundError(
            f"Windows manifest not found: '{WINDOWS_MANIFEST_PATH}'. "
            "Run step 04 first."
        )

    manifest = pd.read_csv(WINDOWS_MANIFEST_PATH)
    n_train = (manifest['split'] == 'train').sum()
    n_test  = (manifest['split'] == 'test').sum()
    print(f"\n  {len(manifest)} manifest rows  ({n_train} train, {n_test} test).\n")

    manifest = run_make_spectrograms(manifest)
    manifest.to_csv(WINDOWS_MANIFEST_PATH, index=False)
    print(f"\nManifest updated: '{WINDOWS_MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
