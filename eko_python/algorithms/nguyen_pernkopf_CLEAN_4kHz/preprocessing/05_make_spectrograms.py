"""
05_make_spectrograms.py

Converts every wav in the manifest to a 224×224 greyscale PNG spectrogram.

For each row in manifest.csv:
    1. Load the wav file from wav_path.
    2. Compute an STFT magnitude spectrogram (WIN_LENGTH=64, HOP_LENGTH=64,
       N_FFT=512 → 257 freq bins × 251 frames at 4 kHz for a 4 s cycle).
    3. Apply a VTLP-warped mel filterbank (N_MELS=128, FMIN=50, FMAX=2000 Hz).
    4. Take log to get the log-mel spectrogram.
    5. Resize to MODEL_INPUT_SIZE (224×224) — pure downsampling, no upsampling.
    6. Scale per-sample to [0, 255] uint8 and save as a greyscale PNG.

VTLP warp parameters:
    - Originals (aug_index == 0) and all test cycles use identity warp
      (alpha=1.0, fhi=FMAX), so the standard mel filterbank is applied.
    - Augmented copies (aug_index >= 1) get a random warp sampled from
      [VTLP_ALPHA_MIN, VTLP_ALPHA_MAX] × [VTLP_FHI_MIN, VTLP_FHI_MAX].
    The sampled alpha and fhi are written back into the manifest so training
    can inspect or re-use them.

PNG storage format:
    Greyscale uint8. Pixel value 0 = min log-mel energy, 255 = max.
    Per-sample min-max scaling is used, so relative amplitude across samples
    is not preserved in the PNG — only the spectral shape. The training
    dataloader applies per-sample mean/std normalisation at load time.
    Because both steps are per-sample, no information useful to the model
    is lost.

Skips rows with no wav_path. Skips rows whose PNG already exists (safe to
resume an interrupted run).

Input:  manifest.csv (from step 04), augmented wavs in AUGMENTED_DIR,
        original cycle wavs in CYCLES_DIR
Output: greyscale PNGs in SPECTROGRAMS_DIR/{split}/{stem}.png
        manifest.csv updated with spec_path, vtlp_alpha, vtlp_fhi columns
"""

import sys
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MANIFEST_PATH,
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

RNG = np.random.default_rng(seed=42)


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
    # Piecewise linear frequency warp:
    #   below fhi: freq * alpha
    #   above fhi: smooth continuation so the warp reaches the Nyquist
    # alpha=1.0, fhi=anything → identity (standard mel filterbank)
    nyquist = sr / 2.0
    # Piecewise linear VTLP warp — continuous at fhi:
    #   f <= fhi : warped = alpha * f           (linear stretch/compression)
    #   f >  fhi : warped = nyquist - (nyquist - alpha*fhi) * (nyquist - f) / (nyquist - fhi)
    # Both pieces agree at fhi (= alpha*fhi) and the upper piece maps nyquist→nyquist.
    warped = np.where(
        freq_bins <= fhi,
        freq_bins * alpha,
        nyquist - (nyquist - alpha * fhi) * (nyquist - freq_bins) / (nyquist - fhi),
    )
    warped = np.clip(warped, 0, nyquist)

    # Bilinear interpolation of the base filterbank at warped bin positions
    indices = np.searchsorted(freq_bins, warped)
    indices = np.clip(indices, 1, len(freq_bins) - 1)
    lo = freq_bins[indices - 1]
    hi = freq_bins[indices]
    t  = (warped - lo) / (hi - lo + 1e-10)

    return (1 - t) * base_filterbank[:, indices - 1] + t * base_filterbank[:, indices]


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------

def compute_logmel(
    audio: np.ndarray,
    filterbank: np.ndarray,
) -> np.ndarray:
    # STFT → magnitude → mel filterbank → log
    stft      = librosa.stft(audio, n_fft=N_FFT, win_length=WIN_LENGTH,
                             hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)                   # (N_FFT//2+1, T)
    mel       = filterbank @ magnitude          # (N_MELS, T)
    return np.log(np.maximum(mel, 1e-10))      # (N_MELS, T)


# ---------------------------------------------------------------------------
# Resize + quantise to uint8
# ---------------------------------------------------------------------------

def to_png_array(logmel: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # Resize (bilinear) — always downsampling from 128×251 to 224×224 requires
    # upsampling on the frequency axis (128→224). PIL handles both directions.
    img = Image.fromarray(logmel.astype(np.float32))
    img = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img)

    # Per-sample min-max to [0, 255]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_make_spectrograms(manifest: pd.DataFrame) -> pd.DataFrame:
    target_h, target_w = MODEL_INPUT_SIZE

    # Ensure output columns exist
    if 'spec_path'  not in manifest.columns:
        manifest['spec_path']  = ''
    if 'vtlp_alpha' not in manifest.columns:
        manifest['vtlp_alpha'] = np.nan
    if 'vtlp_fhi'   not in manifest.columns:
        manifest['vtlp_fhi']   = np.nan

    # Pre-compute the base (un-warped) mel filterbank — reused for every row
    base_filterbank = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

    # Output subdirectories
    (SPECTROGRAMS_DIR / 'train').mkdir(parents=True, exist_ok=True)
    (SPECTROGRAMS_DIR / 'test').mkdir(parents=True, exist_ok=True)

    valid = manifest[manifest['wav_path'].notna() & (manifest['wav_path'] != '')]
    print(f"  {len(valid)} rows with wav_path to process.")

    n_ok = n_fail = n_skip = 0

    for idx, row in tqdm(valid.iterrows(), total=len(valid), desc="  Spectrograms"):
        wav_path = Path(row['wav_path'])
        split    = row['split']
        stem     = wav_path.stem
        out_path = SPECTROGRAMS_DIR / split / f"{stem}.png"

        # Resume: skip if PNG already exists and spec_path is populated
        if out_path.exists() and str(row['spec_path']) == str(out_path):
            n_skip += 1
            continue

        if not wav_path.exists():
            print(f"  Warning: wav not found: '{wav_path.name}'. Skipping.")
            n_fail += 1
            continue

        # ── VTLP parameters ─────────────────────────────────────────────
        # Augmented training copies get a random warp; everything else gets
        # identity (alpha=1.0) so the filterbank is unchanged.
        aug_index = int(row.get('aug_index', 0))
        if aug_index >= 1 and split == 'train':
            alpha = float(RNG.uniform(VTLP_ALPHA_MIN, VTLP_ALPHA_MAX))
            fhi   = float(RNG.uniform(VTLP_FHI_MIN,   VTLP_FHI_MAX))
        else:
            alpha = 1.0
            fhi   = float(FMAX)

        # ── Load audio ───────────────────────────────────────────────────
        try:
            audio, file_sr = sf.read(str(wav_path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if file_sr != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {file_sr} Hz.")
        except Exception as e:
            print(f"  Error loading '{wav_path.name}': {e}")
            n_fail += 1
            continue

        # ── Spectrogram ──────────────────────────────────────────────────
        try:
            if alpha == 1.0:
                # Identity warp — use the pre-computed base filterbank directly
                filterbank = base_filterbank
            else:
                filterbank = get_vtlp_filterbank(
                    n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE,
                    fmin=FMIN, fmax=FMAX,
                    alpha=alpha, fhi=fhi,
                    base_filterbank=base_filterbank,
                    freq_bins=freq_bins,
                )

            logmel  = compute_logmel(audio, filterbank)
            png_arr = to_png_array(logmel, target_h, target_w)
        except Exception as e:
            print(f"  Error computing spectrogram for '{wav_path.name}': {e}")
            n_fail += 1
            continue

        # ── Save PNG ─────────────────────────────────────────────────────
        try:
            Image.fromarray(png_arr, mode='L').save(str(out_path))
        except Exception as e:
            print(f"  Error saving PNG for '{wav_path.name}': {e}")
            n_fail += 1
            continue

        # Write back to manifest
        manifest.at[idx, 'spec_path']  = str(out_path)
        manifest.at[idx, 'vtlp_alpha'] = round(alpha, 4)
        manifest.at[idx, 'vtlp_fhi']   = round(fhi,   2)
        n_ok += 1

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
    print(f"Mel bins:       {N_MELS}  ({FMIN}–{FMAX} Hz)")
    print(f"VTLP range:     alpha=[{VTLP_ALPHA_MIN}, {VTLP_ALPHA_MAX}]  "
          f"fhi=[{VTLP_FHI_MIN}, {VTLP_FHI_MAX}] Hz")
    print(f"Output size:    {MODEL_INPUT_SIZE[0]}×{MODEL_INPUT_SIZE[1]}")
    print(f"Output dir:     {SPECTROGRAMS_DIR}")
    print("=" * 60)

    if not Path(MANIFEST_PATH).exists():
        raise FileNotFoundError(
            f"Manifest not found: '{MANIFEST_PATH}'. Run step 04 first."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    n_train = (manifest['split'] == 'train').sum()
    n_test  = (manifest['split'] == 'test').sum()
    print(f"\n  {len(manifest)} manifest rows  ({n_train} train, {n_test} test).\n")

    manifest = run_make_spectrograms(manifest)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest updated: '{MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
