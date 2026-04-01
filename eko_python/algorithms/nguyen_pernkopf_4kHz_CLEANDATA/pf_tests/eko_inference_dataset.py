"""
eko_inference_dataset.py

Lightweight PyTorch Dataset for running inference on Eko device recordings.

Applies spectrum correction and VTLP log-mel spectrogram computation on-the-fly,
without requiring the offline preprocessing pipeline to have been run on these files.

Per-sample pipeline:
    1. Load chunk from wav file (resamples to SAMPLE_RATE=4kHz if needed)
    2. Reflect-pad to CYCLE_DURATION
    3. Compute STFT magnitude
    4. Apply spectrum correction: magnitude × exp(Meditron_profile − Eko_profile)
    5. Apply VTLP-warped mel filterbank (fixed alpha=1.0 for inference)
    6. Log + resize to MODEL_INPUT_SIZE + per-sample normalisation
    7. Return (3, H, W) tensor + sample index

Use dataset.samples[idx] to retrieve {'file', 'chunk', 'start', 'end'} metadata
for each index, enabling per-file aggregation of predictions.
"""

import json
import sys
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    SAMPLE_RATE,
    CYCLE_DURATION,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    MODEL_INPUT_SIZE,
    REFERENCE_DEVICE,
    DEVICE_PROFILES_PATH,
    VTLP_ALPHA_MIN,
    VTLP_ALPHA_MAX,
    VTLP_FHI_MIN,
    VTLP_FHI_MAX,
)

# Spectrum correction STFT params — must match 02_spectrum_correction.py
_CORR_N_FFT      = 256
_CORR_WIN_LENGTH = 256
_CORR_HOP_LENGTH = 64

EKO_DEVICE_NAME = 'Eko'


# ---------------------------------------------------------------------------
# VTLP filterbank
# ---------------------------------------------------------------------------

def get_vtlp_filterbank(
    n_mels:          int,
    n_fft:           int,
    sr:              int,
    fmin:            float,
    fmax:            float,
    alpha:           float,
    fhi:             float,
    base_filterbank: np.ndarray,
    freq_bins:       np.ndarray,
) -> np.ndarray:
    """
    Return a VTLP-warped mel filterbank of shape (n_mels, n_fft//2+1).

    Frequency warping:
        f_warped = f * alpha                             for f <= fhi
        f_warped = fmax - (fmax - fhi*alpha) * (fmax-f) / (fmax-fhi)   for f > fhi
    """
    scale = np.ones_like(freq_bins)
    lo    = freq_bins <= fhi
    hi    = ~lo

    scale[lo] = alpha
    if fhi < fmax:
        scale[hi] = (fmax - fhi * alpha) / (fmax - fhi)

    warped_bins = np.clip(freq_bins * scale, 0, fmax)

    # Rebuild filterbank by interpolating the base filterbank at warped positions
    warped_filterbank = np.zeros_like(base_filterbank)
    for m in range(n_mels):
        warped_filterbank[m] = np.interp(warped_bins, freq_bins, base_filterbank[m])

    return warped_filterbank


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def _pad_reflect(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) >= target_length:
        return audio[:target_length]
    while len(audio) < target_length:
        pad   = min(target_length - len(audio), len(audio) - 1)
        audio = np.pad(audio, (0, pad), mode='reflect')
    return audio[:target_length]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EkoInferenceDataset(Dataset):
    """
    PyTorch Dataset for inference on Eko recordings.

    Parameters
    ----------
    wav_dir        : directory of wav files (any sample rate — resampled to 4kHz)
    profiles_path  : path to device_profiles.json (produced by step 02)
    chunk_duration : length of each chunk in seconds (default 4.0)
    vtlp_alpha     : VTLP warp factor — 1.0 gives unwarped spectrogram
    vtlp_fhi       : VTLP boundary frequency in Hz
    apply_correction : set False to skip spectrum correction (diagnostic use)
    """

    def __init__(
        self,
        wav_dir:          str | Path,
        profiles_path:    str | Path = DEVICE_PROFILES_PATH,
        chunk_duration:   float = 4.0,
        vtlp_alpha:       float | None = 1.0,
        vtlp_fhi:         float = (VTLP_FHI_MIN + VTLP_FHI_MAX) / 2,
        apply_correction: bool = True,
    ):
        self.wav_dir       = Path(wav_dir)
        self.sr            = SAMPLE_RATE
        self.target_length = int(SAMPLE_RATE * CYCLE_DURATION)
        self.chunk_samples = int(SAMPLE_RATE * chunk_duration)
        self.chunk_duration = chunk_duration
        self.vtlp_alpha    = vtlp_alpha
        self.vtlp_fhi      = vtlp_fhi

        # ── Spectrum correction filter ────────────────────────────────────
        with open(profiles_path) as f:
            profiles = json.load(f)

        if EKO_DEVICE_NAME not in profiles:
            raise KeyError(
                f"'{EKO_DEVICE_NAME}' not found in '{profiles_path}'. "
                f"Run step 02 (spectrum correction) first."
            )

        n_corr_bins = _CORR_N_FFT // 2 + 1
        if apply_correction:
            ref_profile = np.array(profiles[REFERENCE_DEVICE])
            eko_profile = np.array(profiles[EKO_DEVICE_NAME])
            correction  = ref_profile - eko_profile              # log domain
            self._linear_correction = np.exp(correction).astype(np.float32)
        else:
            self._linear_correction = np.ones(n_corr_bins, dtype=np.float32)

        # ── Pre-compute base mel filterbank and FFT frequency bins ────────
        self._base_filterbank = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX,
        )
        self._freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

        if vtlp_alpha is not None:
            self._vtlp_filterbank = get_vtlp_filterbank(
                n_mels=N_MELS, n_fft=N_FFT,
                sr=SAMPLE_RATE, fmin=FMIN, fmax=FMAX,
                alpha=vtlp_alpha, fhi=vtlp_fhi,
                base_filterbank=self._base_filterbank,
                freq_bins=self._freq_bins,
            )
        else:
            self._vtlp_filterbank = None

        # ── Build flat sample list from all wav files ─────────────────────
        self.samples: list[dict] = []
        wav_files = sorted(self.wav_dir.glob('*.wav'))

        for wav_path in wav_files:
            try:
                info      = sf.info(str(wav_path))
                # Duration in samples at the pipeline's SAMPLE_RATE
                n_samples = int(info.frames / info.samplerate * SAMPLE_RATE)
                starts    = range(0, n_samples, self.chunk_samples)
                for i, start in enumerate(starts):
                    end = min(start + self.chunk_samples, n_samples)
                    self.samples.append({
                        'file':        wav_path.name,
                        'path':        str(wav_path),
                        'file_sr':     info.samplerate,
                        'chunk':       i,
                        'start':       start,
                        'end':         end,
                    })
            except Exception as e:
                print(f"Warning: could not read '{wav_path.name}': {e}")

        n_files  = len(wav_files)
        vtlp_str = f"alpha={vtlp_alpha}, fhi={vtlp_fhi} Hz" if vtlp_alpha else "random (TTA)"
        print(f"EkoInferenceDataset: {n_files} file(s) → {len(self.samples)} chunk(s)")
        print(f"  Chunk:    {chunk_duration}s  |  Target: {CYCLE_DURATION}s (reflect-padded)")
        print(f"  VTLP:     {vtlp_str}")
        print(f"  Correction: {'yes' if apply_correction else 'no (diagnostic)'}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # ── Load full file and resample to pipeline SR if needed ──────────
        audio, file_sr = sf.read(sample['path'], always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if file_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=self.sr)

        # ── Slice chunk ───────────────────────────────────────────────────
        audio = audio[sample['start']:sample['end']]

        # ── Reflect-pad to CYCLE_DURATION ─────────────────────────────────
        audio = _pad_reflect(audio, self.target_length)

        # ── Spectrum correction (mirrors 02_spectrum_correction.py) ─────────
        # Correct at _CORR_N_FFT=256, reconstruct via ISTFT, then compute
        # mel spectrogram at N_FFT=512 — same two-stage process as training
        corr_stft      = librosa.stft(audio, n_fft=_CORR_N_FFT,
                                      win_length=_CORR_WIN_LENGTH,
                                      hop_length=_CORR_HOP_LENGTH)
        corr_magnitude = np.abs(corr_stft) * self._linear_correction[:, np.newaxis]
        corr_stft      = corr_magnitude * np.exp(1j * np.angle(corr_stft))
        audio          = librosa.istft(corr_stft,
                                       hop_length=_CORR_HOP_LENGTH,
                                       win_length=_CORR_WIN_LENGTH)

        # Re-pad after ISTFT (length may shift slightly)
        audio = _pad_reflect(audio, self.target_length)

        # ── STFT magnitude for mel spectrogram ────────────────────────────
        stft      = librosa.stft(audio, n_fft=N_FFT,
                                 win_length=WIN_LENGTH, hop_length=HOP_LENGTH)
        magnitude = np.abs(stft)                                 # (n_fft//2+1, T)

        # ── VTLP-warped mel filterbank ────────────────────────────────────
        if self._vtlp_filterbank is not None:
            filterbank = self._vtlp_filterbank
        else:
            alpha = np.random.uniform(VTLP_ALPHA_MIN, VTLP_ALPHA_MAX)
            fhi   = np.random.uniform(VTLP_FHI_MIN,   VTLP_FHI_MAX)
            filterbank = get_vtlp_filterbank(
                n_mels=N_MELS, n_fft=N_FFT,
                sr=self.sr, fmin=FMIN, fmax=FMAX,
                alpha=alpha, fhi=fhi,
                base_filterbank=self._base_filterbank,
                freq_bins=self._freq_bins,
            )

        mel_spec = filterbank @ magnitude                        # (n_mels, T)
        log_mel  = np.log(np.maximum(mel_spec, 1e-10))

        # ── Resize to model input ─────────────────────────────────────────
        target_h, target_w = MODEL_INPUT_SIZE
        img         = Image.fromarray(log_mel)
        img         = img.resize((target_w, target_h), Image.BILINEAR)
        spectrogram = np.array(img)

        # ── Per-sample normalisation ──────────────────────────────────────
        mean        = spectrogram.mean()
        std         = spectrogram.std() + 1e-10
        spectrogram = (spectrogram - mean) / std

        # ── (3, H, W) tensor for ResNet ───────────────────────────────────
        tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).repeat(3, 1, 1)

        return tensor, idx
