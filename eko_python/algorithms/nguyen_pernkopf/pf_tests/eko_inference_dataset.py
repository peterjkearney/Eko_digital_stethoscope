"""
eko_inference_dataset.py

Lightweight PyTorch Dataset for running inference on Eko device recordings.

Rather than applying Griffin-Lim spectrum correction offline and re-saving
wav files, this dataset applies the correction directly in the STFT magnitude
domain during feature extraction.  This is mathematically equivalent to the
training pipeline but avoids the phase-reconstruction artefacts introduced
by Griffin-Lim.

Per-sample pipeline:
    1. Load chunk from preprocessed 16 kHz mono wav
    2. Reflect-pad to CYCLE_DURATION (matches pad_cycles.py)
    3. Compute STFT magnitude
    4. Apply spectrum correction: magnitude × exp(Meditron_profile − Eko_profile)
    5. Apply VTLP-warped mel filterbank  (fixed alpha=1.0 by default)
    6. Log + resize to MODEL_INPUT_SIZE + per-sample normalisation
    7. Return (3, H, W) tensor + sample index

Use dataset.samples[idx] to retrieve the {'file', 'chunk', 'start', 'end'}
metadata for each index, enabling per-file aggregation of predictions.

Usage:
    from pf_tests.eko_inference_dataset import EkoInferenceDataset

    dataset = EkoInferenceDataset(wav_dir='data/pf_samples/preprocessed')
    loader  = DataLoader(dataset, batch_size=8, shuffle=False)

    for tensors, indices in loader:
        logits = model(tensors.to(device))
        preds  = logits.argmax(dim=1)
        for pred, idx in zip(preds.cpu(), indices):
            meta = dataset.samples[int(idx)]
            # meta = {'file': 'Eko_p001_t01_mild_...wav', 'chunk': 0, ...}
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
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    MODEL_INPUT_SIZE,
    REFERENCE_DEVICE,
    SPECTRUM_CORRECTION_PROFILES_PATH,
)
from dataset.icbhi_dataset import get_vtlp_filterbank

EKO_DEVICE_NAME = 'Eko'


# ---------------------------------------------------------------------------
# Padding (mirrors pad_cycles.py)
# ---------------------------------------------------------------------------

def _pad_reflect(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Reflect-pad or truncate audio to exactly target_length samples."""
    if len(audio) >= target_length:
        return audio[:target_length]
    while len(audio) < target_length:
        max_pad = len(audio) - 1
        this_pad = min(target_length - len(audio), max_pad)
        audio = np.pad(audio, (0, this_pad), mode='reflect')
    return audio[:target_length]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EkoInferenceDataset(Dataset):
    """
    PyTorch Dataset for inference on preprocessed Eko recordings.

    Parameters
    ----------
    wav_dir        : directory of preprocessed 16 kHz mono wav files
    profiles_path  : path to device_profiles.json (must contain 'Eko' and
                     REFERENCE_DEVICE keys, produced by analyse_spectrum.py)
    chunk_duration : length of each chunk in seconds (default 4.0)
    vtlp_alpha     : VTLP warp factor — 1.0 gives an unwarped spectrogram
                     (deterministic inference).  Pass None to sample randomly
                     from the training range on each call.
    vtlp_fhi       : VTLP boundary frequency in Hz (default: midpoint of
                     training range, 3500 Hz)
    """

    def __init__(
        self,
        wav_dir: str | Path,
        profiles_path: str | Path = SPECTRUM_CORRECTION_PROFILES_PATH,
        chunk_duration: float = 4.0,
        vtlp_alpha: float | None = 1.0,
        vtlp_fhi: float = 3500.0,
        apply_correction: bool = True,
        device_bandwidth_hz: float = 2000.0,
    ):
        self.wav_dir       = Path(wav_dir)
        self.sr            = SAMPLE_RATE
        self.n_fft         = N_FFT
        self.hop_length    = HOP_LENGTH
        self.n_mels        = N_MELS
        self.fmin          = FMIN
        self.fmax          = FMAX
        self.input_size    = MODEL_INPUT_SIZE
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
                f"Run pf_tests/analyse_spectrum.py first."
            )

        if apply_correction:
            ref_profile = np.array(profiles[REFERENCE_DEVICE])
            eko_profile = np.array(profiles[EKO_DEVICE_NAME])
            correction  = ref_profile - eko_profile             # log domain
            self._linear_correction = np.exp(correction).astype(np.float32)
            # Don't correct above the device's recording bandwidth — bins above
            # the Nyquist of the original device contain no real signal, so
            # applying a large correction gain there only amplifies artefacts.
            if device_bandwidth_hz is not None:
                freq_res   = SAMPLE_RATE / N_FFT           # Hz per bin
                cutoff_bin = int(device_bandwidth_hz / freq_res)
                self._linear_correction[cutoff_bin:] = 1.0
        else:
            n_bins = N_FFT // 2 + 1
            self._linear_correction = np.ones(n_bins, dtype=np.float32)  # identity

        # ── Bandwidth scalar (mirrors ICBHIDataset) ───────────────────────
        # Normalised device Nyquist relative to the pipeline Nyquist.
        pipeline_nyquist       = SAMPLE_RATE / 2
        effective_nyquist      = min(device_bandwidth_hz or pipeline_nyquist,
                                     pipeline_nyquist)
        self._bandwidth_scalar = float(effective_nyquist / pipeline_nyquist)

        # ── Pre-compute base mel filterbank and FFT bins ──────────────────
        self._base_filterbank = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )
        self._freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Pre-compute VTLP filterbank if alpha is fixed; otherwise compute
        # per-sample in __getitem__.
        if vtlp_alpha is not None:
            self._vtlp_filterbank = get_vtlp_filterbank(
                n_mels=self.n_mels, n_fft=self.n_fft,
                sr=self.sr, fmin=self.fmin, fmax=self.fmax,
                alpha=vtlp_alpha, fhi=vtlp_fhi,
                base_filterbank=self._base_filterbank,
                freq_bins=self._freq_bins,
            )
        else:
            self._vtlp_filterbank = None

        # ── Build flat sample list ────────────────────────────────────────
        # Use sf.info() to get file lengths without loading audio.
        self.samples: list[dict] = []
        wav_files = sorted(self.wav_dir.glob('*.wav'))

        for wav_path in wav_files:
            try:
                info      = sf.info(str(wav_path))
                n_samples = info.frames
                starts    = range(0, n_samples, self.chunk_samples)
                for i, start in enumerate(starts):
                    end = min(start + self.chunk_samples, n_samples)
                    self.samples.append({
                        'file':  wav_path.name,
                        'path':  str(wav_path),
                        'chunk': i,
                        'start': start,
                        'end':   end,
                    })
            except Exception as e:
                print(f"Warning: could not read '{wav_path.name}': {e}")

        n_files = len(wav_files)
        print(f"EkoInferenceDataset: {n_files} file(s) → {len(self.samples)} chunk(s)")
        print(f"  Chunk:    {chunk_duration}s  |  Target: {CYCLE_DURATION}s (reflect-padded)")
        vtlp_str = f"alpha={vtlp_alpha}, fhi={vtlp_fhi} Hz" if vtlp_alpha else "random (TTA)"
        print(f"  VTLP:     {vtlp_str}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # ── Load chunk ────────────────────────────────────────────────────
        audio, _ = sf.read(sample['path'], always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[sample['start']:sample['end']].astype(np.float32)

        # ── Reflect-pad to CYCLE_DURATION ─────────────────────────────────
        audio = _pad_reflect(audio, self.target_length)

        # ── STFT magnitude ────────────────────────────────────────────────
        stft      = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)  # (n_fft//2+1, T)

        # ── Spectrum correction (STFT domain — no Griffin-Lim needed) ─────
        magnitude = magnitude * self._linear_correction[:, np.newaxis]

        # ── VTLP-warped mel filterbank ────────────────────────────────────
        if self._vtlp_filterbank is not None:
            filterbank = self._vtlp_filterbank
        else:
            # Random VTLP (test-time augmentation mode)
            from config import VTLP_ALPHA_MIN, VTLP_ALPHA_MAX, VTLP_FHI_MIN, VTLP_FHI_MAX
            alpha = np.random.uniform(VTLP_ALPHA_MIN, VTLP_ALPHA_MAX)
            fhi   = np.random.uniform(VTLP_FHI_MIN, VTLP_FHI_MAX)
            filterbank = get_vtlp_filterbank(
                n_mels=self.n_mels, n_fft=self.n_fft,
                sr=self.sr, fmin=self.fmin, fmax=self.fmax,
                alpha=alpha, fhi=fhi,
                base_filterbank=self._base_filterbank,
                freq_bins=self._freq_bins,
            )

        mel_spec = filterbank @ magnitude               # (n_mels, T)
        log_mel  = np.log(np.maximum(mel_spec, 1e-10))

        # ── Resize to model input ─────────────────────────────────────────
        target_h, target_w = self.input_size
        img = Image.fromarray(log_mel)
        img = img.resize((target_w, target_h), Image.BILINEAR)
        spectrogram = np.array(img)

        # ── Per-sample normalisation (matches ICBHIDataset) ───────────────
        mean        = spectrogram.mean()
        std         = spectrogram.std() + 1e-10
        spectrogram = (spectrogram - mean) / std

        # ── (3, H, W) tensor for ResNet ───────────────────────────────────
        tensor    = torch.from_numpy(spectrogram).float().unsqueeze(0).repeat(3, 1, 1)
        bandwidth = torch.tensor(self._bandwidth_scalar, dtype=torch.float32)

        return tensor, bandwidth, idx
