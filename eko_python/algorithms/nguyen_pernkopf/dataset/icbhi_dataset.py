"""
icbhi_dataset.py

PyTorch Dataset class for the ICBHI ALSC (adventitious lung sound
classification) task.

Loads preprocessed cycle wav files from disk, applies online augmentations,
computes log-mel spectrograms with VTLP, and returns tensors ready for the
ResNet model.

Per-sample pipeline:
    1. Load wav file
    2. Apply random roll (all samples)
    3. Apply volume / noise / pitch / speed augmentations (stretched samples only)
    4. Compute log-mel spectrogram with randomly sampled VTLP warp
    5. Resize to model input dimensions
    6. Convert to tensor

Patient-level splitting:
    The dataset accepts an explicit list of patient IDs to include, making
    it straightforward to construct train and validation folds at the patient
    level without any cycles from the same patient appearing in both.

Label encoding:
    normal  → 0
    crackle → 1
    wheeze  → 2
    both    → 3
"""

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    MANIFEST_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    VTLP_ALPHA_MIN,
    VTLP_ALPHA_MAX,
    VTLP_FHI_MIN,
    VTLP_FHI_MAX,
    MODEL_INPUT_SIZE,
    AUG_PROBABILITY,
    RANDOM_SEED,
)

from dataset.augmentations import (
    random_roll,
    random_volume,
    random_noise,
    random_pitch,
    random_speed,
)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

LABEL_TO_IDX = {
    'normal':  0,
    'crackle': 1,
    'wheeze':  2,
    'both':    3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# VTLP mel filterbank
# ---------------------------------------------------------------------------

def get_vtlp_filterbank(n_mels, n_fft, sr, fmin, fmax, alpha, fhi,
                        base_filterbank=None, freq_bins=None):
    if base_filterbank is None:
        base_filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                              fmin=fmin, fmax=fmax)
    if freq_bins is None:
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    filterbank = base_filterbank
    nyquist   = sr / 2.0

    # Vectorised piecewise linear warp
    warped_bins = np.where(
        freq_bins <= fhi,
        freq_bins * alpha,
        freq_bins + (freq_bins - fhi) * (alpha * fhi / (nyquist - fhi) - 1)
        * (nyquist - freq_bins) / (nyquist - fhi)
    )
    warped_bins = np.clip(warped_bins, 0, nyquist)

    # Vectorised interpolation replacing the Python loop
    indices = np.searchsorted(freq_bins, warped_bins)
    indices = np.clip(indices, 1, len(freq_bins) - 1)

    lo = freq_bins[indices - 1]
    hi = freq_bins[indices]
    t  = (warped_bins - lo) / (hi - lo + 1e-10)

    # Interpolate across all bins simultaneously
    warped_filterbank = (
        (1 - t) * filterbank[:, indices - 1] +
        t       * filterbank[:, indices]
    )

    return warped_filterbank

def compute_vtlp_logmel(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    alpha: float,
    fhi: float,
    base_filterbank: np.ndarray | None = None,
    freq_bins: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram using a VTLP-warped filterbank.

    Parameters
    ----------
    audio      : 1D numpy array of audio samples
    sr         : sample rate
    n_fft      : FFT window size
    hop_length : STFT hop length
    n_mels     : number of mel filters
    fmin       : minimum frequency (Hz)
    fmax       : maximum frequency (Hz)
    alpha      : VTLP warp factor
    fhi        : VTLP warp boundary frequency (Hz)

    Returns
    -------
    2D numpy array of shape (n_mels, T) containing the log-mel spectrogram.
    """
    # Compute STFT magnitude
    stft      = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)  # shape: (1 + n_fft // 2, T)

    # Get VTLP-warped filterbank
    filterbank = get_vtlp_filterbank(
        n_mels=n_mels,
        n_fft=n_fft,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        alpha=alpha,
        fhi=fhi,
        base_filterbank=base_filterbank,
        freq_bins=freq_bins,
    )  # shape: (n_mels, 1 + n_fft // 2)

    # Apply filterbank to magnitude spectrum
    mel_spec = filterbank @ magnitude  # shape: (n_mels, T)

    # Convert to log scale
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    return mel_spec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ICBHIDataset(Dataset):
    """
    PyTorch Dataset for the ICBHI ALSC task.

    Parameters
    ----------
    manifest_path : path to manifest CSV
    patient_ids   : list of patient IDs to include. If None, all patients
                    in the specified split are included.
    split         : 'train' or 'test'. Used to filter the manifest if
                    patient_ids is None.
    augment       : whether to apply online augmentations. Should be True
                    for training, False for validation and test.
    """

    def __init__(
        self,
        manifest_path: str = MANIFEST_PATH,
        patient_ids: list[int] | None = None,
        split: str | None = None,
        augment: bool = True,
        fraction: float = 1.0,
    ):
        manifest = pd.read_csv(manifest_path)

        # Filter to valid cycles only
        manifest = manifest[
            manifest['wav_path'].notna() & (manifest['wav_path'] != '')
        ]

        # Filter by split
        if split is not None:
            manifest = manifest[manifest['split'] == split]

        # Filter by patient IDs (for patient-level train/val splitting)
        if patient_ids is not None:
            manifest = manifest[manifest['patient_id'].isin(patient_ids)]

        # Subsample cycles per patient, preserving patient distribution
        if fraction < 1.0:
            manifest = (
                manifest
                .groupby('patient_id', group_keys=False)
                .apply(lambda g: g.sample(frac=fraction, random_state=RANDOM_SEED))
                .reset_index(drop=True)
            )

        self.manifest = manifest.reset_index(drop=True)
        self.augment  = augment

        # Fixed config
        self.sr          = SAMPLE_RATE
        self.n_fft       = N_FFT
        self.hop_length  = HOP_LENGTH
        self.n_mels      = N_MELS
        self.fmin        = FMIN
        self.fmax        = FMAX
        self.input_size  = MODEL_INPUT_SIZE  # (H, W) e.g. (224, 224)

        # Augmentation probability
        self.aug_prob = AUG_PROBABILITY

        # VTLP parameter ranges
        self.alpha_min = VTLP_ALPHA_MIN
        self.alpha_max = VTLP_ALPHA_MAX
        self.fhi_min   = VTLP_FHI_MIN
        self.fhi_max   = VTLP_FHI_MAX

        # Pre-compute the base mel filterbank and FFT frequencies — these
        # depend only on fixed config values and are constant across all samples
        self._base_filterbank = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )
        self._freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        
        
        row = self.manifest.iloc[idx]

        wav_path     = row['wav_path']
        label        = LABEL_TO_IDX[row['label']]
        is_stretched = bool(row['is_stretched'])

        # ── Load audio ───────────────────────────────────────────────────
        audio, file_sr = sf.read(wav_path, always_2d=False)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        # ── Online augmentations ─────────────────────────────────────────
        if self.augment:
            # Random roll applied to all samples unconditionally
            audio = random_roll(audio)

            # Waveform augmentations applied to stretched samples only,
            # each independently with probability aug_prob
            if is_stretched:
                if np.random.random() < self.aug_prob:
                    audio = random_volume(audio)
                if np.random.random() < self.aug_prob:
                    audio = random_noise(audio)
                if np.random.random() < self.aug_prob:
                    audio = random_pitch(audio, sr=self.sr)
                if np.random.random() < self.aug_prob:
                    audio = random_speed(audio, sr=self.sr)

        # ── Compute log-mel spectrogram with VTLP ────────────────────────
        alpha = np.random.uniform(self.alpha_min, self.alpha_max)
        fhi   = np.random.uniform(self.fhi_min,   self.fhi_max)

        spectrogram = compute_vtlp_logmel(
            audio=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            alpha=alpha,
            fhi=fhi,
            base_filterbank=self._base_filterbank,
            freq_bins=self._freq_bins,
        )  # shape: (n_mels, T)

        # ── Resize to model input size ───────────────────────────────────
        spectrogram = self._resize(spectrogram, self.input_size)

        # ── Normalise ────────────────────────────────────────────────────
        # Per-sample mean/std normalisation
        mean = spectrogram.mean()
        std  = spectrogram.std() + 1e-10
        spectrogram = (spectrogram - mean) / std

        # ── Convert to tensor ────────────────────────────────────────────
        # ResNet expects (C, H, W) — replicate single channel to 3 channels
        spectrogram_tensor = torch.from_numpy(spectrogram).float()
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0)          # (1, H, W)
        spectrogram_tensor = spectrogram_tensor.repeat(3, 1, 1)       # (3, H, W)

        # ── Device bandwidth scalar ───────────────────────────────────────
        # Normalised Nyquist of the recording device relative to the pipeline
        # Nyquist (SAMPLE_RATE / 2).  Tells the model whether high-frequency
        # mel bins contain real signal or just amplified near-zero content.
        pipeline_nyquist = SAMPLE_RATE / 2
        device_nyquist   = min(int(row['native_sr']) / 2, pipeline_nyquist)
        bandwidth_scalar = device_nyquist / pipeline_nyquist          # in (0, 1]
        bandwidth_tensor = torch.tensor(bandwidth_scalar, dtype=torch.float32)

        return spectrogram_tensor, bandwidth_tensor, label

    def _resize(
        self,
        spectrogram: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Resize a spectrogram to the target (H, W) using bilinear interpolation.

        Parameters
        ----------
        spectrogram : 2D array of shape (n_mels, T)
        target_size : (height, width) target dimensions

        Returns
        -------
        2D array of shape target_size.
        """
        from PIL import Image
        target_h, target_w = target_size
        img = Image.fromarray(spectrogram)
        img = img.resize((target_w, target_h), Image.BILINEAR)
        return np.array(img)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss functions.

        Returns a tensor of shape (num_classes,) where each weight is
        inversely proportional to the class frequency in this dataset.
        Useful for addressing class imbalance in the loss function.

        Returns
        -------
        torch.Tensor of shape (4,) with weights for
        [normal, crackle, wheeze, both].
        """
        counts = self.manifest['label'].value_counts()
        weights = []
        for label in LABEL_TO_IDX:
            count = counts.get(label, 1)
            weights.append(1.0 / count)

        weights = torch.tensor(weights, dtype=torch.float32)
        # Normalise so weights sum to number of classes
        weights = weights / weights.sum() * len(LABEL_TO_IDX)
        return weights

    def get_patient_ids(self) -> list[int]:
        """Return the list of unique patient IDs in this dataset."""
        return sorted(self.manifest['patient_id'].unique().tolist())

    def summary(self) -> None:
        """Print a summary of the dataset composition."""
        print(f"Dataset summary:")
        print(f"  Total samples:  {len(self.manifest)}")
        print(f"  Patients:       {self.manifest['patient_id'].nunique()}")
        print(f"  Augment:        {self.augment}")
        print(f"\n  Class distribution:")
        counts = self.manifest.groupby(['label', 'is_stretched']).size()
        print(counts.to_string())
        print(f"\n  Device distribution:")
        print(self.manifest['device'].value_counts().to_string())