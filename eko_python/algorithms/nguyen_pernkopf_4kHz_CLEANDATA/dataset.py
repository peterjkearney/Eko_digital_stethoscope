"""
dataset.py

PyTorch Dataset and DataLoader factory for the ICBHI lung sound classification task.

Spectrograms are pre-computed offline (step 05) and stored as 224×224 greyscale
PNGs. This class simply loads those PNGs and applies a single cheap online
augmentation: a random reflect-padded time shift.

All expensive augmentation (volume, noise, speed, VTLP) is handled offline in
steps 04 and 05. The manifest records exactly which augmentation was applied to
each sample.

Label encoding
--------------
    normal  → 0
    crackle → 1
    wheeze  → 2
    both    → 3

Fold support
------------
Pass explicit patient_id lists to carve out patient-level train/val splits for
cross-validation. Val and test folds use originals only (aug_index == 0) so that
augmented copies of a patient's cycles never leak into the held-out fold.

DataLoader factory
------------------
get_dataloaders() returns a dict with keys 'train', 'test', and optionally 'val'
if val_patient_ids is provided.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    MANIFEST_PATH,
    MODEL_INPUT_SIZE,
    NUM_WORKERS,
    SPECTROGRAMS_DIR,
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
# Online augmentation: reflect roll
# ---------------------------------------------------------------------------

def reflect_roll(arr: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift a 2D spectrogram along the time axis (axis=1) by `shift` columns,
    filling the gap with a mirror reflection of the adjacent content.

    Positive shift moves content to the right; negative moves it to the left.
    Using reflect rather than circular roll avoids a seam where the end of a
    breath cycle abruptly meets the start.
    """
    if shift == 0:
        return arr
    W = arr.shape[1]
    shift = shift % W
    if shift == 0:
        return arr
    # Roll right by `shift`: left gap filled by mirroring the leftmost columns
    # arr[:, shift:] is the main content; arr[:, 1:shift+1][:, ::-1] is the mirror
    mirror = arr[:, 1:shift + 1][:, ::-1]
    return np.concatenate([mirror, arr[:, :-shift]], axis=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ICBHIDataset(Dataset):
    """
    Parameters
    ----------
    manifest_path    : path to manifest CSV (default from config)
    split            : 'train' or 'test' — filters the manifest if
                       patient_ids is None
    patient_ids      : explicit list of patient IDs to include; takes priority
                       over split when provided
    include_augmented: if True, include augmented copies (aug_index > 0);
                       set False for val/test folds so only originals are used
    augment          : if True, apply random reflect roll at load time;
                       should be False for val and test
    max_roll_frac    : maximum roll as a fraction of spectrogram width
                       (default 0.25 → up to ±25% of 224 = ±56 columns)
    """

    def __init__(
        self,
        manifest_path: str | Path = MANIFEST_PATH,
        split: str | None = None,
        patient_ids: list[int] | None = None,
        include_augmented: bool = True,
        augment: bool = True,
        max_roll_frac: float = 0.25,
    ):
        manifest = pd.read_csv(manifest_path)

        # Keep only rows with a computed spectrogram
        manifest = manifest[
            manifest['spec_path'].notna() & (manifest['spec_path'] != '')
        ].copy()

        # Filter by split or explicit patient list
        if patient_ids is not None:
            manifest = manifest[manifest['patient_id'].isin(patient_ids)]
        elif split is not None:
            manifest = manifest[manifest['split'] == split]

        # Optionally restrict to original cycles only
        if not include_augmented:
            manifest = manifest[manifest['aug_index'] == 0]

        self.manifest       = manifest.reset_index(drop=True)
        self.augment        = augment
        self.max_roll_frac  = max_roll_frac
        self._input_h, self._input_w = MODEL_INPUT_SIZE

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]

        # ── Load PNG ─────────────────────────────────────────────────────
        # Reconstruct path from SPECTROGRAMS_DIR so the dataset works on any
        # machine regardless of the absolute prefix stored in the manifest.
        spec_path = SPECTROGRAMS_DIR / row['split'] / Path(row['spec_path']).name
        img = Image.open(spec_path).convert('L')  # greyscale uint8
        arr = np.array(img, dtype=np.float32)            # (H, W)

        # ── Online augmentation: random reflect roll ──────────────────────
        if self.augment:
            max_shift = int(self._input_w * self.max_roll_frac)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                arr = reflect_roll(arr, shift)

        # ── Per-sample normalisation ──────────────────────────────────────
        mean = arr.mean()
        std  = arr.std() + 1e-10
        arr  = (arr - mean) / std

        # ── To tensor: replicate greyscale → 3 channels for ResNet ───────
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        label = LABEL_TO_IDX[row['label']]
        return tensor, label

    # ── Utility methods ───────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for a weighted loss function."""
        counts  = self.manifest['label'].value_counts()
        weights = torch.tensor(
            [1.0 / counts.get(lbl, 1) for lbl in LABEL_TO_IDX],
            dtype=torch.float32,
        )
        return weights / weights.sum() * len(LABEL_TO_IDX)

    def get_patient_ids(self) -> list[int]:
        return sorted(self.manifest['patient_id'].unique().tolist())

    def summary(self) -> None:
        print(f"ICBHIDataset summary:")
        print(f"  Total samples : {len(self.manifest)}")
        print(f"  Patients      : {self.manifest['patient_id'].nunique()}")
        print(f"  Augment       : {self.augment}")
        print(f"  Incl. augmented copies: "
              f"{(self.manifest['aug_index'] > 0).sum()} "
              f"/ {len(self.manifest)} rows")
        print(f"\n  Class distribution:")
        print(self.manifest['label'].value_counts().to_string())
        print(f"\n  Device distribution:")
        print(self.manifest['device'].value_counts().to_string())


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    manifest_path: str | Path = MANIFEST_PATH,
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    train_patient_ids: list[int] | None = None,
    val_patient_ids:   list[int] | None = None,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders for training, validation (optional), and test.

    Full training mode (no fold)
    ----------------------------
    Leave train_patient_ids and val_patient_ids as None.
    Returns {'train': ..., 'test': ...}.
    All train-split rows (originals + augmented) are included in 'train'.

    Fold mode (patient-level cross-validation)
    ------------------------------------------
    Pass explicit patient ID lists for train and val.
    Returns {'train': ..., 'val': ..., 'test': ...}.
    'train' includes augmented copies; 'val' uses originals only so no
    augmented copies of held-out patients can appear in the val set.

    Parameters
    ----------
    manifest_path     : path to manifest CSV
    batch_size        : samples per batch
    num_workers       : DataLoader worker processes
    train_patient_ids : patient IDs for the training fold (fold mode only)
    val_patient_ids   : patient IDs for the validation fold (fold mode only)
    """
    loaders = {}

    if train_patient_ids is not None:
        # Fold mode — explicit patient lists
        train_ds = ICBHIDataset(
            manifest_path=manifest_path,
            patient_ids=train_patient_ids,
            include_augmented=True,
            augment=True,
        )
    else:
        # Full training mode — all train-split rows
        train_ds = ICBHIDataset(
            manifest_path=manifest_path,
            split='train',
            include_augmented=True,
            augment=True,
        )

    loaders['train'] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if val_patient_ids is not None:
        val_ds = ICBHIDataset(
            manifest_path=manifest_path,
            patient_ids=val_patient_ids,
            include_augmented=False,  # originals only for val
            augment=False,
        )
        loaders['val'] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_ds = ICBHIDataset(
        manifest_path=manifest_path,
        split='test',
        include_augmented=False,
        augment=False,
    )
    loaders['test'] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loaders
