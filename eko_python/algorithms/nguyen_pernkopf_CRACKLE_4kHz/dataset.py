"""
dataset.py

PyTorch Dataset and DataLoader factory for binary crackle detection.

Loads the same pre-computed 224×224 PNGs produced by the CLEAN_4kHz pipeline.
The only difference from that pipeline is the label remapping:

    ICBHI label → binary label
    ─────────────────────────
    normal       → no_crackle  (0)
    wheeze       → no_crackle  (0)
    crackle      → crackle     (1)
    both         → crackle     (1)  ← both contains crackles
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
    LABEL_TO_IDX,
    ICBHI_TO_BINARY,
)

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Online augmentation: reflect roll
# ---------------------------------------------------------------------------

def reflect_roll(arr: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return arr
    W = arr.shape[1]
    shift = shift % W
    if shift == 0:
        return arr
    mirror = arr[:, 1:shift + 1][:, ::-1]
    return np.concatenate([mirror, arr[:, :-shift]], axis=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ICBHIDataset(Dataset):

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

        manifest = manifest[
            manifest['spec_path'].notna() & (manifest['spec_path'] != '')
        ].copy()

        if patient_ids is not None:
            manifest = manifest[manifest['patient_id'].isin(patient_ids)]
        elif split is not None:
            manifest = manifest[manifest['split'] == split]

        if not include_augmented:
            manifest = manifest[manifest['aug_index'] == 0]

        # Remap 4-class ICBHI labels to binary
        manifest['label'] = manifest['label'].map(ICBHI_TO_BINARY)

        self.manifest      = manifest.reset_index(drop=True)
        self.augment       = augment
        self.max_roll_frac = max_roll_frac
        self._input_h, self._input_w = MODEL_INPUT_SIZE

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]

        spec_path = SPECTROGRAMS_DIR / row['split'] / Path(row['spec_path']).name
        img = Image.open(spec_path).convert('L')
        arr = np.array(img, dtype=np.float32)

        if self.augment:
            max_shift = int(self._input_w * self.max_roll_frac)
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift != 0:
                arr = reflect_roll(arr, shift)

        mean = arr.mean()
        std  = arr.std() + 1e-10
        arr  = (arr - mean) / std

        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)

        label = LABEL_TO_IDX[row['label']]
        return tensor, label

    def get_class_weights(self) -> torch.Tensor:
        counts  = self.manifest['label'].value_counts()
        weights = torch.tensor(
            [1.0 / counts.get(lbl, 1) for lbl in LABEL_TO_IDX],
            dtype=torch.float32,
        )
        return weights / weights.sum() * len(LABEL_TO_IDX)

    def get_patient_ids(self) -> list[int]:
        return sorted(self.manifest['patient_id'].unique().tolist())

    def summary(self) -> None:
        print(f"ICBHIDataset (binary crackle) summary:")
        print(f"  Total samples : {len(self.manifest)}")
        print(f"  Patients      : {self.manifest['patient_id'].nunique()}")
        print(f"  Augment       : {self.augment}")
        print(f"\n  Class distribution:")
        print(self.manifest['label'].value_counts().to_string())


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

    loaders = {}

    if train_patient_ids is not None:
        train_ds = ICBHIDataset(
            manifest_path=manifest_path,
            patient_ids=train_patient_ids,
            include_augmented=True,
            augment=True,
        )
    else:
        train_ds = ICBHIDataset(
            manifest_path=manifest_path,
            split='train',
            include_augmented=True,
            augment=True,
        )

    loaders['train'] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    if val_patient_ids is not None:
        val_ds = ICBHIDataset(
            manifest_path=manifest_path,
            patient_ids=val_patient_ids,
            include_augmented=False,
            augment=False,
        )
        loaders['val'] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    test_ds = ICBHIDataset(
        manifest_path=manifest_path,
        split='test',
        include_augmented=False,
        augment=False,
    )
    loaders['test'] = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return loaders
