"""
dataset.py

PyTorch Dataset and DataLoader factory for multi-label lung sound
classification on HF_Lung_V1.

Spectrograms are pre-computed offline (step 05) and stored as 224×224
greyscale PNGs.

Label encoding
--------------
Each sample returns a FloatTensor of shape (2,):
    [das, cas]   — 1.0 if present, 0.0 otherwise
Both labels are independent (multi-label binary classification).

Patient-date grouping
---------------------
The train/test split in HF_Lung_V1 is defined at the patient-day level —
all recordings from the same patient on the same day are assigned to the
same split. The first 8 digits of each recording identifier encode the
recording date (YYYYMMDD), which serves as the grouping unit for
cross-validation via GroupKFold.

Val and test loaders use original windows only (aug_index == 0 when present).
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
    WINDOWS_MANIFEST_PATH,
    SPECTROGRAMS_DIR,
    NUM_WORKERS,
)

LABEL_COLS = ['das', 'cas']


# ---------------------------------------------------------------------------
# Patient-date extraction
# ---------------------------------------------------------------------------

def extract_patient_date(recording_id: str) -> str:
    """
    Extract the 8-digit recording date (YYYYMMDD) from a recording_id.

    steth_20180814_09_37_11          → '20180814'
    trunc_2019-06-03-09-33-45-Tc_1  → '20190603'
    """
    return recording_id.split('_')[1].replace('-', '')[:8]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HFLungDataset(Dataset):
    """
    Parameters
    ----------
    manifest_path    : path to windows_manifest.csv
    recording_ids    : explicit list of recording_ids to include;
                       if None, `split` is used instead
    split            : 'train' or 'test' — used when recording_ids is None
    include_augmented: include augmented copies (aug_index > 0); False for val/test
    """

    def __init__(
        self,
        manifest_path: str | Path = WINDOWS_MANIFEST_PATH,
        recording_ids: list[str] | None = None,
        split: str | None = None,
        include_augmented: bool = True,
    ):
        manifest = pd.read_csv(manifest_path)
        manifest = manifest[
            manifest['spec_path'].notna() & (manifest['spec_path'] != '')
        ].copy()

        manifest['patient_date'] = manifest['recording_id'].apply(extract_patient_date)

        if recording_ids is not None:
            manifest = manifest[manifest['recording_id'].isin(recording_ids)]
        elif split is not None:
            manifest = manifest[manifest['split'] == split]

        if not include_augmented and 'aug_index' in manifest.columns:
            manifest = manifest[manifest['aug_index'] == 0]

        # Store the full filtered manifest so resample_augmented() can draw
        # from the complete pool of augmented copies each epoch.
        self._full_manifest = manifest.reset_index(drop=True)
        self.manifest       = self._full_manifest.copy()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        # Reconstruct path from SPECTROGRAMS_DIR so it works on any machine
        spec_path = SPECTROGRAMS_DIR / row['split'] / Path(row['spec_path']).name
        img = Image.open(spec_path).convert('L')
        arr = np.array(img, dtype=np.float32)   # (H, W)

        mean = arr.mean()
        std  = arr.std() + 1e-10
        arr  = (arr - mean) / std

        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        labels = torch.tensor([float(row[c]) for c in LABEL_COLS],
                               dtype=torch.float32)                    # (2,)
        return tensor, labels

    def get_pos_weights(self) -> torch.Tensor:
        """
        Per-label positive weights for BCEWithLogitsLoss.
        pos_weight[i] = n_negative[i] / n_positive[i]
        """
        weights = []
        for col in LABEL_COLS:
            n_pos = (self.manifest[col] == 1).sum()
            n_neg = (self.manifest[col] == 0).sum()
            weights.append(n_neg / max(n_pos, 1))
        return torch.tensor(weights, dtype=torch.float32)

    def resample_augmented(self, seed: int | None = None) -> None:
        """
        Rebuild self.manifest with originals + one randomly sampled augmented
        copy per window. Call at the start of each training epoch so the model
        sees a different augmented version each time rather than the same fixed
        copies on every pass.

        If augmentation has not been run (no aug_index column), this is a no-op.
        """
        if 'aug_index' not in self._full_manifest.columns:
            return

        originals = self._full_manifest[self._full_manifest['aug_index'] == 0]
        augmented = self._full_manifest[self._full_manifest['aug_index'] > 0]

        if augmented.empty:
            self.manifest = originals.reset_index(drop=True)
            return

        # Assign a random number to every augmented row, sort by it, then
        # keep the first occurrence of each (recording_id, window_start) pair
        # — equivalent to sampling 1 random augmented copy per window.
        rng = np.random.default_rng(seed)
        sampled = (
            augmented
            .assign(_r=rng.random(len(augmented)))
            .sort_values('_r')
            .drop_duplicates(subset=['recording_id', 'window_start'], keep='first')
            .drop(columns=['_r'])
        )

        self.manifest = pd.concat(
            [originals, sampled], ignore_index=True
        )

    def summary(self) -> None:
        print(f"HFLungDataset — {len(self.manifest)} samples")
        aug_count = (self.manifest.get('aug_index',
                     pd.Series(0, index=self.manifest.index)) > 0).sum()
        print(f"  Augmented copies : {aug_count}")
        print(f"  Augment online   : {self.augment}")
        for col in LABEL_COLS:
            n_pos = (self.manifest[col] == 1).sum()
            print(f"  {col}: {n_pos} positive ({100*n_pos/len(self.manifest):.1f}%)")
        print(f"  Device split:")
        print(self.manifest['device'].value_counts().to_string())


# ---------------------------------------------------------------------------
# GroupKFold helpers
# ---------------------------------------------------------------------------

def get_train_groups(
    manifest_path: str | Path = WINDOWS_MANIFEST_PATH,
) -> tuple[list[str], list[str]]:
    """
    Return (recording_ids, patient_dates) for all train recordings.

    recording_ids  : one entry per unique train recording
    patient_dates  : corresponding patient-date group label for each recording

    Pass these to sklearn's GroupKFold to create date-grouped folds:
        gkf = GroupKFold(n_splits=k)
        for train_idx, val_idx in gkf.split(recording_ids, groups=patient_dates):
            train_ids = [recording_ids[i] for i in train_idx]
            val_ids   = [recording_ids[i] for i in val_idx]
    """
    manifest = pd.read_csv(manifest_path)
    train = (
        manifest[manifest['split'] == 'train']
        [['recording_id']]
        .drop_duplicates('recording_id')
        .copy()
    )
    train['patient_date'] = train['recording_id'].apply(extract_patient_date)
    return (
        train['recording_id'].tolist(),
        train['patient_date'].tolist(),
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    manifest_path:     str | Path      = WINDOWS_MANIFEST_PATH,
    batch_size:        int              = 64,
    num_workers:       int              = NUM_WORKERS,
    train_recording_ids: list[str] | None = None,
    val_recording_ids:   list[str] | None = None,
) -> dict[str, DataLoader]:
    """
    Build DataLoaders for train, val, and test.

    Full-training mode (train_recording_ids=None, val_recording_ids=None)
    ----------------------------------------------------------------------
    Uses all train-split recordings for training; no val loader returned.
    Call evaluate() on the test loader at the end.

    Fold mode (pass explicit recording ID lists)
    --------------------------------------------
    train_recording_ids : recordings for this fold's training set
    val_recording_ids   : recordings for this fold's validation set
    Returns {'train': ..., 'val': ..., 'test': ...}.

    In both modes, val/test loaders use originals only (aug_index == 0).
    """
    if train_recording_ids is not None:
        train_ds = HFLungDataset(
            manifest_path=manifest_path,
            recording_ids=train_recording_ids,
            include_augmented=True,
        )
    else:
        train_ds = HFLungDataset(
            manifest_path=manifest_path,
            split='train',
            include_augmented=True,
        )

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
    }

    if val_recording_ids is not None:
        val_ds = HFLungDataset(
            manifest_path=manifest_path,
            recording_ids=val_recording_ids,
            include_augmented=False,
        )
        loaders['val'] = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    test_ds = HFLungDataset(
        manifest_path=manifest_path,
        split='test',
        include_augmented=False,
    )
    loaders['test'] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return loaders
