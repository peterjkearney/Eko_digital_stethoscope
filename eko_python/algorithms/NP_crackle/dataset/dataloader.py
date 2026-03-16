"""
dataloader.py

Constructs PyTorch DataLoaders for the ICBHI ALSC task.

Handles:
    - Patient-level k-fold cross-validation splits for training
    - Separation of official train and test sets
    - Weighted sampling to address class imbalance during training
    - DataLoader configuration (batch size, workers, pinning)

Typical usage:

    # For a specific fold during cross-validation
    train_loader, val_loader = get_fold_dataloaders(fold=0)

    # For final training on all official train data before test evaluation
    train_loader = get_full_train_dataloader()

    # For evaluation on the official test set
    test_loader = get_test_dataloader()
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config_c import (
    MANIFEST_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_FOLDS,
    RANDOM_SEED,
)

from dataset.icbhi_dataset import ICBHIDataset


# ---------------------------------------------------------------------------
# Patient-level fold generation
# ---------------------------------------------------------------------------

def get_patient_folds(
    manifest_path: str = MANIFEST_PATH,
    n_folds: int = NUM_FOLDS,
    random_seed: int = RANDOM_SEED,
) -> list[tuple[list[int], list[int]]]:
    """
    Generate patient-level k-fold splits from the official training set.

    Splitting is done at the patient level — all cycles from a given patient
    appear in either the train fold or the validation fold, never both.

    Parameters
    ----------
    manifest_path : path to manifest CSV
    n_folds       : number of folds
    random_seed   : random seed for reproducibility

    Returns
    -------
    List of (train_patient_ids, val_patient_ids) tuples, one per fold.
    """
    manifest = pd.read_csv(manifest_path)

    # Get unique patients from the official training set only
    train_patients = (
        manifest[manifest['split'] == 'train']['patient_id']
        .unique()
    )
    train_patients = np.sort(train_patients)

    kfold  = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds  = []

    for train_idx, val_idx in kfold.split(train_patients):
        train_ids = train_patients[train_idx].tolist()
        val_ids   = train_patients[val_idx].tolist()
        folds.append((train_ids, val_ids))

    return folds


# ---------------------------------------------------------------------------
# Weighted sampler for class imbalance
# ---------------------------------------------------------------------------

def get_weighted_sampler(dataset: ICBHIDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that upsamples minority classes.

    Each sample is assigned a weight inversely proportional to its class
    frequency, so all classes are seen approximately equally often per epoch
    regardless of their raw counts in the dataset.

    Parameters
    ----------
    dataset : ICBHIDataset instance

    Returns
    -------
    WeightedRandomSampler instance.
    """
    class_weights = dataset.get_class_weights()  # shape: (4,)

    # Assign per-sample weights based on class
 
    sample_weights = torch.zeros(len(dataset))

    for idx in range(len(dataset)):
        label_idx              = int(dataset.manifest.iloc[idx]['crackle'])
        sample_weights[idx]    = class_weights[label_idx]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader constructors
# ---------------------------------------------------------------------------

def get_fold_dataloaders(
    fold: int,
    folds: list[tuple[list[int], list[int]]],
    manifest_path: str = MANIFEST_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
    train_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """
    Construct train and validation DataLoaders for a specific fold.

    Accepts a pre-computed folds list from get_patient_folds() so that
    fold generation is not repeated on every call.

    The validation set uses augment=False so results are deterministic and
    comparable across epochs. The training set uses augment=True.

    Parameters
    ----------
    fold                 : fold index (0 to len(folds) - 1)
    folds                : pre-computed list of (train_ids, val_ids) tuples
                           from get_patient_folds()
    manifest_path        : path to manifest CSV
    batch_size           : number of samples per batch
    num_workers          : number of DataLoader worker processes
    use_weighted_sampler : if True, use WeightedRandomSampler for the train
                           loader to address class imbalance. If False,
                           shuffle randomly.

    Returns
    -------
    (train_loader, val_loader) tuple of DataLoader instances.
    """
    if fold < 0 or fold >= len(folds):
        raise ValueError(f"fold must be between 0 and {len(folds) - 1}, got {fold}.")

    train_patient_ids, val_patient_ids = folds[fold]

    print(f"Fold {fold}: {len(train_patient_ids)} train patients, "
          f"{len(val_patient_ids)} val patients.")

    # Training dataset — augmentations on
    train_dataset = ICBHIDataset(
        manifest_path=manifest_path,
        patient_ids=train_patient_ids,
        split='train',
        augment=True,
        fraction=train_fraction,
    )

    # Validation dataset — augmentations off
    val_dataset = ICBHIDataset(
        manifest_path=manifest_path,
        patient_ids=val_patient_ids,
        split='train',
        augment=False,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # Verify no patient overlap
    train_patients = set(train_dataset.get_patient_ids())
    val_patients   = set(val_dataset.get_patient_ids())
    overlap        = train_patients & val_patients
    if overlap:
        raise RuntimeError(
            f"Patient overlap detected between train and val sets: {overlap}"
        )

    # Build train loader
    if use_weighted_sampler:
        sampler      = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # Build validation loader — no shuffling, no sampler
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_full_train_dataloader(
    manifest_path: str = MANIFEST_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
) -> DataLoader:
    """
    Construct a DataLoader over the entire official training set.

    Used for final training after hyperparameter tuning via cross-validation,
    before evaluating once on the official test set.

    Parameters
    ----------
    manifest_path        : path to manifest CSV
    batch_size           : number of samples per batch
    num_workers          : number of DataLoader worker processes
    use_weighted_sampler : if True, use WeightedRandomSampler

    Returns
    -------
    DataLoader over all official training cycles.
    """
    dataset = ICBHIDataset(
        manifest_path=manifest_path,
        patient_ids=None,
        split='train',
        augment=True,
    )

    print(f"Full train dataset: {len(dataset)} samples.")

    if use_weighted_sampler:
        sampler = get_weighted_sampler(dataset)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return loader


def get_test_dataloader(
    manifest_path: str = MANIFEST_PATH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Construct a DataLoader over the official test set.

    No augmentation, no shuffling, no weighted sampling. This loader should
    only be used for final evaluation — never during training or
    hyperparameter tuning.

    Parameters
    ----------
    manifest_path : path to manifest CSV
    batch_size    : number of samples per batch
    num_workers   : number of DataLoader worker processes

    Returns
    -------
    DataLoader over all official test cycles.
    """
    dataset = ICBHIDataset(
        manifest_path=manifest_path,
        patient_ids=None,
        split='test',
        augment=False,
    )

    print(f"Test dataset: {len(dataset)} samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return loader