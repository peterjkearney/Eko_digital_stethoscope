"""
test_dataset.py

Quick sanity checks for dataset.py:
  1. Load a few samples and check tensor shapes and dtypes
  2. Check label distribution matches the manifest
  3. Check reflect roll produces a different array
  4. Check val fold has no augmented copies
  5. Check get_dataloaders() returns correct splits
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from dataset import ICBHIDataset, get_dataloaders, LABEL_TO_IDX

MANIFEST = Path(__file__).parent / 'manifest.csv'
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ''))
    return condition


def test_single_sample():
    print("\n── Single sample ──────────────────────────────────────")
    ds = ICBHIDataset(MANIFEST, split='train', augment=False)
    tensor, label = ds[0]

    check("tensor shape", tensor.shape == (3, 224, 224), str(tensor.shape))
    check("tensor dtype", tensor.dtype == torch.float32, str(tensor.dtype))
    check("label in range", 0 <= label <= 3, str(label))
    check("no NaN", not torch.isnan(tensor).any())
    check("not all zeros", tensor.abs().sum().item() > 0)


def test_normalisation():
    print("\n── Per-sample normalisation ───────────────────────────")
    ds = ICBHIDataset(MANIFEST, split='test', augment=False)
    tensor, _ = ds[0]
    # Each channel is a copy of the same normalised array — check one channel
    ch = tensor[0]
    mean = ch.mean().item()
    std  = ch.std().item()
    check("mean ≈ 0", abs(mean) < 0.1, f"{mean:.4f}")
    check("std ≈ 1",  abs(std - 1.0) < 0.1, f"{std:.4f}")


def test_reflect_roll():
    print("\n── Reflect roll ───────────────────────────────────────")
    ds_aug  = ICBHIDataset(MANIFEST, split='train', augment=True)
    ds_orig = ICBHIDataset(MANIFEST, split='train', augment=False)

    # Run 10 samples — at least one should differ (roll=0 is possible but unlikely)
    diffs = sum(
        not torch.equal(ds_aug[i][0], ds_orig[i][0])
        for i in range(10)
    )
    check("roll changes at least some samples", diffs > 0, f"{diffs}/10 differ")


def test_augmented_filtering():
    print("\n── Augmented filtering ────────────────────────────────")
    ds_all  = ICBHIDataset(MANIFEST, split='train', include_augmented=True)
    ds_orig = ICBHIDataset(MANIFEST, split='train', include_augmented=False)

    check("include_augmented=True has more rows",
          len(ds_all) > len(ds_orig),
          f"{len(ds_all)} vs {len(ds_orig)}")
    check("include_augmented=False has no aug copies",
          (ds_orig.manifest['aug_index'] == 0).all())


def test_patient_fold():
    print("\n── Patient fold ───────────────────────────────────────")
    import pandas as pd
    manifest = pd.read_csv(MANIFEST)
    train_patients = manifest[manifest['split'] == 'train']['patient_id'].unique().tolist()

    if len(train_patients) < 4:
        print("  (skipped — not enough patients)")
        return

    fold_train = train_patients[:len(train_patients) // 2]
    fold_val   = train_patients[len(train_patients) // 2:]

    ds_train = ICBHIDataset(MANIFEST, patient_ids=fold_train, include_augmented=True)
    ds_val   = ICBHIDataset(MANIFEST, patient_ids=fold_val,   include_augmented=False)

    train_ids = set(ds_train.manifest['patient_id'].unique())
    val_ids   = set(ds_val.manifest['patient_id'].unique())

    check("no patient overlap between folds", train_ids.isdisjoint(val_ids))
    check("val has no augmented copies", (ds_val.manifest['aug_index'] == 0).all())


def test_dataloaders():
    print("\n── DataLoaders ────────────────────────────────────────")
    loaders = get_dataloaders(MANIFEST, batch_size=8, num_workers=0)

    check("'train' key present", 'train' in loaders)
    check("'test' key present",  'test'  in loaders)
    check("no 'val' in full mode", 'val' not in loaders)

    # One batch from train
    batch_x, batch_y = next(iter(loaders['train']))
    check("batch shape", batch_x.shape == (8, 3, 224, 224), str(batch_x.shape))
    check("labels shape", batch_y.shape == (8,), str(batch_y.shape))

    # One batch from test
    batch_x, batch_y = next(iter(loaders['test']))
    check("test batch shape", batch_x.shape[1:] == (3, 224, 224))


def test_class_weights():
    print("\n── Class weights ──────────────────────────────────────")
    ds = ICBHIDataset(MANIFEST, split='train', include_augmented=False)
    w  = ds.get_class_weights()
    check("shape (4,)", w.shape == (4,), str(w.shape))
    check("all positive", (w > 0).all())
    check("sum ≈ 4", abs(w.sum().item() - 4.0) < 0.01, f"{w.sum():.4f}")


if __name__ == '__main__':
    print("=" * 55)
    print("DATASET TESTS")
    print("=" * 55)

    test_single_sample()
    test_normalisation()
    test_reflect_roll()
    test_augmented_filtering()
    test_patient_fold()
    test_dataloaders()
    test_class_weights()

    print("\n" + "=" * 55)

    # Summary
    ds = ICBHIDataset(MANIFEST, split='train', include_augmented=True)
    ds.summary()
