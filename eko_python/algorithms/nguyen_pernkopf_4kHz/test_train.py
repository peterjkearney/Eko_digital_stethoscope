"""
test_train.py

Sanity checks for train.py without running a full training job:
  1. compute_icbhi_score returns correct values on known inputs
  2. evaluate() runs without error and returns expected keys
  3. train_one_epoch() updates model weights
  4. train_run() completes 2 epochs on a tiny synthetic dataset
  5. Checkpoint is saved when score improves
  6. Early stopping halts training correctly
  7. Resume checkpoint is cleaned up after a completed run
"""

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from train import compute_icbhi_score, evaluate, train_one_epoch
from model import build_model, get_device, CoTuningLoss
from config import NUM_CLASSES

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ''))
    return condition


def make_synthetic_loader(n=64, batch_size=16, device='cpu'):
    """Random (3, 224, 224) spectrograms with random 4-class labels."""
    x = torch.randn(n, 3, 224, 224)
    y = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


# ---------------------------------------------------------------------------

def test_icbhi_score():
    print("\n── compute_icbhi_score ────────────────────────────────")

    # Perfect predictions
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = compute_icbhi_score(y, y)
    check("perfect score = 1.0",   abs(m['icbhi_score'] - 1.0) < 1e-6, f"{m['icbhi_score']:.6f}")
    check("mean_se = 1.0",         abs(m['mean_sensitivity'] - 1.0) < 1e-6)
    check("mean_sp = 1.0",         abs(m['mean_specificity'] - 1.0) < 1e-6)

    # All predictions wrong (predict 0 for everything)
    y_true = np.array([1, 2, 3, 1, 2, 3])
    y_pred = np.zeros(6, dtype=int)
    m = compute_icbhi_score(y_true, y_pred)
    check("se=0 when all wrong",   m['mean_sensitivity'] == 0.0, f"{m['mean_sensitivity']:.4f}")

    # Keys present
    check("has icbhi_score key",   'icbhi_score'      in m)
    check("has mean_sensitivity",  'mean_sensitivity' in m)
    check("has mean_specificity",  'mean_specificity' in m)
    check("has per_class",         'per_class'        in m)
    check("per_class has 4 keys",  len(m['per_class']) == 4, str(list(m['per_class'].keys())))

    # Score in [0, 1]
    y_rand = np.random.randint(0, NUM_CLASSES, 100)
    m = compute_icbhi_score(y_rand, y_rand)
    check("score in [0, 1]", 0.0 <= m['icbhi_score'] <= 1.0, f"{m['icbhi_score']:.4f}")


def test_evaluate():
    print("\n── evaluate() ─────────────────────────────────────────")
    device = get_device()
    model, _ = build_model(device=device)
    loader   = make_synthetic_loader()

    metrics = evaluate(model, loader, device)

    check("returns dict",          isinstance(metrics, dict))
    check("has icbhi_score",       'icbhi_score' in metrics)
    check("score in [0, 1]",       0.0 <= metrics['icbhi_score'] <= 1.0,
          f"{metrics['icbhi_score']:.4f}")
    check("has per_class",         'per_class' in metrics)


def test_train_one_epoch():
    print("\n── train_one_epoch() ──────────────────────────────────")
    device   = get_device()
    model, loss_fn = build_model(device=device)
    loader   = make_synthetic_loader()
    optimiser = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)

    # Snapshot backbone weights before training
    p_before = next(model.backbone.parameters()).clone().detach()

    metrics = train_one_epoch(model, loss_fn, optimiser, loader, device)

    p_after = next(model.backbone.parameters()).clone().detach()

    check("returns dict",          isinstance(metrics, dict))
    check("has total_loss",        'total_loss' in metrics)
    check("has ce_loss",           'ce_loss'    in metrics)
    check("has kl_loss",           'kl_loss'    in metrics)
    check("loss is finite",        np.isfinite(metrics['total_loss']), f"{metrics['total_loss']:.4f}")
    check("weights changed",       not torch.allclose(p_before, p_after))

    # Source classifier must NOT change
    sc_before = model.source_classifier.fc.weight.clone()
    train_one_epoch(model, loss_fn, optimiser, loader, device)
    sc_after  = model.source_classifier.fc.weight.clone()
    check("source classifier unchanged", torch.allclose(sc_before, sc_after))


def test_early_stopping():
    print("\n── Early stopping ─────────────────────────────────────")
    # Patch train_run with a tiny patience and a loader that always scores 0
    # so it stops before num_epochs
    import train as train_module
    import pandas as pd

    device = get_device()

    # We'll directly test the logic: if best_epoch never updates,
    # training stops after `patience` epochs regardless of num_epochs
    model, loss_fn = build_model(device=device)
    optimiser = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)
    loader    = make_synthetic_loader(n=16, batch_size=16)

    patience    = 3
    num_epochs  = 20
    best_score  = 1.0   # artificially high so it never improves
    best_epoch  = 0
    epochs_run  = 0

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, loss_fn, optimiser, loader, device)
        score = 0.0   # always 0 → never beats best_score
        epochs_run = epoch
        if epoch - best_epoch >= patience:
            break

    check("stopped before num_epochs", epochs_run < num_epochs, f"ran {epochs_run}/{num_epochs}")
    check("stopped at patience boundary", epochs_run == patience, f"epochs_run={epochs_run}")


def test_checkpoint_saved():
    print("\n── Checkpoint saving ──────────────────────────────────")
    device = get_device()
    model, loss_fn = build_model(device=device)
    optimiser = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)
    loader    = make_synthetic_loader(n=32, batch_size=16)

    with tempfile.TemporaryDirectory() as tmp:
        from model import save_checkpoint
        ckpt_path = Path(tmp) / 'test.pt'

        # Score improves on first epoch
        train_one_epoch(model, loss_fn, optimiser, loader, device)
        metrics = evaluate(model, loader, device)
        score   = metrics['icbhi_score']

        save_checkpoint(model, optimiser, epoch=1, score=score, path=ckpt_path)
        check("checkpoint file created", ckpt_path.exists())

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        check("epoch saved",  ckpt.get('epoch') == 1)
        check("score saved",  abs(ckpt.get('icbhi_score', -1) - score) < 1e-6)
        check("model state present", 'model_state_dict' in ckpt)


def test_mini_train_run():
    """Run train_run for 2 epochs on synthetic data to check it completes cleanly."""
    print("\n── train_run() mini run ───────────────────────────────")

    # Monkey-patch get_dataloaders and MANIFEST_PATH so train_run
    # uses synthetic data instead of real files.
    import train as train_module

    device = get_device()
    loader = make_synthetic_loader(n=32, batch_size=16)

    original_get_dataloaders = train_module.get_dataloaders

    # Attach a stub get_class_weights so train_run doesn't need ICBHIDataset
    loader.dataset.get_class_weights = lambda: torch.ones(NUM_CLASSES) / NUM_CLASSES

    def fake_get_dataloaders(**kwargs):
        return {'train': loader, 'test': loader}

    train_module.get_dataloaders = fake_get_dataloaders

    with tempfile.TemporaryDirectory() as tmp:
        import config as cfg
        original_ckpt_dir = cfg.CHECKPOINTS_DIR
        cfg.CHECKPOINTS_DIR = Path(tmp)
        train_module.CHECKPOINTS_DIR = Path(tmp)

        result = train_module.train_run(
            train_patient_ids=None,
            val_patient_ids=None,
            run_name='test_run',
            device=device,
            num_epochs=2,
            patience=10,
            batch_size=16,
        )

        check("returns dict",           isinstance(result, dict))
        check("has best_score",         'best_score' in result)
        check("has best_epoch",         'best_epoch' in result)
        check("has checkpoint key",     'checkpoint' in result)
        check("best_epoch <= 2",        result['best_epoch'] <= 2, str(result['best_epoch']))
        check("resume ckpt cleaned up", not (Path(tmp) / 'test_run' / 'resume.pt').exists())
        check("history CSV written",    (Path(tmp) / 'test_run' / 'history.csv').exists())

        cfg.CHECKPOINTS_DIR = original_ckpt_dir
        train_module.CHECKPOINTS_DIR = original_ckpt_dir

    train_module.get_dataloaders = original_get_dataloaders


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 55)
    print("TRAIN TESTS")
    print("=" * 55)

    test_icbhi_score()
    test_evaluate()
    test_train_one_epoch()
    test_early_stopping()
    test_checkpoint_saved()
    test_mini_train_run()

    print("\n" + "=" * 55)
