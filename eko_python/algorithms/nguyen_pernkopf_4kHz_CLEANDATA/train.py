"""
train.py

Training loop for the ICBHI lung sound classification model.

Supports two modes:
  - Cross-validation: k-fold patient-level CV over the official train split,
    tracking ICBHI score on the held-out val fold each epoch.
  - Full training: train on all train-split data, evaluate on the official
    test split. Use this after CV to produce the final model.

ICBHI score
-----------
The official evaluation metric is the average of mean sensitivity and mean
specificity across all 4 classes:

    ICBHI score = (mean_sensitivity + mean_specificity) / 2

This macro-averaged metric treats all classes equally regardless of frequency,
appropriate given ICBHI's severe class imbalance.

Usage
-----
    # 5-fold cross-validation
    python train.py

    # Full training on all train data, evaluate on test
    python train.py --mode full
"""

import csv
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    MANIFEST_PATH,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_FOLDS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    CHECKPOINTS_DIR,
    NUM_CLASSES,
)
from dataset import get_dataloaders, IDX_TO_LABEL
from model import CoTuningModel, CoTuningLoss, build_model, get_device, save_checkpoint, print_model_summary

import pandas as pd


# ---------------------------------------------------------------------------
# ICBHI evaluation metric
# ---------------------------------------------------------------------------

def compute_icbhi_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute ICBHI score and per-class sensitivity / specificity.

    Returns a dict with keys:
        icbhi_score, mean_sensitivity, mean_specificity, per_class
    """
    sensitivities = []
    specificities = []
    per_class     = {}

    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        tn = np.sum((y_pred != c) & (y_true != c))

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivities.append(se)
        specificities.append(sp)
        per_class[IDX_TO_LABEL[c]] = {
            'sensitivity': se, 'specificity': sp,
            'tp': int(tp), 'fn': int(fn),
            'fp': int(fp), 'tn': int(tn),
            'support': int(tp + fn),
        }

    mean_se     = float(np.mean(sensitivities))
    mean_sp     = float(np.mean(specificities))
    icbhi_score = (mean_se + mean_sp) / 2.0

    return {
        'icbhi_score':      icbhi_score,
        'mean_sensitivity': mean_se,
        'mean_specificity': mean_sp,
        'per_class':        per_class,
    }


@torch.no_grad()
def evaluate(
    model:   CoTuningModel,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
    loss_fn: CoTuningLoss | None = None,
) -> dict:
    """Run inference over loader and return ICBHI metrics and optional loss."""
    model.eval()
    all_preds, all_labels = [], []
    total_losses, ce_losses, kl_losses = [], [], []

    with torch.no_grad():
        for x, labels in loader:
            x      = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, prior, _ = model(x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if loss_fn is not None:
                total, ce, kl = loss_fn(logits, prior, labels)
                total_losses.append(total.item())
                ce_losses.append(ce.item())
                kl_losses.append(kl.item())

    metrics = compute_icbhi_score(
        y_true=np.concatenate(all_labels),
        y_pred=np.concatenate(all_preds),
    )

    if loss_fn is not None:
        metrics['total_loss'] = float(np.mean(total_losses))
        metrics['ce_loss']    = float(np.mean(ce_losses))
        metrics['kl_loss']    = float(np.mean(kl_losses))

    return metrics


def print_eval_report(metrics: dict, split: str = 'Validation') -> None:
    print(f"\n  {split.upper()} — ICBHI {metrics['icbhi_score']:.4f}  "
          f"(Se {metrics['mean_sensitivity']:.4f}  Sp {metrics['mean_specificity']:.4f})")
    print(f"  {'Class':<10} {'Se':>7} {'Sp':>7} {'Support':>9}")
    print(f"  {'-'*37}")
    for name, m in metrics['per_class'].items():
        print(f"  {name:<10} {m['sensitivity']:>7.4f} {m['specificity']:>7.4f} {m['support']:>9}")


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     CoTuningModel,
    loss_fn:   CoTuningLoss,
    optimiser: torch.optim.Optimizer,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
) -> dict:
    model.train()
    total_losses, ce_losses, kl_losses = [], [], []

    n_batches = len(loader)
    for batch_idx, (x, labels) in enumerate(loader):
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits, prior, _ = model(x)
        total, ce, kl    = loss_fn(logits, prior, labels)
        total.backward()
        optimiser.step()

        total_losses.append(total.item())
        ce_losses.append(ce.item())
        kl_losses.append(kl.item())

        print(f'\r  batch {batch_idx + 1}/{n_batches}', end='', flush=True)
    print()

    return {
        'total_loss': float(np.mean(total_losses)),
        'ce_loss':    float(np.mean(ce_losses)),
        'kl_loss':    float(np.mean(kl_losses)),
    }


# ---------------------------------------------------------------------------
# Single fold / full-training run
# ---------------------------------------------------------------------------

def train_run(
    train_patient_ids: list[int] | None,
    val_patient_ids:   list[int] | None,
    run_name:          str,
    device:            torch.device,
    num_epochs:        int   = NUM_EPOCHS,
    patience:          int   = EARLY_STOPPING_PATIENCE,
    batch_size:        int   = BATCH_SIZE,
    learning_rate:     float = LEARNING_RATE,
    weight_decay:      float = WEIGHT_DECAY,
) -> dict:
    """
    Train for one fold (or full training run) and return the best result.

    Parameters
    ----------
    train_patient_ids : patient IDs for training; None = full train split
    val_patient_ids   : patient IDs for validation; None = use test split
    run_name          : label used for checkpoint directory (e.g. 'fold_1')
    """
    print(f"\n{'='*60}\n{run_name.upper()}\n{'='*60}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    loaders = get_dataloaders(
        manifest_path=MANIFEST_PATH,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        train_patient_ids=train_patient_ids,
        val_patient_ids=val_patient_ids,
    )
    train_loader  = loaders['train']
    val_loader    = loaders.get('val')       # None in full-training mode
    test_loader   = loaders['test']
    is_full_train = val_loader is None       # no val split → full training run
    eval_loader   = val_loader or test_loader
    eval_name     = 'Val' if val_loader else 'Test'

    # ── Model ─────────────────────────────────────────────────────────────
    class_weights = train_loader.dataset.get_class_weights()
    model, loss_fn = build_model(class_weights=class_weights, device=device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimiser = optim.Adam(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs, eta_min=learning_rate * 0.01,
    )

    # ── Checkpoint paths ──────────────────────────────────────────────────
    run_dir        = Path(CHECKPOINTS_DIR) / run_name
    best_path      = run_dir / 'best.pt'
    resume_path    = run_dir / 'resume.pt'
    history_path   = run_dir / 'history.csv'
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────
    best_score       = 0.0
    best_epoch       = 0
    history          = []
    start_epoch      = 1

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimiser.load_state_dict(ckpt['optimiser_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_score  = ckpt['best_score']
        best_epoch  = ckpt['best_epoch']
        history     = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch - 1} "
              f"(best so far: {best_score:.4f} at epoch {best_epoch})")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, loss_fn, optimiser, train_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        if not is_full_train:
            # CV: evaluate on val every epoch (needed for early stopping/checkpointing).
            eval_metrics = evaluate(model, eval_loader, device, loss_fn=loss_fn)
            score    = eval_metrics['icbhi_score']
            val_loss = eval_metrics.get('total_loss', float('nan'))

            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Tr loss {train_metrics['total_loss']:.4f}  "
                f"Val loss {val_loss:.4f}  "
                f"Val ICBHI {score:.4f} "
                f"(Se {eval_metrics['mean_sensitivity']:.4f} "
                f"Sp {eval_metrics['mean_specificity']:.4f})  "
                f"LR {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )

            row = {
                'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'elapsed': elapsed,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                'val_total_loss': val_loss,
                'val_icbhi':      score,
                'val_mean_se':    eval_metrics['mean_sensitivity'],
                'val_mean_sp':    eval_metrics['mean_specificity'],
            }
            history.append(row)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(model, optimiser, epoch=epoch, score=score, path=best_path)
                print(f"  ✓ New best Val ICBHI {score:.4f} — checkpoint saved.")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.")
                break

        else:
            # Full training: no evaluation during training — just log loss.
            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Tr loss {train_metrics['total_loss']:.4f}  "
                f"LR {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )
            row = {
                'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'elapsed': elapsed,
                **{f'train_{k}': v for k, v in train_metrics.items()},
            }
            history.append(row)
            if epoch % 10 == 0 or epoch == num_epochs:
                save_checkpoint(model, optimiser, epoch=epoch, score=0.0, path=best_path)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score, 'best_epoch': best_epoch, 'history': history,
            }, resume_path)

    if is_full_train:
        # Evaluate on test set once, at the end — never touched during training
        test_metrics = evaluate(model, test_loader, device, loss_fn=loss_fn)
        print(f"\n{run_name} complete.")
        print_eval_report(test_metrics, split='Test')
        best_score = test_metrics['icbhi_score']
    else:
        print(f"\n{run_name} complete — best Val ICBHI {best_score:.4f} at epoch {best_epoch}.")

    # Save history CSV
    if history:
        with open(history_path, 'w', newline='') as f:
            fieldnames = list(dict.fromkeys(k for row in history for k in row.keys()))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k, '') for k in fieldnames})

    # Clean up resume checkpoint
    if resume_path.exists():
        resume_path.unlink()

    return {'best_score': best_score, 'best_epoch': best_epoch, 'checkpoint': str(best_path)}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    num_epochs: int   = NUM_EPOCHS,
    patience:   int   = EARLY_STOPPING_PATIENCE,
    batch_size: int   = BATCH_SIZE,
    num_folds:  int   = NUM_FOLDS,
    start_fold: int   = 0,
) -> dict:
    device = get_device()

    manifest     = pd.read_csv(MANIFEST_PATH)
    train_rows   = manifest[(manifest['split'] == 'train') & (manifest['aug_index'] == 0)]
    all_patients = sorted(train_rows['patient_id'].unique().tolist())

    kf    = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_SEED)
    folds = [
        (
            [all_patients[i] for i in train_idx],
            [all_patients[i] for i in val_idx],
        )
        for train_idx, val_idx in kf.split(all_patients)
    ]

    print("=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    print(f"  Folds:         {num_folds}")
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Patients:      {len(all_patients)}")
    print(f"  Device:        {device}")
    print("=" * 60)

    if start_fold == 0:
        model, _ = build_model(device=device)
        print_model_summary(model)
        del model

    results = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        if fold < start_fold:
            continue
        result = train_run(
            train_patient_ids=train_ids,
            val_patient_ids=val_ids,
            run_name=f'fold_{fold + 1}',
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            batch_size=batch_size,
        )
        results.append(result)

    scores    = [r['best_score'] for r in results]
    best_fold = int(np.argmax(scores))

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  Fold {i + 1 + start_fold}: {r['best_score']:.4f}  (epoch {r['best_epoch']})")
    print(f"\n  Mean ICBHI: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Best fold:  {best_fold + 1 + start_fold}  ({scores[best_fold]:.4f})")
    print(f"  Checkpoint: {results[best_fold]['checkpoint']}")
    print("=" * 60)

    return {'results': results, 'mean_score': np.mean(scores), 'std_score': np.std(scores)}


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_full_training(
    num_epochs: int = NUM_EPOCHS,
    patience:   int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
) -> dict:
    device = get_device()
    print("=" * 60)
    print("FULL TRAINING (all train data → test evaluation)")
    print("=" * 60)

    model, _ = build_model(device=device)
    print_model_summary(model)
    del model

    return train_run(
        train_patient_ids=None,
        val_patient_ids=None,
        run_name='full',
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       default='cv', choices=['cv', 'full'],
                        help="'cv' for cross-validation, 'full' for full training")
    parser.add_argument('--epochs',     type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--folds',      type=int, default=NUM_FOLDS)
    parser.add_argument('--start-fold', type=int, default=0,
                        help="Skip earlier folds (0-indexed) to resume a partial CV run")
    parser.add_argument('--patience',   type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    if args.mode == 'cv':
        run_cross_validation(
            num_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_folds=args.folds,
            start_fold=args.start_fold,
        )
    else:
        run_full_training(
            num_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
