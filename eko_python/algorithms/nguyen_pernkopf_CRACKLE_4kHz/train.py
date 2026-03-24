"""
train.py

Training loop for binary crackle detection.

Identical structure to the CLEAN_4kHz train.py except the evaluation metric
is replaced with binary crackle metrics:

    score = (Se_crackle + Sp_crackle) / 2

    Se_crackle : sensitivity for the crackle class
                 (fraction of true crackle cycles correctly identified)
    Sp_crackle : specificity for the crackle class
                 (fraction of true no_crackle cycles correctly rejected)

This mirrors the structure of the ICBHI score (mean of Se and Sp) but applied
to the single crackle class, making cross-pipeline comparisons straightforward.

Usage
-----
    python train.py               # 5-fold cross-validation
    python train.py --mode full   # full training on all train data
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
    OVERFIT_GAP_THRESHOLD,
    OVERFIT_GAP_PATIENCE,
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

CRACKLE_IDX = 1   # index of the positive class


# ---------------------------------------------------------------------------
# Binary crackle evaluation metric
# ---------------------------------------------------------------------------

def compute_crackle_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute binary crackle detection metrics.

    Returns a dict with keys:
        score, sensitivity, specificity, f1, per_class
    """
    per_class = {}
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        tn = np.sum((y_pred != c) & (y_true != c))

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[IDX_TO_LABEL[c]] = {
            'sensitivity': se, 'specificity': sp,
            'tp': int(tp), 'fn': int(fn),
            'fp': int(fp), 'tn': int(tn),
            'support': int(tp + fn),
        }

    se_crackle = per_class['crackle']['sensitivity']
    sp_crackle = per_class['crackle']['specificity']
    score      = (se_crackle + sp_crackle) / 2.0

    tp = per_class['crackle']['tp']
    fp = per_class['crackle']['fp']
    fn = per_class['crackle']['fn']
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        'score':       score,
        'sensitivity': se_crackle,
        'specificity': sp_crackle,
        'f1':          f1,
        'per_class':   per_class,
    }


@torch.no_grad()
def evaluate(
    model:   CoTuningModel,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
    loss_fn: CoTuningLoss | None = None,
) -> dict:
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

    metrics = compute_crackle_score(
        y_true=np.concatenate(all_labels),
        y_pred=np.concatenate(all_preds),
    )

    if loss_fn is not None:
        metrics['total_loss'] = float(np.mean(total_losses))
        metrics['ce_loss']    = float(np.mean(ce_losses))
        metrics['kl_loss']    = float(np.mean(kl_losses))

    return metrics


def print_eval_report(metrics: dict, split: str = 'Validation') -> None:
    print(f"\n  {split.upper()} — Score {metrics['score']:.4f}  "
          f"(Se {metrics['sensitivity']:.4f}  Sp {metrics['specificity']:.4f}  "
          f"F1 {metrics['f1']:.4f})")
    print(f"  {'Class':<12} {'Se':>7} {'Sp':>7} {'Support':>9}")
    print(f"  {'-'*39}")
    for name, m in metrics['per_class'].items():
        print(f"  {name:<12} {m['sensitivity']:>7.4f} {m['specificity']:>7.4f} {m['support']:>9}")


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
    print(f"\n{'='*60}\n{run_name.upper()}\n{'='*60}")

    loaders = get_dataloaders(
        manifest_path=MANIFEST_PATH,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        train_patient_ids=train_patient_ids,
        val_patient_ids=val_patient_ids,
    )
    train_loader  = loaders['train']
    val_loader    = loaders.get('val')
    test_loader   = loaders['test']
    is_full_train = val_loader is None
    eval_loader   = val_loader or test_loader
    eval_name     = 'Val' if val_loader else 'Test'

    class_weights = train_loader.dataset.get_class_weights()
    model, loss_fn = build_model(class_weights=class_weights, device=device)

    optimiser = optim.Adam(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=num_epochs, eta_min=learning_rate * 0.01,
    )

    run_dir      = Path(CHECKPOINTS_DIR) / run_name
    best_path    = run_dir / 'best.pt'
    resume_path  = run_dir / 'resume.pt'
    history_path = run_dir / 'history.csv'
    run_dir.mkdir(parents=True, exist_ok=True)

    best_score     = 0.0
    best_epoch     = 0
    history        = []
    start_epoch    = 1
    overfit_streak = 0

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

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, loss_fn, optimiser, train_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        if not is_full_train:
            eval_metrics = evaluate(model, eval_loader, device, loss_fn=loss_fn)
            score    = eval_metrics['score']
            val_loss = eval_metrics.get('total_loss', float('nan'))

            if epoch % 5 == 0 or epoch == 1:
                train_eval_metrics = evaluate(model, train_loader, device)
                train_score = train_eval_metrics['score']
                gap = train_score - score
                if gap > OVERFIT_GAP_THRESHOLD:
                    overfit_streak += 1
                else:
                    overfit_streak = 0
            else:
                train_score = float('nan')

            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Tr loss {train_metrics['total_loss']:.4f}  "
                f"Val loss {val_loss:.4f}  "
                f"Tr score {train_score:.4f}  "
                f"Val score {score:.4f} "
                f"(Se {eval_metrics['sensitivity']:.4f} "
                f"Sp {eval_metrics['specificity']:.4f} "
                f"F1 {eval_metrics['f1']:.4f})  "
                f"LR {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )

            row = {
                'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'elapsed': elapsed,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                'train_score':   train_score,
                'val_total_loss': val_loss,
                'val_score':     score,
                'val_se':        eval_metrics['sensitivity'],
                'val_sp':        eval_metrics['specificity'],
                'val_f1':        eval_metrics['f1'],
            }
            history.append(row)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(model, optimiser, epoch=epoch, score=score, path=best_path)
                print(f"  ✓ New best Val score {score:.4f} — checkpoint saved.")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.")
                break

            if overfit_streak >= OVERFIT_GAP_PATIENCE:
                print(
                    f"  Early stopping: train−val gap > {OVERFIT_GAP_THRESHOLD:.2f} "
                    f"for {OVERFIT_GAP_PATIENCE} consecutive train-eval epochs "
                    f"(gap = {train_score - score:.3f})."
                )
                break

        else:
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
        test_metrics = evaluate(model, test_loader, device, loss_fn=loss_fn)
        print(f"\n{run_name} complete.")
        print_eval_report(test_metrics, split='Test')
        best_score = test_metrics['score']
    else:
        print(f"\n{run_name} complete — best Val score {best_score:.4f} at epoch {best_epoch}.")

    if history:
        with open(history_path, 'w', newline='') as f:
            fieldnames = list(dict.fromkeys(k for row in history for k in row.keys()))
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k, '') for k in fieldnames})

    if resume_path.exists():
        resume_path.unlink()

    return {'best_score': best_score, 'best_epoch': best_epoch, 'checkpoint': str(best_path)}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    num_epochs: int = NUM_EPOCHS,
    patience:   int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    num_folds:  int = NUM_FOLDS,
    start_fold: int = 0,
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
    print("CROSS-VALIDATION  (binary crackle detection)")
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
    print(f"\n  Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
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
    print("FULL TRAINING  (binary crackle detection)")
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
    parser.add_argument('--mode',       default='cv', choices=['cv', 'full'])
    parser.add_argument('--epochs',     type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--folds',      type=int, default=NUM_FOLDS)
    parser.add_argument('--start-fold', type=int, default=0)
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
