"""
train.py

Training loop for multi-label DAS/CAS lung sound classification on HF_Lung_V1.

Evaluation metric
-----------------
Primary metric: mean AUC-ROC across DAS and CAS labels.
AUC is threshold-independent, making it suitable for the imbalanced label
distributions in this dataset (~15% DAS positive, ~13% CAS positive).
Per-label F1 at threshold=0.5 is also reported for interpretability.

Modes
-----
cv   (default) — k-fold cross-validation using GroupKFold on patient-date
                 groups. All recordings from the same patient-day are kept
                 in the same fold, matching the grouping principle used to
                 define the HF_Lung_V1 train/test split.
                 Reports mean ± std AUC across folds; best fold checkpoint
                 is retained.

full           — Train on all train-split data. Evaluate on the official
                 test split once at the end. Use this after CV to produce
                 the final model.

Usage
-----
    python train.py                          # 5-fold CV with default config
    python train.py --mode full              # full training
    python train.py --folds 3 --epochs 30
    python train.py --start-fold 2           # resume CV from fold 2 (0-indexed)
"""

import csv
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    WINDOWS_MANIFEST_PATH,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    CHECKPOINTS_DIR,
    NUM_FOLDS,
)
from dataset import get_dataloaders, get_train_groups, LABEL_COLS
from model import (
    LungSoundClassifier,
    MultiLabelBCELoss,
    build_model,
    get_device,
    save_checkpoint,
    load_checkpoint,
    print_model_summary,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:   LungSoundClassifier,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
    loss_fn: MultiLabelBCELoss | None = None,
) -> dict:
    """
    Run inference over loader and return per-label AUC, F1, and optional loss.

    Returns a dict with keys:
        mean_auc, das_auc, cas_auc, das_f1, cas_f1,
        total_loss, das_loss, cas_loss  (if loss_fn provided)
    """
    model.eval()
    all_probs  = []
    all_labels = []
    total_losses, das_losses, cas_losses = [], [], []

    for x, labels in loader:
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(x)
        probs  = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if loss_fn is not None:
            total, das_l, cas_l = loss_fn(logits, labels)
            total_losses.append(total.item())
            das_losses.append(das_l.item())
            cas_losses.append(cas_l.item())

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 2)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 2)

    metrics = {}
    aucs = []
    for i, col in enumerate(LABEL_COLS):
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        f1  = f1_score(all_labels[:, i], (all_probs[:, i] >= 0.5).astype(int),
                       zero_division=0)
        metrics[f'{col}_auc'] = float(auc)
        metrics[f'{col}_f1']  = float(f1)
        aucs.append(auc)

    metrics['mean_auc'] = float(np.mean(aucs))

    if loss_fn is not None:
        metrics['total_loss'] = float(np.mean(total_losses))
        metrics['das_loss']   = float(np.mean(das_losses))
        metrics['cas_loss']   = float(np.mean(cas_losses))

    return metrics


def print_eval_report(metrics: dict, split: str = 'Val') -> None:
    print(f"\n  {split.upper()} — mean AUC {metrics['mean_auc']:.4f}")
    print(f"  {'Label':<6} {'AUC':>7} {'F1':>7}")
    print(f"  {'-'*22}")
    for col in LABEL_COLS:
        print(f"  {col:<6} {metrics[f'{col}_auc']:>7.4f} {metrics[f'{col}_f1']:>7.4f}")
    if 'total_loss' in metrics:
        print(f"  Loss: total={metrics['total_loss']:.4f}  "
              f"das={metrics['das_loss']:.4f}  cas={metrics['cas_loss']:.4f}")


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     LungSoundClassifier,
    loss_fn:   MultiLabelBCELoss,
    optimiser: torch.optim.Optimizer,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
) -> dict:
    model.train()
    total_losses, das_losses, cas_losses = [], [], []
    n_batches = len(loader)

    for batch_idx, (x, labels) in enumerate(loader):
        x      = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(x)
        total, das_l, cas_l = loss_fn(logits, labels)
        total.backward()
        optimiser.step()

        total_losses.append(total.item())
        das_losses.append(das_l.item())
        cas_losses.append(cas_l.item())

        print(f'\r  batch {batch_idx + 1}/{n_batches}', end='', flush=True)
    print()

    return {
        'total_loss': float(np.mean(total_losses)),
        'das_loss':   float(np.mean(das_losses)),
        'cas_loss':   float(np.mean(cas_losses)),
    }


# ---------------------------------------------------------------------------
# Single training run (one fold or full training)
# ---------------------------------------------------------------------------

def train_run(
    run_name:            str,
    device:              torch.device,
    train_recording_ids: list[str] | None = None,
    val_recording_ids:   list[str] | None = None,
    num_epochs:          int   = NUM_EPOCHS,
    patience:            int   = EARLY_STOPPING_PATIENCE,
    batch_size:          int   = BATCH_SIZE,
    learning_rate:       float = LEARNING_RATE,
    weight_decay:        float = WEIGHT_DECAY,
    pretrained:          bool  = True,
) -> dict:
    """
    Train for one fold or a full run and return the best result.

    train_recording_ids=None  → use all train-split recordings (full mode)
    val_recording_ids=None    → no val loader; evaluate on test at the end
    """
    print(f"\n{'='*60}\n{run_name.upper()}\n{'='*60}")

    loaders = get_dataloaders(
        manifest_path=WINDOWS_MANIFEST_PATH,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        train_recording_ids=train_recording_ids,
        val_recording_ids=val_recording_ids,
    )
    is_full_train = 'val' not in loaders
    eval_loader   = loaders.get('val') or loaders['test']
    eval_name     = 'Val' if not is_full_train else 'Test'

    pos_weight = loaders['train'].dataset.get_pos_weights()
    print(f"  Pos weights — das: {pos_weight[0]:.2f}  cas: {pos_weight[1]:.2f}")
    print(f"  Train: {len(loaders['train'].dataset)} windows  |  "
          + (f"Val: {len(loaders['val'].dataset)} windows" if not is_full_train
             else f"Test: {len(loaders['test'].dataset)} windows"))

    model, loss_fn = build_model(
        pretrained=pretrained,
        pos_weight=pos_weight,
        device=device,
    )

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

    best_score  = 0.0
    best_epoch  = 0
    history     = []
    start_epoch = 1

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
              f"(best so far: mean AUC {best_score:.4f} at epoch {best_epoch})")

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        # Resample augmented copies so each epoch sees a different random
        # augmented version of every window (original + 1 random copy).
        loaders['train'].dataset.resample_augmented(seed=epoch)
        train_metrics = train_one_epoch(model, loss_fn, optimiser,
                                        loaders['train'], device)
        scheduler.step()
        elapsed = time.time() - t0

        if not is_full_train:
            val_metrics = evaluate(model, eval_loader, device, loss_fn=loss_fn)
            score = val_metrics['mean_auc']
            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Tr loss {train_metrics['total_loss']:.4f}  "
                f"Val loss {val_metrics['total_loss']:.4f}  "
                f"Val AUC {score:.4f} "
                f"(DAS {val_metrics['das_auc']:.4f}  CAS {val_metrics['cas_auc']:.4f})  "
                f"LR {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )
            row = {
                'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'elapsed': elapsed,
                'train_loss': train_metrics['total_loss'],
                'train_das_loss': train_metrics['das_loss'],
                'train_cas_loss': train_metrics['cas_loss'],
                'val_loss': val_metrics['total_loss'],
                'val_mean_auc': score,
                'val_das_auc': val_metrics['das_auc'],
                'val_cas_auc': val_metrics['cas_auc'],
                'val_das_f1':  val_metrics['das_f1'],
                'val_cas_f1':  val_metrics['cas_f1'],
            }
            history.append(row)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                save_checkpoint(model, optimiser, epoch=epoch, score=score,
                                path=best_path)
                print(f"  New best {eval_name} mean AUC {score:.4f} — checkpoint saved.")

            if epoch - best_epoch >= patience:
                print(f"  Early stopping: no improvement for {patience} epochs.")
                break
        else:
            print(
                f"Epoch {epoch:>3}/{num_epochs}  "
                f"Tr loss {train_metrics['total_loss']:.4f}  "
                f"LR {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )
            row = {
                'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'elapsed': elapsed,
                'train_loss': train_metrics['total_loss'],
                'train_das_loss': train_metrics['das_loss'],
                'train_cas_loss': train_metrics['cas_loss'],
            }
            history.append(row)
            if epoch % 10 == 0 or epoch == num_epochs:
                save_checkpoint(model, optimiser, epoch=epoch, score=0.0,
                                path=best_path)

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict':     model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score, 'best_epoch': best_epoch,
                'history': history,
            }, resume_path)

    # Final evaluation on the appropriate split
    print(f"\n{run_name} complete.")
    if is_full_train:
        print("Loading final checkpoint for test evaluation...")
        load_checkpoint(best_path, model, device=device)
        test_metrics = evaluate(model, loaders['test'], device, loss_fn=loss_fn)
        print_eval_report(test_metrics, split='Test')
        best_score = test_metrics['mean_auc']
    else:
        print(f"Best val mean AUC {best_score:.4f} at epoch {best_epoch}.")

    if history:
        with open(history_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    if resume_path.exists():
        resume_path.unlink()

    return {'best_score': best_score, 'best_epoch': best_epoch,
            'checkpoint': str(best_path)}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    num_epochs:  int  = NUM_EPOCHS,
    patience:    int  = EARLY_STOPPING_PATIENCE,
    batch_size:  int  = BATCH_SIZE,
    num_folds:   int  = NUM_FOLDS,
    start_fold:  int  = 0,
    pretrained:  bool = True,
) -> dict:
    device = get_device()

    recording_ids, patient_dates = get_train_groups()
    unique_dates = len(set(patient_dates))

    print("=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    print(f"  Folds:          {num_folds}")
    print(f"  Epochs:         {num_epochs}  (patience={patience})")
    print(f"  Batch size:     {batch_size}")
    print(f"  Train recordings: {len(recording_ids)}")
    print(f"  Patient-date groups: {unique_dates}")
    print(f"  Device:         {device}")
    print("=" * 60)

    # Print model summary once before training
    if start_fold == 0:
        model, _ = build_model(device=device)
        print_model_summary(model)
        del model

    gkf = GroupKFold(n_splits=num_folds)
    folds = [
        (
            [recording_ids[i] for i in train_idx],
            [recording_ids[i] for i in val_idx],
        )
        for train_idx, val_idx in gkf.split(recording_ids, groups=patient_dates)
    ]

    results = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        if fold < start_fold:
            continue
        n_train_dates = len(set(patient_dates[i]
                                for i in range(len(recording_ids))
                                if recording_ids[i] in set(train_ids)))
        n_val_dates = unique_dates - n_train_dates
        print(f"\n  Fold {fold + 1}: {len(train_ids)} train recordings "
              f"({n_train_dates} dates)  |  "
              f"{len(val_ids)} val recordings ({n_val_dates} dates)")

        result = train_run(
            run_name=f'fold_{fold + 1}',
            device=device,
            train_recording_ids=train_ids,
            val_recording_ids=val_ids,
            num_epochs=num_epochs,
            patience=patience,
            batch_size=batch_size,
            pretrained=pretrained,
        )
        results.append(result)

    scores    = [r['best_score'] for r in results]
    best_fold = int(np.argmax(scores))

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  Fold {i + 1 + start_fold}: mean AUC {r['best_score']:.4f}  "
              f"(epoch {r['best_epoch']})")
    print(f"\n  Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Best fold:  {best_fold + 1 + start_fold}  "
          f"({scores[best_fold]:.4f})")
    print(f"  Checkpoint: {results[best_fold]['checkpoint']}")
    print("=" * 60)

    return {'results': results,
            'mean_auc': float(np.mean(scores)),
            'std_auc':  float(np.std(scores))}


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------

def run_full_training(
    num_epochs: int  = NUM_EPOCHS,
    patience:   int  = EARLY_STOPPING_PATIENCE,
    batch_size: int  = BATCH_SIZE,
    pretrained: bool = True,
) -> dict:
    device = get_device()
    print("=" * 60)
    print("FULL TRAINING (all train data → test evaluation)")
    print("=" * 60)

    model, _ = build_model(device=device)
    print_model_summary(model)
    del model

    return train_run(
        run_name='full',
        device=device,
        train_recording_ids=None,
        val_recording_ids=None,
        num_epochs=num_epochs,
        patience=patience,
        batch_size=batch_size,
        pretrained=pretrained,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',         default='cv', choices=['cv', 'full'],
                        help="'cv' for cross-validation, 'full' for full training")
    parser.add_argument('--epochs',       type=int,   default=NUM_EPOCHS)
    parser.add_argument('--batch-size',   type=int,   default=BATCH_SIZE)
    parser.add_argument('--folds',        type=int,   default=NUM_FOLDS)
    parser.add_argument('--start-fold',   type=int,   default=0,
                        help='Skip earlier folds to resume a partial CV run (0-indexed)')
    parser.add_argument('--patience',     type=int,   default=EARLY_STOPPING_PATIENCE)
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train backbone from scratch (no ImageNet weights)')
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if args.mode == 'cv':
        run_cross_validation(
            num_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_folds=args.folds,
            start_fold=args.start_fold,
            pretrained=not args.no_pretrained,
        )
    else:
        run_full_training(
            num_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            pretrained=not args.no_pretrained,
        )
