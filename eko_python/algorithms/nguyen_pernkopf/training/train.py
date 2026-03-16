"""
train.py

Training loop for the ICBHI ALSC co-tuning model.

Runs k-fold cross-validation over the official training set, tracking
the ICBHI score on the validation fold at each epoch. The best checkpoint
per fold is saved to disk.

Each epoch:
    1. Train one pass over the training DataLoader
    2. Evaluate on the validation DataLoader
    3. Compute ICBHI score (average of sensitivity and specificity)
    4. Save checkpoint if ICBHI score improved

After all folds:
    - Reports mean and std of best ICBHI score across folds
    - Best fold checkpoint can be used for final test evaluation
"""

import csv
import time
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_FOLDS,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
    NUM_WORKERS,
)

from dataset.icbhi_dataset import ICBHIDataset
from dataset.dataloader import get_patient_folds, get_fold_dataloaders
from models.model import build_model, get_device, save_checkpoint, print_model_summary
from training.evaluate import evaluate


# ---------------------------------------------------------------------------
# Single epoch training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loss_fn,
    optimiser,
    loader,
    device: torch.device,
    epoch: int,
) -> dict:
    """
    Run one training epoch.

    Parameters
    ----------
    model     : CoTuningModel instance
    loss_fn   : CoTuningLoss instance
    optimiser : torch optimiser
    loader    : training DataLoader
    device    : torch device
    epoch     : current epoch number (for logging)

    Returns
    -------
    Dict with mean losses for the epoch:
        total_loss, ce_loss, kl_loss
    """
    model.train()

    total_losses = []
    ce_losses    = []
    kl_losses    = []

    for batch_idx, (spectrograms, bandwidths, labels) in enumerate(loader):

        print(f'\rbatch_idx: {batch_idx}, len(loader) = {len(loader)}', end='', flush=True)

        spectrograms = spectrograms.to(device, non_blocking=True)
        bandwidths   = bandwidths.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        optimiser.zero_grad()

        target_logits, target_prior, _ = model(spectrograms, bandwidth=bandwidths)
       
        total_loss, ce_loss, kl_loss = loss_fn(
            target_logits=target_logits,
            target_prior=target_prior,
            labels=labels,
        )
        
        total_loss.backward()
       
        optimiser.step()
       
        total_losses.append(total_loss.item())
        ce_losses.append(ce_loss.item())
        kl_losses.append(kl_loss.item())

    print() #clearining batch index line at end of epoch

    return {
        'total_loss': np.mean(total_losses),
        'ce_loss':    np.mean(ce_losses),
        'kl_loss':    np.mean(kl_losses),
    }


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold: int,
    folds: list,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    num_workers: int = NUM_WORKERS,
    train_fraction: float = 1.0,
) -> dict:
    """
    Train the model for a single cross-validation fold.

    Parameters
    ----------
    fold            : fold index
    folds           : pre-computed list of (train_ids, val_ids) from
                      get_patient_folds()
    device          : torch device
    num_epochs      : number of training epochs
    batch_size      : batch size
    learning_rate   : initial learning rate
    checkpoints_dir : directory to save checkpoints
    num_workers     : DataLoader worker processes

    Returns
    -------
    Dict with:
        best_icbhi_score : best validation ICBHI score achieved
        best_epoch       : epoch at which best score was achieved
        history          : list of per-epoch metric dicts
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1} / {len(folds)}")
    print(f"{'='*60}")

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_loader, val_loader = get_fold_dataloaders(
        fold=fold,
        folds=folds,
        batch_size=batch_size,
        num_workers=num_workers,
        train_fraction=train_fraction,
    )

    # ── Class weights from training fold ─────────────────────────────────
    train_dataset  = train_loader.dataset
    class_weights  = train_dataset.get_class_weights()

    # ── Model ────────────────────────────────────────────────────────────
    model, loss_fn = build_model(
        class_weights=class_weights,
        device=device,
    )

    if fold == 0:
        print_model_summary(model)

    print('start optimiser')
    # ── Optimiser ────────────────────────────────────────────────────────
    # Only pass trainable parameters — source classifier is excluded
    optimiser = optim.Adam(
        model.get_trainable_params(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print('start cosine annealing')
    # Cosine annealing scheduler — decays LR smoothly to near zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01,
    )

    print('start training loop')
    # ── Training loop ────────────────────────────────────────────────────
    best_icbhi_score = 0.0
    best_epoch       = 0
    history          = []
    checkpoint_path  = str(
        Path(checkpoints_dir) / f"fold_{fold}" / "best.pt"
    )
    resume_path = str(
        Path(checkpoints_dir) / f"fold_{fold}" / "resume.pt"
    )

    # Resume from periodic checkpoint if one exists for this fold
    start_epoch = 1
    if Path(resume_path).exists():
        print(f"  Resuming from {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        optimiser.load_state_dict(resume_ckpt['optimiser_state_dict'])
        scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
        best_icbhi_score = resume_ckpt['best_icbhi_score']
        best_epoch       = resume_ckpt['best_epoch']
        history          = resume_ckpt['history']
        start_epoch      = resume_ckpt['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch} "
              f"(best so far: {best_icbhi_score:.4f} at epoch {best_epoch})")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            loader=train_loader,
            device=device,
            epoch=epoch,
        )

        # Evaluate on train set (no_grad, eval mode)
        train_icbhi_metrics = evaluate(
            model=model,
            loader=train_loader,
            device=device,
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        scheduler.step()

        elapsed = time.time() - epoch_start

        # Log
        epoch_log = {
            'epoch':       epoch,
            'lr':          scheduler.get_last_lr()[0],
            'elapsed':     elapsed,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            'train_icbhi_score':      train_icbhi_metrics['icbhi_score'],
            'train_mean_sensitivity': train_icbhi_metrics['mean_sensitivity'],
            'train_mean_specificity': train_icbhi_metrics['mean_specificity'],
            **{f'val_{k}':   v for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        icbhi_score = val_metrics['icbhi_score']

        print(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"Loss {train_metrics['total_loss']:.4f} "
            f"(CE {train_metrics['ce_loss']:.4f} "
            f"KL {train_metrics['kl_loss']:.4f}) | "
            f"Tr ICBHI {train_icbhi_metrics['icbhi_score']:.4f} | "
            f"Val ICBHI {icbhi_score:.4f} "
            f"(Se {val_metrics['mean_sensitivity']:.4f} "
            f"Sp {val_metrics['mean_specificity']:.4f}) | "
            f"LR {epoch_log['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save checkpoint if improved
        if icbhi_score > best_icbhi_score:
            best_icbhi_score = icbhi_score
            best_epoch       = epoch
            save_checkpoint(
                model=model,
                optimiser=optimiser,
                epoch=epoch,
                score=icbhi_score,
                checkpoint_path=checkpoint_path,
            )
            print(f"  ✓ New best ICBHI score: {icbhi_score:.4f} — checkpoint saved.")

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"  Early stopping: no improvement for {patience} epochs.")
            break

        # Periodic resume checkpoint every 10 epochs
        if epoch % 10 == 0:
            Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_icbhi_score':     best_icbhi_score,
                'best_epoch':           best_epoch,
                'history':              history,
            }, resume_path)
            print(f"  [resume checkpoint saved at epoch {epoch}]")

    print(f"\nFold {fold + 1} complete. "
          f"Best ICBHI score: {best_icbhi_score:.4f} at epoch {best_epoch}.")

    # Clean up resume checkpoint now that the fold completed successfully
    if Path(resume_path).exists():
        Path(resume_path).unlink()

    # Save history to CSV alongside the checkpoint
    history_path = str(Path(checkpoint_path).parent / "history.csv")
    if history:
        with open(history_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"  History saved to {history_path}")

    return {
        'best_icbhi_score': best_icbhi_score,
        'best_epoch':       best_epoch,
        'history':          history,
        'checkpoint_path':  checkpoint_path,
    }


# ---------------------------------------------------------------------------
# Full cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(
    num_epochs: int      = NUM_EPOCHS,
    patience: int        = EARLY_STOPPING_PATIENCE,
    batch_size: int      = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float  = WEIGHT_DECAY,
    num_folds: int       = NUM_FOLDS,
    random_seed: int     = RANDOM_SEED,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    num_workers: int     = NUM_WORKERS,
    start_fold: int      = 0,
) -> dict:
    """
    Run k-fold cross-validation over the official training set.

    Parameters
    ----------
    num_epochs      : number of training epochs per fold
    batch_size      : batch size
    learning_rate   : initial learning rate
    num_folds       : number of cross-validation folds
    random_seed     : random seed for fold generation
    checkpoints_dir : directory to save per-fold checkpoints
    num_workers     : DataLoader worker processes
    start_fold      : first fold to train (0-indexed); skips earlier folds

    Returns
    -------
    Dict with:
        fold_results  : list of per-fold result dicts
        mean_score    : mean best ICBHI score across folds
        std_score     : std of best ICBHI score across folds
        best_fold     : fold index with highest best ICBHI score
    """
    device = get_device()

    print("="*60)
    print("ICBHI CO-TUNING CROSS-VALIDATION")
    print("="*60)
    print(f"Folds:          {num_folds}")
    print(f"Epochs:         {num_epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {learning_rate}")
    print(f"Weight decay:   {weight_decay}")
    print(f"Device:         {device}")
    print(f"Checkpoints:    {checkpoints_dir}")
    print("="*60)

    # Compute folds once
    folds = get_patient_folds(n_folds=num_folds, random_seed=random_seed)

    fold_results = []

    for fold in range(start_fold, num_folds):
        result = train_fold(
            fold=fold,
            folds=folds,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            checkpoints_dir=checkpoints_dir,
            num_workers=num_workers,
        )
        fold_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    best_scores = [r['best_icbhi_score'] for r in fold_results]
    mean_score  = np.mean(best_scores)
    std_score   = np.std(best_scores)
    best_fold   = int(np.argmax(best_scores))

    print("\n" + "="*60)
    print("CROSS-VALIDATION COMPLETE")
    print("="*60)
    for fold, result in enumerate(fold_results):
        print(f"  Fold {fold + 1}: {result['best_icbhi_score']:.4f} "
              f"(epoch {result['best_epoch']})")
    print(f"\n  Mean ICBHI score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"  Best fold:        {best_fold + 1} "
          f"({fold_results[best_fold]['best_icbhi_score']:.4f})")
    print(f"  Best checkpoint:  "
          f"{fold_results[best_fold]['checkpoint_path']}")
    print("="*60)

    return {
        'fold_results': fold_results,
        'mean_score':   mean_score,
        'std_score':    std_score,
        'best_fold':    best_fold,
    }