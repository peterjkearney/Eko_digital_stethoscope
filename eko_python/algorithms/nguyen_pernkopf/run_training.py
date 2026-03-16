"""
run_training.py

Single entry point for training the ICBHI ALSC co-tuning model.

Two modes:

    Cross-validation (default):
        Runs k-fold cross-validation over the official training set to
        tune hyperparameters and estimate generalisation performance.
        Reports mean ± std ICBHI score across folds.
        The best checkpoint from each fold is saved to disk.

    Final evaluation (--final):
        Trains on the full official training set, then evaluates once
        on the official test set. This should only be run after
        hyperparameters are fixed via cross-validation.

Usage:
    python run_training.py                  # cross-validation only
    python run_training.py --final          # cross-val then final eval
    python run_training.py --final-only     # final eval only (skip cross-val)
                                            # requires a checkpoint path
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_FOLDS,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
    NUM_WORKERS,
    MANIFEST_PATH,
    WEIGHT_DECAY
)

from training.train import run_cross_validation, train_one_epoch
from training.evaluate import evaluate, print_evaluation_report
from dataset.icbhi_dataset import ICBHIDataset
from dataset.dataloader import get_full_train_dataloader, get_test_dataloader
from models.model import (
    build_model,
    get_device,
    save_checkpoint,
    load_checkpoint,
    print_model_summary,
)


# ---------------------------------------------------------------------------
# Final training on full train set
# ---------------------------------------------------------------------------

def run_final_training(
    device: torch.device,
    num_epochs: int      = NUM_EPOCHS,
    batch_size: int      = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    num_workers: int     = NUM_WORKERS,
    weight_decay:float   = WEIGHT_DECAY
) -> str:
    """
    Train on the full official training set and save the final checkpoint.

    Parameters
    ----------
    device          : torch device
    num_epochs      : number of training epochs
    batch_size      : batch size
    learning_rate   : initial learning rate
    checkpoints_dir : directory to save checkpoint
    num_workers     : DataLoader worker processes

    Returns
    -------
    Path to the saved final checkpoint.
    """
    print("\n" + "="*60)
    print("FINAL TRAINING ON FULL TRAINING SET")
    print("="*60)

    train_loader  = get_full_train_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    class_weights = train_loader.dataset.get_class_weights()

    model, loss_fn = build_model(
        class_weights=class_weights,
        device=device,
    )
    print_model_summary(model)

    optimiser = torch.optim.Adam(
        model.get_trainable_params(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01,
    )

    checkpoint_path = str(Path(checkpoints_dir) / 'final' / 'best.pt')

    for epoch in range(1, num_epochs + 1):
        epoch_start   = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            loader=train_loader,
            device=device,
            epoch=epoch,
        )
        scheduler.step()
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"Loss {train_metrics['total_loss']:.4f} "
            f"(CE {train_metrics['ce_loss']:.4f} "
            f"KL {train_metrics['kl_loss']:.4f}) | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimiser=optimiser,
        epoch=num_epochs,
        score=0.0,  # no validation score available for final model
        checkpoint_path=checkpoint_path,
    )
    print(f"\nFinal checkpoint saved to '{checkpoint_path}'.")

    return checkpoint_path


# ---------------------------------------------------------------------------
# Test set evaluation
# ---------------------------------------------------------------------------

def run_test_evaluation(
    checkpoint_path: str,
    device: torch.device,
    batch_size: int  = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> dict:
    """
    Load a checkpoint and evaluate once on the official test set.

    This should only be called after hyperparameters are finalised.
    Calling it multiple times constitutes test set leakage.

    Parameters
    ----------
    checkpoint_path : path to model checkpoint
    device          : torch device
    batch_size      : batch size
    num_workers     : DataLoader worker processes

    Returns
    -------
    Dict of ICBHI metrics from evaluate().
    """
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    print("NOTE: This should only be run once. Repeated evaluation on the")
    print("      test set constitutes data leakage.")
    print("="*60)

    test_loader = get_test_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model, _ = build_model(device=device)
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
    )

    metrics = evaluate(model=model, loader=test_loader, device=device)
    print_evaluation_report(metrics, split='Test')

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the ICBHI ALSC co-tuning model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--final',
        action='store_true',
        help=(
            "Run cross-validation then train on full training set "
            "and evaluate on test set."
        )
    )
    group.add_argument(
        '--final-only',
        action='store_true',
        help=(
            "Skip cross-validation and run final training + test evaluation "
            "only. Use when hyperparameters are already fixed."
        )
    )
    parser.add_argument(
        '--start-fold',
        type=int,
        default=0,
        help=(
            "0-indexed fold to start from. Use to resume after a crash "
            "(e.g. --start-fold 1 to skip fold 0)."
        )
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            "Path to an existing checkpoint to use for test evaluation "
            "instead of training a new final model. "
            "Only used with --final-only."
        )
    )
    args = parser.parse_args()

    device     = get_device()
    start_time = time.time()

    cv_results        = None
    final_checkpoint  = args.checkpoint

    # ── Cross-validation ─────────────────────────────────────────────────
    if not args.final_only:
        cv_results = run_cross_validation(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_folds=NUM_FOLDS,
            random_seed=RANDOM_SEED,
            checkpoints_dir=CHECKPOINTS_DIR,
            num_workers=NUM_WORKERS,
            start_fold=args.start_fold,
        )

    # ── Final training + test evaluation ─────────────────────────────────
    if args.final or args.final_only:

        # Use provided checkpoint or train a new final model
        if final_checkpoint is not None and Path(final_checkpoint).exists():
            print(f"\nUsing existing checkpoint: '{final_checkpoint}'")
        else:
            final_checkpoint = run_final_training(
                device=device,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                checkpoints_dir=CHECKPOINTS_DIR,
                num_workers=NUM_WORKERS,
                weight_decay=WEIGHT_DECAY
            )

        test_metrics = run_test_evaluation(
            checkpoint_path=final_checkpoint,
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

    # ── Total time ───────────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    minutes       = int(total_elapsed // 60)
    seconds       = total_elapsed % 60
    print(f"\nTotal time: {minutes}m {seconds:.1f}s")


if __name__ == '__main__':
    main()