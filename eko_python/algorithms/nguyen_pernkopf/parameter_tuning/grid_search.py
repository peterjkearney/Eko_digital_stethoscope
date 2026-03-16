"""
grid_search.py

Grid search over learning rate and co-tuning lambda.
Runs fold 0 only for a reduced number of epochs per combination.

Usage:
    python grid_search.py
"""

import csv
import itertools
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent))

import config as cfg
from models.model import get_device
from dataset.dataloader import get_patient_folds
from training.train import train_fold


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# LEARNING_RATES   = [1e-3, 5e-4, 1e-4]
# COTUNING_LAMBDAS = [0.1, 0.01, 0.001]
COMBINATIONS = [(5e-4,.01),(1e-3,0.1),(1e-4,0.001),(5e-4,.1),(1e-3,.01)]

SEARCH_EPOCHS    = 80
SEARCH_FOLD      = 0
SEARCH_FRACTION  = 0.5

RESULTS_PATH = Path(cfg.CHECKPOINTS_DIR) / 'grid_search' / 'results_80ep.csv'


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_grid_search() -> None:
    device = get_device()
    folds  = get_patient_folds(n_folds=cfg.NUM_FOLDS, random_seed=cfg.RANDOM_SEED)

    #combos = list(itertools.product(LEARNING_RATES, COTUNING_LAMBDAS))
    combos = COMBINATIONS
    total  = len(combos)

    print("=" * 60)
    print("GRID SEARCH")
    print("=" * 60)
    # print(f"Learning rates:    {LEARNING_RATES}")
    # print(f"Co-tuning lambdas: {COTUNING_LAMBDAS}")
    print(f"Combinations:      {total}")
    print(f"Epochs per run:    {SEARCH_EPOCHS}")
    print(f"Train fraction:    {SEARCH_FRACTION}")
    print(f"Fold:              {SEARCH_FOLD}")
    print("=" * 60)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'lr',
        'cotuning_lambda',
        'best_icbhi_score',
        'best_epoch',
        'final_icbhi_score',
        'final_train_loss',
        'sensitivity_normal',
        'sensitivity_crackle',
        'sensitivity_wheeze',
        'sensitivity_both',
    ]

    # Load already-completed combinations so we can skip them on resume
    completed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline='') as f:
            for row in csv.DictReader(f):
                completed.add((float(row['lr']), float(row['cotuning_lambda'])))
        print(f"  Resuming — {len(completed)}/{total} combinations already done.")
    else:
        with open(RESULTS_PATH, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for i, (lr, lam) in enumerate(combos, 1):
        if (lr, lam) in completed:
            print(f"\n[{i}/{total}] lr={lr:.0e}  lambda={lam}  — already done, skipping.")
            continue

        print(f"\n[{i}/{total}] lr={lr:.0e}  lambda={lam}")
        print("-" * 40)

        # Override config values for this run
        cfg.LEARNING_RATE   = lr
        cfg.COTUNING_LAMBDA = lam

        checkpoints_dir = str(
            Path(cfg.CHECKPOINTS_DIR) / 'grid_search' / f'lr{lr:.0e}_lam{lam}'
        )

        result = train_fold(
            fold=SEARCH_FOLD,
            folds=folds,
            device=device,
            num_epochs=SEARCH_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            learning_rate=lr,
            checkpoints_dir=checkpoints_dir,
            num_workers=cfg.NUM_WORKERS,
            train_fraction=SEARCH_FRACTION,
        )

        history      = result['history']
        best_epoch   = result['best_epoch']
        final_record = history[-1]

        # Per-class sensitivity at best epoch (1-indexed in history list)
        best_record    = history[best_epoch - 1]
        per_class   = best_record['val_per_class']
        sensitivity = {cls: float(per_class[cls]['sensitivity']) for cls in per_class}

        row = {
            'lr':                  lr,
            'cotuning_lambda':     lam,
            'best_icbhi_score':    float(result['best_icbhi_score']),
            'best_epoch':          best_epoch,
            'final_icbhi_score':   float(final_record['val_icbhi_score']),
            'final_train_loss':    float(final_record['train_total_loss']),
            'sensitivity_normal':  sensitivity.get('normal',  ''),
            'sensitivity_crackle': sensitivity.get('crackle', ''),
            'sensitivity_wheeze':  sensitivity.get('wheeze',  ''),
            'sensitivity_both':    sensitivity.get('both',    ''),
        }

        with open(RESULTS_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(
            f"  Best ICBHI: {result['best_icbhi_score']:.4f} (epoch {best_epoch})  "
            f"Final: {final_record['val_icbhi_score']:.4f}  "
            f"Loss: {final_record['train_total_loss']:.4f}"
        )
        print(
            f"  Se — normal: {sensitivity.get('normal', '?'):.3f}  "
            f"crackle: {sensitivity.get('crackle', '?'):.3f}  "
            f"wheeze: {sensitivity.get('wheeze', '?'):.3f}  "
            f"both: {sensitivity.get('both', '?'):.3f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print(f"Results saved to: {RESULTS_PATH}")
    print("=" * 60)

    # Print sorted summary
    results = []
    with open(RESULTS_PATH, newline='') as f:
        for row in csv.DictReader(f):
            results.append(row)

    results.sort(key=lambda r: float(r['best_icbhi_score']), reverse=True)
    print(f"\n{'Rank':<5} {'LR':<8} {'Lambda':<10} {'Best ICBHI':<12} {'Best Ep':<9} {'Final ICBHI':<13} {'Se(both)'}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {float(r['lr']):<8.0e} {float(r['cotuning_lambda']):<10} "
            f"{float(r['best_icbhi_score']):<12.4f} {r['best_epoch']:<9} "
            f"{float(r['final_icbhi_score']):<13.4f} {float(r['sensitivity_both']):.3f}"
        )


if __name__ == '__main__':
    run_grid_search()
