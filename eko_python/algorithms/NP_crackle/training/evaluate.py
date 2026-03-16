"""
evaluate.py

Model evaluation for the ICBHI ALSC task.

The official ICBHI evaluation metric is the ICBHI score, defined as the
average of the mean sensitivity and mean specificity across all classes:

    ICBHI score = (mean_sensitivity + mean_specificity) / 2

Where for each class c:
    sensitivity_c = TP_c / (TP_c + FN_c)   # recall for class c
    specificity_c = TN_c / (TN_c + FP_c)   # true negative rate for class c

And the means are taken across all 4 classes (normal, crackle, wheeze, both).

This is a macro-averaged metric that treats all classes equally regardless
of their frequency, which is appropriate given the severe class imbalance
in the ICBHI dataset.

References:
    Rocha et al. (2019) "An open access database for the evaluation of
    respiratory sound classification algorithms", Physiol. Meas. 40(3).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.icbhi_dataset import IDX_TO_LABEL


# ---------------------------------------------------------------------------
# ICBHI score computation
# ---------------------------------------------------------------------------

def compute_icbhi_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 2,
) -> dict:
    """
    Compute the ICBHI score and per-class metrics from predictions.

    Parameters
    ----------
    y_true      : 1D integer array of ground truth class indices
    y_pred      : 1D integer array of predicted class indices
    num_classes : number of classes (4 for ALSC)

    Returns
    -------
    Dict with:
        icbhi_score      : official ICBHI score
        mean_sensitivity : mean sensitivity across classes
        mean_specificity : mean specificity across classes
        per_class        : dict of per-class metrics keyed by class name
    """
    sensitivities = []
    specificities = []
    per_class     = {}

    for c in range(num_classes):
        # True positives, false negatives, false positives, true negatives
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        tn = np.sum((y_pred != c) & (y_true != c))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        class_name        = IDX_TO_LABEL[c]
        per_class[class_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
            'support': int(tp + fn),
        }

    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)
    icbhi_score      = (mean_sensitivity + mean_specificity) / 2.0

    return {
        'icbhi_score':      icbhi_score,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'per_class':        per_class,
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run model evaluation over a DataLoader and compute ICBHI metrics.

    Parameters
    ----------
    model  : CoTuningModel instance
    loader : DataLoader (validation or test, augment=False)
    device : torch device

    Returns
    -------
    Dict with ICBHI score and per-class metrics from compute_icbhi_score(),
    plus mean validation loss components.
    """
    from models.cotuning import CoTuningLoss

    model.eval()

    all_preds  = []
    all_labels = []

    for spectrograms, bandwidths, labels in loader:
        spectrograms = spectrograms.to(device, non_blocking=True)
        bandwidths   = bandwidths.to(device, non_blocking=True)
        labels       = labels.to(device, non_blocking=True)

        target_logits, _, _ = model(spectrograms, bandwidth=bandwidths)
        preds = torch.argmax(target_logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    return compute_icbhi_score(y_pred=y_pred, y_true=y_true)


# ---------------------------------------------------------------------------
# Detailed evaluation report
# ---------------------------------------------------------------------------

def print_evaluation_report(metrics: dict, split: str = 'Validation') -> None:
    """
    Print a detailed evaluation report to stdout.

    Parameters
    ----------
    metrics : dict returned by compute_icbhi_score()
    split   : name of the split being evaluated (for display)
    """
    print("\n" + "="*60)
    print(f"{split.upper()} EVALUATION REPORT")
    print("="*60)
    print(f"  ICBHI Score:      {metrics['icbhi_score']:.4f}")
    print(f"  Mean Sensitivity: {metrics['mean_sensitivity']:.4f}")
    print(f"  Mean Specificity: {metrics['mean_specificity']:.4f}")
    print()
    print(f"  {'Class':<10} {'Se':>8} {'Sp':>8} {'Support':>10}")
    print(f"  {'-'*40}")

    for class_name, m in metrics['per_class'].items():
        print(
            f"  {class_name:<10} "
            f"{m['sensitivity']:>8.4f} "
            f"{m['specificity']:>8.4f} "
            f"{m['support']:>10}"
        )

    print("="*60)