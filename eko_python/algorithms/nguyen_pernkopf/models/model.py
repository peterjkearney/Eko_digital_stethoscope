"""
model.py

Assembles the full co-tuning model from its components and provides
a single build function for use in the training loop.

This file is the single entry point for model construction. The training
loop should call build_model() rather than importing from resnet.py or
cotuning.py directly.

Also handles:
    - Moving the model to the correct device
    - Saving and loading checkpoints
    - Printing a model summary
"""

import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    NUM_CLASSES,
    NUM_SOURCE_CLASSES,
    RESNET_VARIANT,
    COTUNING_LAMBDA,
    USE_BANDWIDTH_FEATURE,
)

from models.resnet import build_resnet
from models.cotuning import CoTuningModel, CoTuningLoss


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    variant: str = RESNET_VARIANT,
    pretrained: bool = True,
    num_target_classes: int = NUM_CLASSES,
    num_source_classes: int = NUM_SOURCE_CLASSES,
    cotuning_lambda: float = COTUNING_LAMBDA,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
    use_bandwidth: bool = USE_BANDWIDTH_FEATURE,
) -> tuple[CoTuningModel, CoTuningLoss]:
    """
    Build and return the full co-tuning model and its loss function.

    Parameters
    ----------
    variant            : ResNet variant (e.g. 'resnet50')
    pretrained         : whether to load ImageNet pretrained weights
    num_target_classes : number of ICBHI classes (4)
    num_source_classes : number of ImageNet classes (1000)
    cotuning_lambda    : weight for the KL divergence term in the loss
    class_weights      : optional (num_target_classes,) tensor for weighted
                         cross-entropy. Use dataset.get_class_weights().
    device             : device to move model to. If None, auto-detected.

    Returns
    -------
    (model, loss_fn) tuple.
    """
    if device is None:
        device = get_device()

    # Build backbone, source classifier, target classifier
    backbone, source_classifier, target_classifier = build_resnet(
        variant=variant,
        pretrained=pretrained,
        num_target_classes=num_target_classes,
        use_bandwidth=use_bandwidth,
    )

    # Assemble into co-tuning model
    model = CoTuningModel(
        backbone=backbone,
        source_classifier=source_classifier,
        target_classifier=target_classifier,
        num_source_classes=num_source_classes,
        num_target_classes=num_target_classes,
        use_bandwidth=use_bandwidth,
    )

    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Build loss function
    loss_fn = CoTuningLoss(
        cotuning_lambda=cotuning_lambda,
        class_weights=class_weights,
    )

    model = model.to(device)

    return model, loss_fn


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Auto-detect the best available device.

    Returns CUDA if available, MPS (Apple Silicon) if available,
    otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpoint saving and loading
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: CoTuningModel,
    optimiser: torch.optim.Optimizer,
    epoch: int,
    score: float,
    checkpoint_path: str,
) -> None:
    """
    Save a model checkpoint to disk.

    Parameters
    ----------
    model           : CoTuningModel instance
    optimiser       : optimiser instance
    epoch           : current epoch number
    score           : ICBHI score at this checkpoint (for reference)
    checkpoint_path : path to save the checkpoint file
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch':                epoch,
        'icbhi_score':          score,
        'model_state_dict':     model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
    }, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: CoTuningModel,
    optimiser: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[int, float]:
    """
    Load a model checkpoint from disk.

    Parameters
    ----------
    checkpoint_path : path to the checkpoint file
    model           : CoTuningModel instance to load weights into
    optimiser       : optional optimiser to restore state for (pass None
                      when loading for inference only)
    device          : device to map the checkpoint tensors to

    Returns
    -------
    (epoch, icbhi_score) from the checkpoint.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimiser is not None and 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    epoch      = checkpoint.get('epoch', 0)
    icbhi_score = checkpoint.get('icbhi_score', 0.0)

    print(f"Loaded checkpoint from '{checkpoint_path}' "
          f"(epoch {epoch}, ICBHI score {icbhi_score:.4f}).")

    return epoch, icbhi_score


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def print_model_summary(model: CoTuningModel) -> None:
    """
    Print a summary of the model's parameter counts.

    Parameters
    ----------
    model : CoTuningModel instance
    """
    def count_params(module: nn.Module) -> tuple[int, int]:
        total     = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    backbone_total,    backbone_trainable    = count_params(model.backbone)
    source_total,      source_trainable      = count_params(model.source_classifier)
    target_total,      target_trainable      = count_params(model.target_classifier)
    relationship_total, relationship_trainable = count_params(model.relationship)

    total_all     = backbone_total + source_total + target_total + relationship_total
    trainable_all = backbone_trainable + source_trainable + target_trainable + relationship_trainable

    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"  Backbone           : {backbone_trainable:>10,} / {backbone_total:>10,} trainable")
    print(f"  Source classifier  : {source_trainable:>10,} / {source_total:>10,} trainable (frozen)")
    print(f"  Target classifier  : {target_trainable:>10,} / {target_total:>10,} trainable")
    print(f"  Relationship matrix: {relationship_trainable:>10,} / {relationship_total:>10,} trainable")
    print("-"*60)
    print(f"  Total              : {trainable_all:>10,} / {total_all:>10,} trainable")
    print("="*60)