"""
model.py

Co-tuning ResNet model for ICBHI lung sound classification.

Architecture
------------
    Input (B, 3, 224, 224)
        ↓
    ResNet backbone (pretrained on ImageNet, FC removed)
        ↓
    Feature vector (B, feature_dim)     feature_dim = 512 for ResNet18/34,
        ↓                                             2048 for ResNet50
    ├── Source classifier  (1000-class ImageNet head, frozen)
    │       ↓ softmax
    │   source_probs (B, 1000)
    │       ↓ @ relationship matrix R (1000, 4)
    │   target_prior (B, 4)          ← translated prior for KL loss
    │
    └── Target classifier  (4-class ICBHI head, trained from scratch)
            ↓
        target_logits (B, 4)         ← logits for CE loss and prediction

Co-tuning loss
--------------
    L = CE(target_logits, y) + λ * KL(softmax(target_logits) ‖ target_prior)

    The KL term regularises the target classifier against the translated
    prior from the ImageNet source task, preventing the backbone from
    drifting too far from its pretrained representations.

    Reference: You et al. (2021) "Co-Tuning for Transfer Learning", NeurIPS.
               Nguyen & Pernkopf (2022), IEEE Trans. Biomed. Eng. 69(9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RESNET_VARIANT,
    NUM_CLASSES,
    NUM_SOURCE_CLASSES,
    COTUNING_LAMBDA,
    DROPOUT_P,
    CHECKPOINTS_DIR,
)


# ---------------------------------------------------------------------------
# Supported ResNet variants
# ---------------------------------------------------------------------------

_RESNET_CONFIGS = {
    'resnet18': (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    'resnet34': (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
    'resnet50': (models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """ResNet with the final FC layer removed. Outputs (B, feature_dim)."""

    def __init__(self, variant: str = RESNET_VARIANT, pretrained: bool = True):
        super().__init__()
        if variant not in _RESNET_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_RESNET_CONFIGS)}")
        model_fn, weights, self.feature_dim = _RESNET_CONFIGS[variant]
        resnet = model_fn(weights=weights if pretrained else None)
        # All layers except the final FC
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(start_dim=1)   # (B, feature_dim)


class SourceClassifier(nn.Module):
    """
    The pretrained ImageNet FC head — frozen throughout training.
    Used only to produce source_probs for the relationship matrix.
    """

    def __init__(self, feature_dim: int, num_source_classes: int = NUM_SOURCE_CLASSES):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_source_classes)
        for p in self.parameters():
            p.requires_grad = False

    def load_pretrained_weights(self, resnet: nn.Module) -> None:
        with torch.no_grad():
            self.fc.weight.copy_(resnet.fc.weight)
            self.fc.bias.copy_(resnet.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)   # (B, 1000)


class TargetClassifier(nn.Module):
    """New 4-class ICBHI head — randomly initialised, trained from scratch."""

    def __init__(
        self,
        feature_dim: int,
        num_target_classes: int = NUM_CLASSES,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(feature_dim, num_target_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))   # (B, num_target_classes)


class RelationshipMatrix(nn.Module):
    """
    Learnable (num_source_classes, num_target_classes) matrix R.

    Maps source_probs → target_prior:
        target_prior = softmax(source_probs @ R)

    Initialised uniformly so the prior starts uninformative.
    Updated alongside the backbone and target classifier.
    """

    def __init__(
        self,
        num_source_classes: int = NUM_SOURCE_CLASSES,
        num_target_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.R = nn.Parameter(
            torch.ones(num_source_classes, num_target_classes) / num_target_classes
        )

    def forward(self, source_probs: torch.Tensor) -> torch.Tensor:
        return F.softmax(source_probs @ self.R, dim=1)   # (B, num_target_classes)


# ---------------------------------------------------------------------------
# Full co-tuning model
# ---------------------------------------------------------------------------

class CoTuningModel(nn.Module):

    def __init__(
        self,
        backbone:           ResNetBackbone,
        source_classifier:  SourceClassifier,
        target_classifier:  TargetClassifier,
        num_source_classes: int = NUM_SOURCE_CLASSES,
        num_target_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.backbone          = backbone
        self.source_classifier = source_classifier
        self.target_classifier = target_classifier
        self.relationship      = RelationshipMatrix(num_source_classes, num_target_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        target_logits : (B, 4)    — for CE loss and argmax prediction
        target_prior  : (B, 4)    — for KL loss
        source_probs  : (B, 1000) — for diagnostics
        """
        features = self.backbone(x)

        with torch.no_grad():
            source_logits = self.source_classifier(features)
        source_probs = F.softmax(source_logits, dim=1)

        target_logits = self.target_classifier(features)
        target_prior  = self.relationship(source_probs)

        return target_logits, target_prior, source_probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (B,)."""
        logits, _, _ = self.forward(x)
        return torch.argmax(logits, dim=1)

    def trainable_parameters(self) -> list:
        """All parameters except the frozen source classifier."""
        return (
            list(self.backbone.parameters()) +
            list(self.target_classifier.parameters()) +
            list(self.relationship.parameters())
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CoTuningLoss(nn.Module):
    """
    L = CE(target_logits, y) + λ * KL(softmax(target_logits) ‖ target_prior)
    """

    def __init__(
        self,
        cotuning_lambda: float = COTUNING_LAMBDA,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lam     = cotuning_lambda
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        target_logits: torch.Tensor,
        target_prior:  torch.Tensor,
        labels:        torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, ce_loss, kl_loss) — CE and KL are returned
        separately for logging.
        """
        ce = self.ce_loss(target_logits, labels)

        target_probs = F.softmax(target_logits, dim=1)
        kl = F.kl_div(
            input=torch.log(target_probs + 1e-10),
            target=target_prior,
            reduction='batchmean',
        )

        return ce + self.lam * kl, ce, kl


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    variant:            str   = RESNET_VARIANT,
    pretrained:         bool  = True,
    num_target_classes: int   = NUM_CLASSES,
    num_source_classes: int   = NUM_SOURCE_CLASSES,
    cotuning_lambda:    float = COTUNING_LAMBDA,
    class_weights: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[CoTuningModel, CoTuningLoss]:
    """Build and return (model, loss_fn), both moved to device."""
    if device is None:
        device = get_device()

    model_fn, weights, feature_dim = _RESNET_CONFIGS[variant]
    resnet = model_fn(weights=weights if pretrained else None)

    backbone          = ResNetBackbone(variant=variant, pretrained=pretrained)
    source_classifier = SourceClassifier(feature_dim=feature_dim,
                                         num_source_classes=num_source_classes)
    target_classifier = TargetClassifier(feature_dim=feature_dim,
                                         num_target_classes=num_target_classes)

    if pretrained:
        source_classifier.load_pretrained_weights(resnet)

    model = CoTuningModel(
        backbone=backbone,
        source_classifier=source_classifier,
        target_classifier=target_classifier,
        num_source_classes=num_source_classes,
        num_target_classes=num_target_classes,
    ).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = CoTuningLoss(cotuning_lambda=cotuning_lambda, class_weights=class_weights)

    return model, loss_fn


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     CoTuningModel,
    optimiser: torch.optim.Optimizer,
    epoch:     int,
    score:     float,
    path:      str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'icbhi_score':          score,
        'model_state_dict':     model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
    }, path)


def load_checkpoint(
    path:      str | Path,
    model:     CoTuningModel,
    optimiser: torch.optim.Optimizer | None = None,
    device:    torch.device | None = None,
) -> tuple[int, float]:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimiser is not None and 'optimiser_state_dict' in ckpt:
        optimiser.load_state_dict(ckpt['optimiser_state_dict'])
    epoch = ckpt.get('epoch', 0)
    score = ckpt.get('icbhi_score', 0.0)
    print(f"Loaded checkpoint '{path}' (epoch {epoch}, ICBHI score {score:.4f})")
    return epoch, score


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_model_summary(model: CoTuningModel) -> None:
    def _count(m):
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    rows = [
        ('Backbone',            *_count(model.backbone)),
        ('Source classifier',   *_count(model.source_classifier)),
        ('Target classifier',   *_count(model.target_classifier)),
        ('Relationship matrix', *_count(model.relationship)),
    ]
    total_all     = sum(r[1] for r in rows)
    trainable_all = sum(r[2] for r in rows)

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    for name, total, trainable in rows:
        frozen = '' if trainable else '  (frozen)'
        print(f"  {name:<22}: {trainable:>8,} / {total:>8,} trainable{frozen}")
    print("-" * 60)
    print(f"  {'Total':<22}: {trainable_all:>8,} / {total_all:>8,} trainable")
    print("=" * 60)
