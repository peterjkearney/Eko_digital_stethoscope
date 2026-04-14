"""
model.py

ResNet-based multi-label classifier for DAS/CAS lung sound detection.

Architecture
------------
    Input (B, 3, 224, 224)
        ↓
    ResNet backbone (pretrained on ImageNet, FC removed)
        ↓
    Feature vector (B, feature_dim)   512 for ResNet18/34, 2048 for ResNet50
        ↓
    Dropout
        ↓
    Linear(feature_dim, 2)
        ↓
    Logits (B, 2)   — [das_logit, cas_logit]

Loss
----
    BCEWithLogitsLoss with per-label pos_weight to handle class imbalance.
    Sigmoid is applied internally by the loss — not in the forward pass.

Co-tuning is not used here: it relies on a softmax-based relationship matrix
that does not translate cleanly to independent sigmoid outputs.
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
    DROPOUT_P,
    FREEZE_BACKBONE_UNTIL,
    CHECKPOINTS_DIR,
)

_RESNET_CONFIGS = {
    'resnet18': (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    'resnet34': (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
    'resnet50': (models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}

# ResNet children (excluding FC): conv1(0) bn1(1) relu(2) maxpool(3)
#                                  layer1(4) layer2(5) layer3(6) layer4(7) avgpool(8)
_FREEZE_UNTIL_INDEX = {
    'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7,
}

LABEL_NAMES = ['das', 'cas']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LungSoundClassifier(nn.Module):
    """
    ResNet backbone with a two-output linear head for DAS/CAS multi-label
    classification.

    Parameters
    ----------
    variant       : 'resnet18', 'resnet34', or 'resnet50'
    pretrained    : load ImageNet weights for the backbone
    freeze_until  : freeze all backbone layers up to and including this layer
                    ('layer1', 'layer2', 'layer3', 'layer4', or None)
    dropout_p     : dropout probability before the classification head
    """

    def __init__(
        self,
        variant:      str        = RESNET_VARIANT,
        pretrained:   bool       = True,
        freeze_until: str | None = FREEZE_BACKBONE_UNTIL,
        dropout_p:    float      = DROPOUT_P,
    ):
        super().__init__()
        if variant not in _RESNET_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. "
                             f"Choose from {list(_RESNET_CONFIGS)}")

        model_fn, weights, feature_dim = _RESNET_CONFIGS[variant]
        resnet = model_fn(weights=weights if pretrained else None)

        # Backbone: all layers except the final FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_until is not None:
            if freeze_until not in _FREEZE_UNTIL_INDEX:
                raise ValueError(f"freeze_until must be one of "
                                 f"{list(_FREEZE_UNTIL_INDEX)} or None")
            freeze_idx = _FREEZE_UNTIL_INDEX[freeze_until]
            for i in range(freeze_idx + 1):
                for p in self.backbone[i].parameters():
                    p.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_p)
        self.head    = nn.Linear(feature_dim, 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, 2) — sigmoid applied at loss/inference time."""
        features = self.backbone(x).flatten(start_dim=1)
        return self.head(self.dropout(features))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities (B, 2)."""
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary predictions (B, 2) at the given threshold."""
        return (self.predict_proba(x) >= threshold).float()

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MultiLabelBCELoss(nn.Module):
    """
    BCEWithLogitsLoss with per-label pos_weight.

    pos_weight should be a FloatTensor of shape (2,):
        pos_weight[i] = n_negative[i] / n_positive[i]
    Upweights positive examples to counteract class imbalance.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    def forward(
        self,
        logits: torch.Tensor,   # (B, 2)
        labels: torch.Tensor,   # (B, 2) float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total_loss, das_loss, cas_loss).
        Per-label losses are for logging; total is the mean across both labels.
        """
        das_loss = F.binary_cross_entropy_with_logits(
            logits[:, 0], labels[:, 0],
            pos_weight=self.bce.pos_weight[0:1] if self.bce.pos_weight is not None else None,
        )
        cas_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1], labels[:, 1],
            pos_weight=self.bce.pos_weight[1:2] if self.bce.pos_weight is not None else None,
        )
        total = (das_loss + cas_loss) / 2.0
        return total, das_loss, cas_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    variant:      str                    = RESNET_VARIANT,
    pretrained:   bool                   = True,
    pos_weight:   torch.Tensor | None    = None,
    device:       torch.device | None    = None,
) -> tuple['LungSoundClassifier', MultiLabelBCELoss]:
    """Build and return (model, loss_fn), both moved to device."""
    if device is None:
        device = get_device()

    model = LungSoundClassifier(
        variant=variant,
        pretrained=pretrained,
    ).to(device)

    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    loss_fn = MultiLabelBCELoss(pos_weight=pos_weight)

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
    model:     LungSoundClassifier,
    optimiser: torch.optim.Optimizer,
    epoch:     int,
    score:     float,
    path:      str | Path,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':                epoch,
        'mean_auc':             score,
        'model_state_dict':     model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
    }, path)


def load_checkpoint(
    path:      str | Path,
    model:     LungSoundClassifier,
    optimiser: torch.optim.Optimizer | None = None,
    device:    torch.device | None = None,
) -> tuple[int, float]:
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimiser is not None and 'optimiser_state_dict' in ckpt:
        optimiser.load_state_dict(ckpt['optimiser_state_dict'])
    epoch = ckpt.get('epoch', 0)
    score = ckpt.get('mean_auc', 0.0)
    print(f"Loaded checkpoint '{path}' (epoch {epoch}, mean AUC {score:.4f})")
    return epoch, score


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_BACKBONE_LAYER_NAMES = [
    'conv1', 'bn1', 'relu', 'maxpool',
    'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',
]


def print_model_summary(model: LungSoundClassifier) -> None:
    def _count(m):
        total     = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return total, trainable

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print("  Backbone:")
    bb_total = bb_trainable = 0
    for i, layer in enumerate(model.backbone):
        name = (_BACKBONE_LAYER_NAMES[i] if i < len(_BACKBONE_LAYER_NAMES)
                else f'layer_{i}')
        total, trainable = _count(layer)
        if total == 0:
            continue
        frozen = '  (frozen)' if trainable == 0 else ''
        print(f"    {name:<18}: {trainable:>8,} / {total:>8,} trainable{frozen}")
        bb_total     += total
        bb_trainable += trainable
    print(f"    {'[backbone total]':<18}: {bb_trainable:>8,} / {bb_total:>8,} trainable")

    head_total, head_trainable = _count(model.head)
    print(f"  {'Head (das+cas)':<22}: {head_trainable:>8,} / {head_total:>8,} trainable")

    total_all     = bb_total     + head_total
    trainable_all = bb_trainable + head_trainable
    print("-" * 60)
    print(f"  {'Total':<22}: {trainable_all:>8,} / {total_all:>8,} trainable")
    print("=" * 60)
