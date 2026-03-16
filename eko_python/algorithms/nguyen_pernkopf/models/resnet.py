"""
resnet.py

ResNet backbone setup for the ICBHI ALSC task.

Loads a pretrained ResNet50 from torchvision, adapts it for use with
log-mel spectrogram inputs, and exposes the feature extractor and source
classifier (ImageNet) separately to support co-tuning.

The final fully-connected classification layer is removed from the backbone
so that the feature vector can be passed to both the source classifier
(kept from pretraining, used in co-tuning) and the target classifier
(new 4-class ICBHI head).

Architecture:
    Input (3, 224, 224)
        ↓
    ResNet50 backbone (conv layers + avgpool)
        ↓
    Feature vector (2048,)
        ↓
    ├── Source classifier (1000-class ImageNet head, frozen weights)
    └── Target classifier (4-class ICBHI head, randomly initialised)

The source classifier weights are loaded from the pretrained checkpoint
and kept fixed — they are never updated during fine-tuning. Only the
backbone and target classifier are trained.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import NUM_CLASSES, RESNET_VARIANT, DROPOUT_P


# ---------------------------------------------------------------------------
# Supported variants
# ---------------------------------------------------------------------------

RESNET_CONFIGS = {
    'resnet18': (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    'resnet34': (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
    'resnet50': (models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """
    ResNet backbone with the final classification layer removed.

    Outputs a feature vector of shape (batch_size, feature_dim) which
    is fed into both the source and target classifiers.

    Parameters
    ----------
    variant    : ResNet variant name, must be a key in RESNET_CONFIGS
    pretrained : if True, load ImageNet pretrained weights
    """

    def __init__(
        self,
        variant: str = RESNET_VARIANT,
        pretrained: bool = True,
    ):
        super().__init__()

        if variant not in RESNET_CONFIGS:
            raise ValueError(
                f"Unsupported ResNet variant '{variant}'. "
                f"Choose from: {list(RESNET_CONFIGS.keys())}"
            )

        model_fn, weights, self.feature_dim = RESNET_CONFIGS[variant]

        # Load pretrained model
        resnet = model_fn(weights=weights if pretrained else None)

        # Remove the final FC layer — everything up to and including avgpool
        # becomes the backbone feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor of shape (B, 3, H, W)

        Returns
        -------
        features : tensor of shape (B, feature_dim)
        """
        x = self.features(x)       # (B, feature_dim, 1, 1)
        x = x.flatten(start_dim=1) # (B, feature_dim)
        return x


# ---------------------------------------------------------------------------
# Source classifier (ImageNet head)
# ---------------------------------------------------------------------------

class SourceClassifier(nn.Module):
    """
    The original ImageNet classification head from the pretrained ResNet.

    Weights are loaded from the pretrained checkpoint and frozen — they
    are never updated during fine-tuning. Used in co-tuning to provide
    a translated prior over target classes.

    Parameters
    ----------
    feature_dim   : input feature dimension (2048 for ResNet50)
    num_source_classes : number of ImageNet classes (1000)
    """

    def __init__(
        self,
        feature_dim: int,
        num_source_classes: int = 1000,
    ):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_source_classes)

        # Freeze all parameters — source classifier is never updated
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : tensor of shape (B, feature_dim)

        Returns
        -------
        logits : tensor of shape (B, num_source_classes)
        """
        return self.fc(features)

    def load_pretrained_weights(self, resnet: nn.Module) -> None:
        """
        Copy the FC weights from a pretrained ResNet into this classifier.

        Parameters
        ----------
        resnet : a torchvision ResNet model with pretrained weights loaded
        """
        with torch.no_grad():
            self.fc.weight.copy_(resnet.fc.weight)
            self.fc.bias.copy_(resnet.fc.bias)


# ---------------------------------------------------------------------------
# Target classifier (ICBHI head)
# ---------------------------------------------------------------------------

class TargetClassifier(nn.Module):
    """
    New classification head for the ICBHI 4-class ALSC task.

    Randomly initialised. Trained from scratch alongside the backbone
    during fine-tuning.

    Parameters
    ----------
    feature_dim      : input feature dimension (2048 for ResNet50)
    num_target_classes : number of ICBHI classes (4: normal/crackle/wheeze/both)
    """

    def __init__(
        self,
        feature_dim: int,
        num_target_classes: int = NUM_CLASSES,
        dropout_p: float = DROPOUT_P,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc      = nn.Linear(feature_dim, num_target_classes)

        # Initialise with small random weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : tensor of shape (B, feature_dim)

        Returns
        -------
        logits : tensor of shape (B, num_target_classes)
        """
        return self.fc(self.dropout(features))


# ---------------------------------------------------------------------------
# Full model assembly
# ---------------------------------------------------------------------------

def build_resnet(
    variant: str = RESNET_VARIANT,
    pretrained: bool = True,
    num_target_classes: int = NUM_CLASSES,
    use_bandwidth: bool = False,
) -> tuple[ResNetBackbone, SourceClassifier, TargetClassifier]:
    """
    Build the full ResNet model for co-tuning, returning the three
    components separately so the training loop can apply the correct
    loss to each.

    The source classifier weights are copied from the pretrained ResNet
    FC layer before it is discarded from the backbone.

    Parameters
    ----------
    variant            : ResNet variant (e.g. 'resnet50')
    pretrained         : whether to load ImageNet pretrained weights
    num_target_classes : number of target classes

    Returns
    -------
    (backbone, source_classifier, target_classifier) tuple.
    """
    if variant not in RESNET_CONFIGS:
        raise ValueError(
            f"Unsupported ResNet variant '{variant}'. "
            f"Choose from: {list(RESNET_CONFIGS.keys())}"
        )

    model_fn, weights, feature_dim = RESNET_CONFIGS[variant]

    # Load full pretrained ResNet temporarily to extract FC weights
    resnet = model_fn(weights=weights if pretrained else None)

    # Build components
    # Target classifier input is feature_dim + 1 when the bandwidth scalar
    # is concatenated to features before the classification head.
    target_feature_dim = feature_dim + (1 if use_bandwidth else 0)
    backbone           = ResNetBackbone(variant=variant, pretrained=pretrained)
    source_classifier  = SourceClassifier(feature_dim=feature_dim)
    target_classifier  = TargetClassifier(
        feature_dim=target_feature_dim,
        num_target_classes=num_target_classes,
    )

    # Copy pretrained FC weights into source classifier
    if pretrained:
        source_classifier.load_pretrained_weights(resnet)

    return backbone, source_classifier, target_classifier


def get_trainable_params(
    backbone: ResNetBackbone,
    target_classifier: TargetClassifier,
) -> list:
    """
    Return only the trainable parameters for the optimiser.

    The source classifier is frozen and excluded. The backbone and target
    classifier are both trainable.

    Parameters
    ----------
    backbone           : ResNetBackbone instance
    target_classifier  : TargetClassifier instance

    Returns
    -------
    List of parameter tensors to pass to the optimiser.
    """
    return (
        list(backbone.parameters()) +
        list(target_classifier.parameters())
    )