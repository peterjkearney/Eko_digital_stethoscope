"""
cotuning.py

Implements the co-tuning mechanism from:

    Nguyen & Pernkopf (2022) "Lung Sound Classification Using Co-Tuning
    and Stochastic Normalization", IEEE Trans. Biomed. Eng. 69(9):2872-2882.

    Original co-tuning paper:
    You et al. (2021) "Co-Tuning for Transfer Learning", NeurIPS 2020.

Co-tuning keeps the pretrained source classifier (ImageNet) active during
fine-tuning and learns a relationship matrix R that maps source class
probabilities to a translated prior over target classes. This prior is used
as an additional supervision signal via KL divergence, regularising the
target classifier to stay consistent with the source task structure.

Components:
    RelationshipMatrix  — learnable (num_source_classes, num_target_classes)
                          matrix mapping source probs → target prior
    CoTuningLoss        — combined CE + KL loss
    CoTuningModel       — wraps backbone, source classifier, target classifier,
                          and relationship matrix into a single forward pass

Forward pass:
    features      = backbone(x)
    source_probs  = softmax(source_classifier(features))   # (B, 1000)
    target_logits = target_classifier(features)            # (B, 4)
    target_probs  = softmax(target_logits)                 # (B, 4)
    target_prior  = softmax(source_probs @ R)              # (B, 4)

Loss:
    L = CE(target_logits, y) + lambda * KL(target_probs || target_prior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config_c import (
    NUM_CLASSES,
    COTUNING_LAMBDA,
    NUM_SOURCE_CLASSES,
)

from models.resnet import ResNetBackbone, SourceClassifier, TargetClassifier


# ---------------------------------------------------------------------------
# Relationship matrix
# ---------------------------------------------------------------------------

class RelationshipMatrix(nn.Module):
    """
    Learnable relationship matrix R of shape (num_source_classes, num_target_classes).

    Maps source class probabilities to a prior distribution over target
    classes. Each row corresponds to one source class and contains the
    learned affinity of that source class towards each target class.

    R is initialised uniformly so the prior starts as uninformative,
    and is updated during training alongside the backbone and target
    classifier.

    Parameters
    ----------
    num_source_classes : number of source (ImageNet) classes (1000)
    num_target_classes : number of target (ICBHI) classes (4)
    """

    def __init__(
        self,
        num_source_classes: int = NUM_SOURCE_CLASSES,
        num_target_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Initialise uniformly — prior starts as uninformative
        self.R = nn.Parameter(
            torch.ones(num_source_classes, num_target_classes) / num_target_classes
        )

    def forward(self, source_probs: torch.Tensor) -> torch.Tensor:
        """
        Translate source class probabilities into a prior over target classes.

        Parameters
        ----------
        source_probs : tensor of shape (B, num_source_classes)
                       softmax probabilities from the source classifier

        Returns
        -------
        target_prior : tensor of shape (B, num_target_classes)
                       softmax-normalised prior over target classes
        """
        # Matrix multiply: (B, num_source) @ (num_source, num_target) → (B, num_target)
        raw_prior    = source_probs @ self.R
        target_prior = F.softmax(raw_prior, dim=1)
        return target_prior


# ---------------------------------------------------------------------------
# Co-tuning loss
# ---------------------------------------------------------------------------

class CoTuningLoss(nn.Module):
    """
    Combined cross-entropy and KL divergence loss for co-tuning.

    L = CE(target_logits, y) + lambda * KL(target_probs || target_prior)

    The CE term trains the target classifier on ground truth labels.
    The KL term regularises the target classifier to stay consistent
    with the translated prior from the source task, preventing the
    backbone from drifting too far from its pretrained representations.

    KL(P || Q) = sum(P * log(P / Q))
    Here P = target_probs (model output) and Q = target_prior (from R).

    Parameters
    ----------
    cotuning_lambda : weight for the KL divergence term
    class_weights   : optional tensor of shape (num_target_classes,) for
                      weighted cross-entropy to address class imbalance
    """

    def __init__(
        self,
        cotuning_lambda: float = COTUNING_LAMBDA,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.cotuning_lambda = cotuning_lambda
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        target_logits: torch.Tensor,
        target_prior:  torch.Tensor,
        labels:        torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined co-tuning loss.

        Parameters
        ----------
        target_logits : tensor of shape (B, num_target_classes)
                        raw logits from the target classifier
        target_prior  : tensor of shape (B, num_target_classes)
                        translated prior from the relationship matrix
        labels        : tensor of shape (B,) with integer class labels

        Returns
        -------
        total_loss : weighted sum of CE and KL losses
        ce_loss    : cross-entropy component (for logging)
        kl_loss    : KL divergence component (for logging)
        """
        # Cross-entropy loss against ground truth
        ce = self.ce_loss(target_logits, labels)

        # Target probabilities from softmax of logits
        target_probs = F.softmax(target_logits, dim=1)

        # KL divergence: KL(target_probs || target_prior)
        # F.kl_div expects log-probabilities as input and probabilities as target
        kl = F.kl_div(
            input=torch.log(target_probs + 1e-10),
            target=target_prior,
            reduction='batchmean',
        )

        total = ce + self.cotuning_lambda * kl

        return total, ce, kl


# ---------------------------------------------------------------------------
# Full co-tuning model
# ---------------------------------------------------------------------------

class CoTuningModel(nn.Module):
    """
    Full co-tuning model wrapping backbone, source classifier, target
    classifier, and relationship matrix into a single nn.Module.

    Exposes a single forward pass that returns everything needed for
    the co-tuning loss and evaluation.

    Parameters
    ----------
    backbone           : ResNetBackbone instance
    source_classifier  : SourceClassifier instance (frozen)
    target_classifier  : TargetClassifier instance
    num_source_classes : number of ImageNet classes (1000)
    num_target_classes : number of ICBHI classes (4)
    """

    def __init__(
        self,
        backbone:           ResNetBackbone,
        source_classifier:  SourceClassifier,
        target_classifier:  TargetClassifier,
        num_source_classes: int = NUM_SOURCE_CLASSES,
        num_target_classes: int = NUM_CLASSES,
        use_bandwidth:      bool = False,
    ):
        super().__init__()
        self.backbone           = backbone
        self.source_classifier  = source_classifier
        self.target_classifier  = target_classifier
        self.use_bandwidth      = use_bandwidth
        self.relationship       = RelationshipMatrix(
            num_source_classes=num_source_classes,
            num_target_classes=num_target_classes,
        )

    def forward(
        self,
        x:         torch.Tensor,
        bandwidth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full co-tuning forward pass.

        Parameters
        ----------
        x         : tensor of shape (B, 3, H, W)
        bandwidth : optional tensor of shape (B,) containing the normalised
                    device Nyquist scalar for each sample in [0, 1].
                    Concatenated to features before the target classifier when
                    use_bandwidth=True.  Ignored otherwise.

        Returns
        -------
        target_logits : tensor of shape (B, num_target_classes)
                        raw logits for prediction and CE loss
        target_prior  : tensor of shape (B, num_target_classes)
                        translated prior for KL loss
        source_probs  : tensor of shape (B, num_source_classes)
                        source classifier probabilities (for diagnostics)
        """
        # Extract features
        features = self.backbone(x)                          # (B, feature_dim)

        # Source classifier — frozen, no gradients needed
        with torch.no_grad():
            source_logits = self.source_classifier(features) # (B, 1000)
        source_probs  = F.softmax(source_logits, dim=1)      # (B, 1000)

        # Optionally append device bandwidth scalar before target classifier
        if self.use_bandwidth and bandwidth is not None:
            features_for_target = torch.cat(
                [features, bandwidth.unsqueeze(1)], dim=1    # (B, feature_dim+1)
            )
        else:
            features_for_target = features

        # Target classifier
        target_logits = self.target_classifier(features_for_target)  # (B, 4)

        # Translate source probs to target prior via relationship matrix
        target_prior  = self.relationship(source_probs)      # (B, 4)

        return target_logits, target_prior, source_probs

    def get_trainable_params(self) -> list:
        """
        Return trainable parameters for the optimiser.

        Excludes the source classifier (frozen).
        Includes backbone, target classifier, and relationship matrix.

        Returns
        -------
        List of parameter tensors.
        """
        return (
            list(self.backbone.parameters()) +
            list(self.target_classifier.parameters()) +
            list(self.relationship.parameters())
        )

    def predict(
        self,
        x:         torch.Tensor,
        bandwidth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return predicted class indices for a batch.

        Parameters
        ----------
        x         : tensor of shape (B, 3, H, W)
        bandwidth : optional tensor of shape (B,) — see forward()

        Returns
        -------
        tensor of shape (B,) with predicted class indices.
        """
        target_logits, _, _ = self.forward(x, bandwidth=bandwidth)
        return torch.argmax(target_logits, dim=1)