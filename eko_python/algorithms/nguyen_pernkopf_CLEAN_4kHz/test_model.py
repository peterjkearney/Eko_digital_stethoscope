"""
test_model.py

Quick sanity checks for model.py:
  1. build_model() runs without error
  2. Forward pass produces correct output shapes
  3. Source classifier is frozen
  4. Backbone and target classifier are trainable
  5. Loss returns finite values
  6. Gradients flow to trainable parameters only
  7. predict() returns valid class indices
  8. Checkpoint save/load round-trip
"""

import torch
import tempfile
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from model import (
    build_model, get_device, print_model_summary,
    save_checkpoint, load_checkpoint,
)
from config import NUM_CLASSES, NUM_SOURCE_CLASSES

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
B    = 4   # batch size for tests

def check(name, condition, detail=''):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ''))
    return condition


def make_batch(device):
    return torch.randn(B, 3, 224, 224, device=device)


def test_build():
    print("\n── build_model ────────────────────────────────────────")
    device = get_device()
    model, loss_fn = build_model(device=device)
    check("model built", model is not None)
    check("loss_fn built", loss_fn is not None)
    check("model on device", next(model.parameters()).device.type == device.type)
    return model, loss_fn, device


def test_forward(model, device):
    print("\n── Forward pass ───────────────────────────────────────")
    x = make_batch(device)
    model.eval()
    with torch.no_grad():
        logits, prior, source_probs = model(x)

    check("logits shape",      logits.shape      == (B, NUM_CLASSES),        str(logits.shape))
    check("prior shape",       prior.shape       == (B, NUM_CLASSES),        str(prior.shape))
    check("source_probs shape",source_probs.shape == (B, NUM_SOURCE_CLASSES), str(source_probs.shape))
    check("logits finite",     torch.isfinite(logits).all().item())
    check("prior sums to 1",   (prior.sum(dim=1) - 1.0).abs().max().item() < 1e-5)
    check("source sums to 1",  (source_probs.sum(dim=1) - 1.0).abs().max().item() < 1e-5)


def test_frozen_source(model):
    print("\n── Frozen source classifier ───────────────────────────")
    for p in model.source_classifier.parameters():
        check("requires_grad=False", not p.requires_grad)
        break  # one check is enough


def test_trainable_params(model):
    print("\n── Trainable parameters ───────────────────────────────")
    trainable = set(id(p) for p in model.trainable_parameters())
    source    = set(id(p) for p in model.source_classifier.parameters())

    check("source classifier excluded from trainable",
          len(trainable & source) == 0)
    check("backbone included",
          any(id(p) in trainable for p in model.backbone.parameters()))
    check("target classifier included",
          any(id(p) in trainable for p in model.target_classifier.parameters()))
    check("relationship matrix included",
          any(id(p) in trainable for p in model.relationship.parameters()))


def test_loss(model, loss_fn, device):
    print("\n── Loss ───────────────────────────────────────────────")
    x      = make_batch(device)
    labels = torch.randint(0, NUM_CLASSES, (B,), device=device)

    model.train()
    logits, prior, _ = model(x)
    total, ce, kl    = loss_fn(logits, prior, labels)

    check("total loss finite", torch.isfinite(total).item(), f"{total.item():.4f}")
    check("ce loss finite",    torch.isfinite(ce).item(),    f"{ce.item():.4f}")
    check("kl loss finite",    torch.isfinite(kl).item(),    f"{kl.item():.4f}")
    check("total > 0",         total.item() > 0)


def test_gradients(model, loss_fn, device):
    print("\n── Gradients ──────────────────────────────────────────")
    x      = make_batch(device)
    labels = torch.randint(0, NUM_CLASSES, (B,), device=device)

    model.train()
    model.zero_grad()
    logits, prior, _ = model(x)
    total, _, _      = loss_fn(logits, prior, labels)
    total.backward()

    backbone_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.backbone.parameters()
    )
    target_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.target_classifier.parameters()
    )
    source_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.source_classifier.parameters()
    )

    check("backbone receives gradients",          backbone_grad)
    check("target classifier receives gradients", target_grad)
    check("source classifier has no gradients",   not source_grad)


def test_predict(model, device):
    print("\n── predict() ──────────────────────────────────────────")
    x = make_batch(device)
    model.eval()
    with torch.no_grad():
        preds = model.predict(x)
    check("preds shape",    preds.shape == (B,), str(preds.shape))
    check("preds in range", preds.min().item() >= 0 and preds.max().item() < NUM_CLASSES,
          f"min={preds.min().item()} max={preds.max().item()}")


def test_checkpoint(model, device):
    print("\n── Checkpoint round-trip ──────────────────────────────")
    optimiser = torch.optim.Adam(model.trainable_parameters(), lr=1e-4)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name

    save_checkpoint(model, optimiser, epoch=3, score=0.72, path=path)
    check("checkpoint saved", Path(path).exists())

    # Load into a fresh model and check weights match
    model2, _ = build_model(device=device)
    epoch, score = load_checkpoint(path, model2, device=device)

    check("epoch restored",  epoch == 3,  str(epoch))
    check("score restored",  abs(score - 0.72) < 1e-5, str(score))

    # Compare a backbone parameter
    p1 = next(model.backbone.parameters()).cpu()
    p2 = next(model2.backbone.parameters()).cpu()
    check("weights match after load", torch.allclose(p1, p2))

    Path(path).unlink()


if __name__ == '__main__':
    print("=" * 55)
    print("MODEL TESTS")
    print("=" * 55)

    model, loss_fn, device = test_build()
    test_forward(model, device)
    test_frozen_source(model)
    test_trainable_params(model)
    test_loss(model, loss_fn, device)
    test_gradients(model, loss_fn, device)
    test_predict(model, device)
    test_checkpoint(model, device)

    print("\n" + "=" * 55)
    print_model_summary(model)
