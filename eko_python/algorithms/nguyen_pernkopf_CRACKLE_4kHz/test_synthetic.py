"""
test_synthetic.py

Sanity check: can the model overfit a synthetic binary dataset?

Generates 400 fake 224×224 spectrograms in memory:
  - Class 0 (no_crackle): Gaussian noise
  - Class 1 (crackle):    Gaussian noise + a bright horizontal band
                          (simulates a frequency-localised pattern)

If the pipeline is correct the model should reach F1 > 0.95 within ~10 epochs.
If it can't, there is a bug in the model, loss, or training loop — not a data
problem.

Usage:
    python test_synthetic.py
"""

import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from model import build_model, get_device
from train import train_one_epoch, compute_crackle_score

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

N_SAMPLES   = 400     # 200 per class
IMG_SIZE    = 224
BAND_ROW    = 50      # row index of the bright band for class 1
BAND_WIDTH  = 10      # rows
BAND_VALUE  = 5.0     # standard deviations above noise — very obvious signal


class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int = N_SAMPLES, seed: int = 0):
        rng = np.random.default_rng(seed)
        half = n_samples // 2

        images = rng.standard_normal((n_samples, IMG_SIZE, IMG_SIZE)).astype(np.float32)
        labels = np.array([0] * half + [1] * half, dtype=np.int64)

        # Add a bright horizontal band to all class-1 samples
        images[half:, BAND_ROW:BAND_ROW + BAND_WIDTH, :] += BAND_VALUE

        # Per-sample normalisation (mirrors ICBHIDataset)
        means = images.reshape(n_samples, -1).mean(axis=1)[:, None, None]
        stds  = images.reshape(n_samples, -1).std(axis=1)[:, None, None] + 1e-10
        images = (images - means) / stds

        self.images = torch.from_numpy(images).unsqueeze(1).repeat(1, 3, 1, 1)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("SYNTHETIC SANITY CHECK")
    print("=" * 60)
    print(f"  Samples:    {N_SAMPLES} ({N_SAMPLES//2} per class)")
    print(f"  Signal:     bright band at rows {BAND_ROW}–{BAND_ROW+BAND_WIDTH} "
          f"({BAND_VALUE:.0f}σ above noise)")
    print(f"  Expected:   F1 > 0.95 within ~10 epochs")
    print("=" * 60)

    device = get_device()

    dataset = SyntheticDataset(n_samples=N_SAMPLES)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    # Equal class weights — dataset is balanced
    class_weights = torch.ones(2, dtype=torch.float32).to(device)
    model, loss_fn = build_model(class_weights=class_weights, device=device)

    optimiser = optim.Adam(model.trainable_parameters(), lr=1e-4, weight_decay=1e-3)

    for epoch in range(1, 21):
        train_metrics = train_one_epoch(model, loss_fn, optimiser, loader, device)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                logits, _, _ = model(x.to(device))
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.numpy())

        import numpy as np
        metrics = compute_crackle_score(
            y_true=np.concatenate(all_labels),
            y_pred=np.concatenate(all_preds),
        )
        print(
            f"Epoch {epoch:>2}/20  "
            f"Loss {train_metrics['total_loss']:.4f}  "
            f"F1 {metrics['f1']:.4f}  "
            f"Se {metrics['sensitivity']:.4f}  "
            f"Sp {metrics['specificity']:.4f}"
        )

        if metrics['f1'] > 0.95:
            print(f"\n  PASS — model reached F1 {metrics['f1']:.4f} at epoch {epoch}.")
            print("  Pipeline is functioning correctly.")
            return

    print(f"\n  FAIL — model did not reach F1 > 0.95 after 20 epochs.")
    print("  There is likely a bug in the model, loss, or training loop.")


if __name__ == '__main__':
    main()
