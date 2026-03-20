"""
gradcam_icbhi.py

Generates Grad-CAM visualisations for a random sample of held-out ICBHI
Meditron test cycles.

This script is the ICBHI counterpart of gradcam.py (which processes Eko
recordings).  Its purpose is to determine whether the recurring spatial
activation pattern seen in the Eko Grad-CAMs is:
    (a) a model-level artefact — present in both Eko and ICBHI heatmaps, or
    (b) specific to the Eko preprocessing pipeline.

Key differences from gradcam.py:
    - Input files are already padded ICBHI cycles (no chunking required).
    - No spectrum correction applied — Meditron is the reference device.
    - True labels come from manifest.csv, not from an inference results CSV.
    - One PNG is saved per class, showing N randomly selected cycles.

Output: pf_tests/results/gradcam_icbhi/<label>.png

Usage:
    python pf_tests/gradcam_icbhi.py
    python pf_tests/gradcam_icbhi.py --n-per-class 10 --seed 42
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.model import build_model, get_device, load_checkpoint
from dataset.icbhi_dataset import get_vtlp_filterbank

from config import (
    CHECKPOINTS_DIR,
    MANIFEST_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    MODEL_INPUT_SIZE,
)

CHECKPOINT_PATH = Path(CHECKPOINTS_DIR) / 'final' / 'best_lr_5e4_cl_0_3_e10_bw_cap.pt'
TEST_DIR        = Path(MANIFEST_PATH).parent / 'prepared' / 'test'
OUTPUT_DIR      = Path(__file__).parent / 'results' / 'gradcam_icbhi'

LABELS = ['normal', 'crackle', 'wheeze', 'both']

LABEL_COLORS = {
    'normal':  '#4CAF50',
    'crackle': '#FF8C00',
    'wheeze':  '#1E90FF',
    'both':    '#DC143C',
}

VTLP_ALPHA = 1.0
VTLP_FHI   = 3500.0


# ---------------------------------------------------------------------------
# Grad-CAM  (identical logic to gradcam.py)
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer, input_size=(224, 224)):
        self.model       = model
        self.input_size  = input_size
        self._activations = None
        self._gradients   = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._activations = output

    def _bwd_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def compute(
        self,
        tensor:    torch.Tensor,
        class_idx: int,
        bandwidth: torch.Tensor | None = None,
    ) -> np.ndarray:
        self.model.zero_grad()
        target_logits, _, _ = self.model(tensor, bandwidth=bandwidth)
        score = target_logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=self.input_size,
                                mode='bilinear', align_corners=False)
        cam     = cam[0, 0].detach().cpu().numpy()

        vmin, vmax = cam.min(), cam.max()
        if vmax > vmin:
            cam = (cam - vmin) / (vmax - vmin)
        else:
            cam = np.zeros_like(cam)
        return cam


# ---------------------------------------------------------------------------
# Audio → tensor (same pipeline as ICBHIDataset, no augmentation)
# ---------------------------------------------------------------------------

def load_cycle_tensor(wav_path: Path, vtlp_filterbank: np.ndarray,
                      device: torch.device) -> tuple[torch.Tensor, float]:
    """
    Load a prepared ICBHI cycle wav and convert to a (1, 3, 224, 224) tensor.

    Returns (tensor, original_duration_seconds).
    """
    audio, _ = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    orig_duration = len(audio) / SAMPLE_RATE

    # STFT → VTLP mel filterbank → log
    stft      = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    mel_spec  = vtlp_filterbank @ magnitude
    log_mel   = np.log(np.maximum(mel_spec, 1e-10))

    # Resize → normalise → 3-channel tensor
    h, w = MODEL_INPUT_SIZE
    img  = Image.fromarray(log_mel)
    img  = img.resize((w, h), Image.BILINEAR)
    spec = np.array(img)

    mean = spec.mean()
    std  = spec.std() + 1e-10
    spec = (spec - mean) / std

    tensor = torch.from_numpy(spec).float().unsqueeze(0).repeat(3, 1, 1)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, orig_duration


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def load_test_meditron_samples(manifest_path: Path,
                                test_dir: Path) -> dict[str, list[dict]]:
    """
    Return {label: [{'path': Path, 'recording': str, 'duration': float}, ...]}
    for non-stretched Meditron test cycles whose wav files exist locally.
    """
    by_label: dict[str, list[dict]] = {lbl: [] for lbl in LABELS}

    with open(manifest_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] != 'test':
                continue
            if row['device'] != 'Meditron':
                continue
            if row['is_stretched'] == 'True':
                continue
            wav_name = Path(row['wav_path']).name
            local_path = test_dir / wav_name
            if not local_path.exists():
                continue
            lbl = row['label']
            if lbl not in by_label:
                continue
            by_label[lbl].append({
                'path':      local_path,
                'recording': row['recording_id'],
                'duration':  float(row['duration']),
            })

    return by_label


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_class_grid(label: str, samples: list[dict],
                    tensors: list[torch.Tensor],
                    heatmaps: list[np.ndarray],
                    predictions: list[str],
                    confidences: list[float],
                    out_path: Path) -> None:
    n     = len(tensors)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.8),
                             squeeze=False)
    fig.suptitle(
        f'ICBHI Meditron test — true label: {label}',
        fontsize=11, fontweight='bold', y=1.01,
    )

    extent = [0, CYCLE_DURATION, FMIN, FMAX]

    for i, (tensor, heatmap, sample, pred, conf) in enumerate(
            zip(tensors, heatmaps, samples, predictions, confidences)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        spectrogram = tensor[0].cpu().numpy()  # first channel (224, 224)
        ax.imshow(spectrogram, origin='lower', aspect='auto',
                  extent=extent, cmap='magma', interpolation='bilinear')
        ax.imshow(heatmap, origin='lower', aspect='auto',
                  extent=extent, cmap='jet', alpha=0.45, interpolation='bilinear')

        # Dashed line where real cycle audio ends and reflect-padding begins
        if sample['duration'] < CYCLE_DURATION:
            ax.axvline(x=sample['duration'], color='white', linewidth=1.0,
                       linestyle='--', alpha=0.8)

        color = LABEL_COLORS[pred]
        rec   = sample['recording']
        ax.set_title(f"{rec}\n{pred} ({conf:.2f})",
                     fontsize=6, color=color, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=5)
        ax.set_ylabel('Freq (Hz)', fontsize=5)
        ax.tick_params(labelsize=4)

    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-per-class', type=int, default=8,
                        help='Number of cycles to visualise per class (default: 8)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for sample selection (default: 0)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────
    model, _ = build_model(pretrained=False, device=device)
    load_checkpoint(str(CHECKPOINT_PATH), model, optimiser=None, device=device)
    model.eval()

    target_layer = model.backbone.features[7]
    gradcam      = GradCAM(model, target_layer, input_size=(224, 224))

    # ── Pre-compute fixed VTLP filterbank (alpha=1.0, same as Eko inference)
    base_filterbank = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    freq_bins = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    vtlp_filterbank = get_vtlp_filterbank(
        n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE,
        fmin=FMIN, fmax=FMAX,
        alpha=VTLP_ALPHA, fhi=VTLP_FHI,
        base_filterbank=base_filterbank,
        freq_bins=freq_bins,
    )

    # ── Load manifest and select samples ──────────────────────────────────
    by_label = load_test_meditron_samples(Path(MANIFEST_PATH), TEST_DIR)

    for lbl, all_samples in by_label.items():
        print(f"\n  {lbl}: {len(all_samples)} available — selecting {args.n_per_class}")

    print(f"\nGenerating Grad-CAM for ICBHI Meditron test cycles...")

    for label in LABELS:
        pool = by_label[label]
        if not pool:
            print(f"\n[{label}] No samples available — skipping.")
            continue

        # Prefer samples from different recordings; shuffle then deduplicate
        random.shuffle(pool)
        seen_recs: set[str] = set()
        selected = []
        for s in pool:
            if s['recording'] not in seen_recs:
                selected.append(s)
                seen_recs.add(s['recording'])
            if len(selected) >= args.n_per_class:
                break
        # If not enough unique recordings, pad with any remaining
        if len(selected) < args.n_per_class:
            remaining = [s for s in pool if s not in selected]
            selected += remaining[:args.n_per_class - len(selected)]

        print(f"\n[{label}]  {len(selected)} cycle(s):")

        tensors_list:  list[torch.Tensor] = []
        heatmaps_list: list[np.ndarray]   = []
        preds_list:    list[str]          = []
        confs_list:    list[float]        = []

        # Iterate through a larger pool so skipped files don't reduce count
        candidates = selected + [s for s in pool if s not in selected]
        processed = 0
        for sample in candidates:
            if processed >= args.n_per_class:
                break
            try:
                tensor, orig_dur = load_cycle_tensor(
                    sample['path'], vtlp_filterbank, device,
                )
            except Exception as e:
                print(f"  Skipping {Path(sample['path']).name}: {e}")
                continue

            # Meditron is the reference device — normalised bandwidth = 1.0
            bandwidth = torch.tensor([1.0], device=device)

            # Run forward to get predicted class (no grad needed here)
            with torch.no_grad():
                logits, _, _ = model(tensor, bandwidth=bandwidth)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx  = int(probs.argmax())
            pred_lbl  = LABELS[pred_idx]
            conf      = float(probs[pred_idx])

            # Grad-CAM for the model's predicted class
            heatmap = gradcam.compute(tensor, class_idx=pred_idx, bandwidth=bandwidth)

            tensors_list.append(tensor.squeeze(0))  # (3, 224, 224)
            heatmaps_list.append(heatmap)
            preds_list.append(pred_lbl)
            confs_list.append(conf)
            processed += 1

            print(f"  {Path(sample['path']).name}  →  {pred_lbl} ({conf:.2f})")

        out_path = OUTPUT_DIR / f"{label}.png"
        plot_class_grid(label, selected, tensors_list, heatmaps_list,
                        preds_list, confs_list, out_path)

    print(f"\nGrad-CAM plots saved to '{OUTPUT_DIR}'.")


if __name__ == '__main__':
    main()
