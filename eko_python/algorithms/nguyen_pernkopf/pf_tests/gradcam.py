"""
gradcam.py

Generates Grad-CAM visualisations for each chunk of every Eko recording.

For each chunk the script:
    1. Runs a forward pass with gradients enabled.
    2. Backpropagates through the predicted class score.
    3. Computes Grad-CAM from layer4 activations and gradients.
    4. Overlays the heatmap on the log-mel spectrogram.

Output: one PNG per recording saved to pf_tests/results/gradcam/.
Each PNG shows all chunks of that recording in a grid.

Reads per-chunk predictions from pf_tests/results/inference_chunk_results.csv
(produced by run_inference.py) to annotate each subplot with the predicted
class and confidence.

Usage:
    python pf_tests/gradcam.py
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.model import build_model, get_device, load_checkpoint
from dataset.icbhi_dataset import IDX_TO_LABEL
from pf_tests.eko_inference_dataset import EkoInferenceDataset

from config import (
    CHECKPOINTS_DIR,
    EKO_PROJECT_ROOT,
    SPECTRUM_CORRECTION_PROFILES_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    FMIN,
    FMAX,
)

CHECKPOINT_PATH   = Path(CHECKPOINTS_DIR) / 'final' / 'best.pt'
INPUT_DIR         = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples' / 'preprocessed'
CHUNK_RESULTS_CSV = Path(__file__).parent / 'results' / 'inference_chunk_results.csv'
OUTPUT_DIR        = Path(__file__).parent / 'results' / 'gradcam'

LABEL_COLORS = {
    'normal':  '#4CAF50',
    'crackle': '#FF8C00',
    'wheeze':  '#1E90FF',
    'both':    '#DC143C',
}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for a CoTuningModel.

    Hooks into target_layer to capture forward activations and backward
    gradients, then computes a spatially-upsampled heatmap.

    Parameters
    ----------
    model        : CoTuningModel in eval mode
    target_layer : nn.Module to hook (e.g. model.backbone.features[7])
    input_size   : (H, W) to upsample the heatmap to (default: (224, 224))
    """

    def __init__(self, model, target_layer, input_size=(224, 224)):
        self.model       = model
        self.input_size  = input_size
        self._activations = None
        self._gradients   = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._activations = output  # (B, C, h, w)

    def _bwd_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]  # (B, C, h, w)

    def compute(
        self,
        tensor:    torch.Tensor,
        class_idx: int,
        bandwidth: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single input.

        Parameters
        ----------
        tensor    : (1, 3, H, W) input tensor (on same device as model)
        class_idx : target class index to explain
        bandwidth : optional (1,) bandwidth scalar tensor

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.zero_grad()

        target_logits, _, _ = self.model(tensor, bandwidth=bandwidth)
        score = target_logits[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=self.input_size,
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()   # (H, W)

        # Normalise to [0, 1]
        vmin, vmax = cam.min(), cam.max()
        if vmax > vmin:
            cam = (cam - vmin) / (vmax - vmin)
        else:
            cam = np.zeros_like(cam)

        return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunk_meta(csv_path: Path) -> tuple[dict[str, list[dict]], float]:
    """
    Return ({filename: [chunk_row, ...]}, chunk_duration_seconds).

    Chunk duration is inferred from the first row's sample range so that
    the dataset is always rebuilt with the same chunking used for inference.
    """
    by_file: dict[str, list[dict]] = defaultdict(list)
    chunk_duration = 4.0  # fallback

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            row['chunk']        = int(row['chunk'])
            row['start_sample'] = int(row['start_sample'])
            row['end_sample']   = int(row['end_sample'])
            for key in ('prob_normal', 'prob_crackle', 'prob_wheeze', 'prob_both'):
                row[key] = float(row[key])
            by_file[row['file']].append(row)

    for chunks in by_file.values():
        chunks.sort(key=lambda r: r['chunk'])

    # Infer chunk duration from a chunk-0 row (end - start of the raw slice,
    # before reflect-padding, so use start_sample of chunk 1 minus chunk 0).
    for chunks in by_file.values():
        if len(chunks) >= 2:
            chunk_duration = (chunks[1]['start_sample'] - chunks[0]['start_sample']) / SAMPLE_RATE
            break
        elif len(chunks) == 1:
            chunk_duration = (chunks[0]['end_sample'] - chunks[0]['start_sample']) / SAMPLE_RATE
            break

    print(f"Inferred chunk duration from CSV: {chunk_duration:.1f}s")
    return dict(by_file), chunk_duration


def plot_recording(fname: str, tensors: list[torch.Tensor],
                   heatmaps: list[np.ndarray], chunk_meta: list[dict],
                   chunk_duration: float, out_path: Path) -> None:
    n = len(tensors)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6),
                             squeeze=False)
    fig.suptitle(fname, fontsize=10, fontweight='bold', y=1.01)

    # Extent for imshow: [xmin, xmax, ymin, ymax]
    # X axis spans the full padded duration the model sees (CYCLE_DURATION).
    # A vertical line at chunk_duration marks where real audio ends.
    extent = [0, CYCLE_DURATION, FMIN, FMAX]

    for i, (tensor, heatmap, meta) in enumerate(zip(tensors, heatmaps, chunk_meta)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        # Spectrogram (channel 0, normalised — still shows pattern)
        spectrogram = tensor[0].cpu().numpy()  # (224, 224)
        ax.imshow(spectrogram, origin='lower', aspect='auto',
                  extent=extent, cmap='magma', interpolation='bilinear')

        # Grad-CAM overlay
        ax.imshow(heatmap, origin='lower', aspect='auto',
                  extent=extent, cmap='jet', alpha=0.45, interpolation='bilinear')

        # Mark where real audio ends and reflect-padding begins
        real_end = meta['end_sample'] - meta['start_sample']
        real_end_s = real_end / SAMPLE_RATE
        if real_end_s < CYCLE_DURATION:
            ax.axvline(x=real_end_s, color='white', linewidth=1.0,
                       linestyle='--', alpha=0.8)

        # Chunk time range as subtitle
        t_start = meta['start_sample'] / SAMPLE_RATE
        t_end   = meta['end_sample']   / SAMPLE_RATE
        label   = meta['prediction']
        conf    = max(meta['prob_normal'], meta['prob_crackle'],
                      meta['prob_wheeze'], meta['prob_both'])
        color   = LABEL_COLORS[label]

        ax.set_title(f"{t_start:.1f}–{t_end:.1f}s\n{label} ({conf:.2f})",
                     fontsize=7, color=color, fontweight='bold')
        ax.set_xlabel(f'Time (s, dashed = padding)', fontsize=5)
        ax.set_ylabel('Freq (Hz)', fontsize=6)
        ax.tick_params(labelsize=5)

    # Hide unused subplots
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
    parser.add_argument('--no-correction', action='store_true',
                        help='Skip spectrum correction (compare against corrected run)')
    args = parser.parse_args()

    suffix     = '_nocorr' if args.no_correction else ''
    chunk_csv  = CHUNK_RESULTS_CSV.parent / f'inference_chunk_results{suffix}.csv'
    output_dir = OUTPUT_DIR.parent / f'gradcam{suffix}'

    if not chunk_csv.exists():
        script = f"python pf_tests/run_inference.py{'  --no-correction' if args.no_correction else ''}"
        print(f"Chunk results not found: '{chunk_csv}'")
        print(f"Run '{script}' first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────
    model, _ = build_model(pretrained=False, device=device)
    load_checkpoint(str(CHECKPOINT_PATH), model, optimiser=None, device=device)
    model.eval()

    # Hook layer4 (index 7 in backbone.features Sequential)
    target_layer = model.backbone.features[7]
    gradcam = GradCAM(model, target_layer, input_size=(224, 224))

    # ── Load per-chunk metadata ───────────────────────────────────────────
    chunk_meta_by_file, chunk_duration = load_chunk_meta(chunk_csv)

    # ── Build dataset with the same chunk duration used for inference ─────
    dataset = EkoInferenceDataset(
        wav_dir=INPUT_DIR,
        profiles_path=SPECTRUM_CORRECTION_PROFILES_PATH,
        chunk_duration=chunk_duration,
        apply_correction=not args.no_correction,
    )

    # Build a lookup: (file, chunk_idx) → dataset sample index
    sample_index: dict[tuple[str, int], int] = {
        (s['file'], s['chunk']): i for i, s in enumerate(dataset.samples)
    }

    print(f"\nGenerating Grad-CAM for {len(chunk_meta_by_file)} recording(s)...")

    for fname, chunks in sorted(chunk_meta_by_file.items()):
        tensors_list:  list[torch.Tensor] = []
        heatmaps_list: list[np.ndarray]   = []

        for meta in chunks:
            idx = sample_index.get((fname, meta['chunk']))
            if idx is None:
                print(f"  Warning: chunk {meta['chunk']} of '{fname}' not in dataset.")
                continue

            tensor, bandwidth, _ = dataset[idx]               # (3, 224, 224), scalar
            tensor    = tensor.unsqueeze(0).to(device)        # (1, 3, 224, 224)
            bandwidth = bandwidth.unsqueeze(0).to(device)     # (1,)
            tensor.requires_grad_(False)                      # hooks handle grad flow

            pred_idx = ['normal', 'crackle', 'wheeze', 'both'].index(meta['prediction'])
            heatmap  = gradcam.compute(tensor, class_idx=pred_idx, bandwidth=bandwidth)

            tensors_list.append(tensor.squeeze(0))
            heatmaps_list.append(heatmap)

        if not tensors_list:
            continue

        out_path = output_dir / f"{Path(fname).stem}.png"
        plot_recording(fname, tensors_list, heatmaps_list, chunks, chunk_duration, out_path)

    print(f"\nGrad-CAM plots saved to '{output_dir}'.")


if __name__ == '__main__':
    main()
