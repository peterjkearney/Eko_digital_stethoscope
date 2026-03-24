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
(produced by run_inference.py) to annotate each subplot.

Usage:
    python pf_tests/gradcam.py
    python pf_tests/gradcam.py --no-correction
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
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model import build_model, get_device, load_checkpoint
from dataset import IDX_TO_LABEL
from pf_tests.eko_inference_dataset import EkoInferenceDataset

from config import (
    CHECKPOINTS_DIR,
    EKO_PROJECT_ROOT,
    DEVICE_PROFILES_PATH,
    SAMPLE_RATE,
    CYCLE_DURATION,
    FMIN,
    FMAX,
)

CHECKPOINT_PATH   = CHECKPOINTS_DIR / 'full' /'best.pt'
INPUT_DIR         = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples'
CHUNK_RESULTS_CSV = Path(__file__).parent / 'results' / 'inference_chunk_results.csv'
OUTPUT_DIR        = Path(__file__).parent / 'results' / 'gradcam'

LABEL_COLORS = {
    'no_crackle': '#4CAF50',
    'crackle':    '#FF8C00',
}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for CoTuningModel.

    Hooks into target_layer to capture forward activations and backward
    gradients, then computes a spatially-upsampled heatmap.
    """

    def __init__(self, model, target_layer, input_size=(224, 224)):
        self.model        = model
        self.input_size   = input_size
        self._activations = None
        self._gradients   = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._activations = output

    def _bwd_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def compute(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Parameters
        ----------
        tensor    : (1, 3, H, W) input tensor on the same device as model
        class_idx : target class index to explain

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.zero_grad()

        target_logits, _, _ = self.model(tensor)
        score = target_logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        cam = F.interpolate(cam, size=self.input_size,
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

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
    by_file: dict[str, list[dict]] = defaultdict(list)
    chunk_duration = 4.0

    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            row['chunk']        = int(row['chunk'])
            row['start_sample'] = int(row['start_sample'])
            row['end_sample']   = int(row['end_sample'])
            for key in ('prob_no_crackle', 'prob_crackle'):
                row[key] = float(row[key])
            by_file[row['file']].append(row)

    for chunks in by_file.values():
        chunks.sort(key=lambda r: r['chunk'])

    for chunks in by_file.values():
        if len(chunks) >= 2:
            chunk_duration = (chunks[1]['start_sample'] - chunks[0]['start_sample']) / SAMPLE_RATE
            break
        elif len(chunks) == 1:
            chunk_duration = (chunks[0]['end_sample'] - chunks[0]['start_sample']) / SAMPLE_RATE
            break

    print(f"Inferred chunk duration from CSV: {chunk_duration:.1f}s")
    return dict(by_file), chunk_duration


def plot_recording(
    fname:          str,
    tensors:        list[torch.Tensor],
    heatmaps:       list[np.ndarray],
    chunk_meta:     list[dict],
    chunk_duration: float,
    out_path:       Path,
) -> None:
    n     = len(tensors)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.6),
                             squeeze=False)
    fig.suptitle(fname, fontsize=10, fontweight='bold', y=1.01)

    extent = [0, CYCLE_DURATION, FMIN, FMAX]

    for i, (tensor, heatmap, meta) in enumerate(zip(tensors, heatmaps, chunk_meta)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]

        spectrogram = tensor[0].cpu().numpy()
        ax.imshow(spectrogram, origin='lower', aspect='auto',
                  extent=extent, cmap='magma', interpolation='bilinear')
        ax.imshow(heatmap, origin='lower', aspect='auto',
                  extent=extent, cmap='jet', alpha=0.45, interpolation='bilinear')

        real_end_s = (meta['end_sample'] - meta['start_sample']) / SAMPLE_RATE
        if real_end_s < CYCLE_DURATION:
            ax.axvline(x=real_end_s, color='white', linewidth=1.0,
                       linestyle='--', alpha=0.8)

        t_start = meta['start_sample'] / SAMPLE_RATE
        t_end   = meta['end_sample']   / SAMPLE_RATE
        label   = meta['prediction']
        conf    = max(meta['prob_normal'], meta['prob_crackle'],
                      meta['prob_wheeze'], meta['prob_both'])
        color   = LABEL_COLORS[label]

        ax.set_title(f"{t_start:.1f}–{t_end:.1f}s\n{label} ({conf:.2f})",
                     fontsize=7, color=color, fontweight='bold')
        ax.set_xlabel('Time (s, dashed = padding)', fontsize=5)
        ax.set_ylabel('Freq (Hz)', fontsize=6)
        ax.tick_params(labelsize=5)

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
    parser.add_argument('--no-correction', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
    suffix          = '_nocorr' if args.no_correction else ''
    chunk_csv       = CHUNK_RESULTS_CSV.parent / f'inference_chunk_results{suffix}.csv'
    output_dir      = OUTPUT_DIR.parent / f'gradcam{suffix}'

    if not chunk_csv.exists():
        script = f"python pf_tests/run_inference.py{'  --no-correction' if args.no_correction else ''}"
        print(f"Chunk results not found: '{chunk_csv}'")
        print(f"Run '{script}' first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────
    model, _ = build_model(pretrained=False, device=device)
    load_checkpoint(str(checkpoint_path), model, optimiser=None, device=device)
    model.eval()

    # Hook layer4 (index 7 in backbone.features Sequential)
    target_layer = model.backbone.features[7]
    gradcam = GradCAM(model, target_layer, input_size=(224, 224))

    # ── Load per-chunk metadata and rebuild dataset ───────────────────────
    chunk_meta_by_file, chunk_duration = load_chunk_meta(chunk_csv)

    dataset = EkoInferenceDataset(
        wav_dir=INPUT_DIR,
        profiles_path=DEVICE_PROFILES_PATH,
        chunk_duration=chunk_duration,
        apply_correction=not args.no_correction,
    )

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

            tensor, _ = dataset[idx]                           # (3, 224, 224)
            tensor    = tensor.unsqueeze(0).to(device)         # (1, 3, 224, 224)

            pred_idx = ['no_crackle', 'crackle'].index(meta['prediction'])
            heatmap  = gradcam.compute(tensor, class_idx=pred_idx)

            tensors_list.append(tensor.squeeze(0))
            heatmaps_list.append(heatmap)

        if not tensors_list:
            continue

        out_path = output_dir / f"{Path(fname).stem}.png"
        plot_recording(fname, tensors_list, heatmaps_list, chunks,
                       chunk_duration, out_path)

    print(f"\nGrad-CAM plots saved to '{output_dir}'.")


if __name__ == '__main__':
    main()
