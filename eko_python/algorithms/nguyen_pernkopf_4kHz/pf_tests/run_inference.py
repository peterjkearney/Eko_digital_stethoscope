"""
run_inference.py

Runs inference on all Eko recordings using the trained model from
checkpoints/best.pt.

Each recording is divided into fixed-length chunks. The model produces softmax
probabilities for each chunk; these are averaged across all chunks of the same
recording to give a single per-recording prediction.

Pipeline:
    1. Build model and load checkpoints/best.pt
    2. Load EkoInferenceDataset (spectrum correction + VTLP log-mel, on-the-fly)
    3. Forward pass on every chunk
    4. Average softmax probabilities per recording → predicted class
    5. Print results table and save to pf_tests/results/inference_results.csv

Usage:
    python pf_tests/run_inference.py
    python pf_tests/run_inference.py --chunk-duration 4.0
    python pf_tests/run_inference.py --no-correction
"""

import argparse
import csv
import sys
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from model import build_model, get_device, load_checkpoint
from dataset import IDX_TO_LABEL
from pf_tests.eko_inference_dataset import EkoInferenceDataset

from config import (
    CHECKPOINTS_DIR,
    EKO_PROJECT_ROOT,
    DEVICE_PROFILES_PATH,
    NUM_WORKERS,
)

CHECKPOINT_PATH   = CHECKPOINTS_DIR / 'full' / 'best.pt'
INPUT_DIR         = Path(EKO_PROJECT_ROOT) / 'data' / 'pf_samples'
RESULTS_DIR       = Path(__file__).parent / 'results'


# ---------------------------------------------------------------------------
# Filename parser  (Eko_p001_t01_mild_20250214.wav)
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> dict:
    stem    = Path(name).stem
    parts   = stem.split('_')
    patient = parts[1] if len(parts) > 1 else '?'
    trial   = parts[2] if len(parts) > 2 else '?'
    date_idx = next(
        (i for i in range(len(parts) - 1, -1, -1) if parts[i].isdigit()), None
    )
    severity = '_'.join(parts[3:date_idx]) if date_idx and date_idx > 3 else '?'
    return {'patient': patient, 'trial': trial, 'severity': severity}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-duration', type=float, default=4.0,
                        help='Chunk length in seconds (default: 4.0)')
    parser.add_argument('--no-correction', action='store_true',
                        help='Skip spectrum correction (diagnostic comparison)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: checkpoints/best.pt)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_PATH
    suffix          = '_nocorr' if args.no_correction else ''
    results_csv     = RESULTS_DIR / f'inference_results{suffix}.csv'
    chunk_csv       = RESULTS_DIR / f'inference_chunk_results{suffix}.csv'

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # ── Build model and load checkpoint ──────────────────────────────────
    print("=" * 60)
    print("EKO INFERENCE  (4 kHz pipeline)")
    print("=" * 60)
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Input dir:   {INPUT_DIR}")
    print(f"Chunk:       {args.chunk_duration}s")
    print("=" * 60)

    model, _ = build_model(pretrained=False, device=device)
    epoch, icbhi_score = load_checkpoint(str(checkpoint_path), model,
                                         optimiser=None, device=device)
    model.eval()

    # ── Build dataset and dataloader ─────────────────────────────────────
    dataset = EkoInferenceDataset(
        wav_dir=INPUT_DIR,
        profiles_path=DEVICE_PROFILES_PATH,
        chunk_duration=args.chunk_duration,
        apply_correction=not args.no_correction,
    )

    if len(dataset) == 0:
        print(f"\nNo chunks found. Check that '{INPUT_DIR}' contains wav files.")
        return

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == 'cuda',
    )

    # ── Inference ─────────────────────────────────────────────────────────
    file_probs:  dict[str, list[np.ndarray]] = defaultdict(list)
    file_chunks: dict[str, int]              = defaultdict(int)
    chunk_rows:  list[dict]                  = []

    print(f"\nRunning inference on {len(dataset)} chunks...")

    with torch.no_grad():
        for tensors, indices in loader:
            tensors = tensors.to(device)
            target_logits, _, _ = model(tensors)
            probs = F.softmax(target_logits, dim=1).cpu().numpy()   # (B, 4)

            for prob, idx in zip(probs, indices.numpy()):
                sample = dataset.samples[int(idx)]
                fname  = sample['file']
                file_probs[fname].append(prob)
                file_chunks[fname] += 1
                chunk_rows.append({
                    'file':         fname,
                    'chunk':        sample['chunk'],
                    'start_sample': sample['start'],
                    'end_sample':   sample['end'],
                    'prediction':   IDX_TO_LABEL[int(prob.argmax())],
                    'prob_normal':  round(float(prob[0]), 4),
                    'prob_crackle': round(float(prob[1]), 4),
                    'prob_wheeze':  round(float(prob[2]), 4),
                    'prob_both':    round(float(prob[3]), 4),
                })

    # ── Aggregate and print ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'File':<45} {'Pred':<8} {'Conf':>6}  "
          f"{'normal':>7} {'crackle':>7} {'wheeze':>7} {'both':>7}  N")
    print(f"{'-'*70}")

    rows = []
    for fname in sorted(file_probs.keys()):
        avg_probs  = np.stack(file_probs[fname]).mean(axis=0)
        pred_idx   = int(avg_probs.argmax())
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = float(avg_probs[pred_idx])
        n_chunks   = file_chunks[fname]
        meta       = parse_filename(fname)

        print(
            f"{fname:<45} {pred_label:<8} {confidence:>6.3f}  "
            f"{avg_probs[0]:>7.3f} {avg_probs[1]:>7.3f} "
            f"{avg_probs[2]:>7.3f} {avg_probs[3]:>7.3f}  {n_chunks}"
        )

        rows.append({
            'file':         fname,
            'patient':      meta['patient'],
            'trial':        meta['trial'],
            'severity':     meta['severity'],
            'prediction':   pred_label,
            'confidence':   round(confidence, 4),
            'prob_normal':  round(float(avg_probs[0]), 4),
            'prob_crackle': round(float(avg_probs[1]), 4),
            'prob_wheeze':  round(float(avg_probs[2]), 4),
            'prob_both':    round(float(avg_probs[3]), 4),
            'n_chunks':     n_chunks,
        })

    print(f"{'='*70}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    fieldnames = ['file', 'patient', 'trial', 'severity', 'prediction',
                  'confidence', 'prob_normal', 'prob_crackle',
                  'prob_wheeze', 'prob_both', 'n_chunks']

    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to '{results_csv}'.")

    chunk_fieldnames = ['file', 'chunk', 'start_sample', 'end_sample',
                        'prediction', 'prob_normal', 'prob_crackle',
                        'prob_wheeze', 'prob_both']
    with open(chunk_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=chunk_fieldnames)
        writer.writeheader()
        writer.writerows(chunk_rows)
    print(f"Chunk results saved to '{chunk_csv}'.")

    # ── Summary by severity ───────────────────────────────────────────────
    by_severity: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_severity[r['severity']].append(r['prediction'])

    print(f"\n{'Severity':<12}  Predictions")
    print("-" * 40)
    for sev in sorted(by_severity):
        preds = by_severity[sev]
        print(f"  {sev:<10}  {', '.join(preds)}")


if __name__ == '__main__':
    main()
