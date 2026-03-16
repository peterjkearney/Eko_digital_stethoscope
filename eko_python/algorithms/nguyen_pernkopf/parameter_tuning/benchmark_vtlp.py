"""
benchmark_vtlp.py

Benchmarks the original (loop-based) and vectorised VTLP filterbank
implementations against each other.

Usage:
    python benchmark_vtlp.py
    python benchmark_vtlp.py --n-runs 100 --n-mels 128 --n-fft 512 --sr 16000
"""

import argparse
import time
import numpy as np
import librosa


# ---------------------------------------------------------------------------
# Original implementation (loop-based)
# ---------------------------------------------------------------------------

def get_vtlp_filterbank_original(n_mels, n_fft, sr, fmin, fmax, alpha, fhi):
    filterbank = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyquist   = sr / 2.0

    warped_bins = np.where(
        freq_bins <= fhi,
        freq_bins * alpha,
        freq_bins + (freq_bins - fhi) * (alpha * fhi / (nyquist - fhi) - 1)
        * (nyquist - freq_bins) / (nyquist - fhi)
    )
    warped_bins = np.clip(warped_bins, 0, nyquist)

    warped_filterbank = np.zeros_like(filterbank)
    for i, wf in enumerate(warped_bins):
        idx = np.searchsorted(freq_bins, wf)
        if idx == 0:
            warped_filterbank[:, i] = filterbank[:, 0]
        elif idx >= len(freq_bins):
            warped_filterbank[:, i] = filterbank[:, -1]
        else:
            lo, hi = freq_bins[idx - 1], freq_bins[idx]
            t      = (wf - lo) / (hi - lo + 1e-10)
            warped_filterbank[:, i] = (
                (1 - t) * filterbank[:, idx - 1] + t * filterbank[:, idx]
            )

    return warped_filterbank


# ---------------------------------------------------------------------------
# Vectorised implementation
# ---------------------------------------------------------------------------

def get_vtlp_filterbank_vectorised(n_mels, n_fft, sr, fmin, fmax, alpha, fhi):
    filterbank = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )

    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyquist   = sr / 2.0

    warped_bins = np.where(
        freq_bins <= fhi,
        freq_bins * alpha,
        freq_bins + (freq_bins - fhi) * (alpha * fhi / (nyquist - fhi) - 1)
        * (nyquist - freq_bins) / (nyquist - fhi)
    )
    warped_bins = np.clip(warped_bins, 0, nyquist)

    indices = np.searchsorted(freq_bins, warped_bins)
    indices = np.clip(indices, 1, len(freq_bins) - 1)

    lo = freq_bins[indices - 1]
    hi = freq_bins[indices]
    t  = (warped_bins - lo) / (hi - lo + 1e-10)

    warped_filterbank = (
        (1 - t) * np.take(filterbank, indices - 1, axis=1) +
        t       * np.take(filterbank, indices,     axis=1)
    )

    return warped_filterbank


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(fn, n_runs, kwargs, label):
    # Warm-up run to avoid first-call overhead
    fn(**kwargs)

    times = []
    for _ in range(n_runs):
        # Fresh random alpha and fhi each run to simulate real usage
        kwargs['alpha'] = np.random.uniform(0.9, 1.1)
        kwargs['fhi']   = np.random.uniform(3200, 3800)

        start = time.perf_counter()
        fn(**kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # convert to ms

    times   = np.array(times)
    print(f"\n{label}")
    print(f"  Runs:    {n_runs}")
    print(f"  Mean:    {times.mean():.3f} ms")
    print(f"  Std:     {times.std():.3f} ms")
    print(f"  Min:     {times.min():.3f} ms")
    print(f"  Max:     {times.max():.3f} ms")
    print(f"  Total:   {times.sum():.1f} ms")

    return times


def verify_outputs_match(kwargs):
    """Check that both implementations produce the same output."""
    out_original   = get_vtlp_filterbank_original(**kwargs)
    out_vectorised = get_vtlp_filterbank_vectorised(**kwargs)
    max_diff       = np.max(np.abs(out_original - out_vectorised))
    match          = np.allclose(out_original, out_vectorised, atol=1e-6)
    print(f"\nOutput verification:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Outputs match (atol=1e-6): {match}")
    return match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark VTLP filterbank implementations.")
    parser.add_argument('--n-runs',  type=int,   default=50,    help="Number of timed runs per implementation")
    parser.add_argument('--n-mels',  type=int,   default=128,   help="Number of mel filters")
    parser.add_argument('--n-fft',   type=int,   default=512,   help="FFT window size")
    parser.add_argument('--sr',      type=int,   default=16000, help="Sample rate")
    parser.add_argument('--fmin',    type=float, default=50.0,  help="Minimum frequency")
    parser.add_argument('--fmax',    type=float, default=8000.0,help="Maximum frequency")
    args = parser.parse_args()

    kwargs = dict(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        alpha=1.0,
        fhi=3500.0,
    )

    print("="*60)
    print("VTLP FILTERBANK BENCHMARK")
    print("="*60)
    print(f"  n_mels:  {args.n_mels}")
    print(f"  n_fft:   {args.n_fft}")
    print(f"  sr:      {args.sr}")
    print(f"  fmin:    {args.fmin}")
    print(f"  fmax:    {args.fmax}")
    print(f"  n_runs:  {args.n_runs}")

    # Verify outputs match before benchmarking
    verify_outputs_match(kwargs)

    # Benchmark both
    times_original   = run_benchmark(
        get_vtlp_filterbank_original,
        n_runs=args.n_runs,
        kwargs=dict(**kwargs),
        label="ORIGINAL (loop-based)",
    )
    times_vectorised = run_benchmark(
        get_vtlp_filterbank_vectorised,
        n_runs=args.n_runs,
        kwargs=dict(**kwargs),
        label="VECTORISED (numpy)",
    )

    # Summary
    speedup = times_original.mean() / times_vectorised.mean()
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Original mean:   {times_original.mean():.3f} ms")
    print(f"  Vectorised mean: {times_vectorised.mean():.3f} ms")
    if speedup >= 1:
        print(f"  Speedup:         {speedup:.1f}x faster (vectorised)")
    else:
        print(f"  Speedup:         {1/speedup:.1f}x slower (vectorised) ← unexpected")
    print("="*60)


if __name__ == '__main__':
    main()