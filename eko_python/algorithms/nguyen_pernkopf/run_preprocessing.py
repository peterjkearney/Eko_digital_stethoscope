"""
run_preprocessing.py

Single entry point for the full ICBHI preprocessing pipeline.

Runs the following steps in order:

    1. parse_annotations.py  — parse annotation files, build manifest
    2. spectrum_correction.py — normalise recordings across devices
    3. split_cycles.py        — extract individual cycles from corrected recordings
    4. time_stretch.py        — create stretched copies of minority classes
    5. pad_cycles.py          — reflect pad all cycles to fixed length

Each step is idempotent — if output files already exist they are skipped,
so the pipeline can be resumed after interruption without reprocessing
everything from scratch.

Usage:
    python run_preprocessing.py              # run all steps
    python run_preprocessing.py --from 3     # resume from step 3
    python run_preprocessing.py --only 2     # run only step 2
"""

import argparse
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    RAW_DATA_PATH,
    OFFICIAL_SPLIT_PATH,
    DIAGNOSIS_FILE_PATH,
    MANIFEST_PATH,
    PREPARED_DIR,
    SAMPLE_RATE,
    CYCLE_DURATION,
    TIME_STRETCH_RATES,
    MINORITY_CLASSES,
    REFERENCE_DEVICE,
    SPECTRUM_CORRECTION_PROFILES_PATH,
    N_FFT,
    HOP_LENGTH,
)

from preprocessing.parse_annotations import build_manifest
from preprocessing.spectrum_correction import run_spectrum_correction
from preprocessing.split_cycles import run_split_cycles
from preprocessing.time_stretch import run_time_stretch
from preprocessing.pad_cycles import run_pad_cycles


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = {
    1: "Parse annotations → build manifest",
    2: "Spectrum correction",
    3: "Split cycles",
    4: "Time stretch minority classes",
    5: "Pad cycles to fixed length",
}


def print_header(step_num: int, description: str) -> None:
    print("\n" + "="*60)
    print(f"STEP {step_num}: {description}")
    print("="*60)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def step_1_parse_annotations() -> None:
    manifest = build_manifest(
        raw_data_path=RAW_DATA_PATH,
        official_split_path=OFFICIAL_SPLIT_PATH,
        diagnosis_file_path=DIAGNOSIS_FILE_PATH,
    )
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest saved to '{MANIFEST_PATH}'.")


def step_2_spectrum_correction() -> None:
    run_spectrum_correction(
        manifest_path=MANIFEST_PATH,
        raw_data_path=RAW_DATA_PATH,
        output_dir=str(Path(PREPARED_DIR) / 'corrected'),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        reference_device=REFERENCE_DEVICE,
        profiles_save_path=SPECTRUM_CORRECTION_PROFILES_PATH,
    )


def step_3_split_cycles() -> None:
    run_split_cycles(
        manifest_path=MANIFEST_PATH,
        corrected_dir=str(Path(PREPARED_DIR) / 'corrected'),
        output_dir=PREPARED_DIR,
        sr=SAMPLE_RATE,
    )


def step_4_time_stretch() -> None:
    run_time_stretch(
        manifest_path=MANIFEST_PATH,
        output_dir=PREPARED_DIR,
        sr=SAMPLE_RATE,
        stretch_rate_range=TIME_STRETCH_RATES,
        minority_classes=MINORITY_CLASSES,
    )


def step_5_pad_cycles() -> None:
    run_pad_cycles(
        manifest_path=MANIFEST_PATH,
        sr=SAMPLE_RATE,
        cycle_duration=CYCLE_DURATION,
    )


STEP_FUNCTIONS = {
    1: step_1_parse_annotations,
    2: step_2_spectrum_correction,
    3: step_3_split_cycles,
    4: step_4_time_stretch,
    5: step_5_pad_cycles,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs() -> None:
    """Check that required input files exist before starting."""
    required = {
        "Raw data directory":  RAW_DATA_PATH,
        "Official split file": OFFICIAL_SPLIT_PATH,
        "Diagnosis file":      DIAGNOSIS_FILE_PATH,
    }
    missing = [
        f"  {label}: '{path}'"
        for label, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "The following required inputs were not found:\n" +
            "\n".join(missing)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ICBHI preprocessing pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--from', dest='from_step', type=int, metavar='N',
        help="Start from step N (default: 1)"
    )
    group.add_argument(
        '--only', dest='only_step', type=int, metavar='N',
        help="Run only step N"
    )
    args = parser.parse_args()

    # Determine which steps to run
    if args.only_step is not None:
        if args.only_step not in STEPS:
            parser.error(f"--only must be between 1 and {len(STEPS)}")
        steps_to_run = [args.only_step]
    elif args.from_step is not None:
        if args.from_step not in STEPS:
            parser.error(f"--from must be between 1 and {len(STEPS)}")
        steps_to_run = list(range(args.from_step, len(STEPS) + 1))
    else:
        steps_to_run = list(range(1, len(STEPS) + 1))

    # Print pipeline overview
    print("="*60)
    print("ICBHI PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Steps to run: {steps_to_run}")
    print(f"Raw data:     {RAW_DATA_PATH}")
    print(f"Output:       {PREPARED_DIR}")
    print("="*60)

    # Validate inputs before doing any work
    # (only needed if step 1 is included, but check always to catch config errors)
    validate_inputs()

    # Run steps
    pipeline_start = time.time()
    step_times     = {}

    for step_num in steps_to_run:
        description  = STEPS[step_num]
        step_fn      = STEP_FUNCTIONS[step_num]

        print_header(step_num, description)
        step_start = time.time()

        try:
            step_fn()
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"PIPELINE FAILED at step {step_num}: {description}")
            print(f"Error: {e}")
            print(f"{'='*60}")
            raise

        elapsed = time.time() - step_start
        step_times[step_num] = elapsed
        print(f"\nStep {step_num} completed in {format_elapsed(elapsed)}.")

    # Final summary
    total_elapsed = time.time() - pipeline_start
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*60)
    for step_num, elapsed in step_times.items():
        print(f"  Step {step_num} ({STEPS[step_num]}): {format_elapsed(elapsed)}")
    print(f"  Total: {format_elapsed(total_elapsed)}")
    print("="*60)


if __name__ == '__main__':
    main()