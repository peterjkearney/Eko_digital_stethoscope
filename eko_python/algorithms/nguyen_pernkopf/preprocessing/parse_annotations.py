"""
parse_annotations.py

Parses the ICBHI 2017 dataset annotation files and produces a master manifest
CSV containing one row per respiratory cycle with all relevant metadata.

ICBHI file naming convention:
    {patient_id}_{recording_index}_{chest_location}_{acquisition_mode}_{device}.wav
    e.g. 101_1b1_Al_sc_Meditron.wav

    patient_id        : integer patient identifier
    recording_index   : recording session identifier
    chest_location    : Tc / Al / Ar / Pl / Pr / Ll / Lr
    acquisition_mode  : sc (single channel) / mc (multi-channel)
    device            : AKGC417L / Meditron / LittC2SE / Litt3200

Annotation files (.txt) contain one row per respiratory cycle:
    {cycle_start} {cycle_end} {crackle} {wheeze}
    e.g. 0.036 1.115 1 0
    crackle and wheeze are binary (0/1)

Official split file contains one row per recording:
    {filename} {split}
    e.g. 101_1b1_Al_sc_Meditron train

Diagnosis file (patient_diagnosis.csv) contains:
    {patient_id},{diagnosis}
    e.g. 101,COPD

Output manifest.csv columns:
    recording_id      : base filename without extension
    patient_id        : integer
    cycle_index       : cycle number within the recording (0-indexed)
    cycle_start       : start time in seconds
    cycle_end         : end time in seconds
    crackle           : 0/1
    wheeze            : 0/1
    label             : normal / crackle / wheeze / both
    chest_location    : recording chest location
    acquisition_mode  : sc or mc
    device            : recording device name
    diagnosis         : patient-level disease diagnosis
    split             : train or test (from official split)
    duration          : cycle duration in seconds
    native_sr         : sample rate of the original wav file (Hz)
"""

import os
import re
import pandas as pd
import soundfile as sf
from pathlib import Path

# ---------------------------------------------------------------------------
# Import config — adjust the import path if your project layout differs
# ---------------------------------------------------------------------------
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_PATH, OFFICIAL_SPLIT_PATH, DIAGNOSIS_FILE_PATH, MANIFEST_PATH, FILENAME_CORRECTIONS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from (crackle, wheeze) binary tuple to human-readable label
LABEL_MAP = {
    (0, 0): 'normal',
    (1, 0): 'crackle',
    (0, 1): 'wheeze',
    (1, 1): 'both',
}

# Expected devices — used for validation
KNOWN_DEVICES = {'AKGC417L', 'Meditron', 'LittC2SE', 'Litt3200'}

# Expected chest locations — used for validation
KNOWN_LOCATIONS = {'Tc', 'Al', 'Ar', 'Pl', 'Pr', 'Ll', 'Lr'}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """
    Parse an ICBHI wav filename into its component metadata fields.

    Parameters
    ----------
    filename : str
        Base filename without extension, e.g. '101_1b1_Al_sc_Meditron'

    Returns
    -------
    dict with keys: patient_id, recording_index, chest_location,
                    acquisition_mode, device

    Raises
    ------
    ValueError if the filename does not match the expected pattern.
    """
    parts = filename.split('_')

    if len(parts) != 5:
        raise ValueError(
            f"Unexpected filename format '{filename}': "
            f"expected 5 underscore-separated fields, got {len(parts)}."
        )

    patient_id_str, recording_index, chest_location, acquisition_mode, device = parts

    # Validate patient_id is numeric
    if not patient_id_str.isdigit():
        raise ValueError(
            f"Expected numeric patient_id in '{filename}', got '{patient_id_str}'."
        )

    # Soft validation — warn but don't fail on unknown values
    # (the dataset has some inconsistencies)
    if chest_location not in KNOWN_LOCATIONS:
        print(f"  Warning: unknown chest location '{chest_location}' in '{filename}'.")
    if device not in KNOWN_DEVICES:
        print(f"  Warning: unknown device '{device}' in '{filename}'.")
    if acquisition_mode not in {'sc', 'mc'}:
        print(f"  Warning: unknown acquisition mode '{acquisition_mode}' in '{filename}'.")

    return {
        'patient_id':       int(patient_id_str),
        'recording_index':  recording_index,
        'chest_location':   chest_location,
        'acquisition_mode': acquisition_mode,
        'device':           device,
    }


# ---------------------------------------------------------------------------
# Annotation file parsing
# ---------------------------------------------------------------------------

def parse_annotation_file(txt_path: str) -> list[dict]:
    """
    Parse a single ICBHI annotation (.txt) file.

    Each line represents one respiratory cycle:
        cycle_start  cycle_end  crackle  wheeze

    Parameters
    ----------
    txt_path : str
        Full path to the annotation text file.

    Returns
    -------
    List of dicts, one per cycle, with keys:
        cycle_index, cycle_start, cycle_end, crackle, wheeze, label, duration
    """
    cycles = []

    with open(txt_path, 'r') as f:
        for cycle_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) != 4:
                print(
                    f"  Warning: unexpected number of fields ({len(parts)}) "
                    f"on line {cycle_index + 1} of '{txt_path}'. Skipping."
                )
                continue

            try:
                cycle_start = float(parts[0])
                cycle_end   = float(parts[1])
                crackle     = int(parts[2])
                wheeze      = int(parts[3])
            except ValueError as e:
                print(
                    f"  Warning: could not parse line {cycle_index + 1} "
                    f"of '{txt_path}': {e}. Skipping."
                )
                continue

            # Validate binary labels
            if crackle not in (0, 1) or wheeze not in (0, 1):
                print(
                    f"  Warning: non-binary crackle/wheeze values on line "
                    f"{cycle_index + 1} of '{txt_path}'. Skipping."
                )
                continue

            # Validate timing
            if cycle_end <= cycle_start:
                print(
                    f"  Warning: cycle_end <= cycle_start on line "
                    f"{cycle_index + 1} of '{txt_path}'. Skipping."
                )
                continue

            duration = round(cycle_end - cycle_start, 6)
            label    = LABEL_MAP[(crackle, wheeze)]

            cycles.append({
                'cycle_index': cycle_index,
                'cycle_start': cycle_start,
                'cycle_end':   cycle_end,
                'crackle':     crackle,
                'wheeze':      wheeze,
                'label':       label,
                'duration':    duration,
            })

    return cycles


# ---------------------------------------------------------------------------
# Official split file parsing
# ---------------------------------------------------------------------------

def parse_official_split(split_path: str) -> dict[str, str]:
    """
    Parse the official ICBHI train/test split file.

    Expected format (one recording per line):
        {filename_without_extension} {train|test}

    Parameters
    ----------
    split_path : str
        Full path to the official split file.

    Returns
    -------
    Dict mapping recording_id (str) → split (str: 'train' or 'test')
    """
    split_map = {}

    with open(split_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                print(
                    f"  Warning: unexpected format on line {line_num} "
                    f"of split file: '{line}'. Skipping."
                )
                continue

            filename, split = parts
            # Strip .wav extension if present
            recording_id = filename.replace('.wav', '')
            split = split.lower()

            if split not in ('train', 'test'):
                print(
                    f"  Warning: unknown split value '{split}' for "
                    f"'{recording_id}'. Skipping."
                )
                continue

            split_map[recording_id] = split

    return split_map


# ---------------------------------------------------------------------------
# Diagnosis file parsing
# ---------------------------------------------------------------------------

def parse_diagnosis_file(diagnosis_path: str) -> dict[int, str]:
    """
    Parse the ICBHI patient diagnosis CSV file.

    Expected format:
        {patient_id},{diagnosis}

    Parameters
    ----------
    diagnosis_path : str
        Full path to the diagnosis CSV file.

    Returns
    -------
    Dict mapping patient_id (int) → diagnosis (str)
    """
    diagnosis_map = {}

    with open(diagnosis_path, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Skip header if present
            if line_num == 1 and not line[0].isdigit():
                continue

            parts = line.split('\t') if '\t' in line else line.split(',')

            if len(parts) < 2:
                print(
                    f"  Warning: unexpected format on line {line_num} "
                    f"of diagnosis file: '{line}'. Skipping."
                )
                continue

            try:
                patient_id = int(parts[0].strip())
                diagnosis  = parts[1].strip()
            except ValueError as e:
                print(
                    f"  Warning: could not parse line {line_num} "
                    f"of diagnosis file: {e}. Skipping."
                )
                continue

            diagnosis_map[patient_id] = diagnosis

    return diagnosis_map


# ---------------------------------------------------------------------------
# Main manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    raw_data_path: str,
    official_split_path: str,
    diagnosis_file_path: str,
) -> pd.DataFrame:
    """
    Walk the raw data directory, parse all annotation files, and assemble
    a master manifest DataFrame with one row per respiratory cycle.

    Parameters
    ----------
    raw_data_path       : path to ICBHI raw data directory (wav + txt files)
    official_split_path : path to official train/test split file
    diagnosis_file_path : path to patient diagnosis CSV

    Returns
    -------
    pd.DataFrame with columns defined in the module docstring.
    """
    raw_data_path = Path(raw_data_path)




    print("Parsing official split file...")
    split_map = parse_official_split(official_split_path)
    print(f"  Found {len(split_map)} recordings in split file.")

    print("Parsing diagnosis file...")
    diagnosis_map = parse_diagnosis_file(diagnosis_file_path)
    print(f"  Found diagnoses for {len(diagnosis_map)} patients.")

    # Find all annotation txt files
    txt_files = sorted(raw_data_path.glob('*.txt'))
    print(f"\nFound {len(txt_files)} annotation files in '{raw_data_path}'.")

    rows = []
    skipped_recordings  = []
    skipped_no_split    = []
    skipped_no_diagnosis = []

    for txt_path in txt_files:
        recording_id = txt_path.stem  # filename without extension

        # Apply known filename corrections
        corrected_id = FILENAME_CORRECTIONS.get(recording_id, recording_id)
        if corrected_id != recording_id:
            print(f"  Applying filename correction: '{recording_id}' → '{corrected_id}'")
        # Use corrected_id for all subsequent lookups (split file, diagnosis, etc.)
        # but keep recording_id for locating the actual file on disk

        # ── Parse filename metadata ──────────────────────────────────────
        try:
            file_meta = parse_filename(corrected_id)
        except ValueError as e:
            print(f"  Skipping '{corrected_id}': {e}")
            skipped_recordings.append(corrected_id)
            continue

        patient_id = file_meta['patient_id']

        # ── Check wav file exists ────────────────────────────────────────
        wav_path = raw_data_path / f"{recording_id}.wav"
        if not wav_path.exists():
            print(f"  Skipping '{recording_id}': no matching wav file found.")
            skipped_recordings.append(recording_id)
            continue

        native_sr = sf.info(str(wav_path)).samplerate

        # ── Look up official split ───────────────────────────────────────
        if corrected_id not in split_map:
            print(f"  Skipping '{corrected_id}': not found in official split file.")
            skipped_no_split.append(corrected_id)
            continue

        split = split_map[corrected_id]

        # ── Look up patient diagnosis ────────────────────────────────────
        if patient_id not in diagnosis_map:
            print(f"  Warning: no diagnosis for patient {patient_id} "
                  f"(recording '{corrected_id}'). Setting to 'Unknown'.")
            skipped_no_diagnosis.append(corrected_id)
            diagnosis = 'Unknown'
        else:
            diagnosis = diagnosis_map[patient_id]

        # ── Parse annotation file ────────────────────────────────────────
        cycles = parse_annotation_file(str(txt_path))

        if not cycles:
            print(f"  Warning: no valid cycles found in '{txt_path.name}'. Skipping.")
            skipped_recordings.append(recording_id)
            continue

        # ── Build one row per cycle ──────────────────────────────────────
        for cycle in cycles:
            rows.append({
                'recording_id':    recording_id,
                'patient_id':      patient_id,
                'cycle_index':     cycle['cycle_index'],
                'cycle_start':     cycle['cycle_start'],
                'cycle_end':       cycle['cycle_end'],
                'crackle':         cycle['crackle'],
                'wheeze':          cycle['wheeze'],
                'label':           cycle['label'],
                'chest_location':  file_meta['chest_location'],
                'acquisition_mode': file_meta['acquisition_mode'],
                'device':          file_meta['device'],
                'diagnosis':       diagnosis,
                'split':           split,
                'duration':        cycle['duration'],
                'native_sr':       native_sr,
                # These columns will be populated by later preprocessing steps
                'wav_path':        '',   # filled by split_cycles.py
                'is_stretched':    False,  # filled by time_stretch.py
            })

    manifest = pd.DataFrame(rows)

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("MANIFEST SUMMARY")
    print("="*60)
    print(f"Total cycles:        {len(manifest)}")
    print(f"Total recordings:    {manifest['recording_id'].nunique()}")
    print(f"Total patients:      {manifest['patient_id'].nunique()}")
    print(f"\nCycles by split:")
    print(manifest.groupby('split').size().to_string())
    print(f"\nCycles by label:")
    print(manifest.groupby('label').size().to_string())
    print(f"\nCycles by device:")
    print(manifest.groupby('device').size().to_string())
    print(f"\nCycles by diagnosis:")
    print(manifest.groupby('diagnosis').size().to_string())
    print(f"\nCycle duration (seconds):")
    print(manifest['duration'].describe().round(3).to_string())

    if skipped_recordings:
        print(f"\nSkipped recordings (parse error / missing wav): "
              f"{len(skipped_recordings)}")
    if skipped_no_split:
        print(f"Skipped recordings (not in split file): {len(skipped_no_split)}")
    if skipped_no_diagnosis:
        print(f"Recordings with unknown diagnosis: {len(skipped_no_diagnosis)}")

    print("="*60)

    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print("ICBHI ANNOTATION PARSER")
    print("="*60)
    print(f"Raw data path:    {RAW_DATA_PATH}")
    print(f"Official split:   {OFFICIAL_SPLIT_PATH}")
    print(f"Diagnosis file:   {DIAGNOSIS_FILE_PATH}")
    print(f"Output manifest:  {MANIFEST_PATH}")
    print("="*60 + "\n")

    # Validate input paths exist before doing any work
    for label, path in [
        ("Raw data directory", RAW_DATA_PATH),
        ("Official split file", OFFICIAL_SPLIT_PATH),
        ("Diagnosis file", DIAGNOSIS_FILE_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: '{path}'")

    # Ensure output directory exists
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        raw_data_path=RAW_DATA_PATH,
        official_split_path=OFFICIAL_SPLIT_PATH,
        diagnosis_file_path=DIAGNOSIS_FILE_PATH,
    )

    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest saved to '{MANIFEST_PATH}'.")


if __name__ == '__main__':
    main()