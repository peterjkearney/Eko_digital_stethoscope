"""
01_parse_annotations.py

Parses the ICBHI 2017 annotation files and produces manifest.csv — one row
per respiratory cycle with all metadata needed by downstream steps.

Cycles longer than CYCLE_DURATION (config.py) are dropped here. All
remaining cycles will be padded to CYCLE_DURATION in step 03.

ICBHI filename format:
    {patient_id}_{recording_index}_{chest_location}_{acquisition_mode}_{device}
    e.g. 101_1b1_Al_sc_Meditron

Annotation file format (one line per cycle):
    {cycle_start}  {cycle_end}  {crackle}  {wheeze}
    e.g. 0.036 1.115 1 0

Output manifest.csv columns:
    recording_id      : base filename without extension
    patient_id        : integer
    cycle_index       : 0-indexed position within the recording
    cycle_start       : start time in seconds
    cycle_end         : end time in seconds
    duration          : cycle_end - cycle_start in seconds
    crackle           : 0 or 1
    wheeze            : 0 or 1
    label             : normal / crackle / wheeze / both
    chest_location    : Tc / Al / Ar / Pl / Pr / Ll / Lr
    acquisition_mode  : sc or mc
    device            : AKGC417L / Meditron / LittC2SE / Litt3200
    native_sr         : sample rate of the raw wav file (Hz)
    diagnosis         : patient-level diagnosis
    split             : train or test
"""

import sys
import pandas as pd
import soundfile as sf
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    RAW_DATA_PATH,
    OFFICIAL_SPLIT_PATH,
    DIAGNOSIS_FILE_PATH,
    MANIFEST_PATH,
    FILENAME_CORRECTIONS,
    CYCLE_DURATION,
)

LABEL_MAP = {
    (0, 0): 'normal',
    (1, 0): 'crackle',
    (0, 1): 'wheeze',
    (1, 1): 'both',
}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_official_split(split_path: Path) -> dict[str, str]:
    #official split file gives recording name and either 'test' or 'train'
    split_map = {}
    for line in split_path.read_text().splitlines():
        parts = line.strip().split() #remove white space and split consituents parts into list i.e. [106_2b1_Pr_mc_LittC2SE, train]
        if len(parts) != 2:
            continue
        recording_id = parts[0].replace('.wav', '') #remove file type (if present) and record recording name
        split = parts[1].lower() #ensure 'test' or 'train' is in lower case
        if split in ('train', 'test'):
            split_map[recording_id] = split
    return split_map

def parse_diagnosis_file(diagnosis_path: Path) -> dict[int, str]:
    #creates a dictionary of patient number and disease from the diagnosis file
    diagnosis_map = {}
    for line_num, line in enumerate(diagnosis_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or (line_num == 1 and not line[0].isdigit()):
            continue
        parts = line.split('\t') if '\t' in line else line.split(',')
        if len(parts) < 2:
            continue
        try:
            diagnosis_map[int(parts[0].strip())] = parts[1].strip()
        except ValueError:
            continue
    return diagnosis_map

def parse_filename(recording_id: str) -> dict:
    #Takes filename of wav file and splits it into its constituent parts (patient, loc, device etc)
    parts = recording_id.split('_')
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields, got {len(parts)} in '{recording_id}'")
    patient_id_str, recording_index, chest_location, acquisition_mode, device = parts
    if not patient_id_str.isdigit():
        raise ValueError(f"Non-numeric patient_id '{patient_id_str}' in '{recording_id}'")
    return {
        'patient_id':       int(patient_id_str),
        'chest_location':   chest_location,
        'acquisition_mode': acquisition_mode,
        'device':           device,
    }


def parse_annotation_file(txt_path: Path) -> list[dict]:
    #for each wav file there is a text file with the start/end time of the respiration cycle
    #this function reads this text file and builds a table with the cycle number, start, end time of 
    #each cycle, duration as well as columns for the presence of crackle, wheeze and the associated label ('crackle','wheeze','both')
    cycles = []
    for cycle_index, line in enumerate(txt_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split() #list of consituent parts of 1 line (i.e. [0.2,0.6,0,1])
        if len(parts) != 4:
            print(f"  Warning: {txt_path.name} line {cycle_index + 1}: "
                  f"expected 4 fields, got {len(parts)}. Skipping.")
            continue
        try:
            start   = float(parts[0])
            end     = float(parts[1])
            crackle = int(parts[2])
            wheeze  = int(parts[3])
        except ValueError as e:
            print(f"  Warning: {txt_path.name} line {cycle_index + 1}: {e}. Skipping.")
            continue
        if end <= start:
            print(f"  Warning: {txt_path.name} line {cycle_index + 1}: "
                  f"end <= start. Skipping.")
            continue
        if crackle not in (0, 1) or wheeze not in (0, 1):
            print(f"  Warning: {txt_path.name} line {cycle_index + 1}: "
                  f"non-binary crackle/wheeze. Skipping.")
            continue
        cycles.append({
            'cycle_index': cycle_index,
            'cycle_start': start,
            'cycle_end':   end,
            'duration':    round(end - start, 6),
            'crackle':     crackle,
            'wheeze':      wheeze,
            'label':       LABEL_MAP[(crackle, wheeze)],
        })
    return cycles








# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    raw_data_path: Path,
    official_split_path: Path,
    diagnosis_file_path: Path,
    max_cycle_duration: float,
) -> pd.DataFrame:

    split_map     = parse_official_split(official_split_path)
    diagnosis_map = parse_diagnosis_file(diagnosis_file_path)

    print(f"  {len(split_map)} recordings in split file.")
    print(f"  {len(diagnosis_map)} patients in diagnosis file.")

    txt_files = sorted(raw_data_path.glob('*.txt'))
    print(f"  {len(txt_files)} annotation files found.\n")

    rows = []
    n_dropped_duration = 0
    n_skipped_recordings = 0

    for txt_path in txt_files:
        recording_id = txt_path.stem
        #some filenames are incorrect in the ICBHI dataset (wrong device)
        #Want to pull the corrected name (if it is int the corrections table), otherwise just pull the name
        corrected_id = FILENAME_CORRECTIONS.get(recording_id, recording_id) 

        try:
            file_meta = parse_filename(corrected_id)
        except ValueError as e:
            print(f"  Skipping '{recording_id}': {e}")
            n_skipped_recordings += 1
            continue

        wav_path = raw_data_path / f"{recording_id}.wav"
        if not wav_path.exists():
            print(f"  Skipping '{recording_id}': wav file not found.")
            n_skipped_recordings += 1
            continue

        if corrected_id not in split_map:
            print(f"  Skipping '{corrected_id}': not in split file.")
            n_skipped_recordings += 1
            continue

        native_sr = sf.info(str(wav_path)).samplerate
        split     = split_map[corrected_id]
        diagnosis = diagnosis_map.get(file_meta['patient_id'], 'Unknown')

        cycles = parse_annotation_file(txt_path)
        if not cycles:
            print(f"  Skipping '{recording_id}': no valid cycles.")
            n_skipped_recordings += 1
            continue

        for cycle in cycles:
            if cycle['duration'] > max_cycle_duration:
                n_dropped_duration += 1
                continue
            rows.append({
                'recording_id':    recording_id,
                'patient_id':      file_meta['patient_id'],
                'cycle_index':     cycle['cycle_index'],
                'cycle_start':     cycle['cycle_start'],
                'cycle_end':       cycle['cycle_end'],
                'duration':        cycle['duration'],
                'crackle':         cycle['crackle'],
                'wheeze':          cycle['wheeze'],
                'label':           cycle['label'],
                'chest_location':  file_meta['chest_location'],
                'acquisition_mode': file_meta['acquisition_mode'],
                'device':          file_meta['device'],
                'native_sr':       native_sr,
                'diagnosis':       diagnosis,
                'split':           split,
            })

    manifest = pd.DataFrame(rows)

    print("=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Cycles kept:         {len(manifest)}")
    print(f"Dropped (>{max_cycle_duration}s):    {n_dropped_duration}")
    print(f"Skipped recordings:  {n_skipped_recordings}")
    print(f"\nBy split:\n{manifest.groupby('split').size().to_string()}")
    print(f"\nBy label:\n{manifest.groupby('label').size().to_string()}")
    print(f"\nBy device:\n{manifest.groupby('device').size().to_string()}")
    print(f"\nDuration (s):\n{manifest['duration'].describe().round(3).to_string()}")
    print("=" * 60)

    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 1: PARSE ANNOTATIONS")
    print("=" * 60)
    print(f"Raw data:   {RAW_DATA_PATH}")
    print(f"Manifest:   {MANIFEST_PATH}")
    print(f"Max cycle duration: {CYCLE_DURATION}s")
    print("=" * 60 + "\n")

    #Checking folder paths to raw data, split data and diagnosis file are valid
    for label, path in [
        ("Raw data directory", RAW_DATA_PATH),
        ("Official split file", OFFICIAL_SPLIT_PATH),
        ("Diagnosis file",      DIAGNOSIS_FILE_PATH),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: '{path}'")

    #Making sure destination folder for manifest file exists before creating manifest
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)

    #creating manifest file
    manifest = build_manifest(
        raw_data_path=Path(RAW_DATA_PATH),
        official_split_path=Path(OFFICIAL_SPLIT_PATH),
        diagnosis_file_path=Path(DIAGNOSIS_FILE_PATH),
        max_cycle_duration=CYCLE_DURATION,
    )

    #saving manifest to csv file
    manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nSaved: '{MANIFEST_PATH}'")


if __name__ == '__main__':
    main()
