"""
01_parse_annotations.py

Parses the HF_Lung_V1 annotation files and produces parsed_segments.csv — one row
per labelled segment with all metadata needed by downstream steps. HF_Lung_V1 has start/end labelling
for inhalation/exhalation as well as adventitious sounds, so segments are overlapping (i.e. a segment of
wheeze could straddle multiple inhalation and exhalation segments)

HF_Lung_V1 filename format (2 variations):
    Littman 3200 format: steth_{date}_{HH}_{MM}_{SS}_label.txt
    Proprietary device format: trunc_{YYYY}-{MM}-{DD}-{HH}-{MM}-{SS}-{Position}_{Rep}_label.txt

Annotation file format (one line per cycle):
    {label}  {start}  {end}
    e.g. Wheeze 00:00:10.072 00:00:11.266

Output parsed_segments.csv columns:
    recording_id        : wav filename without extension
    Device              : 'steth'or 'trunc'
    Identifier          : date/time data merged into single string (e.g. 20190603093345)
    Position            : position of recording device (only for 'trunc' device)
    Rep                 : number of rep (only for 'trunc' device)
    StartTime           : start time in seconds
    EndTime             : end time in seconds
    Duration            : duration of segment
    Label               : 'I' - inhalation, 'E' - exhalation - 'D' - crackle, 'Wheeze', 'Rhonchii', 'Stridor'
    Split               : Train or Test
"""

import sys
import pandas as pd
import soundfile as sf
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_DIR,
    PARSED_SEGMENTS_PATH
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_annotation_file(txt_path: Path) -> list[dict]:
    #for each wav file there is a text file with the start/end time of the respiration cycle
    #this function reads this text file and builds a table with the cycle number, start, end time of 
    #each cycle, duration as well as columns for the presence of crackle, wheeze and the associated label ('crackle','wheeze','both')
  

    events = []
    fileText = txt_path.read_text().splitlines()

    for lineIdx, line in enumerate(fileText):

        line.strip()
        if not line:
            continue

        lineSegs = line.split(' ')
        if len(lineSegs) != 3:
            print(f"  Warning: {txt_path.name} line {lineIdx + 1}: "
                f"expected 3 fields, got {len(lineSegs)}. Skipping.")
            continue
        
        try:
            event = lineSegs[0]
            t = datetime.strptime(lineSegs[1],"%H:%M:%S.%f")
            eventStart = (t - datetime(1900,1,1)).total_seconds()
            t = datetime.strptime(lineSegs[2],"%H:%M:%S.%f")
            eventEnd = (t - datetime(1900,1,1)).total_seconds()
            eventDuration = round(eventEnd - eventStart,3)

        except ValueError as e:
            print(f"  Warning: {txt_path.name} line {lineIdx + 1}: {e}. Skipping.")
            continue

        if eventDuration <= 0:
            print(f"  Warning: {txt_path.name} line {lineIdx + 1}: "
                f"invalid duration, got {eventDuration}. Skipping.")
            continue

        das = 1 if event == 'D' else 0 # adding 1-hot label for Discontinuous Adventitious Sounds (crackles)
        cas = 1 if event in ['Wheeze','Rhonchi','Stridor'] else 0 # adding 1-hot label for Continuous Adventitious Sounds (wheezes etc)

        events.append({
            'line_index':   lineIdx,
            'event_start':  eventStart,
            'event_end':    eventEnd,
            'duration':     eventDuration,
            'label':        event,
            'DAS':          das,
            'CAS':          cas
        })
    return events


def parseTextFile(fileName):

    nameSegs = fileName.split('_')

    device = nameSegs[0]

    if device == 'steth':
        # filename is steth_year_month_date etc, combining all date/time params into one identifier
        # first convert to tuple then join tuple elements
        identifierTuple = tuple(nameSegs[1:-1])
        identifier = ''.join(identifierTuple)
        
        position = 'NA' # no position information for Litt 3200 device
        rep = 'NA'
    elif device == 'trunc':
        # filename is trunc_YYYY-MM-DD-HH-MM-SS-Postion_rep_label
        datepos = nameSegs[1].split('-')
        
        identifierTuple = tuple(datepos[:6])
        identifier = ''.join(identifierTuple)
        
        position = datepos[6]
        rep = nameSegs[2]
    else:
        raise ValueError(f"Invalid device for file {fileName}")
    
    return device, identifier, position, rep

def getWavFileName(txtFileStub):

    # text files have the same name as corresponding wav file with an additional '_label' suffix
    nameSegs = txtFileStub.split('_')

    if len(nameSegs) >=2:

        # taking all segments of file name up to last '_', thus excluding 'label'
        wavFileSegs = nameSegs[:-1]

        # rebuilding file name by separating segments with '_'
        wavFileStub = '_'.join(tuple(wavFileSegs))
        wavFileName = wavFileStub + '.wav'

    else:
        raise ValueError(f"Invalid text file name for file {txtFileStub}")
    
    return wavFileName




# ---------------------------------------------------------------------------
# parsed_segments.csv builder
# ---------------------------------------------------------------------------

def build_parsed_segments_table(raw_data_path: Path) -> pd.DataFrame:

    dataPaths = {'train': raw_data_path / 'train', 'test': raw_data_path / 'test'}
    datasets = ['train', 'test']
    parsedSet = {'train': [], 'test': []}

    for dataset in datasets:
        allTextFiles = sorted(dataPaths[dataset].glob('*.txt'))
        print(f"  {len(allTextFiles)} annotation files found.\n")

        rows = []
        n_skipped_recordings = 0

        for txtFileName in tqdm(allTextFiles,total=len(allTextFiles),desc=f'Parsing {dataset} files'):
        #for txtFileName in allTextFiles:
            txtFileStub = txtFileName.stem
        
            # parsing text file name
            try:
                device, identifier, position, rep  = parseTextFile(txtFileStub)
            except ValueError as e:
                print(f"  Skipping '{txtFileName}': {e}")
                n_skipped_recordings += 1
                continue
            
            # finding corresponding wav file name
            try:
                wav_path = getWavFileName(txtFileStub)
            except ValueError as e:
                print(f"  Skipping '{txtFileName}': {e}")
                n_skipped_recordings += 1
                continue
            
            # adding full file path to wav_path
            wav_path = dataPaths[dataset] / wav_path

            # checking wav file exists
            if not wav_path.exists():
                print(f"  Skipping '{txtFileName}': wav file not found.")
                n_skipped_recordings += 1
                continue

            native_sr = sf.info(str(wav_path)).samplerate
            split     = dataset
            
            events = parse_annotation_file(dataPaths[dataset] / txtFileName)
            if not events:
                print(f"  Skipping '{txtFileName}': no valid events.")
                n_skipped_recordings += 1
                continue

            for event in events:
                rows.append({
                    'identifier':       identifier,
                    'line_index':       event['line_index'],
                    'device':           device,
                    'position':         position,
                    'rep':              rep,
                    'event_start':      event['event_start'],
                    'event_end':        event['event_end'],
                    'event_duration':   event['duration'],
                    'label':            event['label'],
                    'das':              event['DAS'],
                    'cas':              event['CAS'],
                    'native_sr':        native_sr,
                    'split':            split,
                    'wav_path':         wav_path
                })

            parsedSet[dataset] = pd.DataFrame(rows)
    
    # Now have 2 DataFrames in parsedSet (train and test)
    parsed_segments = pd.concat([parsedSet['train'],parsedSet['test']],ignore_index=True)

    print("=" * 60)
    print("PARSED SUMMARY")
    print("=" * 60)
    print(f"Events kept: {len(parsed_segments)}")
    print(f"Skipped recordings:  {n_skipped_recordings}")
    print(f"\nBy split:\n{parsed_segments.groupby('split').size().to_string()}")
    print(f"\nBy label:\n{parsed_segments.groupby('label').size().to_string()}")
    print(f"\nBy device:\n{parsed_segments.groupby('device').size().to_string()}")
    print(f"\nBy sampling rate:\n{parsed_segments.groupby('native_sr').size().to_string()}")
    print("=" * 60)

    return parsed_segments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 1: PARSE ANNOTATIONS")
    print("=" * 60)
    print(f"Raw data:   {DATA_DIR}")
    print(f"Parsed table:   {PARSED_SEGMENTS_PATH}")
    print("=" * 60 + "\n")

    #Checking folder paths to raw data, split data and diagnosis file are valid
    for label, path in [
        ("Raw data directory", DATA_DIR),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: '{path}'")

    #Making sure destination folder for parsed_segments file exists before creating parsed_segments file
    Path(PARSED_SEGMENTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    #creating parsed_segments file
    parsed_segments = build_parsed_segments_table(
        raw_data_path=Path(DATA_DIR)
    )

    #saving parsed_segments to csv file
    parsed_segments.to_csv(PARSED_SEGMENTS_PATH, index=False)
    print(f"\nSaved: '{PARSED_SEGMENTS_PATH}'")


if __name__ == '__main__':
    main()
