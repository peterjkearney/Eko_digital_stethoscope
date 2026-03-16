from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import Config
import os
import json
from datetime import datetime

def data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample


def get_unique_IDs_and_events():

    allDatFiles = sorted(os.listdir(Config.dir_preprocessed))
    allDatFiles = [f for f in allDatFiles if f.endswith('.dat')]

    # Extract patient IDs from filenames
    patients = np.array([s[:3] for s in allDatFiles])
    uniquePatients = np.unique(patients)
    
    # Initialise arrays for segment-level labels
    crackles = np.zeros(len(allDatFiles),dtype=int)
    wheezes = np.zeros(len(allDatFiles),dtype=int)

    for idx,datFile in enumerate(allDatFiles):

        path_datFile = os.path.join(Config.dir_preprocessed, datFile)
        samples = data_Acq(path_datFile)
        
        sample_label = samples['label']

        if sample_label == 1:
            crackles[idx] = 1
        elif sample_label == 2:
            wheezes[idx] = 1
        elif sample_label == 3:
            crackles[idx] = 1
            wheezes[idx] = 1

    uniqueCrackles = np.zeros(len(uniquePatients),dtype = bool)
    uniqueWheezes = np.zeros(len(uniquePatients),dtype=bool)

    for idx,uniquePatient in enumerate(uniquePatients):
        patient_mask = (patients == uniquePatient)
        uniqueCrackles[idx] = crackles[patient_mask].any()
        uniqueWheezes[idx] = wheezes[patient_mask].any()

    return uniquePatients, uniqueCrackles, uniqueWheezes, patients, crackles, wheezes
        



def get_testtrain_split():
    # Get patient and segment level information
    uniquePatients, uniqueCrackles, uniqueWheezes, segment_patients, segment_crackles, segment_wheezes = get_unique_IDs_and_events()

    print(f"Total patients: {len(uniquePatients)}")
    print(f"Patients with crackles: {uniqueCrackles.sum()}")
    print(f"Patients with wheezes: {uniqueWheezes.sum()}")
    print(f"Patients with both: {(uniqueCrackles & uniqueWheezes).sum()}")

    # Create combined label for stratification
    patient_combined_label = uniqueCrackles.astype(int) + 2 * uniqueWheezes.astype(int)

    randomState = 999

    # Split patients with stratification
    train_patients, test_patients = train_test_split(
        uniquePatients, 
        test_size=0.2,
        random_state=randomState,
        stratify=patient_combined_label
    )

    train_patients, val_patients = train_test_split(
        train_patients,
        test_size=0.25,
        random_state=randomState,
        stratify=patient_combined_label[np.isin(uniquePatients, train_patients)]
    )

    # Get segment-level masks
    train_mask = np.isin(segment_patients, train_patients)
    val_mask = np.isin(segment_patients, val_patients)
    test_mask = np.isin(segment_patients, test_patients)

    # Use these masks to split your segment data
    y_crackles_train = segment_crackles[train_mask]
    y_crackles_val = segment_crackles[val_mask]
    y_crackles_test = segment_crackles[test_mask]

    y_wheezes_train = segment_wheezes[train_mask]
    y_wheezes_val = segment_wheezes[val_mask]
    y_wheezes_test = segment_wheezes[test_mask]

    print(f"\nTrain: {train_mask.sum()} segments from {len(train_patients)} patients")
    print(f"  Crackle segments: {y_crackles_train.sum()}")
    print(f"  Wheeze segments: {y_wheezes_train.sum()}")

    print(f"\nVal: {val_mask.sum()} segments from {len(val_patients)} patients")
    print(f"  Crackle segments: {y_crackles_val.sum()}")
    print(f"  Wheeze segments: {y_wheezes_val.sum()}")

    print(f"\nTest: {test_mask.sum()} segments from {len(test_patients)} patients")
    print(f"  Crackle segments: {y_crackles_test.sum()}")
    print(f"  Wheeze segments: {y_wheezes_test.sum()}")

    # Get all dat files
    allDatFiles = sorted(os.listdir(Config.dir_preprocessed))
    allDatFiles = np.array([f for f in allDatFiles if f.endswith('.dat')])
    
    # Create file lists for each split
    train_files = allDatFiles[train_mask]
    val_files = allDatFiles[val_mask]
    test_files = allDatFiles[test_mask]

    # Save the splits
    split_info = {
        'train_files': train_files.tolist(),
        'val_files': val_files.tolist(),
        'test_files': test_files.tolist(),
        'train_patients': train_patients.tolist(),
        'val_patients': val_patients.tolist(),
        'test_patients': test_patients.tolist(),
        'random_state': randomState,
        'date_created': datetime.now().isoformat()
    }
    
    # Save to JSON (human-readable)
    with open(os.path.join(Config.dir_testTrainData, 'train_val_test_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Also save as pickle for easy loading
    with open(os.path.join(Config.dir_testTrainData, 'train_val_test_split.pkl'), 'wb') as f:
        pickle.dump(split_info, f)
    
    print(f"\nSplit saved to {Config.dir_testTrainData}")
    print(f"  - train_val_test_split.json (human-readable)")
    print(f"  - train_val_test_split.pkl (for loading)")



if __name__ == '__main__':
    get_testtrain_split()

    