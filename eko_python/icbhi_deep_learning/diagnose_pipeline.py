import pickle
import os
import numpy as np

# === RUN THIS DIAGNOSTIC SCRIPT ===
# It will identify the root cause of your training issues

def diagnose_data_pipeline():
    """Comprehensive diagnostic for ICBHI pipeline"""
    
    # Update these paths to match your Config
    import Config
    
    print("="*70)
    print("ICBHI DATA PIPELINE DIAGNOSTIC")
    print("="*70)
    
    # 1. Load split info
    with open(os.path.join(Config.dir_testTrainData, 'train_val_test_split.pkl'), 'rb') as f:
        split_info = pickle.load(f)
    
    train_files = split_info['train_files']
    val_files = split_info['val_files']
    test_files = split_info['test_files']
    
    train_patients = split_info['train_patients']
    val_patients = split_info['val_patients']
    test_patients = split_info['test_patients']
    
    print(f"\n1. SPLIT SIZES")
    print(f"   Train: {len(train_files)} files, {len(train_patients)} patients")
    print(f"   Val:   {len(val_files)} files, {len(val_patients)} patients")
    print(f"   Test:  {len(test_files)} files, {len(test_patients)} patients")
    
    # 2. CHECK FOR PATIENT OVERLAP (CRITICAL!)
    print(f"\n2. PATIENT OVERLAP CHECK")
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        print(f"   ⚠️  CRITICAL: {len(train_val_overlap)} patients in BOTH train and val!")
        print(f"      Patients: {list(train_val_overlap)[:5]}...")
    else:
        print(f"   ✓ No train/val patient overlap")
    
    if train_test_overlap:
        print(f"   ⚠️  CRITICAL: {len(train_test_overlap)} patients in BOTH train and test!")
    else:
        print(f"   ✓ No train/test patient overlap")
    
    if val_test_overlap:
        print(f"   ⚠️  CRITICAL: {len(val_test_overlap)} patients in BOTH val and test!")
    else:
        print(f"   ✓ No val/test patient overlap")
    
    # 3. Check label distribution per split
    print(f"\n3. LABEL DISTRIBUTION BY SPLIT")
    
    for split_name, files in [('Train', train_files), ('Val', val_files), ('Test', test_files)]:
        labels = []
        for f in files:
            filepath = os.path.join(Config.dir_preprocessed, f)
            with open(filepath, 'rb') as fp:
                sample = pickle.load(fp)
            labels.append(sample['label'])
        
        labels = np.array(labels)
        crackles = np.sum((labels == 1) | (labels == 3))
        wheezes = np.sum((labels == 2) | (labels == 3))
        normal = np.sum(labels == 0)
        both = np.sum(labels == 3)
        
        print(f"   {split_name}:")
        print(f"      Normal (0):  {normal} ({100*normal/len(labels):.1f}%)")
        print(f"      Crackle (1): {np.sum(labels==1)} ({100*np.sum(labels==1)/len(labels):.1f}%)")
        print(f"      Wheeze (2):  {np.sum(labels==2)} ({100*np.sum(labels==2)/len(labels):.1f}%)")
        print(f"      Both (3):    {both} ({100*both/len(labels):.1f}%)")
        print(f"      → Has crackle (1 or 3): {crackles} ({100*crackles/len(labels):.1f}%)")
    
    # 4. Check spectrogram and feature stats
    print(f"\n4. DATA STATISTICS (first 100 samples from train)")
    
    spec_mins, spec_maxs, spec_means = [], [], []
    feat_mins, feat_maxs, feat_means = [], [], []
    feat_nans, feat_infs = 0, 0
    
    for f in train_files[:100]:
        filepath = os.path.join(Config.dir_preprocessed, f)
        with open(filepath, 'rb') as fp:
            sample = pickle.load(fp)
        
        spec = sample['mel_spectrogram_narrow']
        feat = sample['statistics_feature']
        
        spec_mins.append(spec.min())
        spec_maxs.append(spec.max())
        spec_means.append(spec.mean())
        
        feat_mins.append(feat.min())
        feat_maxs.append(feat.max())
        feat_means.append(feat.mean())
        
        if np.isnan(feat).any():
            feat_nans += 1
        if np.isinf(feat).any():
            feat_infs += 1
    
    print(f"   Spectrograms:")
    print(f"      Min range:  [{np.min(spec_mins):.3f}, {np.max(spec_mins):.3f}]")
    print(f"      Max range:  [{np.min(spec_maxs):.3f}, {np.max(spec_maxs):.3f}]")
    print(f"      Mean range: [{np.min(spec_means):.3f}, {np.max(spec_means):.3f}]")
    
    print(f"   Features:")
    print(f"      Min range:  [{np.min(feat_mins):.3f}, {np.max(feat_mins):.3f}]")
    print(f"      Max range:  [{np.min(feat_maxs):.3f}, {np.max(feat_maxs):.3f}]")
    print(f"      Mean range: [{np.min(feat_means):.3f}, {np.max(feat_means):.3f}]")
    print(f"      NaN samples: {feat_nans}")
    print(f"      Inf samples: {feat_infs}")
    
    # 5. Check spectrogram shape consistency
    print(f"\n5. SPECTROGRAM SHAPE CHECK")
    shapes = set()
    for f in train_files[:50]:
        filepath = os.path.join(Config.dir_preprocessed, f)
        with open(filepath, 'rb') as fp:
            sample = pickle.load(fp)
        shapes.add(sample['mel_spectrogram_narrow'].shape)
    
    if len(shapes) == 1:
        print(f"   ✓ All spectrograms have consistent shape: {list(shapes)[0]}")
    else:
        print(f"   ⚠️  INCONSISTENT SHAPES: {shapes}")
    
    # 6. Verify file-patient mapping
    print(f"\n6. FILE-PATIENT MAPPING CHECK")
    mismatches = 0
    for f in train_files[:20]:
        patient_from_file = f[:3]
        if patient_from_file not in train_patients:
            print(f"   ⚠️  File {f} has patient {patient_from_file} not in train_patients!")
            mismatches += 1
    
    if mismatches == 0:
        print(f"   ✓ File-patient mapping looks correct")
    
    print(f"\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)

if __name__ == '__main__':
    diagnose_data_pipeline()