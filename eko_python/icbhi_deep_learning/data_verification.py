import os
import pickle
import numpy as np
from collections import defaultdict
try:
    from DeepLearning_ICBHI import Config
except ModuleNotFoundError:
    import Config

def verify_all_labels():
    """
    Check every preprocessed .dat file against original .txt annotations
    """
    
    preprocessed_dir = Config.dir_preprocessed
    raw_dir = Config.dir_rawData
    
    # Track results
    results = {
        'match': 0,
        'mismatch': 0,
        'txt_not_found': 0,
        'segment_not_found': 0,
    }
    
    mismatches = []
    
    # Get all .dat files
    dat_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.dat')])
    
    print(f"Checking {len(dat_files)} preprocessed files...")
    print("="*70)
    
    for dat_filename in dat_files:
        # Load preprocessed sample
        dat_path = os.path.join(preprocessed_dir, dat_filename)
        with open(dat_path, 'rb') as f:
            sample = pickle.load(f)
        
        stored_label = sample['label']
        
        # Parse filename to get original recording info
        # Format: "101_1b1_Al_sc_Meditron_0.dat" -> base="101_1b1_Al_sc_Meditron", segment=0
        parts = dat_filename.rsplit('_', 1)
        base_name = parts[0]
        segment_idx = int(parts[1].replace('.dat', ''))
        
        # Find original txt file
        txt_path = os.path.join(raw_dir, base_name + '.txt')
        
        if not os.path.exists(txt_path):
            results['txt_not_found'] += 1
            continue
        
        # Read original annotations
        with open(txt_path, 'r') as f:
            annotations = f.readlines()
        
        # Get the annotation for this segment
        # Note: We need to match segments accounting for filtering in preprocessing
        # (segments < 1s or > 5s were discarded)
        
        valid_segment_idx = 0
        found = False
        original_crackle = None
        original_wheeze = None
        
        for annotation in annotations:
            annotation = annotation.strip()
            if not annotation:
                continue
                
            parts = annotation.split('\t')
            if len(parts) != 4:
                continue
            
            start_time = float(parts[0])
            end_time = float(parts[1])
            crackle = int(parts[2])
            wheeze = int(parts[3])
            
            seg_length = end_time - start_time
            
            # Apply same filtering as preprocessing
            if seg_length < Config.min_valid_segment_length or seg_length > Config.max_valid_segment_length:
                continue
            
            # This is a valid segment
            if valid_segment_idx == segment_idx:
                original_crackle = crackle
                original_wheeze = wheeze
                found = True
                break
            
            valid_segment_idx += 1
        
        if not found:
            results['segment_not_found'] += 1
            continue
        
        # Compute expected label
        if original_crackle == 0 and original_wheeze == 0:
            expected_label = 0  # normal
        elif original_crackle == 1 and original_wheeze == 0:
            expected_label = 1  # crackle
        elif original_crackle == 0 and original_wheeze == 1:
            expected_label = 2  # wheeze
        elif original_crackle == 1 and original_wheeze == 1:
            expected_label = 3  # both
        else:
            expected_label = -1  # unknown
        
        # Compare
        if stored_label == expected_label:
            results['match'] += 1
        else:
            results['mismatch'] += 1
            mismatches.append({
                'file': dat_filename,
                'stored_label': stored_label,
                'expected_label': expected_label,
                'original_crackle': original_crackle,
                'original_wheeze': original_wheeze,
            })
    
    # Print results
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"Total files checked: {len(dat_files)}")
    print(f"  ✓ Matching labels: {results['match']}")
    print(f"  ✗ Mismatched labels: {results['mismatch']}")
    print(f"  ? TXT file not found: {results['txt_not_found']}")
    print(f"  ? Segment not found: {results['segment_not_found']}")
    
    if mismatches:
        print("\n" + "="*70)
        print(f"MISMATCHES (showing first 20 of {len(mismatches)})")
        print("="*70)
        print(f"{'File':<40} {'Stored':<10} {'Expected':<10} {'Crackle':<10} {'Wheeze':<10}")
        print("-"*70)
        
        for m in mismatches[:20]:
            print(f"{m['file']:<40} {m['stored_label']:<10} {m['expected_label']:<10} {m['original_crackle']:<10} {m['original_wheeze']:<10}")
        
        # Analyze mismatch patterns
        print("\n" + "="*70)
        print("MISMATCH PATTERN ANALYSIS")
        print("="*70)
        
        pattern_counts = defaultdict(int)
        for m in mismatches:
            pattern = f"stored={m['stored_label']} vs expected={m['expected_label']}"
            pattern_counts[pattern] += 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} occurrences")
    
    else:
        print("\n✓ All labels match! No issues found.")
    
    return results, mismatches


def label_distribution_check():
    """
    Compare label distributions between preprocessed data and original annotations
    """
    preprocessed_dir = Config.dir_preprocessed
    raw_dir = Config.dir_rawData
    
    # Count from preprocessed
    preprocessed_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    dat_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.dat')]
    
    for dat_filename in dat_files:
        dat_path = os.path.join(preprocessed_dir, dat_filename)
        with open(dat_path, 'rb') as f:
            sample = pickle.load(f)
        preprocessed_counts[sample['label']] += 1
    
    # Count from original txt files (only valid segments)
    original_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    txt_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    
    for txt_filename in txt_files:
        txt_path = os.path.join(raw_dir, txt_filename)
        
        with open(txt_path, 'r') as f:
            annotations = f.readlines()
        
        for annotation in annotations:
            annotation = annotation.strip()
            if not annotation:
                continue
            
            parts = annotation.split('\t')
            if len(parts) != 4:
                continue
            
            start_time = float(parts[0])
            end_time = float(parts[1])
            crackle = int(parts[2])
            wheeze = int(parts[3])
            
            seg_length = end_time - start_time
            
            # Apply same filtering as preprocessing
            if seg_length < Config.min_valid_segment_length or seg_length > Config.max_valid_segment_length:
                continue
            
            # Compute label
            if crackle == 0 and wheeze == 0:
                label = 0
            elif crackle == 1 and wheeze == 0:
                label = 1
            elif crackle == 0 and wheeze == 1:
                label = 2
            else:
                label = 3
            
            original_counts[label] += 1
    
    # Print comparison
    print("="*70)
    print("LABEL DISTRIBUTION COMPARISON")
    print("="*70)
    print(f"{'Label':<20} {'Preprocessed':<15} {'Original (valid)':<15} {'Match?':<10}")
    print("-"*70)
    
    label_names = {0: 'Normal', 1: 'Crackle', 2: 'Wheeze', 3: 'Both'}
    
    for label in [0, 1, 2, 3]:
        pre_count = preprocessed_counts[label]
        orig_count = original_counts[label]
        match = "✓" if pre_count == orig_count else "✗"
        print(f"{label_names[label]:<20} {pre_count:<15} {orig_count:<15} {match:<10}")
    
    print("-"*70)
    print(f"{'TOTAL':<20} {sum(preprocessed_counts.values()):<15} {sum(original_counts.values()):<15}")
    
    return preprocessed_counts, original_counts


if __name__ == '__main__':
    # Run both checks
    print("\n" + "="*70)
    print("STEP 1: LABEL DISTRIBUTION CHECK")
    print("="*70)
    label_distribution_check()
    
    print("\n\n")
    
    print("="*70)
    print("STEP 2: INDIVIDUAL FILE VERIFICATION")
    print("="*70)
    verify_all_labels()