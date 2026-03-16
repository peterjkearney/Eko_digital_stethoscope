import os
import pickle
import Config

def check_label_keys():
    """
    Check all .dat files in preprocessed directory for 'label' key
    """
    dir_preprocessed = Config.dir_preprocessed
    
    print(f"Checking files in: {dir_preprocessed}\n")
    
    # Get all .dat files
    all_files = [f for f in os.listdir(dir_preprocessed) if f.endswith('.dat')]
    
    if not all_files:
        print("❌ No .dat files found!")
        return
    
    print(f"Found {len(all_files)} .dat files\n")
    
    # Track results
    files_with_label = []
    files_without_label = []
    files_with_error = []
    
    # Check each file
    for filename in all_files:
        filepath = os.path.join(dir_preprocessed, filename)
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # Check what keys exist
            keys = list(data.keys())
            
            if 'label' in keys:
                files_with_label.append(filename)
            else:
                files_without_label.append((filename, keys))
        
        except Exception as e:
            files_with_error.append((filename, str(e)))
    
    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files checked: {len(all_files)}")
    print(f"Files WITH 'label' key: {len(files_with_label)} ({100*len(files_with_label)/len(all_files):.1f}%)")
    print(f"Files WITHOUT 'label' key: {len(files_without_label)} ({100*len(files_without_label)/len(all_files):.1f}%)")
    print(f"Files with errors: {len(files_with_error)}")
    print()
    
    # Show details of files without label
    if files_without_label:
        print("="*70)
        print("FILES MISSING 'label' KEY")
        print("="*70)
        
        # Show first 20
        for filename, keys in files_without_label[:20]:
            print(f"\n{filename}:")
            print(f"  Keys present: {keys}")
        
        if len(files_without_label) > 20:
            print(f"\n... and {len(files_without_label) - 20} more files")
        
        print("\n" + "="*70)
    
    # Show files with errors
    if files_with_error:
        print("="*70)
        print("FILES WITH ERRORS")
        print("="*70)
        for filename, error in files_with_error[:10]:
            print(f"{filename}: {error}")
        
        if len(files_with_error) > 10:
            print(f"... and {len(files_with_error) - 10} more errors")
        
        print("\n" + "="*70)
    
    # Show sample of files WITH label (to verify structure)
    if files_with_label:
        print("\n" + "="*70)
        print("SAMPLE OF FILES WITH 'label' KEY (First 3)")
        print("="*70)
        
        for filename in files_with_label[:3]:
            filepath = os.path.join(dir_preprocessed, filename)
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            print(f"\n{filename}:")
            print(f"  All keys: {list(data.keys())}")
            print(f"  Label value: {data['label']}")
            print(f"  Signal shape: {data['signal'].shape if 'signal' in data else 'N/A'}")
            if 'diagnosis' in data:
                print(f"  Diagnosis: {data['diagnosis']}")
    
    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if len(files_without_label) == len(all_files):
        print("❌ NO files have 'label' key!")
        print("   → You MUST rerun Step_1_Preprocessing.py")
    elif len(files_without_label) > 0:
        print(f"⚠️  {len(files_without_label)} files missing 'label' key")
        print("   → You should rerun Step_1_Preprocessing.py")
        print("   → Then rerun Step_2_Feature_extraction.py")
    else:
        print("✅ All files have 'label' key!")
        print("   → The error must be coming from somewhere else")
        print("   → Check Step_3_TestTrainSplit.py logic")
    
    return {
        'total': len(all_files),
        'with_label': len(files_with_label),
        'without_label': len(files_without_label),
        'with_error': len(files_with_error)
    }


if __name__ == '__main__':
    check_label_keys()