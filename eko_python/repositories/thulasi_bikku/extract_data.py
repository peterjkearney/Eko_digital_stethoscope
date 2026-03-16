import os
import shutil
import json
import librosa
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Enhanced constants for better audio processing
SAMPLE_RATE = 22050
DURATION = 10  # seconds
CHUNK_SIZE = SAMPLE_RATE * DURATION

def process_audio_file(file_path, output_dir, metadata=None):
    try:
        # Load audio with resampling
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad if shorter than desired length
        if len(audio) < CHUNK_SIZE:
            audio = np.pad(audio, (0, CHUNK_SIZE - len(audio)))
        
        # Save processed audio
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        librosa.save(output_path, audio, sr=SAMPLE_RATE)
        
        # Save metadata if available
        if metadata:
            metadata_path = output_path.replace('.wav', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def extract_coswara_data(input_dir='Coswara-Data', output_dir='Extracted_Coswara'):
    """Extract and process Coswara dataset with improved parallel processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each date folder
    for date_dir in os.listdir(input_dir):
        date_path = os.path.join(input_dir, date_dir)
        if not os.path.isdir(date_path):
            continue
            
        print(f"\nProcessing data from {date_dir}")
        date_output_dir = os.path.join(output_dir, date_dir)
        os.makedirs(date_output_dir, exist_ok=True)
        
        # Process each participant
        for participant in tqdm(os.listdir(date_path), desc="Processing participants"):
            participant_path = os.path.join(date_path, participant)
            if not os.path.isdir(participant_path):
                continue
                
            # Create output directory for participant
            participant_output = os.path.join(date_output_dir, participant)
            os.makedirs(participant_output, exist_ok=True)
            
            # Load metadata if available
            metadata_path = os.path.join(participant_path, 'metadata.json')
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Copy metadata
            if metadata:
                with open(os.path.join(participant_output, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            # Process audio files
            audio_files = [f for f in os.listdir(participant_path) if f.endswith('.wav')]
            
            with ProcessPoolExecutor() as executor:
                futures = []
                for audio_file in audio_files:
                    input_path = os.path.join(participant_path, audio_file)
                    futures.append(
                        executor.submit(process_audio_file, input_path, participant_output, metadata)
                    )
                
                # Wait for all processing to complete
                for future in futures:
                    future.result()

def extract_icbhi_data(input_dir='h', output_dir='Extracted_ICBHI'):
    """Extract and process ICBHI dataset with improved parallel processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"\nProcessing {len(wav_files)} ICBHI files")
    
    with ProcessPoolExecutor() as executor:
        futures = []
        for wav_file in wav_files:
            # Check for corresponding txt file
            txt_file = wav_file.replace('.wav', '.txt')
            txt_path = os.path.join(input_dir, txt_file)
            
            metadata = None
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r') as f:
                        content = f.read()
                        metadata = {'annotations': content}
                except:
                    pass
            
            input_path = os.path.join(input_dir, wav_file)
            futures.append(
                executor.submit(process_audio_file, input_path, output_dir, metadata)
            )
        
        # Wait for all processing to complete
        for future in tqdm(futures, desc="Processing files"):
            future.result()

if os.path.exists('Coswara-Data'):
    print("\nExtracting Coswara dataset...")
    extract_coswara_data()
else:
    print("\nCoswara dataset not found.")

if os.path.exists('h'):
    print("\nExtracting ICBHI dataset...")
    extract_icbhi_data()
else:
    print("\nICBHI dataset not found.")

print("\nData extraction complete!")