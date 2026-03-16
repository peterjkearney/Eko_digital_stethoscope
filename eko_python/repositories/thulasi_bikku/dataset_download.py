"""
Dataset download script for Lung Disease Prediction Project.
This script downloads the ICBHI and Coswara datasets needed for the project.
"""

import os
import requests
import subprocess
import zipfile
import tarfile
import shutil
from tqdm import tqdm

def download_file(url, destination, chunk_size=8192):
    """Download a file from a URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    
    return destination

def setup_datasets():
    """Main function to download and extract all required datasets"""
    print("Setting up datasets for the lung disease prediction project...")
    
    # Create directories if they don't exist
    os.makedirs('h', exist_ok=True)
    os.makedirs('Coswara-Data', exist_ok=True)
    
    # 1. Download and extract ICBHI dataset
    icbhi_url = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip"
    icbhi_zip = "ICBHI_final_database.zip"
    
    if not os.path.exists(icbhi_zip) and not os.path.exists('h/101_1b1_Al_sc_Meditron.wav'):
        print("Downloading ICBHI dataset...")
        download_file(icbhi_url, icbhi_zip)
        
        print("Extracting ICBHI dataset...")
        with zipfile.ZipFile(icbhi_zip, 'r') as zip_ref:
            zip_ref.extractall('h')
        
        # Cleanup
        os.remove(icbhi_zip)
    else:
        print("ICBHI dataset already exists.")
    
    # 2. Clone or update Coswara dataset repository
    if not os.path.exists('Coswara-Data/.git'):
        print("Cloning Coswara dataset repository...")
        subprocess.run(["git", "clone", "https://github.com/iiscleap/Coswara-Data.git"], check=True)
    else:
        print("Updating Coswara dataset repository...")
        subprocess.run(["git", "-C", "Coswara-Data", "pull"], check=True)
    
    # Process and extract the Coswara data
    process_coswara_dates = ['20210507', '20220224']  # Select specific dates to process for faster setup
    
    if not os.path.exists('Extracted_Coswara'):
        print("Processing selected Coswara data...")
        os.makedirs('Extracted_Coswara', exist_ok=True)
        
        for date in process_coswara_dates:
            date_folder = os.path.join('Coswara-Data', date)
            if os.path.exists(date_folder):
                output_folder = os.path.join('Extracted_Coswara', date)
                os.makedirs(output_folder, exist_ok=True)
                
                # Extract and copy files - note this is a simplified version
                for participant in os.listdir(date_folder):
                    src_path = os.path.join(date_folder, participant)
                    if os.path.isdir(src_path):
                        dst_path = os.path.join(output_folder, participant)
                        if not os.path.exists(dst_path):
                            shutil.copytree(src_path, dst_path)
    else:
        print("Coswara extracted data already exists.")

    print("Dataset setup complete! Ready to continue with processing...")

if __name__ == "__main__":
    setup_datasets()
