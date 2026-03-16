# main.py - Put it all together
from feature_extraction import EmbeddingExtractor
from data_splitting import DataSplitter
from train import ModelTrainer
import config as config
import numpy as np

def main():
   
    embedding_size = 6144
    embedding_version = "v2"  # Increment for new experiments
    
    # Step 1: Extract embeddings (do once, reuse)
    print("Step 1: Extracting embeddings...")
    extractor = EmbeddingExtractor(embedding_size=embedding_size)
    extractor.extract_and_save_embeddings(version = embedding_version)
 
    # Step 2: Create train/test split (do once, reuse)
    split_version = "v2"
    print("\nStep 2: Creating train/test split...")
    data_splitter = DataSplitter()
    split_data = data_splitter.create_and_save_split(embedding_size=embedding_size, embedding_version=embedding_version,split_version=split_version, split_on = 'crackle')
    
    # Step 3: Train models
    train_version = "v2"
    print("\nStep 3: Training models...")
    trainer = ModelTrainer(split_data)
    results = trainer.train_all_models(version=train_version)
    
    print("\n✅ Pipeline complete!")

# For subsequent runs, just load existing data:
def retrain_with_existing_data(embedding_size,embedding_version, split_version, train_version="v1"):
    """Load existing embeddings and split, train new models"""
    splitter = DataSplitter()
    split_data = splitter.load_split(embedding_size=embedding_size,embedding_version=embedding_version,split_version=split_version)
    
    trainer = ModelTrainer(split_data)
    results = trainer.train_all_models(version=f"{train_version}_retrain")
    
    return results

if __name__ == "__main__":
    #main()
    retrain_with_existing_data(embedding_size=6144,embedding_version='v2',split_version='v2')

