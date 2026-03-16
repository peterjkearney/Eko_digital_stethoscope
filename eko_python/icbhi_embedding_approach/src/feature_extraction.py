# src/feature_extraction.py
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import openl3
try:
    from src import config
except ModuleNotFoundError:
    import config
import os

class EmbeddingExtractor():
    def __init__(self, embedding_size=512):
        self.embedding_size = embedding_size

    def extract_and_save_embeddings(self, version="v1"):
        """
        Extract embeddings and save with full metadata
        
        Args:
            dataset_dict: Dict with 'train', 'val', 'test' RespiratoryDataset objects
            version: Version string for tracking
        """

        all_files = []
        all_embeddings = []
        all_crackle_labels = []
        all_wheeze_labels = []

        dataset = self._get_all_samples()

        # for each dat file in dataset, load audio signal and label, preprocess audio and calculate embedding
        for idx in tqdm(range(len(dataset))):
            filename = dataset[idx]
            filepath = Path(config.PREPROCESSED_DIR) / filename
            
            sample = self._load_sample(filepath)
            audio_signal = self._preprocess_audio(sample['signal'])
            
            # Extract embedding (matching training procedure)
            embedding = self._extract_embedding_with_averaging(
                audio_signal, 
                sample_rate=config.SAMPLE_RATE
            )
            
            all_files.append(filename)
            all_embeddings.append(embedding)
            all_crackle_labels.append(1 if sample['label'] in [1, 3] else 0)
            all_wheeze_labels.append(1 if sample['label'] in [2,3] else 0)
        
        # Convert to arrays
        X_all = np.array(all_embeddings)
        y_crackle_all = np.array(all_crackle_labels)
        y_wheeze_all = np.array(all_wheeze_labels)
        
        # Save embeddings
        output_path = (config.EMBEDDINGS_DIR / 
                      f"openl3_{self.embedding_size}_{version}.npz")
        
        np.savez_compressed(
            output_path,
            X=X_all,
            y_crackle = y_crackle_all,
            y_wheeze = y_wheeze_all,
            files=all_files
        )
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": config.get_timestamp(),
            "embedding_size": self.embedding_size,
            "n_samples": len(X_all),
            "positive_rate_crackle": float(y_crackle_all.mean()),
            "positive_rate_wheeze": float(y_wheeze_all.mean()),
            "sample_rate": config.SAMPLE_RATE,
            "openl3_content_type": config.OPENL3_CONTENT_TYPE,
            "preprocessing": {
                "filter_order": config.FILTER_ORDER,
                "filter_lowcut": config.FILTER_LOWCUT,
                "filter_highcut": config.FILTER_HIGHCUT,
                "filter_btype": config.FILTER_BTYPE
            }
        }
        
        metadata_path = (config.EMBEDDINGS_DIR / 
                        f"openl3_{self.embedding_size}_{version}_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSaved embeddings to: {output_path}")
        print(f"Saved metadata to: {metadata_path}")
        
        return output_path
    

    def _get_all_samples(self):

        return [f for f in os.listdir(config.PREPROCESSED_DIR) if f.endswith('.dat')]

    def _extract_embedding_with_averaging(self, audio_signal, sample_rate):
        """Extract embedding with 0.1s hop and averaging (match training)"""
        
        embedding, _ = openl3.get_audio_embedding(
            audio_signal, 
            sample_rate,
            content_type=config.OPENL3_CONTENT_TYPE,
            embedding_size=self.embedding_size,
            verbose=False
            )
        
        return np.mean(embedding, axis=0)
    
    def _preprocess_audio(self, audio_signal):
        """Audio samples have already been filtered, however, samples have been padded to a uniform length of 5s. These zero values 
        will have an effect on the embedding calculation.
        Samples should have zero padding removed so embeddings are calculated solely on real audio data"""
        
        
        # Remove trailing zeros
        endIdx = len(audio_signal)
        while endIdx > 1 and audio_signal[endIdx-1] == 0:
            endIdx -= 1
        audio_signal = audio_signal[:endIdx]
        
        return audio_signal
    
    def _load_sample(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1')


if __name__ == '__main__':

    extractor512 = EmbeddingExtractor(embedding_size=512)
    extractor512.extract_and_save_embeddings(version = 'v1')

    # extractor6144 = EmbeddingExtractor(embedding_size=6144)
    # extractor6144.extract_and_save_embeddings()
