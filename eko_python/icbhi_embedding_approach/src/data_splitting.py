# src/data_splitting.py
import numpy as np
import json
from sklearn.model_selection import train_test_split
try:
    from src import config
except ModuleNotFoundError:
    import config
import os

# Splitting data into 70/30 train/test groups with stratification based on patient label

class DataSplitter:
    
    def create_and_save_split(self, embedding_size=512, embedding_version="v1", split_version = "v1",split_on = 'crackle'):
        """
        Create patient-stratified split and save
        
        Returns:
            Dict with train/test data and paths to saved files
        """

        if split_on not in ['crackle','wheeze']:
            print('Invalid string given for split_on parameter. Must be either "crackle" or "wheeze"')
            return
        

        # Load embeddings
        embeddings_folder = config.EMBEDDINGS_DIR
        embeddings_filename = f'openl3_{embedding_size}_{embedding_version}.npz'
        embeddings_path = os.path.join(embeddings_folder,embeddings_filename)

        if not os.path.exists(embeddings_path):
            print(f'No .npz can be found with embedding_size={embedding_size} and version = {embedding_version}')
            return
        
        data = np.load(embeddings_path, allow_pickle=True)
        
        X_all = data['X']
        y_crackle_all = data['y_crackle']
        y_wheeze_all = data['y_wheeze']
        all_files = data['files']
        
        # Extract patient IDs
        patient_ids_all = self._get_patient_ids(all_files)
        
        # Get unique patients and their labels
        unique_patients = np.unique(patient_ids_all)
        patient_labels = []
        

        # Patients will have a mix of healthy and unhealthy recordings. We want to stratify our data at the patient level,
        # so we need to classify a patient as having predominantly healthy vs unhealthy recordings. We classify a patient by 
        # taking a count of crackle (or wheeze) recordings vs healthy recordings, and picking the dominant class.
        for patient in unique_patients:
            patient_mask = patient_ids_all == patient
            if split_on == 'crackle':
                patient_label = np.bincount(y_crackle_all[patient_mask]).argmax()
            else:
                patient_label = np.bincount(y_wheeze_all[patient_mask]).argmax()
            patient_labels.append(patient_label)
        
        patient_labels = np.array(patient_labels)
        
        # Split patients
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=patient_labels
        )
        
        # Get sample indices
        train_mask = np.isin(patient_ids_all, train_patients)
        test_mask = np.isin(patient_ids_all, test_patients)
        
        X_train = X_all[train_mask]
        y_crackle_train = y_crackle_all[train_mask]
        y_wheeze_train = y_wheeze_all[train_mask]
        X_test = X_all[test_mask]
        y_crackle_test = y_crackle_all[test_mask]
        y_wheeze_test = y_wheeze_all[test_mask]


        # Save split configuration
        split_config = {
            "split_version": split_version,
            "embedding_size": embedding_size,
            "embedding_version": embedding_version,
            "timestamp": config.get_timestamp(),
            "embeddings_file": str(embeddings_path),
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "split_on": split_on,
            "train_crackle_positive_rate": float(y_crackle_train.mean()),
            "test_crackle_positive_rate": float(y_crackle_test.mean()),
            "train_wheeze_positive_rate": float(y_wheeze_train.mean()),
            "test_wheeze_positive_rate": float(y_wheeze_test.mean()),
            "train_patients": train_patients.tolist(),
            "test_patients": test_patients.tolist()
        }
        
        config_path = (config.FEATURE_SPLITS_DIR / 
                      f"train_test_split_{embedding_size}_{split_version}_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(split_config, f, indent=2)
        

        # Save split
        split_path = (config.FEATURE_SPLITS_DIR / 
                     f"train_test_split_{embedding_size}_{split_version}.npz")
        
        np.savez_compressed(
            split_path, #save location followed by variables to be saved
            X_train=X_train,
            y_crackle_train=y_crackle_train,
            y_wheeze_train=y_wheeze_train,
            X_test=X_test,
            y_crackle_test=y_crackle_test,
            y_wheeze_test=y_wheeze_test,
            split_path=split_path,
            config_path=config_path,
            train_files=all_files[train_mask],
            test_files=all_files[test_mask],
            train_patients=train_patients,
            test_patients=test_patients
        )


        print(f"\nSplit saved to: {split_path}")
        print(f"Config saved to: {config_path}")
        print(f"Train: {len(X_train)} samples, {len(train_patients)} patients")
        print(f"Test: {len(X_test)} samples, {len(test_patients)} patients")
        print(f"Split on: {split_on}")
        print(f"Crackle: Train positive rate: {y_crackle_train.mean():.2%}")
        print(f"Crackle: Test positive rate: {y_crackle_test.mean():.2%}")
        print(f"Wheeze: Train positive rate: {y_wheeze_train.mean():.2%}")
        print(f"Wheeze: Test positive rate: {y_wheeze_test.mean():.2%}")
        
        return {
            'X_train': X_train,
            'y_crackle_train': y_crackle_train,
            'y_wheeze_train': y_wheeze_train,
            'X_test': X_test,
            'y_crackle_test': y_crackle_test,
            'y_wheeze_test': y_wheeze_test,
            'split_path': split_path,
            'config_path': config_path,
            'train_files':all_files[train_mask],
            'test_files':all_files[test_mask],
            'train_patients':train_patients,
            'test_patients':test_patients
        }


    def load_split(self, embedding_size = 512, embedding_version="v1",split_version="v1"):
        """Load an existing split"""
        split_path = (config.FEATURE_SPLITS_DIR / 
                     f"train_test_split_{embedding_size}_{split_version}.npz")
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split not found: {split_path}")
        
        data = np.load(split_path, allow_pickle=True)
        
        return {
            'X_train': data['X_train'],
            'y_crackle_train': data['y_crackle_train'],
            'y_wheeze_train': data['y_wheeze_train'],
            'X_test': data['X_test'],
            'y_crackle_test': data['y_crackle_test'],
            'y_wheeze_test': data['y_wheeze_test'],
            'split_path': data['split_path'],
            'config_path': data['config_path'],
            'train_files': data['train_files'],
            'test_files': data['test_files'],
            'train_patients': data['train_patients'],
            'test_patients': data['test_patients']
        }
    
    def _get_patient_ids(self, filenames):
        """Extract patient ID from filename (first 3 chars)"""
        return np.array([filename[:3] for filename in filenames])


if __name__ == '__main__':

    data_splitter = DataSplitter()
    data_splitter.create_and_save_split(embedding_size=6144, embedding_version="v2", split_version="v2",split_on = 'crackle')