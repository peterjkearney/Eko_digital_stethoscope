from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from DeepLearning_ICBHI import Config
except ModuleNotFoundError:
    import Config
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, recall_score, precision_score, 
                             classification_report, confusion_matrix, roc_auc_score)
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

def data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()
    return sample


def create_stratified_patient_folds(n_folds=5, save_path=None):
    """
    Create k folds where:
    1. All samples from a patient stay in the same fold
    2. Each fold has similar crackle/no-crackle ratio
    """
    
    # Load all your preprocessed files
    preprocessed_dir = Config.dir_preprocessed
    all_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.dat')]
    
    # Extract patient IDs and labels
    patients = []
    labels = []
    filenames = []
    
    for filename in all_files:
        filepath = os.path.join(preprocessed_dir, filename)
        sample = data_Acq(filepath)
        
        # Extract patient ID from filename (adjust based on your naming convention)
        # ICBHI format is typically: PatientID_RecordingIndex_...
        patient_id = filename.split('_')[0]
        
        # Extract label (crackle = 1 or 3)
        label_code = sample['label']
        label = 1 if label_code in [1, 3] else 0
        
        patients.append(patient_id)
        labels.append(label)
        filenames.append(filename)

        # Extract device and location from filename or sample
        # ICBHI format: PatientID_RecordingIndex_ChestLocation_Device_...
        parts = filename.split('_')
        location = parts[2] if len(parts) > 2 else 'unknown'
        device = parts[3] if len(parts) > 3 else 'unknown'
        
        print(f"{filename}: Location={location}, Device={device}")
    
    patients = np.array(patients)
    labels = np.array(labels)
    filenames = np.array(filenames)
    
    # Get unique patients and their majority label (for stratification)
    unique_patients = np.unique(patients)
    patient_labels = {}
    
    for patient in unique_patients:
        patient_mask = patients == patient
        patient_sample_labels = labels[patient_mask]
        # Use majority label for this patient (or proportion > threshold)
        patient_labels[patient] = 1 if patient_sample_labels.mean() > 0.5 else 0
    
    # Create arrays for StratifiedGroupKFold
    # We need: X (dummy), y (patient-level label), groups (patient ID)
    patient_level_labels = np.array([patient_labels[p] for p in unique_patients])
    
    # Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Generate folds at patient level
    folds = []
    dummy_X = np.zeros(len(unique_patients))  # Dummy X, not used
    
    for fold_idx, (train_patient_idx, test_patient_idx) in enumerate(
        sgkf.split(dummy_X, patient_level_labels, groups=unique_patients)
    ):
        train_patients = set(unique_patients[train_patient_idx])
        test_patients = set(unique_patients[test_patient_idx])
        
        # Map back to files
        train_files = filenames[np.isin(patients, list(train_patients))]
        test_files = filenames[np.isin(patients, list(test_patients))]
        
        # Get labels for verification
        train_labels = labels[np.isin(patients, list(train_patients))]
        test_labels = labels[np.isin(patients, list(test_patients))]
        
        fold_info = {
            'fold': fold_idx,
            'train_files': train_files.tolist(),
            'test_files': test_files.tolist(),
            'train_patients': list(train_patients),
            'test_patients': list(test_patients),
            'train_crackle_rate': train_labels.mean(),
            'test_crackle_rate': test_labels.mean(),
            'n_train': len(train_files),
            'n_test': len(test_files)
        }
        folds.append(fold_info)
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_files)} samples, {len(train_patients)} patients, "
              f"{train_labels.mean():.1%} crackles")
        print(f"  Test:  {len(test_files)} samples, {len(test_patients)} patients, "
              f"{test_labels.mean():.1%} crackles")
    
    # Save folds
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(folds, f)
        print(f"\n✓ Folds saved to {save_path}")
    
    return folds


def run_cross_validation(n_folds=5, **training_kwargs):
    """
    Run k-fold cross-validation and aggregate results
    """
    
    # Create or load folds
    folds_path = os.path.join(Config.dir_testTrainData, f'cv_{n_folds}_folds.pkl')
    
    if os.path.exists(folds_path):
        with open(folds_path, 'rb') as f:
            folds = pickle.load(f)
        print(f"Loaded existing folds from {folds_path}")
    else:
        folds = create_stratified_patient_folds(n_folds, save_path=folds_path)
    
    # Store results for each fold
    fold_results = []
    
    for fold_idx, fold_info in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Run training with this fold's split
        model, results, output_path = training_with_fold(
            fold_info=fold_info,
            fold_idx=fold_idx,
            **training_kwargs
        )
        
        fold_results.append({
            'fold': fold_idx,
            'train_f1': results['train_metrics']['f1'],
            'test_f1': results['test_metrics']['f1'],
            'test_recall': results['test_metrics']['recall'],
            'test_precision': results['test_metrics']['precision'],
            'test_roc_auc': results['test_metrics']['roc_auc'],
            'full_results': results
        })
        
        save_experiment_results(results, output_path, 
                                experiment_name=f"cv_fold{fold_idx}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    test_f1s = [r['test_f1'] for r in fold_results]
    test_recalls = [r['test_recall'] for r in fold_results]
    test_precisions = [r['test_precision'] for r in fold_results]
    
    print(f"\nTest F1:        {np.mean(test_f1s):.3f} ± {np.std(test_f1s):.3f}")
    print(f"Test Recall:    {np.mean(test_recalls):.3f} ± {np.std(test_recalls):.3f}")
    print(f"Test Precision: {np.mean(test_precisions):.3f} ± {np.std(test_precisions):.3f}")
    
    print("\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1={r['test_f1']:.3f}, "
              f"Recall={r['test_recall']:.3f}, Precision={r['test_precision']:.3f}")
    
    return fold_results

def save_experiment_results(training_results, output_path, experiment_name=None):
    """
    Save experiment results to a pickle file with optional custom name
    """
    if experiment_name is None:
        task = training_results['task']
        streams = []
        if training_results['use_narrow']:
            streams.append('narrow')
        if training_results['use_wide']:
            streams.append('wide')
        stream_str = '+'.join(streams)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{task}_{stream_str}_{timestamp}"
    
    filename = f"experiment_{experiment_name}.pkl"
    filepath = os.path.join(output_path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(training_results, f)
    
    print(f"\n✓ Experiment results saved to: {filepath}")
    
    # Save summary text file
    summary_path = os.path.join(output_path, f"summary_{experiment_name}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {training_results['timestamp']}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write(f"  Task: {training_results['task']}\n")
        f.write(f"  Streams: Narrow={training_results['use_narrow']}, Wide={training_results['use_wide']}\n")
        f.write(f"  Class weights: {training_results['class_weights']}\n")
        f.write(f"  Dropouts: CNN={training_results['dropout_cnn']}, Fusion={training_results['dropout_fusion']}\n")
        f.write(f"  Learning rate: {training_results['learning_rate']}\n")
        f.write(f"  Weight decay: {training_results['weight_decay']}\n")
        f.write(f"  Augmentation: {training_results['augmentation_enabled']}\n")
        f.write(f"  Random seed: {training_results['random_seed']}\n\n")
        
        f.write("TRAINING INFO\n")
        f.write(f"  Epochs trained: {training_results['epochs_trained']}\n")
        f.write(f"  Best epoch: {training_results['best_epoch']}\n")
        f.write(f"  Training time: {training_results['total_training_time_seconds']/60:.1f} minutes\n")
        f.write(f"  Parameters: {training_results['model_architecture']['parameters']}\n\n")
        
        f.write("RESULTS (threshold=0.5)\n")
        for split in ['train', 'val', 'test']:
            metrics = training_results[f'{split}_metrics']
            f.write(f"  {split.upper()}:\n")
            f.write(f"    F1: {metrics['f1']:.4f}\n")
            f.write(f"    Recall: {metrics['recall']:.4f}\n")
            f.write(f"    Precision: {metrics['precision']:.4f}\n")
            roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else 'N/A'
            f.write(f"    ROC-AUC: {roc_auc_str}\n")
            f.write(f"    Actual positive rate: {metrics['actual_distribution']['positive_rate']:.2%}\n")
            f.write(f"    Predicted positive rate: {metrics['predicted_distribution']['positive_rate']:.2%}\n")
        
        f.write("\nTEST THRESHOLD SWEEP\n")
        for tm in training_results['threshold_metrics']['test']:
            f.write(f"  {tm['threshold']:.2f}: F1={tm['f1']:.3f}, "
                    f"Recall={tm['recall']:.3f}, Precision={tm['precision']:.3f}\n")
    
    print(f"✓ Summary saved to: {summary_path}")
    
    return filepath

class RespiratoryDatasetCV(Dataset):
    """
    Dataset that accepts explicit file lists (for cross-validation)
    """
    def __init__(self, file_list, task='crackles', augment=False, augment_params=None):
        self.task = task
        self.augment = augment
        self.files = file_list
        self.preprocessed_dir = Config.dir_preprocessed
        
        self.augment_params = augment_params or {
            'time_shift_prob': 0.3,
            'time_shift_range': (-10, 10),
            'freq_mask_prob': 0.0,
            'freq_mask_width_narrow': (1, 3),
            'freq_mask_width_wide': (1, 6),
            'time_mask_prob': 0.0,
            'time_mask_width_narrow': (3, 10),
            'time_mask_width_wide': (2, 8),
            'noise_prob': 0.0,
            'noise_std_factor': 0.01
        }
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.preprocessed_dir, filename)
        
        sample = data_Acq(filepath)
        
        spec_narrow = sample['mel_spectrogram_narrow'].copy()
        spec_wide = sample['mel_spectrogram_wide'].copy()
        features = sample['statistics_feature']
        
        label_code = sample['label']
        if self.task == 'crackles':
            label = 1 if label_code in [1, 3] else 0
        elif self.task == 'wheezes':
            label = 1 if label_code in [2, 3] else 0
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Apply augmentation (same as before)
        if self.augment:
            p = self.augment_params
            
            if np.random.rand() < p['time_shift_prob']:
                shift_narrow = np.random.randint(p['time_shift_range'][0], p['time_shift_range'][1])
                spec_narrow = np.roll(spec_narrow, shift_narrow, axis=1)
                shift_wide = shift_narrow // 4
                spec_wide = np.roll(spec_wide, shift_wide, axis=1)
            
            if np.random.rand() < p['freq_mask_prob']:
                f_mask_width = np.random.randint(p['freq_mask_width_narrow'][0], p['freq_mask_width_narrow'][1])
                f_start = np.random.randint(0, spec_narrow.shape[0] - f_mask_width)
                spec_narrow[f_start:f_start + f_mask_width, :] = 0
            
            if np.random.rand() < p['freq_mask_prob']:
                f_mask_width = np.random.randint(p['freq_mask_width_wide'][0], p['freq_mask_width_wide'][1])
                f_start = np.random.randint(0, spec_wide.shape[0] - f_mask_width)
                spec_wide[f_start:f_start + f_mask_width, :] = 0
            
            if np.random.rand() < p['time_mask_prob']:
                t_mask_width = np.random.randint(p['time_mask_width_narrow'][0], p['time_mask_width_narrow'][1])
                t_start = np.random.randint(0, spec_narrow.shape[1] - t_mask_width)
                spec_narrow[:, t_start:t_start + t_mask_width] = 0
            
            if np.random.rand() < p['time_mask_prob']:
                t_mask_width = np.random.randint(p['time_mask_width_wide'][0], p['time_mask_width_wide'][1])
                t_start = np.random.randint(0, spec_wide.shape[1] - t_mask_width)
                spec_wide[:, t_start:t_start + t_mask_width] = 0
            
            if np.random.rand() < p['noise_prob']:
                noise_narrow = np.random.normal(0, p['noise_std_factor'] * spec_narrow.std(), spec_narrow.shape)
                spec_narrow = spec_narrow + noise_narrow
                noise_wide = np.random.normal(0, p['noise_std_factor'] * spec_wide.std(), spec_wide.shape)
                spec_wide = spec_wide + noise_wide
        
        spec_narrow = torch.FloatTensor(spec_narrow)
        spec_wide = torch.FloatTensor(spec_wide)
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return (spec_narrow, spec_wide, features), label
    
    def get_augment_params(self):
        return self.augment_params if self.augment else None


def get_predictions_and_probs(model, dataloader, device):
    """Get predictions, probabilities, and labels for a dataloader"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (spec_narrow, spec_wide, features), labels in dataloader:
            spec_narrow = spec_narrow.to(device)
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            
            outputs = model(spec_narrow, spec_wide, features)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            labels_np = labels.squeeze().cpu().numpy()
            if labels_np.ndim == 0:  # Single sample, 0-d array
                all_labels.append(labels_np.item())
            else:
                all_labels.extend(labels_np)
    
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_metrics_at_threshold(labels, probs, threshold):
    """Compute metrics at a specific threshold"""
    preds = (probs >= threshold).astype(int)
    
    if len(np.unique(preds)) == 1:
        precision = 0.0 if preds[0] == 0 else (labels == 1).sum() / len(labels)
    else:
        precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    
    return {
        'threshold': threshold,
        'f1': f1_score(labels, preds, pos_label=1, zero_division=0),
        'recall': recall_score(labels, preds, pos_label=1, zero_division=0),
        'precision': precision,
        'predicted_positive_rate': preds.mean()
    }


def compute_all_metrics(labels, probs, preds):
    """Compute comprehensive metrics"""
    metrics = {
        'f1': f1_score(labels, preds, pos_label=1, zero_division=0),
        'recall': recall_score(labels, preds, pos_label=1, zero_division=0),
        'precision': precision_score(labels, preds, pos_label=1, zero_division=0),
        'accuracy': (preds == labels).mean(),
        'confusion_matrix': confusion_matrix(labels, preds).tolist(),
        'actual_distribution': {
            'negative': int((labels == 0).sum()),
            'positive': int((labels == 1).sum()),
            'positive_rate': float((labels == 1).mean())
        },
        'predicted_distribution': {
            'negative': int((preds == 0).sum()),
            'positive': int((preds == 1).sum()),
            'positive_rate': float((preds == 1).mean())
        }
    }
    
    # ROC-AUC (only if we have both classes)
    if len(np.unique(labels)) > 1:
        metrics['roc_auc'] = roc_auc_score(labels, probs)
    else:
        metrics['roc_auc'] = None
    
    return metrics


def training_with_fold(fold_info, fold_idx, task='crackles', use_narrow=True, use_wide=True,
                       class_weights=None, dropout_cnn=0.3, dropout_fusion=0.1,
                       augment=False, augment_params=None, learning_rate=None, weight_decay=None,
                       random_seed=None, show_plots=False):
    """
    Training function that uses explicit train/test file lists from fold_info
    """
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
    
    if learning_rate is None:
        learning_rate = Config.lr_crackle
    if weight_decay is None:
        weight_decay = Config.weight_decay_crackle
    if class_weights is None:
        class_weights = [1.0, 1.5]
    
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    
    # Output folder for this fold
    output_folder = f"Record_CV_{task.capitalize()}_Fold{fold_idx}"
    output_path = os.path.join(Config.dir_testTrainData, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = time.time()
    
    # Get train patients
    train_patients = fold_info['train_patients'].copy()
    all_train_files = fold_info['train_files'].copy()
    
    # Build patient-level labels for stratification
    patient_labels = {}
    for f in all_train_files:
        patient_id = f.split('_')[0]
        sample = data_Acq(os.path.join(Config.dir_preprocessed, f))
        label = 1 if sample['label'] in [1, 3] else 0  # crackle
        
        if patient_id not in patient_labels:
            patient_labels[patient_id] = []
        patient_labels[patient_id].append(label)
    
    # Use majority label per patient for stratification
    patient_majority_label = {p: int(np.mean(labels) > 0.5) 
                              for p, labels in patient_labels.items()}
    
    # Stratified split
    train_patient_list = list(patient_majority_label.keys())
    train_patient_labels = [patient_majority_label[p] for p in train_patient_list]
    
    actual_train_patients, val_patients = train_test_split(
        train_patient_list,
        test_size=0.2,
        random_state=random_seed,
        stratify=train_patient_labels
    )
    
    actual_train_patients = set(actual_train_patients)
    val_patients = set(val_patients)
    
    # Map to files
    actual_train_files = [f for f in all_train_files if f.split('_')[0] in actual_train_patients]
    val_files = [f for f in all_train_files if f.split('_')[0] in val_patients]
    test_files = fold_info['test_files']
    
    # Debug: print crackle rates
    val_labels = [1 if data_Acq(os.path.join(Config.dir_preprocessed, f))['label'] in [1,3] else 0 
                  for f in val_files]
    print(f"  Val crackle rate: {np.mean(val_labels):.1%}")
    
    # Create datasets using the CV dataset class
    train_dataset = RespiratoryDatasetCV(actual_train_files, task=task, augment=augment, augment_params=augment_params)
    train_dataset_clean = RespiratoryDatasetCV(actual_train_files, task=task, augment=False)
    val_dataset = RespiratoryDatasetCV(val_files, task=task, augment=False)
    test_dataset = RespiratoryDatasetCV(test_files, task=task, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    train_loader_clean = DataLoader(train_dataset_clean, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    # Get feature count
    sample = train_dataset[0]
    n_features = len(sample[0][2])
    
    # Initialize model
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_mels_wide=Config.n_mels_wide,
        n_features=n_features,
        num_classes=2,
        use_narrow=use_narrow,
        use_wide=use_wide,
        dropout_cnn=dropout_cnn,
        dropout_fusion=dropout_fusion
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Training tracking
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = Config.patience
    best_model_state = None
    
    epoch_times = []
    train_losses, train_f1_scores, train_recalls = [], [], []
    val_losses, val_f1_scores, val_recalls = [], [], []
    
    print(f"\nFold {fold_idx}: Train={len(actual_train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    epochs_trained = 0
    for epoch in range(Config.num_epochs):
        epoch_start = time.time()
        epochs_trained = epoch + 1
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, ((spec_narrow, spec_wide, features), labels) in enumerate(train_loader):
            spec_narrow = spec_narrow.to(device)
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            labels = labels.to(device).view(-1)
            
            optimizer.zero_grad()
            outputs = model(spec_narrow, spec_wide, features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # Evaluate
        train_probs, train_preds, train_labels = get_predictions_and_probs(model, train_loader_clean, device)
        train_f1 = f1_score(train_labels, train_preds, pos_label=1)
        train_recall = recall_score(train_labels, train_preds, pos_label=1)
        
        val_probs, val_preds, val_labels = get_predictions_and_probs(model, val_loader, device)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (spec_narrow, spec_wide, features), labels in val_loader:
                spec_narrow = spec_narrow.to(device)
                spec_wide = spec_wide.to(device)
                features = features.to(device)
                labels_gpu = labels.to(device).view(-1)
                outputs = model(spec_narrow, spec_wide, features)
                loss = criterion(outputs, labels_gpu)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)
        val_recall = recall_score(val_labels, val_preds, pos_label=1)
        
        print(f"  Epoch {epoch+1}/{Config.num_epochs}: "
          f"Train F1={train_f1:.3f}, Val F1={val_f1:.3f}, "
          f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Patience={patience_counter}/{patience}")

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_losses.append(avg_train_loss)
        train_f1_scores.append(train_f1)
        train_recalls.append(train_recall)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        val_recalls.append(val_recall)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
            }, os.path.join(output_path, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    total_training_time = time.time() - start_time
    
    # Final evaluation
    model.load_state_dict(best_model_state)
    model.to(device)
    
    train_probs, train_preds, train_labels = get_predictions_and_probs(model, train_loader_clean, device)
    val_probs, val_preds, val_labels = get_predictions_and_probs(model, val_loader, device)
    test_probs, test_preds, test_labels = get_predictions_and_probs(model, test_loader, device)
    
    train_metrics = compute_all_metrics(train_labels, train_probs, train_preds)
    val_metrics = compute_all_metrics(val_labels, val_probs, val_preds)
    test_metrics = compute_all_metrics(test_labels, test_probs, test_preds)
    
    thresholds = np.arange(0.2, 0.75, 0.05)
    threshold_metrics = {
        'train': [compute_metrics_at_threshold(train_labels, train_probs, t) for t in thresholds],
        'val': [compute_metrics_at_threshold(val_labels, val_probs, t) for t in thresholds],
        'test': [compute_metrics_at_threshold(test_labels, test_probs, t) for t in thresholds]
    }
    
    print(f"  Fold {fold_idx} Test: F1={test_metrics['f1']:.3f}, "
          f"Recall={test_metrics['recall']:.3f}, Precision={test_metrics['precision']:.3f}")
    
    # Compile results
    training_results = {
        'timestamp': datetime.now().isoformat(),
        'fold': fold_idx,
        'random_seed': random_seed,
        'task': task,
        'use_narrow': use_narrow,
        'use_wide': use_wide,
        'class_weights': class_weights,
        'dropout_cnn': dropout_cnn,
        'dropout_fusion': dropout_fusion,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': Config.batch_size,
        'augmentation_enabled': augment,
        'augmentation_params': train_dataset.get_augment_params(),
        'model_architecture': model.get_architecture_config(),
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch,
        'total_training_time_seconds': total_training_time,
        'avg_epoch_time_seconds': np.mean(epoch_times),
        'fold_info': {
            'n_train': len(actual_train_files),
            'n_val': len(val_files),
            'n_test': len(test_files),
            'train_crackle_rate': fold_info['train_crackle_rate'],
            'test_crackle_rate': fold_info['test_crackle_rate']
        },
        'training_history': {
            'train_losses': train_losses,
            'train_f1_scores': train_f1_scores,
            'val_losses': val_losses,
            'val_f1_scores': val_f1_scores,
        },
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'threshold_metrics': threshold_metrics,
        'predictions': {
            'train': {'probs': train_probs.tolist(), 'preds': train_preds.tolist(), 'labels': train_labels.tolist()},
            'val': {'probs': val_probs.tolist(), 'preds': val_preds.tolist(), 'labels': val_labels.tolist()},
            'test': {'probs': test_probs.tolist(), 'preds': test_preds.tolist(), 'labels': test_labels.tolist()}
        }
    }
    
    # Save plot
    if show_plots or True:  # Always save, optionally show
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, 'b-', label='Train Loss')
        ax1.plot(val_losses, 'r-', label='Val Loss')
        ax1.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best ({best_epoch})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Fold {fold_idx} Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_f1_scores, 'b-', label='Train F1')
        ax2.plot(val_f1_scores, 'g-', label='Val F1')
        ax2.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1')
        ax2.set_title(f'Fold {fold_idx} F1')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'training_curves_fold{fold_idx}.png'))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return model, training_results, output_path


class RespiratoryModel(nn.Module):
    def __init__(self, n_mels_narrow=24, n_mels_wide=48, n_features=45, num_classes=2,
                 use_narrow=True, use_wide=True,
                 dropout_cnn=0.3, dropout_fusion=0.1):
        """
        Dual-stream CNN for respiratory sound classification
        
        Args:
            n_mels_narrow: Number of mel bands in narrow spectrogram (24)
            n_mels_wide: Number of mel bands in wide spectrogram (48)
            n_features: Number of handcrafted features (~45)
            num_classes: 2 for binary classification
            use_narrow: Whether to use narrow spectrogram stream
            use_wide: Whether to use wide spectrogram stream
            dropout_cnn: Dropout applied to CNN stream outputs (before fusion)
            dropout_fusion: Dropout applied to fusion layers
        """
        super(RespiratoryModel, self).__init__()
        
        self.use_narrow = use_narrow
        self.use_wide = use_wide
        self.dropout_cnn = dropout_cnn
        self.dropout_fusion = dropout_fusion
        
        fusion_input_size = 64  # From feature MLP
        
        # ====== Narrow Spectrogram CNN ======
        if use_narrow:
            self.narrow_conv1 = nn.Conv2d(1, 24, kernel_size=(5, 5), padding=2)
            self.narrow_bn1 = nn.BatchNorm2d(24)
            self.narrow_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.narrow_conv2 = nn.Conv2d(24, 48, kernel_size=(5,5), padding=2)
            self.narrow_bn2 = nn.BatchNorm2d(48)
            self.narrow_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.narrow_conv3 = nn.Conv2d(48, 48, kernel_size=(5,5), padding=2)
            self.narrow_bn3 = nn.BatchNorm2d(48)
            self.narrow_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.narrow_gap = nn.AdaptiveAvgPool2d((1, 1))
            self.narrow_dropout = nn.Dropout(dropout_cnn)
            fusion_input_size += 48
        
        # ====== Wide Spectrogram CNN ======
        if use_wide:
            self.wide_conv1 = nn.Conv2d(1, 24, kernel_size=(5,5), padding=2)
            self.wide_bn1 = nn.BatchNorm2d(24)
            self.wide_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.wide_conv2 = nn.Conv2d(24, 48, kernel_size=(5,5), padding=2)
            self.wide_bn2 = nn.BatchNorm2d(48)
            self.wide_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.wide_conv3 = nn.Conv2d(48, 48, kernel_size=(5,5), padding=2)
            self.wide_bn3 = nn.BatchNorm2d(48)
            self.wide_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
            
            self.wide_gap = nn.AdaptiveAvgPool2d((1, 1))
            self.wide_dropout = nn.Dropout(dropout_cnn)
            fusion_input_size += 48
        
        # ====== Feature MLP (no dropout - these are fixed statistical features) ======
        self.feature_fc1 = nn.Linear(n_features, 64)
        self.feature_bn1 = nn.BatchNorm1d(64)
        
        # ====== Fusion and Classification ======
        self.fusion_fc1 = nn.Linear(fusion_input_size, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.fusion_dropout1 = nn.Dropout(dropout_fusion)
        
        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_bn2 = nn.BatchNorm1d(64)
        self.fusion_dropout2 = nn.Dropout(dropout_fusion)
        
        self.output = nn.Linear(64, num_classes)
    
    def forward(self, spec_narrow, spec_wide, features):
        """Forward pass"""
        streams = []
        
        # ====== Narrow Stream ======
        if self.use_narrow:
            x_narrow = spec_narrow.unsqueeze(1)
            x_narrow = F.relu(self.narrow_bn1(self.narrow_conv1(x_narrow)))
            x_narrow = self.narrow_pool1(x_narrow)
            x_narrow = F.relu(self.narrow_bn2(self.narrow_conv2(x_narrow)))
            x_narrow = self.narrow_pool2(x_narrow)
            x_narrow = F.relu(self.narrow_bn3(self.narrow_conv3(x_narrow)))
            x_narrow = self.narrow_pool3(x_narrow)
            x_narrow = self.narrow_gap(x_narrow)
            x_narrow = x_narrow.view(x_narrow.size(0), -1)
            x_narrow = self.narrow_dropout(x_narrow)
            streams.append(x_narrow)
        
        # ====== Wide Stream ======
        if self.use_wide:
            x_wide = spec_wide.unsqueeze(1)
            x_wide = F.relu(self.wide_bn1(self.wide_conv1(x_wide)))
            x_wide = self.wide_pool1(x_wide)
            x_wide = F.relu(self.wide_bn2(self.wide_conv2(x_wide)))
            x_wide = self.wide_pool2(x_wide)
            x_wide = F.relu(self.wide_bn3(self.wide_conv3(x_wide)))
            x_wide = self.wide_pool3(x_wide)
            x_wide = self.wide_gap(x_wide)
            x_wide = x_wide.view(x_wide.size(0), -1)
            x_wide = self.wide_dropout(x_wide)
            streams.append(x_wide)
        
        # ====== Feature Stream (no dropout) ======
        x_feat = F.relu(self.feature_bn1(self.feature_fc1(features)))
        streams.append(x_feat)
        
        # ====== Fusion ======
        x_fused = torch.cat(streams, dim=1)
        
        x_fused = F.relu(self.fusion_bn1(self.fusion_fc1(x_fused)))
        x_fused = self.fusion_dropout1(x_fused)
        
        x_fused = F.relu(self.fusion_bn2(self.fusion_fc2(x_fused)))
        x_fused = self.fusion_dropout2(x_fused)
        
        output = self.output(x_fused)
        return output
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def get_architecture_config(self):
        """Return architecture configuration for logging"""
        return {
            'use_narrow': self.use_narrow,
            'use_wide': self.use_wide,
            'dropout_cnn': self.dropout_cnn,
            'dropout_fusion': self.dropout_fusion,
            'parameters': self.count_parameters()
        }
    

if __name__ == '__main__':

    # Run 5-fold cross-validation
    fold_results = run_cross_validation(
        n_folds=5,
        task='crackles',
        use_narrow=True,
        use_wide=True,
        class_weights=[1.0, 1.5],
        dropout_cnn=0.4,
        dropout_fusion=0.1,
        augment=False,
        random_seed=42
    )

   