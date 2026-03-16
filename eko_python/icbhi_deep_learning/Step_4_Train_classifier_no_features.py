import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
try:
    from DeepLearning_ICBHI import Config
except ModuleNotFoundError:
    import Config
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix


def data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()
    return sample


class RespiratoryDataset(Dataset):
    """
    PyTorch Dataset for respiratory sound classification
    """
    def __init__(self, split_name='train', task='crackles', augment=False):
        """
        Args:
            split_name: 'train', 'val', or 'test'
            task: 'crackles' or 'wheezes'
            augment: Whether to apply data augmentation
        """
        self.task = task
        self.augment = augment and (split_name == 'train')
        
        with open(os.path.join(Config.dir_testTrainData, 'train_val_test_split.pkl'), 'rb') as f:
            split_info = pickle.load(f)
        
        self.files = split_info[f'{split_name}_files']
        self.preprocessed_dir = Config.dir_preprocessed
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load a single sample
        Returns: (spectrogram_narrow, spectrogram_wide), label
        """
        filename = self.files[idx]
        filepath = os.path.join(self.preprocessed_dir, filename)
        
        sample = data_Acq(filepath)
        
        spec_narrow = sample['mel_spectrogram_narrow'].copy()
        spec_wide = sample['mel_spectrogram_wide'].copy()
        
        # Extract label based on task
        label_code = sample['label']
        if self.task == 'crackles':
            label = 1 if label_code in [1, 3] else 0
        elif self.task == 'wheezes':
            label = 1 if label_code in [2, 3] else 0
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # AUGMENTATION
        if self.augment:
            # Time shifting
            if np.random.rand() < 0.3:
                shift_narrow = np.random.randint(-15, 15)
                spec_narrow = np.roll(spec_narrow, shift_narrow, axis=1)
                shift_wide = shift_narrow // 4
                spec_wide = np.roll(spec_wide, shift_wide, axis=1)
            
            # Frequency masking (narrow)
            if np.random.rand() < 0.3:
                f_mask_width = np.random.randint(1, 4)
                f_start = np.random.randint(0, spec_narrow.shape[0] - f_mask_width)
                spec_narrow[f_start:f_start + f_mask_width, :] = 0
            
            # Frequency masking (wide)
            if np.random.rand() < 0.3:
                f_mask_width = np.random.randint(1, 6)
                f_start = np.random.randint(0, spec_wide.shape[0] - f_mask_width)
                spec_wide[f_start:f_start + f_mask_width, :] = 0
            
            # Time masking (narrow)
            if np.random.rand() < 0.3:
                t_mask_width = np.random.randint(5, 30)
                t_start = np.random.randint(0, spec_narrow.shape[1] - t_mask_width)
                spec_narrow[:, t_start:t_start + t_mask_width] = 0
            
            # Time masking (wide)
            if np.random.rand() < 0.3:
                t_mask_width = np.random.randint(2, 10)
                t_start = np.random.randint(0, spec_wide.shape[1] - t_mask_width)
                spec_wide[:, t_start:t_start + t_mask_width] = 0
            
            # Additive noise
            if np.random.rand() < 0.2:
                noise_narrow = np.random.normal(0, 0.02 * spec_narrow.std(), spec_narrow.shape)
                spec_narrow = spec_narrow + noise_narrow
                noise_wide = np.random.normal(0, 0.02 * spec_wide.std(), spec_wide.shape)
                spec_wide = spec_wide + noise_wide
        
        spec_narrow = torch.FloatTensor(spec_narrow)
        spec_wide = torch.FloatTensor(spec_wide)
        label = torch.LongTensor([label])
        
        return (spec_narrow, spec_wide), label


class RespiratoryModel(nn.Module):
    def __init__(self, n_mels_narrow=24, n_mels_wide=48, num_classes=2):
        """
        Dual-stream CNN for respiratory sound classification
        
        Args:
            n_mels_narrow: Number of mel bands in narrow spectrogram (24)
            n_mels_wide: Number of mel bands in wide spectrogram (48)
            num_classes: 2 for binary classification
        """
        super(RespiratoryModel, self).__init__()
        
        # ====== Narrow Spectrogram CNN (short window, high temporal res) ======
        self.narrow_conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.narrow_bn1 = nn.BatchNorm2d(16)
        self.narrow_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.narrow_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.narrow_bn2 = nn.BatchNorm2d(32)
        self.narrow_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # self.narrow_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        # self.narrow_bn3 = nn.BatchNorm2d(128)
        # self.narrow_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.narrow_gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # ====== Wide Spectrogram CNN (long window, high frequency res) ======
        self.wide_conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.wide_bn1 = nn.BatchNorm2d(16)
        self.wide_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.wide_conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.wide_bn2 = nn.BatchNorm2d(32)
        self.wide_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # self.wide_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        # self.wide_bn3 = nn.BatchNorm2d(128)
        # self.wide_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.wide_gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # ====== Fusion and Classification ======
        # After GAP: narrow=128, wide=128 → total=256
        fusion_input_size = 32+32
        
        self.fusion_fc1 = nn.Linear(fusion_input_size, 32)
        self.fusion_bn1 = nn.BatchNorm1d(32)
        self.fusion_dropout1 = nn.Dropout(0.3)
        
        # self.fusion_fc2 = nn.Linear(128, 64)
        # self.fusion_bn2 = nn.BatchNorm1d(64)
        # self.fusion_dropout2 = nn.Dropout(0.5)
        
        self.output = nn.Linear(32, num_classes)
    
    def forward(self, spec_narrow, spec_wide):
        """
        Forward pass
        """
        # Add channel dimension for Conv2D
        spec_narrow = spec_narrow.unsqueeze(1)
        spec_wide = spec_wide.unsqueeze(1)
        
        # ====== Narrow Stream ======
        x_narrow = F.relu(self.narrow_bn1(self.narrow_conv1(spec_narrow)))
        x_narrow = self.narrow_pool1(x_narrow)
        
        x_narrow = F.relu(self.narrow_bn2(self.narrow_conv2(x_narrow)))
        x_narrow = self.narrow_pool2(x_narrow)
        
        # x_narrow = F.relu(self.narrow_bn3(self.narrow_conv3(x_narrow)))
        # x_narrow = self.narrow_pool3(x_narrow)
        
        x_narrow = self.narrow_gap(x_narrow)
        x_narrow = x_narrow.view(x_narrow.size(0), -1)
        
        # ====== Wide Stream ======
        x_wide = F.relu(self.wide_bn1(self.wide_conv1(spec_wide)))
        x_wide = self.wide_pool1(x_wide)
        
        x_wide = F.relu(self.wide_bn2(self.wide_conv2(x_wide)))
        x_wide = self.wide_pool2(x_wide)
        
        # x_wide = F.relu(self.wide_bn3(self.wide_conv3(x_wide)))
        # x_wide = self.wide_pool3(x_wide)
        
        x_wide = self.wide_gap(x_wide)
        x_wide = x_wide.view(x_wide.size(0), -1)
        
        # ====== Fusion ======
        x_fused = torch.cat([x_narrow, x_wide], dim=1)
        
        x_fused = F.relu(self.fusion_bn1(self.fusion_fc1(x_fused)))
        x_fused = self.fusion_dropout1(x_fused)
        
        # x_fused = F.relu(self.fusion_bn2(self.fusion_fc2(x_fused)))
        # x_fused = self.fusion_dropout2(x_fused)
        
        output = self.output(x_fused)
        
        return output


def training():
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create record folder
    record_dir = os.path.join(Config.dir_testTrainData, 'Record_Crackle_DualStream')
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    
    # Create datasets
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    train_dataset_clean = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    val_dataset = RespiratoryDataset(split_name='val', task='crackles', augment=False)
    test_dataset = RespiratoryDataset(split_name='test', task='crackles', augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    train_loader_clean = DataLoader(train_dataset_clean, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Print data info
    sample = train_dataset[0]
    print(f"\nData shapes:")
    print(f"  Narrow spectrogram: {sample[0][0].shape}")
    print(f"  Wide spectrogram: {sample[0][1].shape}")
    
    # Class distribution
    train_labels = [train_dataset_clean[i][1].item() for i in range(len(train_dataset_clean))]
    count_no_crackle = train_labels.count(0)
    count_crackle = train_labels.count(1)
    print(f"\nTraining set distribution:")
    print(f"  No Crackle: {count_no_crackle} ({100*count_no_crackle/len(train_labels):.1f}%)")
    print(f"  Crackle: {count_crackle} ({100*count_crackle/len(train_labels):.1f}%)")
    
    # Initialize model
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_mels_wide=Config.n_mels_wide,
        num_classes=2
    ).to(device)
    
    # Loss function - no class weighting
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_crackle, weight_decay=Config.weight_decay_crackle)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Training tracking
    best_val_f1 = 0.0
    patience_counter = 0
    patience = Config.patience
    best_model_state = None
    
    train_losses = []
    train_f1_scores = []
    train_recalls = []
    val_losses = []
    val_f1_scores = []
    val_recalls = []
    
    print("\n" + "="*60)
    print("TRAINING DUAL-STREAM MODEL (Spectrograms Only)")
    print("="*60)
    
    for epoch in range(Config.num_epochs):
        # ====== TRAINING ======
        model.train()
        train_loss = 0.0
        
        for batch_idx, ((spec_narrow, spec_wide), labels) in enumerate(train_loader):
            spec_narrow = spec_narrow.to(device)
            spec_wide = spec_wide.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(spec_narrow, spec_wide)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.num_epochs}], '
                      f'Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # ====== EVALUATE ON CLEAN TRAIN DATA ======
        model.eval()
        clean_train_preds, clean_train_labels = [], []
        with torch.no_grad():
            for (spec_narrow, spec_wide), labels in train_loader_clean:
                spec_narrow = spec_narrow.to(device)
                spec_wide = spec_wide.to(device)
                
                outputs = model(spec_narrow, spec_wide)
                _, preds = torch.max(outputs, 1)
                clean_train_preds.extend(preds.cpu().numpy())
                clean_train_labels.extend(labels.squeeze().cpu().numpy())
        
        train_f1 = f1_score(clean_train_labels, clean_train_preds, pos_label=1)
        train_recall = recall_score(clean_train_labels, clean_train_preds, pos_label=1)
        train_precision = precision_score(clean_train_labels, clean_train_preds, pos_label=1)
        
        # ====== VALIDATION ======
        val_preds, val_labels_list = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for (spec_narrow, spec_wide), labels in val_loader:
                spec_narrow = spec_narrow.to(device)
                spec_wide = spec_wide.to(device)
                labels_gpu = labels.to(device).squeeze()
                
                outputs = model(spec_narrow, spec_wide)
                loss = criterion(outputs, labels_gpu)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.squeeze().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, pos_label=1)
        val_recall = recall_score(val_labels_list, val_preds, pos_label=1)
        val_precision = precision_score(val_labels_list, val_preds, pos_label=1)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_f1_scores.append(train_f1)
        train_recalls.append(train_recall)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        val_recalls.append(val_recall)
        
        print(f'\nEpoch [{epoch+1}/{Config.num_epochs}]:')
        print(f'  TRAIN Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}')
        print(f'  VAL   Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}')
        
        # Early stopping based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_recall': val_recall,
                'val_loss': avg_val_loss,
            }, os.path.join(record_dir, 'best_model.pth'))
            
            print(f'  → Best model saved! (F1: {val_f1:.4f}, Recall: {val_recall:.4f})')
        else:
            patience_counter += 1
            print(f'  → No improvement. Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    # ====== FINAL TEST EVALUATION ======
    print('\n' + '='*60)
    print('FINAL TEST EVALUATION')
    print('='*60)
    
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for (spec_narrow, spec_wide), labels in test_loader:
            spec_narrow = spec_narrow.to(device)
            spec_wide = spec_wide.to(device)
            
            outputs = model(spec_narrow, spec_wide)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            test_probs.extend(probs.cpu().numpy())
            test_labels_list.extend(labels.squeeze().numpy())
    
    test_probs = np.array(test_probs)
    test_labels_arr = np.array(test_labels_list)
    
    # Default threshold results
    test_preds = (test_probs >= 0.5).astype(int)
    print("\n--- Default Threshold (0.5) ---")
    print(classification_report(test_labels_arr, test_preds, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    # Threshold sweep
    print("--- Threshold Sweep ---")
    for thresh in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
        preds = (test_probs >= thresh).astype(int)
        f1 = f1_score(test_labels_arr, preds, pos_label=1)
        rec = recall_score(test_labels_arr, preds, pos_label=1)
        prec = precision_score(test_labels_arr, preds, pos_label=1, zero_division=0)
        pred_pct = 100 * preds.mean()
        print(f"  {thresh:.2f}: F1={f1:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, Pred%={pred_pct:.1f}%")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_f1_scores': train_f1_scores,
        'train_recalls': train_recalls,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'val_recalls': val_recalls,
    }
    
    with open(os.path.join(record_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, 'b-', label='Train Loss')
    ax1.plot(val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_f1_scores, 'b-', label='Train F1')
    ax2.plot(val_f1_scores, 'g-', label='Val F1')
    ax2.plot(train_recalls, 'b--', label='Train Recall')
    ax2.plot(val_recalls, 'g--', label='Val Recall')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('F1 and Recall Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(record_dir, 'training_curves.png'))
    plt.show()
    
    print('\nTraining complete!')
    return model, history


def evaluate_saved_model():
    """Load saved model and run detailed evaluation with threshold analysis"""
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    test_dataset = RespiratoryDataset(split_name='test', task='crackles', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_mels_wide=Config.n_mels_wide,
        num_classes=2
    ).to(device)
    
    record_dir = os.path.join(Config.dir_testTrainData, 'Record_Crackle_DualStream')
    checkpoint_path = os.path.join(record_dir, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val F1: {checkpoint['val_f1']:.4f}, Val Recall: {checkpoint['val_recall']:.4f}")
    
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for (spec_narrow, spec_wide), labels in test_loader:
            spec_narrow = spec_narrow.to(device)
            spec_wide = spec_wide.to(device)
            
            outputs = model(spec_narrow, spec_wide)
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    all_preds = (all_probs >= 0.5).astype(int)
    
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE (threshold=0.5)")
    print("="*60)
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  [[TN={tn}  FP={fp}]")
    print(f"   [FN={fn}  TP={tp}]]")
    
    print("\n--- Threshold Sweep ---")
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, pos_label=1)
        rec = recall_score(all_labels, preds, pos_label=1)
        prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        pred_pct = 100 * preds.mean()
        print(f"  {thresh:.2f}: F1={f1:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, Pred%={pred_pct:.1f}%")
    
    return model

def sanity_check():
    """Verify spectrograms look different for crackle vs no-crackle"""
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    
    # Get some samples
    crackle_specs = []
    no_crackle_specs = []
    
    for i in range(len(train_dataset)):
        (spec_narrow, spec_wide), label = train_dataset[i]
        if label.item() == 1 and len(crackle_specs) < 10:
            crackle_specs.append(spec_narrow.numpy())
        elif label.item() == 0 and len(no_crackle_specs) < 10:
            no_crackle_specs.append(spec_narrow.numpy())
        
        if len(crackle_specs) >= 10 and len(no_crackle_specs) >= 10:
            break
    
  
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    
        # Row 0: Crackle spectrograms
    for i in range(4):
        axes[0, i].imshow(crackle_specs[i], aspect='auto', origin='lower')
        axes[0, i].set_title(f'Crackle {i}')
    
    # Row 1: No-crackle spectrograms
    for i in range(4):
        axes[1, i].imshow(no_crackle_specs[i], aspect='auto', origin='lower')
        axes[1, i].set_title(f'No-Crackle {i}')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Crackle spec range: [{np.min(crackle_specs):.3f}, {np.max(crackle_specs):.3f}]")
    print(f"No-crackle spec range: [{np.min(no_crackle_specs):.3f}, {np.max(no_crackle_specs):.3f}]")

def verify_labels():
    """Trace labels back to original annotations"""
    import os
    
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    
    # Get a few crackle and no-crackle samples
    for i in range(len(train_dataset)):
        (spec_narrow, spec_wide), label = train_dataset[i]
        
        # Get the filename
        filename = train_dataset.files[i]
        
        # Load the raw sample to see original label
        filepath = os.path.join(train_dataset.preprocessed_dir, filename)
        with open(filepath, 'rb') as f:
            sample = pickle.load(f)
        
        original_label = sample['label']
        
        # Print first few of each class
        if label.item() == 1:  # "Crackle" according to dataset
            print(f"CRACKLE - File: {filename}")
            print(f"  Original label code: {original_label}")
            print(f"  (0=normal, 1=crackle, 2=wheeze, 3=both)")
            print()
            
        if i > 50 and i < 70:  # Just check first 20
            break
    
    print("="*50)
    
    for i in range(len(train_dataset)):
        (spec_narrow, spec_wide), label = train_dataset[i]
        filename = train_dataset.files[i]
        
        filepath = os.path.join(train_dataset.preprocessed_dir, filename)
        with open(filepath, 'rb') as f:
            sample = pickle.load(f)
        
        original_label = sample['label']
        
        if label.item() == 0:  # "No Crackle" according to dataset
            print(f"NO-CRACKLE - File: {filename}")
            print(f"  Original label code: {original_label}")
            print(f"  (0=normal, 1=crackle, 2=wheeze, 3=both)")
            print()
            
        if i > 40:
            break

if __name__ == '__main__':
    training()
    # evaluate_saved_model()

    #sanity_check()

    #verify_labels()