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
import math

def data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample

# not used with PyTorch (DataLoader used instead), but keeping for other ML methods
def load_split_data(split_name='train'):
    """
    Load spectrograms, features, and labels for a specific split
    
    Args:
        split_name: 'train', 'val', or 'test'
    """
    # Load split info
    with open(os.path.join(Config.output_dir, 'train_val_test_split.pkl'), 'rb') as f:
        split_info = pickle.load(f)
    
    files = split_info[f'{split_name}_files']
    
    # Load data from files
    spectrograms_narrow = []
    spectrograms_wide = []
    features = []
    labels_crackle = []

    
    for filename in files:
        filepath = os.path.join(Config.dir_preprocessed, filename)
        sample = data_Acq(filepath)
        
        spectrograms_narrow.append(sample['mel_spectrogram_narrow'])
        spectrograms_wide.append(sample['mel_spectrogram_wide'])
        features.append(sample['statistics_feature'])
        
        # Extract labels
        label = sample['label']
        labels_crackle.append(1 if label in [1, 3] else 0)
        
    
    # Convert to arrays
    spectrograms_narrow = np.array(spectrograms_narrow)
    spectrograms_wide = np.array(spectrograms_wide)
    features = np.array(features)
    labels_crackle = np.array(labels_crackle)

    
    return spectrograms_narrow, spectrograms_wide, features, labels_crackle

class RespiratoryDataset(Dataset):
    """
    PyTorch Dataset for respiratory sound classification
    """
    def __init__(self, split_name='train', task='crackles',augment=False):
        """
        Args:
            split_name: 'train', 'val', or 'test'
            task: 'crackles' or 'wheezes'
        """
        self.task = task
        self.augment = augment and (split_name == 'train')

        # Load split info
        with open(os.path.join(Config.dir_testTrainData, 'train_val_test_split.pkl'), 'rb') as f:
            split_info = pickle.load(f)
        
        self.files = split_info[f'{split_name}_files']
        self.preprocessed_dir = Config.dir_preprocessed
        
        # Optionally: preload all data into memory (if dataset fits)
        # self.preload_data()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Load a single sample
        Returns: (spectrogram_narrow, features), label
        """
        filename = self.files[idx]
        filepath = os.path.join(self.preprocessed_dir, filename)
        
        # Load from pickle
        sample = data_Acq(filepath)
        
        # Extract data
        spec_narrow = sample['mel_spectrogram_narrow']
        #spec_wide = sample['mel_spectrogram_wide']
        features = sample['statistics_feature']
        
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
            # Time shifting (already have)
            if np.random.rand() < 0.3:
                shift = np.random.randint(-15, 15)
                spec_narrow = np.roll(spec_narrow, shift, axis=1)
            
            # SpecAugment-style frequency masking
            if np.random.rand() < 0.15:
                f_mask_width = np.random.randint(1, 3)
                f_start = np.random.randint(0, spec_narrow.shape[0] - f_mask_width)
                spec_narrow[f_start:f_start + f_mask_width, :] = 0
            
            # Time masking
            if np.random.rand() < 0.15:
                t_mask_width = np.random.randint(3, 10)
                t_start = np.random.randint(0, spec_narrow.shape[1] - t_mask_width)
                spec_narrow[:, t_start:t_start + t_mask_width] = 0
            
            # Additive noise (small)
            if np.random.rand() < 0.15:
                noise = np.random.normal(0, 0.01 * spec_narrow.std(), spec_narrow.shape)
                spec_narrow = spec_narrow + noise

        # Convert to torch tensors
        spec_narrow = torch.FloatTensor(spec_narrow)
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return (spec_narrow, features), label
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        """
        Args:
            alpha: Class weights as a list/tensor [weight_class_0, weight_class_1]
                   or single float for positive class weight
            gamma: Focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        # Convert to tensor if provided as list
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
        self.register_buffer('alpha', alpha)  # Ensures it moves to correct device
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            # Select alpha for each sample based on its target class
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()
    
class RespiratoryModel(nn.Module):
    def __init__(self, n_mels_narrow=24, n_features=45, num_classes=2):
        """
        Dual-stream CNN for respiratory sound classification
        
        Args:
            n_mels_narrow: Number of mel bands in narrow spectrogram (24)
            n_mels_wide: Number of mel bands in wide spectrogram (48)
            n_features: Number of handcrafted features (~45)
            num_classes: 2 for binary classification
        """
        super(RespiratoryModel, self).__init__()
        
        # ====== Narrow Spectrogram CNN (short window, high temporal res) ======
        self.narrow_conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.narrow_bn1 = nn.BatchNorm2d(32)
        self.narrow_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.narrow_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.narrow_bn2 = nn.BatchNorm2d(64)
        self.narrow_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.narrow_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.narrow_bn3 = nn.BatchNorm2d(128)
        self.narrow_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Global average pooling for narrow stream
        self.narrow_gap = nn.AdaptiveAvgPool2d((1, 1))
        #self.narrow_dropout1 = nn.Dropout(0.3)
              
        # ====== Feature MLP ======
        self.feature_fc1 = nn.Linear(n_features, 64)
        self.feature_bn1 = nn.BatchNorm1d(64)
        self.feature_dropout1 = nn.Dropout(0.1)
        
        # ====== Fusion and Classification ======
        # After GAP: narrow=128, wide=128, features=64 → total=320
        #fusion_input_size = 128 + 128 + 64
        fusion_input_size = 128 + 64

        self.fusion_fc1 = nn.Linear(fusion_input_size, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_dropout1 = nn.Dropout(0.3)
        
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_dropout2 = nn.Dropout(0.3)
        
        self.output = nn.Linear(128, num_classes)
    
    def forward(self, spec_narrow, features):
        """
        Forward pass
        
        Args:
            spec_narrow: (batch, n_mels_narrow, time_frames) e.g., (32, 24, 624)
            spec_wide: (batch, n_mels_wide, time_frames) e.g., (32, 48, 153)
            features: (batch, n_features) e.g., (32, 45)
        
        Returns:
            output: (batch, num_classes)
        """
        # Add channel dimension for Conv2D
        spec_narrow = spec_narrow.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        
        # ====== Narrow Stream ======
        x_narrow = F.relu(self.narrow_bn1(self.narrow_conv1(spec_narrow)))
        x_narrow = self.narrow_pool1(x_narrow)
        
        x_narrow = F.relu(self.narrow_bn2(self.narrow_conv2(x_narrow)))
        x_narrow = self.narrow_pool2(x_narrow)
        
        x_narrow = F.relu(self.narrow_bn3(self.narrow_conv3(x_narrow)))
        x_narrow = self.narrow_pool3(x_narrow)
        
        # Global average pooling
        x_narrow = self.narrow_gap(x_narrow)  # (batch, 128, 1, 1)
        x_narrow = x_narrow.view(x_narrow.size(0), -1)  # (batch, 128)
        #x_narrow = self.narrow_dropout1(x_narrow)
        
        # ====== Feature Stream ======
        x_feat = F.relu(self.feature_bn1(self.feature_fc1(features)))
        x_feat = self.feature_dropout1(x_feat)  # (batch, 64)
        
        # ====== Fusion ======
        #x_fused = torch.cat([x_narrow, x_wide, x_feat], dim=1)  # (batch, 320)
        x_fused = torch.cat([x_narrow, x_feat], dim=1)  # (batch, 320)
        
        x_fused = F.relu(self.fusion_bn1(self.fusion_fc1(x_fused)))
        x_fused = self.fusion_dropout1(x_fused)
        
        x_fused = F.relu(self.fusion_bn2(self.fusion_fc2(x_fused)))
        x_fused = self.fusion_dropout2(x_fused)
        
        # ====== Output ======
        output = self.output(x_fused)  # (batch, 2)
        
        return output

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Don't compute gradients during validation
        for (spec_narrow, features), labels in val_loader:
            spec_narrow = spec_narrow.to(device)
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            # Forward pass
            outputs = model(spec_narrow, features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def training():

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=True)  # Augmentation ON
    # Separate dataset WITHOUT augmentation for clean metric computation
    train_dataset_clean = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    val_dataset = RespiratoryDataset(split_name='val', task='crackles', augment=False)
    test_dataset = RespiratoryDataset(split_name='test', task='crackles', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    train_loader_clean = DataLoader(train_dataset_clean, batch_size=Config.batch_size, shuffle=False, num_workers=0)  # For evaluation
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    n_features = len(train_dataset[0][0][1])

    # Initialise model
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_features=n_features,  
        num_classes=2
    ).to(device)

    # Mild class weighting
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.5]).to(device))

    # Define optimiser
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr_crackle, weight_decay=Config.weight_decay_crackle)

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

    # Training loop
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    for epoch in range(Config.num_epochs):
        # ====== TRAINING ======
        model.train()
        train_loss = 0.0
        all_train_preds = []  # Accumulate across all batches
        all_train_labels = []

        for batch_idx, ((spec_narrow, features), labels) in enumerate(train_loader):
            # Move to device
            spec_narrow = spec_narrow.to(device)
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            # Forward pass
            outputs = model(spec_narrow, features)
            _, train_preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            all_train_preds.extend(train_preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.num_epochs}], '
                      f'Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)


        scheduler.step()
        
        
        # Compute train metrics on CLEAN data for fair comparison
        model.eval()
        clean_train_preds, clean_train_labels = [], []
        with torch.no_grad():
            for (spec, feat), labels in train_loader_clean:
                spec = spec.to(device)
                feat = feat.to(device)
                
                outputs = model(spec, feat)
                _, preds = torch.max(outputs, 1)
                clean_train_preds.extend(preds.cpu().numpy())
                clean_train_labels.extend(labels.squeeze().cpu().numpy())
        

        train_f1 = f1_score(clean_train_labels, clean_train_preds, pos_label=1)
        train_recall = recall_score(clean_train_labels, clean_train_preds, pos_label=1)
        train_precision = precision_score(clean_train_labels, clean_train_preds, pos_label=1)

        val_preds = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for (spec_narrow, features), labels in val_loader:
                spec_narrow = spec_narrow.to(device)
                features = features.to(device)
                labels_gpu = labels.to(device).squeeze()  # For loss calculation
                labels = labels.squeeze()
                
                outputs = model(spec_narrow, features)
                loss = criterion(outputs, labels_gpu)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)  # F1 for crackles
        val_recall = recall_score(val_labels, val_preds, pos_label=1)
        val_precision = precision_score(val_labels, val_preds, pos_label=1)
        
        train_losses.append(avg_train_loss)
        train_f1_scores.append(train_f1)
        train_recalls.append(train_recall)
        val_losses.append(avg_val_loss)
        val_f1_scores.append(val_f1)
        val_recalls.append(val_recall)
        
        print(f'\nEpoch [{epoch+1}/{Config.num_epochs}]:')
        print(f'  TRAIN Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}')
        print(f'  VAL Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}') 
        
        # EARLY STOPPING based on F1
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
            }, os.path.join(Config.dir_testTrainData, 'Record_Crackle', 'best_model.pth'))
            
            print(f'  → Best model saved! (F1: {val_f1:.4f}, Recall: {val_recall:.4f})')
        else:
            patience_counter += 1
            print(f'  → No improvement. Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    # ====== FINAL EVALUATION ON TEST SET ======
    print('\n' + '='*50)
    print('Loading best model for final test evaluation...')
    checkpoint = torch.load(os.path.join(Config.dir_testTrainData, 'Record_Crackle', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
    # Add detailed evaluation
    all_preds, all_labels = detailed_evaluation(model, test_loader, device)

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores,
        'val_recalls': val_recalls,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    import pickle
    with open(os.path.join(Config.dir_testTrainData, 'Record_Crackle', 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(train_losses, 'b-', label='Train Loss')
    ax1.plot(val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # F1 and Recall curves
    ax2.plot(val_f1_scores, 'g-', label='Val F1')
    ax2.plot(train_f1_scores, 'b-', label='Train F1')
    ax2.plot(val_recalls, 'm-', label='Val Recall')
    ax2.plot(train_recalls,'k-',label = 'Train Recall')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print('\nTraining complete!')
    return model, history

def detailed_evaluation(model, test_loader, device):
    """
    Compute detailed metrics: precision, recall, F1, confusion matrix
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (spec_narrow, features), labels in test_loader:
            spec_narrow = spec_narrow.to(device)
            features = features.to(device)
            labels = labels.squeeze()
            
            outputs = model(spec_narrow, features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED TEST SET PERFORMANCE")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Crackle', 'Crackle'],
                               digits=3))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\n  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Metrics:")
    print(f"  True Negatives:  {tn} (correct 'no crackle')")
    print(f"  False Positives: {fp} (predicted crackle incorrectly)")
    print(f"  False Negatives: {fn} (missed crackles)")
    print(f"  True Positives:  {tp} (correctly detected crackles)")
    
    if tp + fn > 0:
        recall = tp/(tp+fn)
        print(f"\nCrackle Detection:")
        print(f"  Recall:    {recall:.1%} (caught {tp}/{tp+fn} crackles)")
    
    if tp + fp > 0:
        precision = tp/(tp+fp)
        print(f"  Precision: {precision:.1%} (when predicting crackles, right {tp}/{tp+fp} times)")
    
    print(f"\nPrediction Distribution:")
    print(f"  Predicted 'No Crackle': {len([p for p in all_preds if p==0])} ({100*sum(p==0 for p in all_preds)/len(all_preds):.1f}%)")
    print(f"  Predicted 'Crackle':    {len([p for p in all_preds if p==1])} ({100*sum(p==1 for p in all_preds)/len(all_preds):.1f}%)")
    
    return all_preds, all_labels

def evaluate_saved_model():
    """
    Load saved model and run detailed evaluation
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import f1_score, recall_score, precision_score
    import torch.nn.functional as F
    
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = RespiratoryDataset(split_name='test', task='crackles')
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Get number of features
    sample_features = test_dataset[0][0][1]
    n_features = len(sample_features)
    
    # Initialize model
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_features=n_features,
        num_classes=2
    ).to(device)
    
    # Load saved weights
    checkpoint_path = os.path.join(Config.dir_testTrainData, 'Record_Crackle', 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    
    # Get predictions AND probabilities
    all_probs = []  # Add this
    all_labels = []
    
    with torch.no_grad():
        for (spec_narrow, features), labels in test_loader:
            spec_narrow = spec_narrow.to(device)
            features = features.to(device)
            
            outputs = model(spec_narrow, features)
            probs = F.softmax(outputs, dim=1)[:, 1]  # P(crackle)
            
            all_probs.extend(probs.cpu().numpy())  # Collect all probabilities
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Default predictions (threshold = 0.5)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Print detailed metrics
    print("="*60)
    print("DETAILED TEST SET PERFORMANCE (threshold=0.5)")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Crackle', 'Crackle'],
                               digits=3))
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={tn}  FP={fp}]")
    print(f"   [FN={fn}  TP={tp}]]")
    
    print(f"\nPrediction Distribution:")
    print(f"  Predicted 'No Crackle': {(all_preds==0).sum()} ({100*(all_preds==0).mean():.1f}%)")
    print(f"  Predicted 'Crackle':    {(all_preds==1).sum()} ({100*(all_preds==1).mean():.1f}%)")
    
    print(f"\nActual Distribution:")
    print(f"  Actually 'No Crackle': {(all_labels==0).sum()} ({100*(all_labels==0).mean():.1f}%)")
    print(f"  Actually 'Crackle':    {(all_labels==1).sum()} ({100*(all_labels==1).mean():.1f}%)")
    
    # THRESHOLD SWEEP
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    for thresh in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, pos_label=1)
        rec = recall_score(all_labels, preds, pos_label=1)
        prec = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        pred_pct = 100 * preds.mean()
        print(f"  {thresh:.2f}: F1={f1:.3f}, Recall={rec:.3f}, Precision={prec:.3f}, Pred%={pred_pct:.1f}%")

def debug_training():
    """Debug what's happening in training"""
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=False)  # No augment for debugging
    val_dataset = RespiratoryDataset(split_name='val', task='crackles')
    
    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    count_no_crackle = labels.count(0)
    count_crackle = labels.count(1)
    
    print(f"Dataset distribution:")
    print(f"  No crackles: {count_no_crackle} ({100*count_no_crackle/len(labels):.1f}%)")
    print(f"  Crackles: {count_crackle} ({100*count_crackle/len(labels):.1f}%)")
    
    # Check a single sample
    (spec, feat), label = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Spec shape: {spec.shape}")
    print(f"  Spec range: [{spec.min():.3f}, {spec.max():.3f}]")
    print(f"  Feat shape: {feat.shape}")
    print(f"  Feat range: [{feat.min():.3f}, {feat.max():.3f}]")
    print(f"  Label: {label.item()}")
    
    # Check if features have NaN or Inf
    print(f"\nChecking for NaN/Inf in features:")
    has_nan = False
    has_inf = False
    for i in range(len(train_dataset)):
        (spec, feat), label = train_dataset[i]
        if torch.isnan(feat).any():
            has_nan = True
            print(f"  NaN found in sample {i}")
            break
        if torch.isinf(feat).any():
            has_inf = True
            print(f"  Inf found in sample {i}")
            break
    if not has_nan and not has_inf:
        print("  ✓ No NaN or Inf found")
    
    # Create balanced sampler
    from torch.utils.data import WeightedRandomSampler
    class_sample_counts = [count_no_crackle, count_crackle]
    sample_weights = [1.0 / class_sample_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Check what sampler produces
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )
    
    print(f"\nChecking first 5 batches from sampler:")
    for batch_idx, ((spec, feat), labels) in enumerate(train_loader):
        if batch_idx >= 5:
            break
        num_crackles = (labels == 1).sum().item()
        num_no_crackles = (labels == 0).sum().item()
        print(f"  Batch {batch_idx}: {num_crackles} crackles, {num_no_crackles} no-crackles")
    
    # Test model forward pass
    n_features = len(train_dataset[0][0][1])
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_features=n_features,
        num_classes=2
    ).to(device)
    
    print(f"\nTesting model forward pass:")
    (spec, feat), label = train_dataset[0]
    spec = spec.unsqueeze(0).to(device)
    feat = feat.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(spec, feat)
        probs = F.softmax(output, dim=1)
        print(f"  Output logits: {output[0].cpu().numpy()}")
        print(f"  Probabilities: {probs[0].cpu().numpy()}")
        print(f"  Predicted class: {output.argmax(dim=1).item()}")
        print(f"  True label: {label.item()}")
    
    # Train for 1 epoch and check predictions on validation
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    
    print(f"\nTraining for 1 epoch...")
    model.train()
    for batch_idx, ((spec, feat), labels) in enumerate(train_loader):
        if batch_idx >= 50:  # Just 50 batches
            break
        spec = spec.to(device)
        feat = feat.to(device)
        labels = labels.to(device).squeeze()
        
        outputs = model(spec, feat)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check validation predictions
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for (spec, feat), labels in val_loader:
            spec = spec.to(device)
            feat = feat.to(device)
            
            outputs = model(spec, feat)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1 (crackle)
    
    print(f"\nValidation predictions after 50 batches:")
    print(f"  Predicted 0: {sum(p==0 for p in all_preds)} ({100*sum(p==0 for p in all_preds)/len(all_preds):.1f}%)")
    print(f"  Predicted 1: {sum(p==1 for p in all_preds)} ({100*sum(p==1 for p in all_preds)/len(all_preds):.1f}%)")
    print(f"  Actual 0: {sum(l==0 for l in all_labels)} ({100*sum(l==0 for l in all_labels)/len(all_labels):.1f}%)")
    print(f"  Actual 1: {sum(l==1 for l in all_labels)} ({100*sum(l==1 for l in all_labels)/len(all_labels):.1f}%)")
    print(f"\nProbability statistics for crackle class:")
    print(f"  Mean: {np.mean(all_probs):.4f}")
    print(f"  Median: {np.median(all_probs):.4f}")
    print(f"  Min: {np.min(all_probs):.4f}")
    print(f"  Max: {np.max(all_probs):.4f}")
    
    # Check actual vs predicted for crackle samples
    crackle_indices = [i for i, l in enumerate(all_labels) if l == 1]
    if len(crackle_indices) > 0:
        crackle_probs = [all_probs[i] for i in crackle_indices[:10]]
        print(f"\nFirst 10 crackle probabilities: {[f'{p:.3f}' for p in crackle_probs]}")

def training_with_threshold_tuning():
    """Training with optimal threshold selection"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=True)
    val_dataset = RespiratoryDataset(split_name='val', task='crackles', augment=False)
    test_dataset = RespiratoryDataset(split_name='test', task='crackles', augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    n_features = len(train_dataset[0][0][1])
    
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_features=n_features,
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.2]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    print("\n" + "="*60)
    print("TRAINING WITH THRESHOLD TUNING")
    print("="*60)
    
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(50):
        # === TRAIN ===
        model.train()
        for (spec, feat), labels in train_loader:
            spec = spec.to(device)
            feat = feat.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(spec, feat)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # === VALIDATE with probabilities ===
        model.eval()
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for (spec, feat), labels in val_loader:
                spec = spec.to(device)
                feat = feat.to(device)
                outputs = model(spec, feat)
                probs = F.softmax(outputs, dim=1)[:, 1]  # P(crackle)
                val_probs.extend(probs.cpu().numpy())
                val_labels_list.extend(labels.squeeze().numpy())
        
        val_probs = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        
        # Default threshold metrics
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = f1_score(val_labels_arr, val_preds, pos_label=1)
        val_recall = recall_score(val_labels_arr, val_preds, pos_label=1)
        val_precision = precision_score(val_labels_arr, val_preds, pos_label=1, zero_division=0)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs = val_probs.copy()
            best_val_labels = val_labels_arr.copy()
            patience_counter = 0
            marker = " ← BEST"
        else:
            patience_counter += 1
            marker = ""
        
        if epoch % 3 == 0 or "BEST" in marker:
            print(f"Epoch {epoch+1:2d}: Val F1={val_f1:.3f} R={val_recall:.3f} P={val_precision:.3f}{marker}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # === FIND OPTIMAL THRESHOLD ===
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    precisions, recalls, thresholds = precision_recall_curve(best_val_labels, best_val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find threshold that maximizes F1
    best_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[best_idx]
    
    print(f"\nThreshold analysis on validation set:")
    print(f"  Default (0.50): F1={f1_score(best_val_labels, (best_val_probs >= 0.5).astype(int), pos_label=1):.3f}")
    print(f"  Optimal ({optimal_threshold:.2f}): F1={f1_scores[best_idx]:.3f}")
    
    # Test multiple thresholds
    print(f"\nThreshold sweep:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (best_val_probs >= thresh).astype(int)
        f1 = f1_score(best_val_labels, preds, pos_label=1)
        rec = recall_score(best_val_labels, preds, pos_label=1)
        prec = precision_score(best_val_labels, preds, pos_label=1, zero_division=0)
        print(f"  {thresh:.1f}: F1={f1:.3f}, Recall={rec:.3f}, Precision={prec:.3f}")
    
    # === TEST WITH BOTH THRESHOLDS ===
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    test_probs, test_labels_list = [], []
    with torch.no_grad():
        for (spec, feat), labels in test_loader:
            spec = spec.to(device)
            feat = feat.to(device)
            outputs = model(spec, feat)
            probs = F.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels_list.extend(labels.squeeze().numpy())
    
    test_probs = np.array(test_probs)
    test_labels_arr = np.array(test_labels_list)
    
    print("\n--- With default threshold (0.5) ---")
    test_preds_default = (test_probs >= 0.5).astype(int)
    print(classification_report(test_labels_arr, test_preds_default, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    print(f"\n--- With optimal threshold ({optimal_threshold:.2f}) ---")
    test_preds_optimal = (test_probs >= optimal_threshold).astype(int)
    print(classification_report(test_labels_arr, test_preds_optimal, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    # Also try a higher threshold to improve precision
    higher_thresh = 0.6
    print(f"\n--- With higher threshold ({higher_thresh}) ---")
    test_preds_higher = (test_probs >= higher_thresh).astype(int)
    print(classification_report(test_labels_arr, test_preds_higher, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    return model, optimal_threshold


if __name__ == '__main__':

    #improved_training()
    #debug_training()

    # Check all three splits
    # train_dataset = RespiratoryDataset(split_name='train', task='crackles')
    # val_dataset = RespiratoryDataset(split_name='val', task='crackles')
    # test_dataset = RespiratoryDataset(split_name='test', task='crackles')

    # for name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
    #     labels = [dataset[i][1].item() for i in range(len(dataset))]
    #     crackles = sum(labels)
    #     total = len(labels)
    #     print(f"{name}: {crackles}/{total} crackles ({100*crackles/total:.1f}%)")   

    training()


