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
    spectrograms_wide = []
    features = []
    labels_wheeze = []
    
    for filename in files:
        filepath = os.path.join(Config.dir_preprocessed, filename)
        sample = data_Acq(filepath)
        
        spectrograms_wide.append(sample['mel_spectrogram_wide'])
        features.append(sample['statistics_feature'])
        
        # Extract labels
        label = sample['label']
        labels_wheeze.append(1 if label in [2, 3] else 0)
    
    # Convert to arrays
    spectrograms_wide = np.array(spectrograms_wide)
    features = np.array(features)
    labels_wheeze = np.array(labels_wheeze)
    
    return spectrograms_wide, features, labels_wheeze

class RespiratoryDataset(Dataset):
    """
    PyTorch Dataset for respiratory sound classification
    """
    def __init__(self, split_name='train', task='wheezes'):
        """
        Args:
            split_name: 'train', 'val', or 'test'
            task: 'crackles' or 'wheezes'
        """
        self.task = task
        
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
        Returns: (spectrogram_narrow, spectrogram_wide, features), label
        """
        filename = self.files[idx]
        filepath = os.path.join(self.preprocessed_dir, filename)
        
        # Load from pickle
        sample = data_Acq(filepath)
        
        # Extract data
        spec_wide = sample['mel_spectrogram_wide']
        features = sample['statistics_feature']
        
        # Extract label based on task
        label_code = sample['label']
        if self.task == 'crackles':
            label = 1 if label_code in [1, 3] else 0
        elif self.task == 'wheezes':
            label = 1 if label_code in [2, 3] else 0
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Convert to torch tensors
        spec_wide = torch.FloatTensor(spec_wide)
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])
        
        return (spec_wide, features), label


class RespiratoryModel(nn.Module):
    def __init__(self, n_mels_wide=48, n_features=45, num_classes=2):
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
        # self.narrow_conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        # self.narrow_bn1 = nn.BatchNorm2d(32)
        # self.narrow_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # self.narrow_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        # self.narrow_bn2 = nn.BatchNorm2d(64)
        # self.narrow_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # self.narrow_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        # self.narrow_bn3 = nn.BatchNorm2d(128)
        # self.narrow_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Global average pooling for narrow stream
        # self.narrow_gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # ====== Wide Spectrogram CNN (long window, high frequency res) ======
        self.wide_conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.wide_bn1 = nn.BatchNorm2d(32)
        self.wide_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.wide_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.wide_bn2 = nn.BatchNorm2d(64)
        self.wide_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.wide_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.wide_bn3 = nn.BatchNorm2d(128)
        self.wide_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Global average pooling for wide stream
        self.wide_gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # ====== Feature MLP ======
        self.feature_fc1 = nn.Linear(n_features, 64)
        self.feature_bn1 = nn.BatchNorm1d(64)
        self.feature_dropout1 = nn.Dropout(0.5)
        
        # ====== Fusion and Classification ======
        # After GAP: narrow=128, wide=128, features=64 → total=320
        fusion_input_size = 128 + 64
        
        self.fusion_fc1 = nn.Linear(fusion_input_size, 256)
        self.fusion_bn1 = nn.BatchNorm1d(256)
        self.fusion_dropout1 = nn.Dropout(0.5)
        
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_bn2 = nn.BatchNorm1d(128)
        self.fusion_dropout2 = nn.Dropout(0.5)
        
        self.output = nn.Linear(128, num_classes)
    
    def forward(self, spec_wide, features):
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
        #spec_narrow = spec_narrow.unsqueeze(1)  # (batch, 1, n_mels, time)
        spec_wide = spec_wide.unsqueeze(1)      # (batch, 1, n_mels, time)
        
        # ====== Wide Stream ======
        x_wide = F.relu(self.wide_bn1(self.wide_conv1(spec_wide)))
        x_wide = self.wide_pool1(x_wide)
        
        x_wide = F.relu(self.wide_bn2(self.wide_conv2(x_wide)))
        x_wide = self.wide_pool2(x_wide)
        
        x_wide = F.relu(self.wide_bn3(self.wide_conv3(x_wide)))
        x_wide = self.wide_pool3(x_wide)
        
        # Global average pooling
        x_wide = self.wide_gap(x_wide)  # (batch, 128, 1, 1)
        x_wide = x_wide.view(x_wide.size(0), -1)  # (batch, 128)
        
        # ====== Feature Stream ======
        x_feat = F.relu(self.feature_bn1(self.feature_fc1(features)))
        x_feat = self.feature_dropout1(x_feat)  # (batch, 64)
        
        # ====== Fusion ======
        x_fused = torch.cat([x_wide, x_feat], dim=1)  # (batch, 320)
        
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
        for (spec_wide, features), labels in val_loader:
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            # Forward pass
            outputs = model(spec_wide, features)
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
    # creating/clearing Record folder for training updates
    if not os.path.exists(f"{Config.dir_testTrainData}/Record_Wheeze"):
        os.makedirs(f"{Config.dir_testTrainData}/Record_Wheeze")

    # Create datasets
    train_dataset = RespiratoryDataset(split_name='train', task='wheezes')
    val_dataset = RespiratoryDataset(split_name='val', task='wheezes')
    test_dataset = RespiratoryDataset(split_name='test', task='wheezes')

    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    num_no_wheeze = labels.count(0)
    num_wheeze = labels.count(1)

    print(f"Class distribution:")
    print(f"  No wheezes: {num_no_wheeze} ({100*num_no_wheeze/len(labels):.1f}%)")
    print(f"  Wheezes: {num_wheeze} ({100*num_wheeze/len(labels):.1f}%)")

    # Calculate class weights
    # Weight the wheeze class more heavily
    weight_for_wheeze = num_no_wheeze / num_wheeze
    class_weights = torch.FloatTensor([1.0, weight_for_wheeze]).to(device)

    print(f"\nUsing class weights: [1.0, {weight_for_wheeze:.2f}]")
    print(f"  → Wheeze predictions penalized {weight_for_wheeze:.1f}x more for being wrong\n")
    


    sample_features = train_dataset[0][0][1]  # Get features from first sample
    n_features = len(sample_features)
    print(f"Number of features: {n_features}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0,  # Parallel loading
        pin_memory=True if torch.cuda.is_available() else False  # Only pin on GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialise model
    model = RespiratoryModel(
        n_mels_wide=Config.n_mels_wide,
        n_features=n_features,  # Adjust based on actual feature count
        num_classes=2
    ).to(device)

    print(f"\nModel initialized:")
    print(f"  - Wide spectrogram: {Config.n_mels_wide} mel bins")
    print(f"  - Features: {n_features}")

    #Define loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.lr_wheeze, 
        weight_decay=Config.weight_decay  # L2 regularization
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    patience = Config.patience  # Early stopping patience
    
    train_losses = []
    val_losses = []
    val_accuracies = []


    # Quick test of data loading
    print("\nTesting data loading...")
    try:
        (spec_wide, features), label = train_dataset[0]
        print(f"  Wide spec shape: {spec_wide.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Label: {label.item()}")
        print("✓ Data loading successful!\n")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # Training loop
    for epoch in range(Config.num_epochs):
        # ====== TRAINING ======
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, ((spec_wide, features), labels) in enumerate(train_loader):
            # Move to device
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            labels = labels.to(device).squeeze()
            
            # Forward pass
            outputs = model(spec_wide, features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{Config.num_epochs}], '
                      f'Step [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # ====== VALIDATION ======
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'\nEpoch [{epoch+1}/{Config.num_epochs}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # ====== EARLY STOPPING ======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, os.path.join(Config.dir_testTrainData, 'Record_Wheeze', 'best_model.pth'))
            
            print(f'  → Best model saved! (Val Loss: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'  → No improvement. Patience: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    # ====== FINAL EVALUATION ON TEST SET ======
    print('\n' + '='*50)
    print('Loading best model for final test evaluation...')
    checkpoint = torch.load(os.path.join(Config.dir_testTrainData, 'Record_Wheeze', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    # Add detailed evaluation
    all_preds, all_labels = detailed_evaluation(model, test_loader, device)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    import pickle
    with open(os.path.join(Config.dir_testTrainData, 'Record_Wheeze', 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print('\nTraining complete!')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_losses,'b')
    ax.plot(val_losses,'r')
    plt.show()

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
        for (spec_wide, features), labels in test_loader:
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            labels = labels.squeeze()
            
            outputs = model(spec_wide, features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("DETAILED TEST SET PERFORMANCE")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Wheeze', 'Wheeze'],
                               digits=3))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\n  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Metrics:")
    print(f"  True Negatives:  {tn} (correct 'no wheeze')")
    print(f"  False Positives: {fp} (predicted wheeze incorrectly)")
    print(f"  False Negatives: {fn} (missed wheezes)")
    print(f"  True Positives:  {tp} (correctly detected wheezes)")
    
    if tp + fn > 0:
        recall = tp/(tp+fn)
        print(f"\nWheeze Detection:")
        print(f"  Recall:    {recall:.1%} (caught {tp}/{tp+fn} wheezes)")
    
    if tp + fp > 0:
        precision = tp/(tp+fp)
        print(f"  Precision: {precision:.1%} (when predicting wheeze, right {tp}/{tp+fp} times)")
    
    print(f"\nPrediction Distribution:")
    print(f"  Predicted 'No Wheeze': {len([p for p in all_preds if p==0])} ({100*sum(p==0 for p in all_preds)/len(all_preds):.1f}%)")
    print(f"  Predicted 'Wheeze':    {len([p for p in all_preds if p==1])} ({100*sum(p==1 for p in all_preds)/len(all_preds):.1f}%)")
    
    return all_preds, all_labels

def evaluate_saved_model():
    """
    Load saved model and run detailed evaluation
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = RespiratoryDataset(split_name='test', task='wheezes')
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
        n_mels_wide=Config.n_mels_wide,
        n_features=n_features,
        num_classes=2
    ).to(device)
    
    # Load saved weights
    checkpoint_path = os.path.join(Config.dir_testTrainData, 'Record_Wheeze', 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  Val Accuracy: {checkpoint['val_accuracy']:.2f}%\n")
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (spec_wide, features), labels in test_loader:
            spec_wide = spec_wide.to(device)
            features = features.to(device)
            
            outputs = model(spec_wide, features)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    # Print detailed metrics
    print("="*60)
    print("DETAILED TEST SET PERFORMANCE")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=['No Wheeze', 'Wheeze'],
                               digits=3))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"\n  [[TN={cm[0,0]}  FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")
    
    tn, fp, fn, tp = cm.ravel()
    print("\nDetailed Metrics:")
    print(f"  True Negatives:  {tn} (correct 'no wheeze')")
    print(f"  False Positives: {fp} (predicted wheeze incorrectly)")
    print(f"  False Negatives: {fn} (missed wheezes)")
    print(f"  True Positives:  {tp} (correctly detected wheezes)")
    
    if tp + fn > 0:
        recall = tp/(tp+fn)
        print(f"\nWheeze Detection:")
        print(f"  Recall:    {recall:.1%} (caught {tp}/{tp+fn} wheezes)")
    
    if tp + fp > 0:
        precision = tp/(tp+fp)
        print(f"  Precision: {precision:.1%} (right {tp}/{tp+fp} times when predicting wheeze)")
    
    print(f"\nPrediction Distribution:")
    no_wheeze_pred = sum(p==0 for p in all_preds)
    wheeze_pred = sum(p==1 for p in all_preds)
    print(f"  Predicted 'No Wheeze': {no_wheeze_pred} ({100*no_wheeze_pred/len(all_preds):.1f}%)")
    print(f"  Predicted 'Wheeze':    {wheeze_pred} ({100*wheeze_pred/len(all_preds):.1f}%)")
    
    no_wheeze_actual = sum(l==0 for l in all_labels)
    wheeze_actual = sum(l==1 for l in all_labels)
    print(f"\nActual Distribution:")
    print(f"  Actually 'No Wheeze': {no_wheeze_actual} ({100*no_wheeze_actual/len(all_labels):.1f}%)")
    print(f"  Actually 'Wheeze':    {wheeze_actual} ({100*wheeze_actual/len(all_labels):.1f}%)")


if __name__ == '__main__':

    training()

    #evaluate_saved_model()



