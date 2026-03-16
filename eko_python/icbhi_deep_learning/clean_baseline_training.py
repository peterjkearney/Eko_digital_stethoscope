from Step_4_Train_classifier_crackle_narrow import RespiratoryDataset, RespiratoryModel
import Config 

def clean_baseline_training():
    """Minimal training to establish true baseline performance"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
    
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # === CLEAN DATASETS - NO AUGMENTATION ===
    train_dataset = RespiratoryDataset(split_name='train', task='crackles', augment=False)
    val_dataset = RespiratoryDataset(split_name='val', task='crackles', augment=False)
    test_dataset = RespiratoryDataset(split_name='test', task='crackles', augment=False)
    
    # === SIMPLE DATALOADERS - NO WEIGHTED SAMPLING ===
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Get feature count
    sample_features = train_dataset[0][0][1]
    n_features = len(sample_features)
    
    # === MODEL ===
    model = RespiratoryModel(
        n_mels_narrow=Config.n_mels_narrow,
        n_features=n_features,
        num_classes=2
    ).to(device)
    
    # === SIMPLE LOSS - mild class weighting ===
    # Train has 64% no-crackle, 36% crackle
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 1.5]).to(device))
    
    # === OPTIMIZER ===
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    print("\n" + "="*60)
    print("CLEAN BASELINE TRAINING (no augmentation, no weighted sampling)")
    print("="*60)
    
    best_val_f1 = 0
    
    for epoch in range(20):
        # === TRAIN ===
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for (spec, feat), labels in train_loader:
            spec = spec.to(device)
            feat = feat.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(spec, feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_f1 = f1_score(train_labels, train_preds, pos_label=1)
        train_recall = recall_score(train_labels, train_preds, pos_label=1)
        
        # === VALIDATE ===
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for (spec, feat), labels in val_loader:
                spec = spec.to(device)
                feat = feat.to(device)
                labels = labels.squeeze()
                
                outputs = model(spec, feat)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, pos_label=1)
        val_recall = recall_score(val_labels, val_preds, pos_label=1)
        val_precision = precision_score(val_labels, val_preds, pos_label=1, zero_division=0)
        
        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            marker = " ← BEST"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}: Train F1={train_f1:.3f} R={train_recall:.3f} | "
              f"Val F1={val_f1:.3f} R={val_recall:.3f} P={val_precision:.3f}{marker}")
    
    # === FINAL TEST ===
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for (spec, feat), labels in test_loader:
            spec = spec.to(device)
            feat = feat.to(device)
            
            outputs = model(spec, feat)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.squeeze().numpy())
    
    print(classification_report(test_labels, test_preds, 
                               target_names=['No Crackle', 'Crackle'], digits=3))
    
    return model


if __name__ == '__main__':
    clean_baseline_training()