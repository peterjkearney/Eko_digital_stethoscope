# src/train.py
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import f1_score, classification_report, make_scorer, confusion_matrix
try:
    from src import config
except ModuleNotFoundError:
    import config
from data_splitting import DataSplitter

class ModelTrainer:
    def __init__(self, split_data):
        self.split_data = split_data
        
        #TEMP - REDUCING DATASETS TO TEST OUTPUT
        # self.split_data['X_train'] = self.split_data['X_train'][:100]
        # self.split_data['y_crackle_train'] = self.split_data['y_crackle_train'][:100]
        # self.split_data['y_wheeze_train'] = self.split_data['y_wheeze_train'][:100]
        # self.split_data['X_test'] = self.split_data['X_test'][:100]
        # self.split_data['y_crackle_test'] = self.split_data['y_crackle_test'][:100]
        # self.split_data['y_wheeze_test'] = self.split_data['y_wheeze_test'][:100]
        # self.split_data['train_files'] = self.split_data['train_files'][:100]

        self.models = {}
        
    def train_all_models(self, version="v1"):
        """Train all models and save"""
        results = {}
        
        # CRACKLE MODELS
        print("CRACKLE MODELS:")
        # Random Forest with GridSearch
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        results['random_forest_crackle'] = self._train_random_forest(version,target='crackle')
        
        # Logistic Regression
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        results['logistic_regression_crackle'] = self._train_logistic_regression(version,target='crackle')
        
        # SVM
        print("\n" + "="*50)
        print("Training SVM...")
        print("="*50)
        results['svm_crackle'] = self._train_svm(version,target='crackle')
        
        # MLP
        print("\n" + "="*50)
        print("Training MLP...")
        print("="*50)
        results['mlp_crackle'] = self._train_mlp(version,target='crackle')


        # WHEEZE MODELS
        print("\nWHEEZE MODELS:")
        # Random Forest with GridSearch
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        results['random_forest_wheeze'] = self._train_random_forest(version,target='wheeze')
        
        # Logistic Regression
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        results['logistic_regression_wheeze'] = self._train_logistic_regression(version,target='wheeze')
        
        # SVM
        print("\n" + "="*50)
        print("Training SVM...")
        print("="*50)
        results['svm_wheeze'] = self._train_svm(version,target='wheeze')
        
        # MLP
        print("\n" + "="*50)
        print("Training MLP...")
        print("="*50)
        results['mlp_wheeze'] = self._train_mlp(version,target='wheeze')
        
        # Save overall results
        self._save_training_results(results, version)
        
        return results
    
    def _train_random_forest(self, version, target):

        if target not in ['crackle','wheeze']:
            print('Invalid target. Must be either "crackle" or "wheeze"')

        """Train RF with GridSearchCV"""
        X_train = self.split_data['X_train']
        X_test = self.split_data['X_test']
        if target == 'crackle':
            y_train = self.split_data['y_crackle_train']
            y_test = self.split_data['y_crackle_test']
        else:
            y_train = self.split_data['y_wheeze_train']
            y_test = self.split_data['y_wheeze_test']
        
        # Get patient IDs for GroupKFold
        train_files = self.split_data['train_files']
        patient_ids_train = np.array([f[:3] for f in train_files])
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        class_weights = config.CLASS_WEIGHTS_CRACKLE if target == 'crackle' else config.CLASS_WEIGHTS_WHEEZE
        class_weights_dict = self._get_class_weights_dict(y_train, class_weights)
        
        # GridSearch with GroupKFold
        gkf = GroupKFold(n_splits=5)
        rf = RandomForestClassifier(
            class_weight=class_weights_dict,
            random_state=config.RANDOM_STATE
        )
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=gkf,
            scoring=make_scorer(f1_score),
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train, groups=patient_ids_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        y_pred_test = best_rf.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Save model
        model_dir = config.MODELS_DIR / "random_forest"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"rf_model_{version}_{target}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(best_rf, f)
        
        # Save config
        config_data = {
            "version": version,
            "target": target,
            "timestamp": config.get_timestamp(),
            "model_type": "RandomForestClassifier",
            "best_params": grid_search.best_params_,
            "best_cv_score": float(grid_search.best_score_),
            "test_f1": float(test_f1),
            "class_weights": class_weights_dict,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test,y_pred_test).tolist()
        }
        
        config_path = model_dir / f"rf_config_{version}_{target}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nRandom Forest Results for {target}s:")
        print(f"Best CV F1: {grid_search.best_score_:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        print(f"Model saved to: {model_path}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test,y_pred_test))
        
        self.models[f'random_forest_{target}'] = best_rf
        
        return config_data
    
    def _train_logistic_regression(self, version,target):
        """Train Logistic Regression"""
        X_train = self.split_data['X_train']
        X_test = self.split_data['X_test']
        if target == 'crackle':
            y_train = self.split_data['y_crackle_train']
            y_test = self.split_data['y_crackle_test']
        else:
            y_train = self.split_data['y_wheeze_train']
            y_test = self.split_data['y_wheeze_test']
        
        class_weights = config.CLASS_WEIGHTS_CRACKLE if target == 'crackle' else config.CLASS_WEIGHTS_WHEEZE
        class_weights_dict = self._get_class_weights_dict(y_train, class_weights)

        lr = LogisticRegression(
            class_weight=class_weights_dict,
            max_iter=1000,
            random_state=config.RANDOM_STATE
        )
        lr.fit(X_train, y_train)
        
        y_pred_test = lr.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Save model
        model_dir = config.MODELS_DIR / "logistic_regression"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"lr_model_{version}_{target}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(lr, f)
        
        # Save config
        config_data = {
            "version": version,
            "target": target,
            "timestamp": config.get_timestamp(),
            "model_type": "LogisticRegression",
            "test_f1": float(test_f1),
            "class_weights": class_weights_dict,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test,y_pred_test).tolist()
        }
        
        config_path = model_dir / f"lr_config_{version}_{target}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nLogistic Regression Results for {target}s:")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Model saved to: {model_path}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test,y_pred_test))
        
        self.models[f'logistic_regression_{target}'] = lr
        
        return config_data
    
    def _train_svm(self, version, target):
        """Train SVM"""
        X_train = self.split_data['X_train']
        X_test = self.split_data['X_test']
        if target == 'crackle':
            y_train = self.split_data['y_crackle_train']
            y_test = self.split_data['y_crackle_test']
        else:
            y_train = self.split_data['y_wheeze_train']
            y_test = self.split_data['y_wheeze_test']
        
        class_weights = config.CLASS_WEIGHTS_CRACKLE if target == 'crackle' else config.CLASS_WEIGHTS_WHEEZE
        class_weights_dict = self._get_class_weights_dict(y_train, class_weights)

        svm = SVC(
            class_weight=class_weights_dict,
            kernel='rbf',
            random_state=config.RANDOM_STATE
        )
        svm.fit(X_train, y_train)
        
        y_pred_test = svm.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Save model
        model_dir = config.MODELS_DIR / "svm"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"svm_model_{version}_{target}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)
        
        # Save config
        config_data = {
            "version": version,
            "target": target,
            "timestamp": config.get_timestamp(),
            "model_type": "SVC",
            "test_f1": float(test_f1),
            "kernel": "rbf",
            "class_weight": class_weights,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test,y_pred_test).tolist()
        }
        
        config_path = model_dir / f"svm_config_{version}_{target}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nSVM Results for {target}s:")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Model saved to: {model_path}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test,y_pred_test))
        
        self.models[f'svm_{target}'] = svm
        
        return config_data
    
    def _train_mlp(self, version, target):
        """Train MLP"""
        X_train = self.split_data['X_train']
        X_test = self.split_data['X_test']
        if target == 'crackle':
            y_train = self.split_data['y_crackle_train']
            y_test = self.split_data['y_crackle_test']
        else:
            y_train = self.split_data['y_wheeze_train']
            y_test = self.split_data['y_wheeze_test']
        
        class_weights = config.CLASS_WEIGHTS_CRACKLE if target == 'crackle' else config.CLASS_WEIGHTS_WHEEZE
        class_weights_dict = self._get_class_weights_dict(y_train, class_weights)

        # Create sample weights
        sample_weights = np.array([class_weights_dict[label] for label in y_train])
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            early_stopping=True
        )
        mlp.fit(X_train, y_train, sample_weight=sample_weights)
        
        y_pred_test = mlp.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # Save model
        model_dir = config.MODELS_DIR / "mlp"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"mlp_model_{version}_{target}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(mlp, f)
        
        # Save config
        config_data = {
            "version": version,
            "target": target,
            "timestamp": config.get_timestamp(),
            "model_type": "MLPClassifier",
            "test_f1": float(test_f1),
            "hidden_layer_sizes": [128, 64],
            "sample_weights": class_weights_dict,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test,y_pred_test).tolist()
        }
        
        config_path = model_dir / f"mlp_config_{version}_{target}.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nMLP Results for {target}s:")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Model saved to: {model_path}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test,y_pred_test))
        
        self.models[f'mlp_{target}'] = mlp
        
        return config_data
    
    
    def _save_training_results(self, results, version):
        """Save summary of all training results"""
        
        summary = {
            "version": version,
            "timestamp": config.get_timestamp(),
            "models": results
        }
        
        results_path = config.RESULTS_DIR / f"training_results_{version}.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        print(f"Results saved to: {results_path}")
        print("\nTest F1 Scores:")
        for model_name, model_config in results.items():
            print(f"  {model_name}: {model_config['test_f1']:.4f}")

    
    def _get_class_weights_dict(self, y, class_weights):
        """
        Convert class_weight parameter to dictionary format.
        Handles both dict and 'balanced' string.
        """
        if isinstance(class_weights, dict):
            return class_weights
        elif class_weights == 'balanced':
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            return {int(k): float(v) for k, v in zip(classes, weights)}
        else:
            raise ValueError(f"Unsupported class_weight format: {class_weights}")


if __name__ == '__main__':

    load_embedding_size = 512
    load_embedding_version = 'v1'
    load_split_version = 'v1'
    
    splitter = DataSplitter()
    split_data = splitter.load_split(embedding_size = load_embedding_size, embedding_version=load_embedding_version,split_version=load_split_version)

    train_version = 'v1'
    trainer = ModelTrainer(split_data)
    trainer.train_all_models(version=train_version)