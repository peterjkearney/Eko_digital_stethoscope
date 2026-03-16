# src/config.py
import os
from pathlib import Path
from datetime import datetime


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data/ext_databases/ICBHI"
RAW_DATA_DIR = DATA_DIR / "audio_and_txt_files_CLEANED"
PREPROCESSED_DIR = DATA_DIR / "preprocessing_CLEANED"

FEATURES_DIR = PROJECT_ROOT / "eko_python/icbhi_embedding_approach/features"
EMBEDDINGS_DIR = FEATURES_DIR / "embeddings"
FEATURE_SPLITS_DIR = FEATURES_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "eko_python/icbhi_embedding_approach/models"
RESULTS_DIR = PROJECT_ROOT / "eko_python/icbhi_embedding_approach/results"
#PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories
# for dir_path in [DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
#                  EMBEDDINGS_DIR, FEATURE_SPLITS_DIR, PREDICTIONS_DIR]:
#     dir_path.mkdir(parents=True, exist_ok=True)

# Feature extraction params
SAMPLE_RATE = 4000
EMBEDDING_SIZE = 6144
OPENL3_CONTENT_TYPE = "env"

# Preprocessing params
FILTER_ORDER = 4
FILTER_LOWCUT = 80
FILTER_HIGHCUT = 1800
FILTER_BTYPE = "bandpass"

# Training params
TEST_SIZE = 0.3
RANDOM_STATE = 42
CLASS_WEIGHTS_CRACKLE = {0: 1, 1: 3}
CLASS_WEIGHTS_WHEEZE = 'balanced'

@staticmethod
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    



