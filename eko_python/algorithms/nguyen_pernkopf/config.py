import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
EKO_PROJECT_ROOT = BASE_DIR.parent.parent.parent
DATA_DIR = EKO_PROJECT_ROOT / 'data/ext_databases/ICBHI'

RAW_DATA_PATH = DATA_DIR / 'audio_and_txt_files'
OFFICIAL_SPLIT_PATH = DATA_DIR / 'ICBHI_challenge_train_test.txt'
DIAGNOSIS_FILE_PATH = DATA_DIR / 'patient_diagnosis.csv'
MANIFEST_PATH = BASE_DIR / 'manifest.csv'
#PREPARED_DIR = BASE_DIR / 'prepared'
PREPARED_DIR = Path('/content/prepared')

REFERENCE_DEVICE = 'Meditron'
SPECTRUM_CORRECTED_PATH = PREPARED_DIR / 'corrected'
SPECTRUM_CORRECTION_PROFILES_PATH = SPECTRUM_CORRECTED_PATH / 'device_profiles.json'

# Known filename errors in the ICBHI dataset.
# Maps the incorrect filename (as it appears on disk) to the corrected filename.
FILENAME_CORRECTIONS = {
    '226_1b1_Pl_sc_LittC2SE': '226_1b1_Pl_sc_Meditron',
}

MINORITY_CLASSES = ['wheeze', 'both']
MODEL_INPUT_SIZE = (224,224)

NUM_WORKERS  = os.cpu_count() or 2
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

SAMPLE_RATE       = 16000   # Hz — for ALSC task
CYCLE_DURATION    = 8.0     # seconds — fixed length after reflect padding

# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

N_FFT       = 512
HOP_LENGTH  = 128
N_MELS      = 128
FMIN        = 50
FMAX        = 8000

# ---------------------------------------------------------------------------
# VTLP
# ---------------------------------------------------------------------------

VTLP_ALPHA_MIN = 0.9
VTLP_ALPHA_MAX = 1.1
VTLP_FHI_MIN   = 3200
VTLP_FHI_MAX   = 3800

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

TIME_STRETCH_RATES = (0.9, 1.1)   # range for offline time stretching

AUG_PROBABILITY         = 0.3   # probability of applying each augmentation step

ROLL_MAX_SHIFT_FRACTION = 0.5

VOLUME_MIN_GAIN         = 0.7
VOLUME_MAX_GAIN         = 1.3

NOISE_MIN_SNR_DB        = 20.0
NOISE_MAX_SNR_DB        = 40.0

PITCH_MIN_SEMITONES     = -1.0
PITCH_MAX_SEMITONES     =  1.0

SPEED_MIN_RATE          = 0.9
SPEED_MAX_RATE          = 1.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH_SIZE      = 512
NUM_EPOCHS      = 10
EARLY_STOPPING_PATIENCE = 15
NUM_FOLDS       = 5

CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
LEARNING_RATE   = 5e-4   # starting point
WEIGHT_DECAY    = 1e-3
DROPOUT_P       = 0.5

# ---------------------------------------------------------------------------
# Resnet
# ---------------------------------------------------------------------------

RESNET_VARIANT = 'resnet34'
NUM_CLASSES    = 4

# ---------------------------------------------------------------------------
# Co-tuning
# ---------------------------------------------------------------------------

NUM_SOURCE_CLASSES = 1000   # ImageNet classes
COTUNING_LAMBDA    = 0.3   # weight for KL divergence term

# ---------------------------------------------------------------------------
# Device bandwidth feature
# ---------------------------------------------------------------------------
# When False, the bandwidth scalar is not concatenated to features and the
# target classifier input dimension is feature_dim only (512 for ResNet18).
USE_BANDWIDTH_FEATURE: bool = True