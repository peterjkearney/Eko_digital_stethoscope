from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR         = Path(__file__).parent
EKO_PROJECT_ROOT = BASE_DIR.parent.parent.parent

# Reuse all preprocessed data from the CLEAN_4kHz sibling pipeline
_CLEAN_DIR       = BASE_DIR.parent / 'nguyen_pernkopf_CLEAN_4kHz'

MANIFEST_PATH    = _CLEAN_DIR / 'manifest.csv'
SPECTROGRAMS_DIR = _CLEAN_DIR / 'data' / 'spectrograms'
DEVICE_PROFILES_PATH = _CLEAN_DIR / 'data' / 'corrected' / 'device_profiles.json'

# ---------------------------------------------------------------------------
# Preprocessing constants — must match CLEAN_4kHz (shared data)
# ---------------------------------------------------------------------------

SAMPLE_RATE      = 4000
CYCLE_DURATION   = 4.0

N_FFT      = 512
WIN_LENGTH = 64
HOP_LENGTH = 64

N_MELS = 128
FMIN   = 50
FMAX   = 2000

VTLP_ALPHA_MIN = 0.9
VTLP_ALPHA_MAX = 1.1
VTLP_FHI_MIN   = 800
VTLP_FHI_MAX   = 950

MODEL_INPUT_SIZE = (224, 224)

REFERENCE_DEVICE = 'Meditron'

# ---------------------------------------------------------------------------
# Label encoding — binary crackle detection
#
#   crackle + both (crackle+wheeze) → 1  (crackle present)
#   normal  + wheeze                → 0  (no crackle)
# ---------------------------------------------------------------------------

LABEL_TO_IDX = {
    'no_crackle': 0,
    'crackle':    1,
}

# Map from the 4-class ICBHI labels to the binary labels above
ICBHI_TO_BINARY = {
    'normal':  'no_crackle',
    'wheeze':  'no_crackle',
    'crackle': 'crackle',
    'both':    'crackle',
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

NUM_WORKERS = min(os.cpu_count() or 1, 4)

RANDOM_SEED = 42

BATCH_SIZE               = 512
NUM_EPOCHS               = 100
EARLY_STOPPING_PATIENCE  = 15
OVERFIT_GAP_THRESHOLD    = 0.20
OVERFIT_GAP_PATIENCE     = 2
NUM_FOLDS                = 5
LEARNING_RATE            = 5e-4
WEIGHT_DECAY             = 1e-3

RESNET_VARIANT     = 'resnet18'
NUM_CLASSES        = 2            # no_crackle, crackle
NUM_SOURCE_CLASSES = 1000
COTUNING_LAMBDA    = 0.3
DROPOUT_P          = 0.5
CHECKPOINTS_DIR    = BASE_DIR / 'checkpoints'
