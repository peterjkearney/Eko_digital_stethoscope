from pathlib import Path

BASE_DIR         = Path(__file__).parent
EKO_PROJECT_ROOT = BASE_DIR.parent.parent.parent
DATA_DIR         = EKO_PROJECT_ROOT / 'data/ext_databases/HF_Lung_V1'

PARSED_SEGMENTS_PATH = BASE_DIR / 'parsed_segments.csv'

SAMPLE_RATE      = 4000
REFERENCE_DEVICE = 'steth'   # Littmann 3200 — used as the spectral reference

LOCAL_DATA_DIR       = BASE_DIR / 'data'
CORRECTED_DIR        = LOCAL_DATA_DIR / 'corrected'
DEVICE_PROFILES_PATH = CORRECTED_DIR / 'device_profiles.json'

WINDOW_DURATION         = 2.0   # seconds
WINDOW_HOP              = 1.0   # seconds (50% overlap)
LABEL_OVERLAP_THRESHOLD = 0.1   # minimum overlap (s) to assign das/cas = 1

WINDOWS_DIR           = LOCAL_DATA_DIR / 'windows'
WINDOWS_MANIFEST_PATH = BASE_DIR / 'windows_manifest.csv'

AUGMENTED_DIR       = LOCAL_DATA_DIR / 'augmented'
N_AUGMENTATIONS     = 5
AUG_VOLUME_RANGE    = (0.7, 1.3)
AUG_NOISE_SNR_RANGE = (30.0, 40.0)   # dB
AUG_SPEED_RANGE     = (0.95, 1.05)

# Spectrogram parameters
# WIN_LENGTH=64 (16 ms) keeps the analysis window short for crackle detection.
# N_FFT=1024 zero-pads that 16 ms window to give 513 frequency bins, which
# supports 224 mel filters cleanly (~2.3 bins/filter, no empty filters).
# N_MELS=224 matches MODEL_INPUT_SIZE height exactly — no height upsampling,
# so no row-boundary banding in the final image.
# hop=32 (8 ms) → 251 time frames from a 2 s window; resize 224×251 → 224×224
# downsamples the time axis only (0.89×), which is clean and artefact-free.
N_FFT            = 1024
WIN_LENGTH       = 64
HOP_LENGTH       = 32
N_MELS           = 224
FMIN             = 50.0
FMAX             = 2000.0            # Nyquist at 4 kHz
VTLP_ALPHA_MIN   = 0.9
VTLP_ALPHA_MAX   = 1.1
VTLP_FHI_MIN     = 1200.0           # Hz
VTLP_FHI_MAX     = 2000.0           # Hz
MODEL_INPUT_SIZE = (224, 224)        # (height, width)

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

SPECTROGRAMS_DIR = (
    Path('/content/data/spectrograms')
    if _in_colab()
    else LOCAL_DATA_DIR / 'spectrograms'
)

# Model / training
RESNET_VARIANT          = 'resnet18'
DROPOUT_P               = 0.5
FREEZE_BACKBONE_UNTIL   = 'layer2'   # freeze conv1–layer2, train layer3 onwards
CHECKPOINTS_DIR         = BASE_DIR / 'checkpoints'

RANDOM_SEED             = 42
BATCH_SIZE              = 64
NUM_EPOCHS              = 50
LEARNING_RATE           = 1e-4
WEIGHT_DECAY            = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_FOLDS               = 5         # k for GroupKFold cross-validation
import os
NUM_WORKERS = min(os.cpu_count() or 1, 4) if not _in_colab() else (os.cpu_count() or 1)