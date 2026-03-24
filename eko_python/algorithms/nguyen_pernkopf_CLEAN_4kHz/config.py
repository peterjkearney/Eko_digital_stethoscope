from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR         = Path(__file__).parent
EKO_PROJECT_ROOT = BASE_DIR.parent.parent.parent
DATA_DIR         = EKO_PROJECT_ROOT / 'data/ext_databases/ICBHI'

RAW_DATA_PATH        = DATA_DIR / 'audio_and_txt_files'
OFFICIAL_SPLIT_PATH  = DATA_DIR / 'ICBHI_challenge_train_test.txt'
DIAGNOSIS_FILE_PATH  = DATA_DIR / 'patient_diagnosis.csv'
MANIFEST_PATH        = BASE_DIR / 'manifest.csv'

# ---------------------------------------------------------------------------
# Known filename errors in the ICBHI dataset.
# Maps the incorrect filename (as on disk) to the corrected filename.
# ---------------------------------------------------------------------------

FILENAME_CORRECTIONS = {
    '226_1b1_Pl_sc_LittC2SE': '226_1b1_Pl_sc_Meditron',
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

SAMPLE_RATE       = 4000    # Hz  — matches Eko device native rate
CYCLE_DURATION    = 4.0     # seconds — cycles longer than this are dropped;
                            # shorter cycles are reflect-padded to this length

# ---------------------------------------------------------------------------
# Spectrum correction  (step 02)
# ---------------------------------------------------------------------------

REFERENCE_DEVICE     = 'Meditron'   # device whose spectral profile all others
                                    # are corrected to match
CORRECTED_DIR        = BASE_DIR / 'data' / 'corrected'
DEVICE_PROFILES_PATH = CORRECTED_DIR / 'device_profiles.json'

# STFT used for spectrum correction and spectrogram computation.
#
# WIN_LENGTH = 64 samples @ 4 kHz = 16 ms  — short enough to resolve crackle transients
# HOP_LENGTH = 64 samples @ 4 kHz = 16 ms  — non-overlapping frames
# N_FFT      = 512                          — zero-pads the 16 ms window before the FFT,
#                                             giving 257 freq bins (0–2 kHz, 7.8 Hz/bin)
#                                             without widening the analysis window
#
# Resulting spectrogram for a 4 s cycle: 257 freq bins × 251 time frames
# Both dims exceed 224, so resizing to 224×224 only downsamples — no upsampling artefacts.
N_FFT      = 512
WIN_LENGTH = 64
HOP_LENGTH = 64

# ---------------------------------------------------------------------------
# Cycle splitting and padding  (step 03)
# ---------------------------------------------------------------------------

CYCLES_DIR = BASE_DIR / 'data' / 'cycles'   # train/ and test/ subdirs created here

# ---------------------------------------------------------------------------
# Spectrogram  (step 05)
# ---------------------------------------------------------------------------

N_MELS = 128
FMIN   = 50     # Hz — ignore sub-bass rumble
FMAX   = 2000   # Hz — Nyquist at 4 kHz; nothing meaningful above this

# VTLP warp parameters applied to augmented copies (aug_index > 0).
# Originals and test samples get identity (alpha=1.0).
# Range mirrors the old 16 kHz pipeline scaled to 4 kHz Nyquist:
#   old fhi range 3200–3800 Hz out of 8000 Hz Nyquist → 40–47% of Nyquist
#   at 4 kHz Nyquist=2000: 40–47% → 800–950 Hz
VTLP_ALPHA_MIN = 0.9
VTLP_ALPHA_MAX = 1.1
VTLP_FHI_MIN   = 800
VTLP_FHI_MAX   = 950

MODEL_INPUT_SIZE = (224, 224)   # (H, W) — ResNet input; spectrogram is
                                # downsampled to fit, never upsampled
SPECTROGRAMS_DIR = BASE_DIR / 'data' / 'spectrograms'

# ---------------------------------------------------------------------------
# Augmentation  (step 04)
# ---------------------------------------------------------------------------

AUGMENTED_DIR   = BASE_DIR / 'data' / 'augmented'
N_AUGMENTATIONS = 5    # augmented copies per training cycle (originals kept as aug_index=0)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

import os
NUM_WORKERS = min(os.cpu_count() or 1, 4)   # DataLoader workers; use available CPUs up to 4

RANDOM_SEED = 42

BATCH_SIZE               = 512
NUM_EPOCHS               = 100
EARLY_STOPPING_PATIENCE  = 15
OVERFIT_GAP_THRESHOLD    = 0.20  # train_icbhi − val_icbhi must exceed this to count
OVERFIT_GAP_PATIENCE     = 2     # consecutive train-eval epochs gap must persist to stop (= 10 training epochs)
NUM_FOLDS                = 5
LEARNING_RATE            = 5e-4
WEIGHT_DECAY             = 1e-3

RESNET_VARIANT     = 'resnet18'   # 'resnet18', 'resnet34', or 'resnet50'
NUM_CLASSES        = 4            # normal, crackle, wheeze, both
NUM_SOURCE_CLASSES = 1000         # ImageNet classes (source classifier)
COTUNING_LAMBDA    = 0.3          # weight of the KL divergence term in the co-tuning loss
DROPOUT_P          = 0.5          # dropout before target classifier head
CHECKPOINTS_DIR    = BASE_DIR / 'checkpoints'

# ---------------------------------------------------------------------------
# Augmentation  (step 04)
# ---------------------------------------------------------------------------

AUG_VOLUME_RANGE    = (0.7,  1.3)    # multiplicative gain
AUG_NOISE_SNR_RANGE = (30.0, 40.0)   # dB — higher = quieter noise
AUG_SPEED_RANGE     = (0.85, 1.15)   # rate < 1 slows down (lower pitch), > 1 speeds up (higher pitch)
                                     # implemented via resampling — no phase vocoder, transients preserved
