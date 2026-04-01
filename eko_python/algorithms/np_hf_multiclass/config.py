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

WINDOW_DURATION         = 1.0   # seconds
WINDOW_HOP              = 0.5   # seconds (50% overlap)
LABEL_OVERLAP_THRESHOLD = 0.1   # minimum overlap (s) to assign das/cas = 1

WINDOWS_DIR           = LOCAL_DATA_DIR / 'windows'
WINDOWS_MANIFEST_PATH = BASE_DIR / 'windows_manifest.csv'