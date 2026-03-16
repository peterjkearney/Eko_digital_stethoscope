import os

# file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the Config.py location
dir_rawData = os.path.join(BASE_DIR, "../Databases/ICBHI/audio_and_txt_files_CLEANED")
dir_preprocessed = os.path.join(BASE_DIR, "../Databases/ICBHI/preprocessing_CLEANED")
dir_testTrainData = os.path.join(BASE_DIR, "../Databases/ICBHI/traintest_CLEANED")

sample_keys = ["signal", "label", "mel_spectrogram", "statistics_feature", "spectrogram"]

# Label Set
normal = 0
crackle = 1
wheezes = 2
both = 3


diagnosis_file_dir = "../Databases/ICBHI/patient_diagnosis.csv"
Healthy = 0
URTI = 1
Asthma = 2
COPD = 3
LRTI = 4
Bronchiectasis = 5
Pneumonia = 6
Bronchiolitis = 7


# signal
sample_rate = 4000
# padding_mode: (str) zero,  sample;
# sample padding ref to the paper "Lung Sound Classification Using Snapshot Ensemble of Convolutional Neural Networks"
padding_mode = "zero"


# filter
filter_lowcut = 80
filter_highcut = 1900
filter_order = 8
filter_btype = "bandpass"           # filter_btype: (str) highpass, bandpass


# samples set
respiratory_cycle = 5   # The length of data as input
#overlap = 5
min_valid_segment_length = 1.0
max_valid_segment_length = 5.0

# Mel spectrogram
winLength_narrow = 64
n_mels_narrow = 24
nfft_narrow = 128
hop_narrow = 32

winLength_wide = 256
n_mels_wide = 48
nfft_wide = 512
hop_wide = 128

f_min = 100
f_max = 1800


# For train and test
Num_classes = 2
device = "cpu"             # device
num_epochs = 80
batch_size = 32
patience = 15

lr = 0.0003                  # learn rate
weight_decay = 0.05     # l2 normalization
lr_crackle = 0.0001
weight_decay_crackle = 0.03
lr_wheeze = 0.0005
weight_decay_wheeze = 0.0005



