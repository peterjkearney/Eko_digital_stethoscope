import matplotlib.pyplot as plt
try:
    from DeepLearning_ICBHI import Config
except ModuleNotFoundError:
    import Config
import librosa
import numpy as np
import os
import pickle
from scipy.stats import kurtosis
from tqdm import tqdm

def compute_spectral_flux(signal, sr, window=32, hop=16):
    """
    Compute spectral flux across frames
    Measures how quickly the power spectrum changes
    """

    n_fft = max(64, window)
    
    # Compute STFT
    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop, 
                            win_length=window))
    
    # Normalize each frame
    S_normalized = S / (S.sum(axis=0) + 1e-10)
    
    # Compute flux between consecutive frames
    flux = np.sqrt(np.sum(np.diff(S_normalized, axis=1)**2, axis=0))
    
    return flux

def extract_short_window_features(signal, sr, window=32, hop=16):
    features = {}
    
    # Zero-crossing rate - captures discontinuous, explosive nature
    features['zcr'] = librosa.feature.zero_crossing_rate(
        signal, frame_length=window, hop_length=hop)[0]
    
    # RMS energy - captures sudden amplitude spikes
    features['rms'] = librosa.feature.rms(
        y=signal, frame_length=window, hop_length=hop)[0]
    
    # Spectral flux - requires custom calculation
    features['flux'] = compute_spectral_flux(signal, sr, window, hop)
    
    # Summary statistics
    summary = {
        'short_zcr_mean': np.mean(features['zcr']),
        'short_zcr_std': np.std(features['zcr']),
        'short_rms_mean': np.mean(features['rms']),
        'short_rms_std': np.std(features['rms']),
        'short_rms_max': np.max(features['rms']),
        'short_rms_kurtosis': kurtosis(features['rms']),
        'short_flux_mean': np.mean(features['flux']),
        'short_flux_std': np.std(features['flux']),
        'short_flux_max': np.max(features['flux'])
    }
    
    return summary

def extract_long_window_features(signal, sr, window=256, hop=128):
    features = {}
    
    # Spectral centroid - wheezes have concentrated frequency content
    features['centroid'] = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=window, hop_length=hop)[0]
    
    # Spectral bandwidth - wheezes are narrow-band
    features['bandwidth'] = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=window, hop_length=hop)[0]
    
    # Spectral rolloff
    features['rolloff'] = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, n_fft=window, hop_length=hop)[0]
    
    # Spectral flatness - wheezes are tonal (low flatness)
    features['flatness'] = librosa.feature.spectral_flatness(
        y=signal, n_fft=window, hop_length=hop)[0]
    
    # MFCCs - capture overall spectral shape
    mfccs = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=13, n_fft=window, hop_length=hop)
    
    # Summary statistics
    summary = {
        'long_centroid_mean': np.mean(features['centroid']),
        'long_centroid_std': np.std(features['centroid']),
        'long_bandwidth_mean': np.mean(features['bandwidth']),
        'long_bandwidth_std': np.std(features['bandwidth']),
        'long_rolloff_mean': np.mean(features['rolloff']),
        'long_flatness_mean': np.mean(features['flatness']),  # Low = tonal
        'long_flatness_std': np.std(features['flatness']),
    }
    
    # Add MFCC statistics
    for i in range(mfccs.shape[0]):
        summary[f'long_mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        summary[f'long_mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    return summary

def extract_respiratory_features(signal, sr,short_window=32,long_window=256):

    short_hop = short_window // 2
    long_hop = long_window // 2
    
    short_feats = extract_short_window_features(signal, sr, short_window, short_hop)
    long_feats = extract_long_window_features(signal, sr, long_window, long_hop)
    
    all_features = {**short_feats, **long_feats}
    feature_vector = np.array(list(all_features.values()))
    feature_names = list(all_features.keys())
    
    return feature_vector, feature_names




def data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample


def dc_normalise(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max()
    return sig_array_norm


def create_mel_spectrogram(data, sample_rate, n_mels=128, f_min=50, f_max=1800, winLength = 256, nfft = 256, hop=256, show_function=False):
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max,
                                       win_length = winLength, n_fft=nfft, hop_length=hop,center=False)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())

    if show_function:
        fig, ax = plt.subplots()
        fig.set_size_inches(8,3)

        img = librosa.display.specshow(S, x_axis='time',
                                       y_axis='mel', sr=sample_rate, hop_length = hop,
                                       fmax=f_max, ax=ax)
        plt.show()

    return S



def feature_extraction(dir_preprocessed):
    dat_files = [f for f in os.listdir(dir_preprocessed) if f.endswith('.dat')]
    
    # First pass: Check all files have required keys
    print("Checking data integrity...")
    problematic_files = []
    for datFileName in dat_files:
        path_datFile = os.path.join(dir_preprocessed, datFileName)
        try:
            samples = data_Acq(path_datFile)
            # Check required keys from Step 1
            if 'signal' not in samples:
                problematic_files.append((datFileName, 'missing signal'))
            if 'label' not in samples:
                problematic_files.append((datFileName, 'missing label'))
            if 'diagnosis' not in samples:
                problematic_files.append((datFileName, 'missing diagnosis'))
        except Exception as e:
            problematic_files.append((datFileName, str(e)))
    
    if problematic_files:
        print(f"\n⚠️  Found {len(problematic_files)} problematic files:")
        for fname, issue in problematic_files[:10]:  # Show first 10
            print(f"  {fname}: {issue}")
        print("\n❌ Please rerun Step_1_Preprocessing.py")
        return
    
    print("✓ All files have required keys\n")
    
    # Collect features for normalization (first pass)
    print("Computing normalization statistics...")
    all_features = []
    for datFileName in tqdm(dat_files, desc="Collecting features"):
        try:
            path_datFile = os.path.join(dir_preprocessed, datFileName)
            samples = data_Acq(path_datFile)
            data = dc_normalise(samples["signal"])
            
            acoustic_features, acoustic_feature_names = extract_respiratory_features(
                data, Config.sample_rate,
                short_window=Config.winLength_narrow,
                long_window=Config.winLength_wide
            )
            all_features.append(acoustic_features)
        except Exception as e:
            print(f"Error processing {datFileName}: {e}")
            continue
    
    all_features = np.array(all_features)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    
    print(f"Feature mean range: [{feature_mean.min():.2f}, {feature_mean.max():.2f}]")
    print(f"Feature std range: [{feature_std.min():.2f}, {feature_std.max():.2f}]")
    
    # Save normalization stats
    norm_stats = {
        'mean': feature_mean,
        'std': feature_std
    }
    with open(os.path.join(dir_preprocessed, 'feature_norm_stats.pkl'), 'wb') as f:
        pickle.dump(norm_stats, f)
    
    # Second pass: normalize and save (second pass)
    print("\nNormalizing and saving features...")
    successful = 0
    failed = 0
    
    for datFileName in tqdm(dat_files, desc="Processing"):
        try:
            path_datFile = os.path.join(dir_preprocessed, datFileName)
            samples = data_Acq(path_datFile)
            
            # Verify keys exist
            assert 'signal' in samples, f"Missing 'signal' in {datFileName}"
            assert 'label' in samples, f"Missing 'label' in {datFileName}"
            assert 'diagnosis' in samples, f"Missing 'diagnosis' in {datFileName}"
            
            data = dc_normalise(samples["signal"])
            
            # Mel spectrograms
            mel_spectrogram_narrow = create_mel_spectrogram(
                data, Config.sample_rate, 
                n_mels=Config.n_mels_narrow,
                f_min=Config.f_min, f_max=Config.f_max, 
                winLength=Config.winLength_narrow,
                nfft=Config.nfft_narrow, 
                hop=Config.hop_narrow
            )
            
            mel_spectrogram_wide = create_mel_spectrogram(
                data, Config.sample_rate, 
                n_mels=Config.n_mels_wide,
                f_min=Config.f_min, f_max=Config.f_max, 
                winLength=Config.winLength_wide,
                nfft=Config.nfft_wide, 
                hop=Config.hop_wide
            )
            
            # Acoustic features
            acoustic_features, acoustic_feature_names = extract_respiratory_features(
                data, Config.sample_rate,
                short_window=Config.winLength_narrow,
                long_window=Config.winLength_wide
            )
            
            # NORMALIZE FEATURES
            acoustic_features = (acoustic_features - feature_mean) / (feature_std + 1e-8)
            
            # Store everything (preserving original keys!)
            samples["statistics_feature"] = acoustic_features
            samples["statistics_feature_names"] = acoustic_feature_names
            samples["mel_spectrogram_narrow"] = mel_spectrogram_narrow
            samples["mel_spectrogram_wide"] = mel_spectrogram_wide
            
            # Verify all keys are present before saving
            required_keys = ['signal', 'label', 'diagnosis', 
                           'statistics_feature', 'statistics_feature_names',
                           'mel_spectrogram_narrow', 'mel_spectrogram_wide']
            for key in required_keys:
                assert key in samples, f"Missing key '{key}' before saving {datFileName}"
            
            # Save
            output_path = os.path.join(dir_preprocessed, datFileName)
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f)
            
            successful += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {datFileName}: {e}")
            failed += 1
            continue
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    feature_extraction(Config.dir_preprocessed)

    
