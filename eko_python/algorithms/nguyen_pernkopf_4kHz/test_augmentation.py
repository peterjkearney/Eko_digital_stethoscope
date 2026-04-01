"""
test_augmentation.py  —  TEMPORARY DIAGNOSTIC

Two sections:

1. RECONSTRUCTION COMPARISON
   Passes the source wav through the spectrum-correction STFT round-trip
   using identical correction parameters (identity filter, so no spectral
   change) and reconstructs via two methods:
       griffinlim/  — old Griffin-Lim phase reconstruction (n_iter=32)
       istft/       — new phase-preserving ISTFT (keeps original phase)
   Compare these two wavs against 00_original.wav to hear how much each
   reconstruction method degrades transient detail (crackles).

2. AUGMENTATION RANGE SWEEP
   Applies each augmentation type individually to the source wav across
   5 values spanning the configured min–max range. Output wavs go into
   augmentation/ for auditing before committing to the ranges in config.py.

Usage:
    python test_augmentation.py

Delete this file once both comparisons are confirmed.
"""

import sys
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    SAMPLE_RATE,
    CYCLE_DURATION,
    AUG_VOLUME_RANGE,
    AUG_NOISE_SNR_RANGE,
    AUG_PITCH_RANGE,
    AUG_SPEED_RANGE,
)

SOURCE_WAV = (
    Path(__file__).parent
    / 'data/cycles/train/103_2b2_Ar_mc_LittC2SE_cycle0000.wav'
)
OUTPUT_DIR     = Path(__file__).parent / 'test_augmentation_output'
TARGET_SAMPLES = int(SAMPLE_RATE * CYCLE_DURATION)
N_STEPS        = 5

# STFT params used by 02_spectrum_correction.py — must match exactly
_CORR_N_FFT      = 256
_CORR_WIN_LENGTH = 256
_CORR_HOP_LENGTH = 64


# ---------------------------------------------------------------------------
# Augmentation helpers (same as 04_augment.py)
# ---------------------------------------------------------------------------

def reflect_pad(audio: np.ndarray, target_length: int) -> np.ndarray:
    while len(audio) < target_length:
        pad   = min(target_length - len(audio), len(audio) - 1)
        audio = np.pad(audio, (0, pad), mode='reflect')
    return audio[:target_length]


def save(audio: np.ndarray, path: Path) -> None:
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.95
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE, subtype='PCM_16')
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Reconstruction comparison
# ---------------------------------------------------------------------------

def reconstruct_griffinlim(audio: np.ndarray, n_iter: int = 32) -> np.ndarray:
    # Old step-02 method: discard phase, reconstruct iteratively via Griffin-Lim.
    # Uses an identity correction filter (all zeros → exp(0)=1 → no spectral change)
    # so any difference from the original is purely due to phase reconstruction.
    stft      = librosa.stft(audio, n_fft=_CORR_N_FFT, win_length=_CORR_WIN_LENGTH,
                             hop_length=_CORR_HOP_LENGTH)
    magnitude = np.abs(stft)  # identity filter: magnitude unchanged
    return librosa.griffinlim(
        magnitude, n_iter=n_iter,
        hop_length=_CORR_HOP_LENGTH, win_length=_CORR_WIN_LENGTH,
    )


def reconstruct_istft(audio: np.ndarray) -> np.ndarray:
    # New step-02 method: keep original phase, apply magnitude correction only.
    # Identity filter → magnitude unchanged → should sound identical to input.
    stft           = librosa.stft(audio, n_fft=_CORR_N_FFT, win_length=_CORR_WIN_LENGTH,
                                  hop_length=_CORR_HOP_LENGTH)
    magnitude      = np.abs(stft)  # identity filter: magnitude unchanged
    corrected_stft = magnitude * np.exp(1j * np.angle(stft))
    return librosa.istft(corrected_stft,
                         hop_length=_CORR_HOP_LENGTH, win_length=_CORR_WIN_LENGTH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SOURCE_WAV.exists():
        raise FileNotFoundError(f"Source wav not found: '{SOURCE_WAV}'")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(SOURCE_WAV), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    print(f"Loaded: {SOURCE_WAV.name}  ({len(audio)} samples, {sr} Hz)\n")

    # Save the unmodified original for direct comparison
    save(audio.copy(), OUTPUT_DIR / '00_original.wav')

    # ── Reconstruction comparison ──────────────────────────────────────────
    # Both methods use an identity correction filter so any audible difference
    # is caused entirely by the reconstruction algorithm, not the correction.
    print("Reconstruction comparison (identity correction, no spectral change):")
    gl   = reconstruct_griffinlim(audio)
    save(gl.astype(np.float32),  OUTPUT_DIR / '01_reconstruction_griffinlim.wav')
    ist  = reconstruct_istft(audio)
    save(ist.astype(np.float32), OUTPUT_DIR / '01_reconstruction_istft.wav')
    print("  Compare 01_reconstruction_griffinlim.wav vs 01_reconstruction_istft.wav")
    print("  against 00_original.wav — listen for smearing of crackle transients.\n")

    # ── Volume ────────────────────────────────────────────────────────────
    print("Augmentation — Volume:")
    for gain in np.linspace(*AUG_VOLUME_RANGE, N_STEPS):
        out = audio * gain
        save(out, OUTPUT_DIR / f'aug_volume_gain{gain:.2f}.wav')

    # ── Noise ─────────────────────────────────────────────────────────────
    print("\nAugmentation — Noise:")
    rng = np.random.default_rng(seed=0)
    signal_power = np.mean(audio ** 2)
    for snr_db in np.linspace(*AUG_NOISE_SNR_RANGE, N_STEPS):
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise       = rng.normal(0.0, np.sqrt(noise_power), size=len(audio))
        out         = (audio + noise).astype(np.float32)
        save(out, OUTPUT_DIR / f'aug_noise_snr{snr_db:.1f}dB.wav')

    # ── Pitch ─────────────────────────────────────────────────────────────
    print("\nAugmentation — Pitch:")
    for semitones in np.linspace(*AUG_PITCH_RANGE, N_STEPS):
        out = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones)
        save(out, OUTPUT_DIR / f'aug_pitch_{semitones:+.1f}st.wav')

    # ── Speed ─────────────────────────────────────────────────────────────
    print("\nAugmentation — Speed:")
    for rate in np.linspace(*AUG_SPEED_RANGE, N_STEPS):
        out = librosa.resample(audio, orig_sr=int(SAMPLE_RATE * rate),
                               target_sr=SAMPLE_RATE)
        out = reflect_pad(out.astype(np.float32), TARGET_SAMPLES)
        save(out, OUTPUT_DIR / f'aug_speed_rate{rate:.2f}.wav')

    print(f"\nAll files written to '{OUTPUT_DIR}'")
    print("Reconstruction: compare 01_reconstruction_*.wav against 00_original.wav.")
    print("Augmentation:   compare aug_*.wav against 00_original.wav.")

    #testing multiple augmentations
    vol_change = 1.1644
    noise_change = 34.3888
    pitch_change = 0.7172
    speed_change = 1.25

    change_vol_only = audio * vol_change
    save(change_vol_only, OUTPUT_DIR / f'aug_vol_only.wav')

    signal_power = np.mean(change_vol_only ** 2)
    noise_power = signal_power / (10 ** (noise_change / 10))
    noise       = rng.normal(0.0, np.sqrt(noise_power), size=len(change_vol_only))
    change_vol_noise  = (change_vol_only + noise).astype(np.float32)
    save(change_vol_noise, OUTPUT_DIR / f'aug_vol_noise.wav')

    change_vol_noise_speed = librosa.resample(change_vol_noise,
                                              orig_sr=int(SAMPLE_RATE * speed_change),
                                              target_sr=SAMPLE_RATE)
    change_vol_noise_speed = reflect_pad(change_vol_noise_speed.astype(np.float32), TARGET_SAMPLES)
    save(change_vol_noise_speed, OUTPUT_DIR / f'aug_vol_noise_speed2.wav')

if __name__ == '__main__':
    main()
