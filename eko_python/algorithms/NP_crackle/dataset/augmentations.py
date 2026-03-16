"""
augmentations.py

Online waveform augmentation functions for the ICBHI ALSC task.

All functions take a 1D numpy float32 audio array and return a 1D numpy
float32 audio array of the same length. They are designed to be composed
in sequence inside the Dataset __getitem__ method.

Functions:
    random_roll   — cyclic time shift (applied to all samples)
    random_volume — scale amplitude randomly (stretched samples only)
    random_noise  — add Gaussian noise (stretched samples only)
    random_pitch  — shift pitch without changing duration (stretched samples only)
    random_speed  — change speed and resample back to original length
                    (stretched samples only)

All randomness uses numpy's default_rng where a seed is needed, but in
normal use each call draws fresh random values so every epoch sees different
augmentations.
"""

import numpy as np
import librosa

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config_c import (
    SAMPLE_RATE,
    ROLL_MAX_SHIFT_FRACTION,
    VOLUME_MIN_GAIN,
    VOLUME_MAX_GAIN,
    NOISE_MIN_SNR_DB,
    NOISE_MAX_SNR_DB,
    PITCH_MIN_SEMITONES,
    PITCH_MAX_SEMITONES,
    SPEED_MIN_RATE,
    SPEED_MAX_RATE,
)


# ---------------------------------------------------------------------------
# Random roll
# ---------------------------------------------------------------------------

def random_roll(audio: np.ndarray, max_shift_fraction: float = ROLL_MAX_SHIFT_FRACTION) -> np.ndarray:
    """
    Cyclically shift the audio along the time axis by a random amount.

    Because cycles are reflect-padded, the boundaries are smooth and the
    wrap-around introduced by the cyclic shift does not create discontinuities.

    Applied to all samples every epoch.

    Parameters
    ----------
    audio               : 1D float32 numpy array
    max_shift_fraction  : maximum shift as a fraction of total length.
                          e.g. 0.5 means shift by up to half the signal.

    Returns
    -------
    1D float32 numpy array of the same length, cyclically shifted.
    """
    max_shift = int(len(audio) * max_shift_fraction)
    shift     = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(audio, shift).astype(np.float32)


# ---------------------------------------------------------------------------
# Random volume
# ---------------------------------------------------------------------------

def random_volume(
    audio: np.ndarray,
    min_gain: float = VOLUME_MIN_GAIN,
    max_gain: float = VOLUME_MAX_GAIN,
) -> np.ndarray:
    """
    Scale the amplitude of the audio by a random gain factor.

    Simulates recordings made at different distances from the microphone
    or with different stethoscope pressure.

    Applied to stretched samples only.

    Parameters
    ----------
    audio    : 1D float32 numpy array
    min_gain : minimum gain multiplier (e.g. 0.7 = 30% quieter)
    max_gain : maximum gain multiplier (e.g. 1.3 = 30% louder)

    Returns
    -------
    1D float32 numpy array with scaled amplitude.
    """
    gain = np.random.uniform(min_gain, max_gain)
    return (audio * gain).astype(np.float32)


# ---------------------------------------------------------------------------
# Random noise
# ---------------------------------------------------------------------------

def random_noise(
    audio: np.ndarray,
    min_snr_db: float = NOISE_MIN_SNR_DB,
    max_snr_db: float = NOISE_MAX_SNR_DB,
) -> np.ndarray:
    """
    Add Gaussian white noise at a random signal-to-noise ratio.

    SNR is sampled uniformly in dB. Higher SNR = less noise.

    Applied to stretched samples only.

    Parameters
    ----------
    audio       : 1D float32 numpy array
    min_snr_db  : minimum SNR in dB (e.g. 20 = noisier)
    max_snr_db  : maximum SNR in dB (e.g. 40 = cleaner)

    Returns
    -------
    1D float32 numpy array with added noise.
    """
    snr_db = np.random.uniform(min_snr_db, max_snr_db)

    # Compute signal power
    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-10:
        return audio

    # Compute required noise power from SNR
    snr_linear  = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise       = np.random.normal(0, np.sqrt(noise_power), size=len(audio))

    return (audio + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Random pitch shift
# ---------------------------------------------------------------------------

def random_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_semitones: float = PITCH_MIN_SEMITONES,
    max_semitones: float = PITCH_MAX_SEMITONES,
) -> np.ndarray:
    """
    Shift the pitch of the audio by a random number of semitones without
    changing its duration.

    Uses librosa's phase vocoder-based pitch shifting. Simulates variation
    in the fundamental frequencies of lung sounds across patients.

    Applied to stretched samples only.

    Parameters
    ----------
    audio         : 1D float32 numpy array
    sr            : sample rate
    min_semitones : minimum pitch shift in semitones (negative = lower pitch)
    max_semitones : maximum pitch shift in semitones (positive = higher pitch)

    Returns
    -------
    1D float32 numpy array with shifted pitch, same length as input.
    """
    n_steps = np.random.uniform(min_semitones, max_semitones)
    shifted = librosa.effects.pitch_shift(
        audio.astype(np.float32),
        sr=sr,
        n_steps=n_steps,
    )
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# Random speed change
# ---------------------------------------------------------------------------

def random_speed(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_rate: float = SPEED_MIN_RATE,
    max_rate: float = SPEED_MAX_RATE,
) -> np.ndarray:
    """
    Change the speed of the audio by a random factor, then resample back
    to the original length.

    Unlike time stretching (which preserves pitch), speed change alters both
    duration and pitch simultaneously — it simulates playing a recording
    slightly faster or slower. The signal is then resampled back to its
    original length so the array size is unchanged.

    Applied to stretched samples only.

    Parameters
    ----------
    audio    : 1D float32 numpy array
    sr       : sample rate
    min_rate : minimum speed factor (e.g. 0.9 = 10% slower)
    max_rate : maximum speed factor (e.g. 1.1 = 10% faster)

    Returns
    -------
    1D float32 numpy array of the same length as input.
    """
    rate          = np.random.uniform(min_rate, max_rate)
    original_len  = len(audio)

    # Resample to simulate speed change:
    # speeding up = resample to fewer samples, then stretch back
    # slowing down = resample to more samples, then compress back
    new_sr        = int(sr * rate)
    speed_changed = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=new_sr)

    # Resample back to original length
    restored = librosa.resample(speed_changed, orig_sr=new_sr, target_sr=sr)

    # Ensure exact original length (resampling can introduce off-by-one)
    if len(restored) > original_len:
        restored = restored[:original_len]
    elif len(restored) < original_len:
        restored = np.pad(restored, (0, original_len - len(restored)), mode='reflect')

    return restored.astype(np.float32)