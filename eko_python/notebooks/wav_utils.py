import os
import scipy.signal as signal
import pandas as pd
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import shutil
import pickle
import pywt
from scipy.stats import kurtosis
from skimage.filters import threshold_otsu

def filter_denoise(raw_audio, sample_rate, filter_order, filter_lowcut=80, filter_highcut=1800, btype="bandpass"):
    
    if btype == "bandpass":
        sos = signal.butter(filter_order, [filter_lowcut, filter_highcut], btype="bandpass",fs = sample_rate, output = 'sos')

    elif btype == "highpass":
        sos = signal.butter(filter_order, filter_lowcut, btype="highpass", fs=sample_rate,output='sos')

    elif btype == "lowpass":
        sos = signal.butter(filter_order, filter_highcut, btype="lowpass", fs=sample_rate,output='sos')

    audio = signal.sosfiltfilt(sos, raw_audio)

    return audio


def dc_normalise(sig_array):
    """Removes DC and normalises to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max()
    return sig_array_norm


def print_DWT_band_info(coeffs, sampling_rate):
    
    numLevels = len(coeffs)-1
    
    # Print info about each band
    for i, coeff in enumerate(coeffs):
        if i == 0:
            freq_max = sampling_rate / (2 ** (len(coeffs)))
            print(f"cA{numLevels}: 0 - {freq_max:.2f} Hz, length: {len(coeff)}")
        else:
            level = numLevels+1 - i
            freq_min = sampling_rate / (2 ** (level + 1))
            freq_max = sampling_rate / (2 ** level)
            print(f"D{level}: {freq_min:.2f} - {freq_max:.2f} Hz, length: {len(coeff)}")





def TKE_otsu_thresholding(coeffs, wavelet):
    
    coeffsPost = []

    for i, coeff in enumerate(coeffs):
        if i ==0 or i==5:
            coeffsPost.append(np.zeros_like(coeff))
            continue
        else:

            rawCoeffs = np.array(coeff)

            # calculating Teager Kaiser Energy of signal
            tke = rawCoeffs[1:-1]**2 - rawCoeffs[:-2]*rawCoeffs[2:]

            # If kurtosis is gaussian-like, dismiss band as noise

            kurt = kurtosis(tke)
            print('Kurtosis: ',kurt)
            
            if kurt > 5:
                adjCoeffs = np.copy(rawCoeffs)

                threshold_value = threshold_otsu(tke)
                mask = tke < threshold_value
                adjCoeffs[1:-1][mask] = 0.

            else:
                
                adjCoeffs = np.zeros_like(rawCoeffs)

            coeffsPost.append(adjCoeffs)

    reconstructed_signal = pywt.waverec(coeffsPost, wavelet)

    return reconstructed_signal

def get_start_end_adventitious(reconstructed_signal):
    inAdventitious = False
    zeroCount = 0

    startIndices = []
    endIndices = []

    # running from index 50 to -50 as start and end are non-zero due to edge effects 
    for i in range(50,len(reconstructed_signal)-50):

        if inAdventitious:
            # looking for run of 3 zeros to be sure we're in a zero zone
            if reconstructed_signal[i] == 0:
                if zeroCount >= 2:
                    # 3 in a row
                    inAdventitious = False
                    endIndices.append(i-2)
                    zeroCount = 0
                    
                else:
                    zeroCount += 1

        else:
            # looking for any non-zero to decide we're in an adventitious zone
            if reconstructed_signal[i] != 0:
                inAdventitious = True
                startIndices.append(i)
                    

    if len(startIndices) > len(endIndices):
        # an adventitious period was started but not finished, set a dummy end point at end-50
        endIndices.append(len(reconstructed_signal)-50)
    
    
    return startIndices, endIndices


def plot_start_end_adventitious(reconstructed_signal,startIndices,endIndices,sampling_rate):
    
    t_total = len(reconstructed_signal)/sampling_rate
    
    t = np.linspace(0,t_total,len(reconstructed_signal))
    
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    ax.plot(t,reconstructed_signal)

    startTimes = [(a/len(reconstructed_signal)) * t_total for a in startIndices]
    endTimes = [(a/len(reconstructed_signal)) * t_total for a in endIndices]
    
    ax2 = ax.twinx()
    ax2.vlines(startTimes, ymin=0, ymax=1, colors='g')
    ax2.vlines(endTimes, ymin=0, ymax=1, colors='r')

def get_dominant_freqs(x,sampling_rate,startIndices,endIndices,plotFigs=True):

    dominantFreqs = []
    n_fft = 4096

    for startIdx, endIdx in zip(startIndices,endIndices):
        segment = x[startIdx:endIdx]
        segment = segment - np.mean(segment)  # Remove DC

        fft_result = np.fft.fft(segment,n=n_fft)
        fft_magnitude = np.abs(fft_result)

        
        # Get frequency bins
        freqs = np.fft.fftfreq(n_fft, 1/sampling_rate)
        
        positive_freq_mask = freqs > 0
        positive_freqs = freqs[positive_freq_mask]
        positive_magnitudes = fft_magnitude[positive_freq_mask]

        # Find peak
        peak_index = np.argmax(positive_magnitudes)
        peak_frequency = positive_freqs[peak_index]

        if plotFigs:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(positive_freqs,positive_magnitudes)

        dominantFreqs.append(peak_frequency)
        #print(f"{startIdx} {endIdx} {peak_frequency:.2f} Hz")

    return dominantFreqs


def getTestInfo(datFileName):
    
    # Removing file type suffic
    fileStub = str.split(datFileName,'.')[0]

    # Splitting file name by '_' delimitter
    strComponents = str.split(fileStub,'_')

    userName = strComponents[1]
    userTest = strComponents[2]
    userSeverity = strComponents[3]

    return userName, userTest, userSeverity


def tke(coeffs,order):

    if order ==1:
        tke = coeffs[1:-1]**2 - coeffs[:-2]*coeffs[2:]
        tks = np.pad(tke,pad_width=1,constant_values=0)
    elif order==2:
        tke = coeffs[2:-2]**2 - coeffs[:-4]*coeffs[4:]
        tks = np.pad(tke,pad_width=2,constant_values=0)
    
    else:
        print('Error: invalid order')
        return -1
        

    return tks