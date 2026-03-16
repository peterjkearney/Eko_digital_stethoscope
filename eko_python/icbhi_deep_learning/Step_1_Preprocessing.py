import os
import shutil
import numpy as np
import scipy.signal as signal
import librosa
try:
    from DeepLearning_ICBHI import Config
except ModuleNotFoundError:
    import Config
import pickle
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import pywt
from PyEMD import EMD


def rawFilter(raw_audio, sample_rate, filter_order, filter_lowcut=80, filter_highcut=1800, btype="bandpass"):
    
    if btype == "bandpass":
        sos = signal.butter(filter_order, [filter_lowcut, filter_highcut], btype="bandpass",fs = sample_rate, output = 'sos')

    elif btype == "highpass":
        sos = signal.butter(filter_order, filter_lowcut, btype="highpass", fs=sample_rate,output='sos')

    elif btype == "lowpass":
        sos = signal.butter(filter_order, filter_highcut, btype="lowpass", fs=sample_rate,output='sos')

    audio = signal.sosfiltfilt(sos, raw_audio)

    return audio


def padding(data, sample_rate, respiratory_cycle, padding_mode):

    if len(data) == (sample_rate * respiratory_cycle):
        return data  # the duration of data is equal to the length of the demand
    else:
        padding = None
        if padding_mode == "zero":
            padding = np.zeros((data.shape[0]))
        elif padding_mode == "sample":
            padding = data.copy()

        while True:  
            data = np.concatenate([data, padding])
            if len(data) > (sample_rate * respiratory_cycle):
                data = data[:int(sample_rate * respiratory_cycle)]

            if len(data) == (sample_rate * respiratory_cycle):
                return data



def segmentation(audio, eventsFilePath, sample_rate, respiratory_cycle, padding_mode, show_function=False):
    
    #Takes the start and end timestamps from the events text files. 
    # If the gap between the start and end time for any period is less than 2s, it gets discarded
    # If it's greater than 2s, it gets padded out to 5s with zeros
    # If it's greater than 5s, the first 5s is stored, then the process is applied again to the remainder

    samples = {
        "signal": [],
        "label": [],
    }

    eventsData = open(f"{eventsFilePath}", "r")
    events = eventsData.readlines()
    eventsData.close()

    for i, event in enumerate(events):
        
        event = event.strip("\n")
        lStart, lEnd, lcrackle, lwheeze = event.split("\t")
        lStart = float(lStart)
        lEnd = float(lEnd)
        lcrackle = int(lcrackle)
        lwheeze = int(lwheeze)

        # Label Construction
        label = None  # normal - 0, crackle - 1, wheezes - 2, both - 3
        if lcrackle == 0 and lwheeze == 0:
            label = Config.normal
        elif lcrackle == 1 and lwheeze == 0:
            label = Config.crackle
        elif lcrackle == 0 and lwheeze == 1:
            label = Config.wheezes
        elif lcrackle == 1 and lwheeze == 1:
            label = Config.both

        # Data Construction
        # If segment length is less than 1s, discard (4% of data)
        # If segement length is between 1 and 5s, pad out to 5s with zeros
        # If segment length is greater than 5s, discard (can't be sure if labelled crackle or wheeze is present in 5s subsection) (3% of data)


        if lStart < lEnd: #i.e. valid timings
            
            seg_length = lEnd - lStart

            if (seg_length < (Config.min_valid_segment_length)) or (seg_length > (Config.max_valid_segment_length)):
                continue

            startIdx = int(lStart * sample_rate)
            endIdx = int(lEnd * sample_rate)

            seg_data = padding(audio[startIdx:endIdx], sample_rate, respiratory_cycle, padding_mode)

            if show_function:
                plt.plot(seg_data)
                plt.show()

            samples["signal"].append(seg_data)
            samples["label"].append(label)

            

    return samples


class Labels:
    
    def __init__(self):
        filelabel = open(f"{Config.diagnosis_file_dir}", "r")
        res = filelabel.readlines()
        filelabel.close()

        self.diagosis_label_list = {}
        for i, cur in enumerate(res):
            cur = cur.strip("\n")
            patientID, diagnosis = cur.split(",")

            if diagnosis == "URTI":
                diagnosis = Config.URTI
            elif diagnosis == "Healthy":
                diagnosis = Config.Healthy
            elif diagnosis == "Asthma":
                diagnosis = Config.Asthma
            elif diagnosis == "COPD":
                diagnosis = Config.COPD
            elif diagnosis == "LRTI":
                diagnosis = Config.LRTI
            elif diagnosis == "Bronchiectasis":
                diagnosis = Config.Bronchiectasis
            elif diagnosis == "Pneumonia":
                diagnosis = Config.Pneumonia
            elif diagnosis == "Bronchiolitis":
                diagnosis = Config.Bronchiolitis

            self.diagosis_label_list[patientID] = diagnosis

    def getLabel(self,patient):
        return self.diagosis_label_list[patient]



def preprocessing(dir_raw_files, dir_preprocessed):
    print('start Preprocessing')

    # Make folder to store preprocessed files (delete and replace if already exists)
    if os.path.exists(f"{dir_preprocessed}"):
        shutil.rmtree(f"{dir_preprocessed}")
        os.makedirs(f"{dir_preprocessed}")
    else:
        os.makedirs(f"{dir_preprocessed}")

    allLabels = Labels()

    
    for file_name in sorted(os.listdir(dir_raw_files)):
        if ".wav" not in file_name:
            continue

        # path
        rawFilePath = f"{dir_raw_files}/{file_name}"
        eventsFilePath = f"{dir_raw_files}/{file_name.split('.')[0]}.txt" # Txt files have same filestub as wav files

        # load data
        raw_audio, sample_rate = librosa.load(path=rawFilePath, sr=Config.sample_rate)

        # Noise reduction method, filter
        audio_data = rawFilter(raw_audio, sample_rate, Config.filter_order,
                                     Config.filter_lowcut, Config.filter_highcut, btype=Config.filter_btype)
        

        # Segmentation - data & label
        samples = segmentation(audio_data, eventsFilePath, sample_rate, Config.respiratory_cycle, Config.padding_mode)
        
        
        if samples["signal"] == []:
            continue

        # diagnosis label
        patient = file_name[:3]

        diagnosis_label = allLabels.getLabel(patient)

        # Save to dir_preprocessed
        for i in range(len(samples["signal"])):
            save_dir = dir_preprocessed + '/' + file_name.split('.')[0] + f"_{i}.dat"

            temp = {
                "signal": samples["signal"][i],
                "label": samples["label"][i],
                "diagnosis": diagnosis_label
            }
            with open(save_dir, 'wb') as f:
                pickle.dump(temp, f)

        print(f"{file_name} over")
    


if __name__ == '__main__':
    preprocessing(Config.dir_rawData, Config.dir_preprocessed)

        
