from scipy.io import wavfile
from scipy.signal import butter, filtfilt 
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import numpy as np
import os

highpass_value = 500

recording_sample_directory = './dataset_generator/sounds/criset_2class_motor/wav_ov1_split1_30db/'
result_directory = './dataset_generator/sounds/criset_2class_motor_schc/wav_ov1_split1_30db/'

if not os.path.exists(result_directory):os.mkdir(result_directory)

for i in os.listdir(recording_sample_directory):
    recording_sample_path = os.path.join(recording_sample_directory, i)
    sr, recording_sample = wavfile.read(recording_sample_path)
    print(recording_sample_path)

    b, a = butter(N=4, Wn=highpass_value, btype='highpass', fs=sr)
    reduced_noise_channel = filtfilt(b, a, recording_sample, axis=0)
    
    # Normalize to the range of int16 to avoid clipping
    max_value = np.max(np.abs(reduced_noise_channel))
    reduced_noise_channel = reduced_noise_channel / max_value
    reduced_noise_channel = reduced_noise_channel * 32768
    reduced_noise_channel = reduced_noise_channel.astype(np.int16)

    result_path = os.path.join(result_directory, i)
    wavfile.write(result_path, sr, reduced_noise_channel)