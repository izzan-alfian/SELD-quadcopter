from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import numpy as np
import os

recording_sample_directory = './dataset_generator/sounds/8_channel_microphone_signals/wav/'
result_directory = './dataset_generator/sounds/filtered_8_channel_microphone_signals/wav_ov1_split1_50db/'

for i in os.listdir(recording_sample_directory):
    recording_sample_path = os.path.join(recording_sample_directory, i)
    recording_sample, sr = sf.read(recording_sample_path)

    reduced_noise_recording_sample = np.zeros_like(recording_sample)
    for index, channel in enumerate(recording_sample.T):
        print(index)
        reduced_noise_channel = nr.reduce_noise(
            y = channel,
            sr=sr,
            prop_decrease=1.0,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            thresh_n_mult_nonstationary=2.0,
            sigmoid_slope_nonstationary=10.0,
            n_std_thresh_stationary=1.5,
            stationary=False
        )
        reduced_noise_recording_sample[:, index] = reduced_noise_channel

    result_path = os.path.join(result_directory, i)
    sf.write(result_path, reduced_noise_recording_sample, sr)