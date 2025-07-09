from scipy.io import wavfile
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import numpy as np
import os

recording_sample_directory = './dataset_generator/sounds/criset_0.120m_3class_motor/wav_ov1_split1_30db/'
result_directory = './dataset_generator/sounds/criset_0.120m_3class_motor/wav_ov1_split1_30db_filtered/'

for i in os.listdir(recording_sample_directory):
    recording_sample_path = os.path.join(recording_sample_directory, i)
    sr, recording_sample = wavfile.read(recording_sample_path)
    print(recording_sample_path)

    reduced_noise_recording_sample = np.zeros_like(recording_sample)
    for index, channel in enumerate(recording_sample.T):
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
    wavfile.write(result_path, sr, reduced_noise_recording_sample)