from scipy.io import wavfile
import noisereduce as nr
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
from multiprocessing import Pool  
import numpy as np
import os

recording_sample_directory = './dataset_generator/sounds/criset_2class_motor/wav_ov1_split1_30db/'
result_directory = './dataset_generator/sounds/criset_2class_motor_filtered/wav_ov1_split1_30db/'
noise_directory = './dataset_generator/sounds/DREGON_individual_motors_recordings/allMotors_70.wav'

sample_rate = 44100
noise_time = 1

sr, noise_data = wavfile.read(noise_directory)
noise_data = noise_data[:sample_rate * noise_time].T * 0.001
# noise_data = np.zeros((8, sample_rate * noise_time))

os.makedirs(result_directory, exist_ok=True)

def process_file(filename):
    path = os.path.join(recording_sample_directory, filename)
    sr, data = wavfile.read(path)
    print(f"Processing {filename}")

    # stationary filtering
    # reduced = nr.reduce_noise(
    #     y = data.T,
    #     y_noise=noise_data,
    #     prop_decrease=1.0,
    #     sr=sr,
    #     n_std_thresh_stationary=1.5,
    #     stationary=True
    # ).T

    # non stationary filtering
    reduced = nr.reduce_noise(
        y = data.T,
        sr=sr,
        prop_decrease=1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=2.0,
        sigmoid_slope_nonstationary=10.0,
        n_std_thresh_stationary=1.5,
        stationary=False
    ).T

    #implement noise normalization
    max_value = np.max(reduced)
    reduced = reduced.astype(np.float64)
    reduced = reduced / max_value
    reduced = reduced * 32768
    reduced = reduced.astype(np.int16)

    out_path = os.path.join(result_directory, filename)
    wavfile.write(out_path, sr, reduced)

if __name__ == "__main__":
    files = os.listdir(recording_sample_directory)

    with Pool() as pool:
        pool.map(process_file, files)
