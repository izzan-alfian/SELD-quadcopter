import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import librosa
import os
import scipy.io.wavfile as wav
from pathlib import Path

# --- CONFIGURATION ---
NB_TRAIN = 240
NB_TEST = 60

# --- Asset and Output Directories ---
# Using Path for better cross-platform compatibility
BASE_DIR = Path("./dataset_generator/sounds/ANSIM_clone")
RAW_DIRECTORY = BASE_DIR / "dcase2016_task2_train_dev/dcase2016_task2_train"
WAV_DIRECTORY = BASE_DIR / "wav_ov1_split1_30db"
DESC_DIRECTORY = BASE_DIR / "desc_ov1_split1"
TEMP_DIRECTORY = BASE_DIR / "temp"


# --- Simulation Parameters ---
SECONDS_LENGTH = 30
SR = 44100
MAX_POSSIBLE_WHISTLE_RADIUS = 100
MIN_POSSIBLE_WHISTLE_RADIUS = 50
SOUND_EVENT_GAP = 1.5
SOUND_EVENT_GAP_TOLERANT = 0.2

# Microphone array geometry (4 microphones)
MIC_POSITIONS = pra.circular_2D_array(center=[0,0], M=4, phi0=0, radius=37.5e-3)
MIC_POSITIONS = np.concatenate((MIC_POSITIONS, [[0, 0, 0 ,0]]), axis=0)


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def read_csv(file_name):
    desc_file = {
        'class': list(), 'start': list(), 'end': list(), 'ele': list(), 'azi': list(), 'dist': list()
    }
    fid = open(os.path.join(DESC_DIRECTORY, file_name), 'r')
    next(fid)
    for line in fid:
        split_line = line.strip().split(',')
        desc_file['class'].append(split_line[0])
        desc_file['start'].append(float(split_line[1]))
        desc_file['end'].append(float(split_line[2]))
        desc_file['ele'].append(float(split_line[3]))
        desc_file['azi'].append(float(split_line[4]))
        desc_file['dist'].append(float(split_line[5]))
    fid.close()
    return desc_file

def generate_sound(csv_dict):
    distance = 5   # for now, we just need to make sound generator to have a white Gaussian noise, a distance of 5 is a middleground of soundrange [0, 10]
    snr_db = 5.    # signal-to-noise ratio
    sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2

    room = pra.AnechoicRoom(fs=SR, sigma2_awgn=sigma2)
    room.add_microphone_array(MIC_POSITIONS)

    for idx, sound_event in enumerate(csv_dict['class']):
        sound_filename = sound_event
        fs, sound = wav.read(RAW_DIRECTORY / sound_filename)
        delay = csv_dict['start'][idx]
        x, y, z = sph2cart(csv_dict['azi'][idx], csv_dict['ele'][idx], csv_dict['dist'][idx])
        room.add_source([x, y, z], signal=sound, delay=delay)

    room.simulate()
    return room.mic_array.signals

def normalize_and_save_all_wavs():
    print("\n--- Normalizing all signals and saving to .wav ---")
    temp_files = list(TEMP_DIRECTORY.glob('*.npy'))
    if not temp_files:
        print("No temporary .npy files found to process.")
        return

    # Find the maximum amplitude across ALL generated files (train and test)
    global_max_amplitude = 0
    for temp_file_path in temp_files:
        signal = np.load(temp_file_path)
        max_abs = np.max(np.abs(signal))
        if max_abs > global_max_amplitude:
            global_max_amplitude = max_abs

    if global_max_amplitude == 0:
        print("Warning: All signals are silent. Setting normalization factor to 1.0.")
        global_max_amplitude = 1.0
        
    print(f"Global max amplitude: {global_max_amplitude:.4f}")

    # Normalize and save each file as a WAV
    for temp_file_path in temp_files:
        print(f"Processing {temp_file_path.name}")
        signal = np.load(temp_file_path)
        signal_normalized = signal / global_max_amplitude
        
        wav_file_name = temp_file_path.stem + ".wav"
        wav_file_path = WAV_DIRECTORY / wav_file_name
        wav.write(wav_file_path, SR, signal_normalized.T)

        # Optional: Clean up the temporary file after conversion
        # os.remove(temp_file_path)

    print("--- WAV file conversion complete. ---")


if __name__ == '__main__':
    all_csv = os.listdir(DESC_DIRECTORY)
    for csv_name in all_csv:
        print(f"Processing {csv_name}")
        csv_dict = read_csv(csv_name)
        sound_array = generate_sound(csv_dict)
        temp_name = csv_name.split('.')[0] + ".npy"
        np.save(TEMP_DIRECTORY / temp_name, sound_array)

    normalize_and_save_all_wavs()