import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import scipy.io.wavfile as wav
from pathlib import Path

# --- Asset and Output Directories ---
# Using Path for better cross-platform compatibility
BASE_DIR = Path("./dataset_generator/sounds/criset_0.120m_3class_motor")
MOTOR_NOISE_DIRECTORY = Path("./dataset_generator/sounds/augmented_drone_motor_noises")
RAW_DIRECTORY = BASE_DIR / "raws"
WAV_DIRECTORY = BASE_DIR / "wav_ov1_split1_30db"
DESC_DIRECTORY = BASE_DIR / "desc_ov1_split1"
TEMP_DIRECTORY = BASE_DIR / "temp"


# --- Simulation Parameters ---
SECONDS_LENGTH = 30
SR = 44100
NORMAL_SNR_DB = 30
MOTOR_SNR_DB = -5

MIC_POSITIONS = np.array([
    [  0.0420,    0.0615,   -0.0410],
    [ -0.0420,    0.0615,    0.0410],
    [ -0.0615,    0.0420,   -0.0410],
    [ -0.0615,   -0.0420,    0.0410],
    [ -0.0420,   -0.0615,   -0.0410],
    [  0.0420,   -0.0615,    0.0410],
    [  0.0615,   -0.0420,   -0.0410],
    [  0.0615,    0.0420,    0.0410]
]).T


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def deg2rad(deg):
    return deg * (np.pi / 180)

def add_noise(signal, snr_db, noise_source=None):

    signal_power = np.mean(signal**2)
    if signal_power < 1e-12: # Avoid division by zero for silent signals
        return signal

    snr_linear = 10**(snr_db / 10.0)
    desired_noise_power = signal_power / snr_linear

    if noise_source is None:
        # Generate Gaussian white noise
        noise = np.random.normal(loc=0, scale=np.sqrt(desired_noise_power), size=signal.shape)
    else:
        # Use a provided noise source (e.g., motor noise)
        # Ensure noise is float64 for calculations
        noise = noise_source.astype(np.float64)

        # Match the length of the noise to the signal
        target_length = signal.shape[1]
        if noise.shape[-1] < target_length:
            noise = np.pad(noise, [(0, 0), (0, target_length - noise.shape[-1])], 'constant')
        elif noise.shape[-1] > target_length:
            noise = noise[:, :target_length]

        # Scale the provided noise to the desired power
        current_noise_power = np.mean(noise**2)
        if current_noise_power > 1e-12:
            scaling_factor = np.sqrt(desired_noise_power / current_noise_power)
            noise *= scaling_factor
        else:
            noise.fill(0) # The provided noise is silent

    return signal + noise


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
    room = pra.AnechoicRoom(fs=SR)
    room.add_microphone_array(MIC_POSITIONS)
    
    for idx, sound_event in enumerate(csv_dict['class']):
        sound_filename = sound_event
        fs, sound = wav.read(RAW_DIRECTORY / sound_filename)
        delay = csv_dict['start'][idx]
        x, y, z = sph2cart(deg2rad(csv_dict['azi'][idx]), deg2rad(csv_dict['ele'][idx]), csv_dict['dist'][idx])
        room.add_source([x, y, z], signal=sound, delay=delay)
    
    room.simulate()
    clean_signal = room.mic_array.signals

    return clean_signal

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
        signal_normalized = np.int16(signal_normalized * 32767)
        
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
        sound_array = add_noise(sound_array, NORMAL_SNR_DB)
        split, count = csv_name.split('_')[0:2]
        motor_noise_name = split + '_' + count + '.wav'
        sound_array = add_noise(sound_array, MOTOR_SNR_DB, noise_source=wav.read(MOTOR_NOISE_DIRECTORY / motor_noise_name)[1].T)

        temp_name = csv_name.split('.')[0] + ".npy"
        np.save(TEMP_DIRECTORY / temp_name, sound_array)

    normalize_and_save_all_wavs()