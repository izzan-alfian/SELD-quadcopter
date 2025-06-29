import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import librosa
import os
from pathlib import Path

# --- CONFIGURATION ---
NB_TRAIN = 240
NB_TEST = 60

# --- Asset and Output Directories ---
# Using Path for better cross-platform compatibility
BASE_DIR = Path("./dataset_generator/sounds/0.002m_8channel_1class_mic_signals")
WHISTLES_DIRECTORY = Path("./dataset_generator/sounds/whistles")
WAV_RESULT_DIRECTORY = BASE_DIR / "wav"
DESCRIPTION_RESULT_DIRECTORY = BASE_DIR / "desc"
TEMP_DIRECTORY = BASE_DIR / "temp"


# --- Simulation Parameters ---
SECONDS_LENGTH = 30
SR = 44100
MAX_POSSIBLE_WHISTLE_RADIUS = 100
MIN_POSSIBLE_WHISTLE_RADIUS = 50
SOUND_EVENT_GAP = 1.5
SOUND_EVENT_GAP_TOLERANT = 0.2

# Microphone array geometry (4 microphones)
MIC_POSITIONS = np.array([
    [ 0.001,  0.001, 0.000],
    [ 0.001, -0.001, 0.000],
    [-0.001, -0.001, 0.000],
    [-0.001, -0.001, 0.000],
]).T

def get_random_3d_pos(min_radius, max_radius):
    """Generates a random 3D position within a spherical shell."""
    u = np.random.rand()
    r = ((max_radius**3 - min_radius**3) * u + min_radius**3)**(1/3)
    theta = np.arccos(2 * np.random.rand() - 1)
    phi = 2 * np.pi * np.random.rand()
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def simulate_whistle_signals(room, whistle_file_names):
    """Adds randomly placed whistle sources to the pyroomacoustics room."""
    seconds_elapsed = 0
    whistle_metadata = []

    while True:
        whistle_file_name = np.random.choice(whistle_file_names)
        whistle_file_path = WHISTLES_DIRECTORY / whistle_file_name
        whistle_signal, _ = librosa.load(whistle_file_path, mono=False, sr=SR)

        signal_length_sec = np.ceil(whistle_signal.shape[-1] / SR)
        random_gap = max(0, np.random.normal(SOUND_EVENT_GAP, SOUND_EVENT_GAP_TOLERANT))

        if seconds_elapsed + random_gap + signal_length_sec >= SECONDS_LENGTH:
            break
        seconds_elapsed += random_gap

        x, y, z = get_random_3d_pos(MIN_POSSIBLE_WHISTLE_RADIUS, MAX_POSSIBLE_WHISTLE_RADIUS)
        room.add_source([x, y, z], signal=whistle_signal, delay=seconds_elapsed)
        
        whistle_metadata.append({
            'sound_event_recording': whistle_file_name,
            'start_time': seconds_elapsed,
            'end_time': seconds_elapsed + signal_length_sec,
            'ele': np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))),
            'azi': np.degrees(np.arctan2(y, x)),
            'distance': np.sqrt(x**2 + y**2 + z**2)
        })
        seconds_elapsed += signal_length_sec
    
    room.compute_rir()
    room.simulate()
    return room, whistle_metadata

def generate_dataset(num_files_to_generate, dataset_type, whistle_file_names):
    """
    Generates a dataset of simulated whistle sounds.

    Args:
        num_files_to_generate (int): The number of WAV/CSV pairs to create.
        dataset_type (str): The type of dataset ('train' or 'test'), used for naming files.
        whistle_file_names (list): A list of available whistle sound filenames.
    """
    print(f"--- Starting generation for {dataset_type} set ({num_files_to_generate} files) ---")
    
    # Part 1: Generate all raw signal files (.npy) and description files (.csv)
    for i in range(num_files_to_generate):
        result_base_name = f"{dataset_type}_{i:0d}"
        print(f"Generating {result_base_name}...")
        
        room = pra.AnechoicRoom(fs=SR, air_absorption=True)
        room.add_microphone_array(MIC_POSITIONS)
        room, whistle_metadata = simulate_whistle_signals(room, whistle_file_names)
        
        whistle_signals = room.mic_array.signals
        
        desired_length_samples = int(SECONDS_LENGTH * SR)
        current_length_samples = whistle_signals.shape[1]

        if current_length_samples < desired_length_samples:
            padding_needed = desired_length_samples - current_length_samples
            microphone_signals = np.pad(whistle_signals, ((0, 0), (0, padding_needed)), 'constant')
        else:
            microphone_signals = whistle_signals[:, :desired_length_samples]

        # Save metadata to CSV
        df = pd.DataFrame(whistle_metadata)
        description_file_path = DESCRIPTION_RESULT_DIRECTORY / f"{result_base_name}.csv"
        df.to_csv(description_file_path, index=False)

        # Save raw signal to a temporary .npy file
        temp_file_path = TEMP_DIRECTORY / f"{result_base_name}.npy"
        np.save(temp_file_path, microphone_signals)

    print(f"--- Raw file generation complete for {dataset_type} set. ---")


def normalize_and_save_all_wavs():
    """
    Normalizes all .npy files in the temp directory based on the global max amplitude,
    then saves them as .wav files.
    """
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
        signal = np.load(temp_file_path)
        signal_normalized = signal / global_max_amplitude
        
        wav_file_name = temp_file_path.stem + ".wav"
        wav_file_path = WAV_RESULT_DIRECTORY / wav_file_name
        sf.write(wav_file_path, signal_normalized.T, SR)

        # Optional: Clean up the temporary file after conversion
        # os.remove(temp_file_path)

    print("--- WAV file conversion complete. ---")


if __name__ == '__main__':
    # Ensure all necessary directories exist
    for directory in [WAV_RESULT_DIRECTORY, DESCRIPTION_RESULT_DIRECTORY, TEMP_DIRECTORY]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get the list of available whistle sound files
    whistle_files = os.listdir(WHISTLES_DIRECTORY)
    if not whistle_files:
        raise FileNotFoundError(f"No whistle sound files found in {WHISTLES_DIRECTORY}")

    # --- Generate Training and Test Sets ---
    generate_dataset(NB_TRAIN, 'train', whistle_files)
    generate_dataset(NB_TEST, 'test', whistle_files)
    
    # --- Normalize all generated files together and save as WAV ---
    normalize_and_save_all_wavs()