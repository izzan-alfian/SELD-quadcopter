from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import os
from multiprocessing import Pool
from pathlib import Path

# --- Directories ---
# Use Path for better cross-platform compatibility
RECORDING_SAMPLE_DIRECTORY = Path('./dataset_generator/sounds/cryset_motor/wav_ov1_split1_30db/')
BASE_DIRECTORY = Path('./dataset_generator/sounds/cryset_motor_filtered/')
RESULT_DIRECTORY = BASE_DIRECTORY / "wav_ov1_split1_30db"
TEMP_DIRECTORY = BASE_DIRECTORY / "temp"
NOISE_DIRECTORY = Path('./dataset_generator/sounds/DREGON_individual_motors_recordings/allMotors_70.wav')

# --- Parameters ---
SAMPLE_RATE = 44100
NOISE_TIME = 1

# --- Pre-load noise sample ---
# This is done once before multiprocessing starts
try:
    sr_noise, noise_data_full = wavfile.read(NOISE_DIRECTORY)
    # Ensure noise data is float for processing and sliced correctly
    noise_data = noise_data_full[:SAMPLE_RATE * NOISE_TIME].T.astype(np.float64) * 0.001
except FileNotFoundError:
    print(f"Warning: Noise file not found at {NOISE_DIRECTORY}. Noise reduction might be affected.")
    noise_data = None # Handle case where noise file is missing

def reduce_noise_and_save_temp(filename):
    """
    Reads a file, applies noise reduction, and saves the result
    as a temporary numpy file without normalization.
    """
    try:
        path = RECORDING_SAMPLE_DIRECTORY / filename
        sr, data = wavfile.read(path)
        print(f"Reducing noise for: {filename}")

        # Non-stationary filtering
        # Data is converted to float for processing by noisereduce
        reduced_float = nr.reduce_noise(
            y=data.T.astype(np.float64),
            sr=sr,
            y_noise=noise_data if noise_data is not None else None,
            prop_decrease=1.0,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            thresh_n_mult_nonstationary=2.0,
            sigmoid_slope_nonstationary=10.0,
            n_std_thresh_stationary=1.5,
            stationary=False
        ).T

        # Save the floating-point, non-normalized result to a temporary file
        temp_path = TEMP_DIRECTORY / (Path(filename).stem + ".npy")
        np.save(temp_path, reduced_float)
        return f"Successfully processed {filename}"
    except Exception as e:
        return f"ERROR processing {filename}: {e}"

def normalize_and_save_all_wavs():
    """
    Finds a global maximum from all temp files, then normalizes
    and saves them as 16-bit WAV files.
    """
    print("\n--- Normalizing all signals and saving to .wav ---")
    temp_files = list(TEMP_DIRECTORY.glob('*.npy'))
    if not temp_files:
        print("No temporary .npy files found to process.")
        return

    # Find the maximum absolute amplitude across ALL generated files
    global_max_amplitude = 0.0
    for temp_file_path in temp_files:
        signal = np.load(temp_file_path)
        max_abs = np.max(np.abs(signal))
        if max_abs > global_max_amplitude:
            global_max_amplitude = max_abs

    if global_max_amplitude < 1e-9: # Check for effective silence
        print("Warning: All signals are silent or near-silent. Setting normalization factor to 1.0.")
        global_max_amplitude = 1.0
        
    print(f"Global max amplitude found: {global_max_amplitude:.4f}")

    # Normalize each file using the global max and save as WAV
    for temp_file_path in temp_files:
        print(f"Normalizing and saving {temp_file_path.name}")
        signal = np.load(temp_file_path)
        
        # Normalize signal to [-1.0, 1.0]
        signal_normalized = signal / global_max_amplitude
        
        # Scale to 16-bit integer range [-32767, 32767]
        signal_int16 = np.int16(signal_normalized * 32767)
        
        # Define final output path and save the WAV file
        wav_file_name = temp_file_path.stem + ".wav"
        wav_file_path = RESULT_DIRECTORY / wav_file_name
        wavfile.write(wav_file_path, SAMPLE_RATE, signal_int16)

        # Optional: Clean up the temporary file after conversion
        # os.remove(temp_file_path)

    print("--- WAV file normalization and saving complete. ---")


if __name__ == "__main__":
    # Create necessary directories
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Get list of files to process
    try:
        files = [f for f in os.listdir(RECORDING_SAMPLE_DIRECTORY) if f.endswith('.wav')]
        if not files:
            print(f"No .wav files found in {RECORDING_SAMPLE_DIRECTORY}")
            exit()
    except FileNotFoundError:
        print(f"Error: Input directory not found at {RECORDING_SAMPLE_DIRECTORY}")
        exit()

    # --- Pass 1: Noise Reduction ---
    # Use a multiprocessing Pool to perform noise reduction in parallel
    num_processes = os.cpu_count()
    print(f"--- Starting noise reduction on {num_processes} cores... ---")
    with Pool(processes=num_processes) as pool:
        results = pool.map(reduce_noise_and_save_temp, files)
    
    print("\n--- Noise Reduction Summary ---")
    for res in results:
        print(res)

    # --- Pass 2: Global Normalization and Saving ---
    # This runs after all noise reduction is complete
    normalize_and_save_all_wavs()