#-------------------------SAR Highperformance-----------------------
#------------------------Spectrogram Generator----------------------

import pyaudio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import joblib
import sys
import time
from scipy import signal

NB_CHANNELS = 8

# FFT parameters
NFFT = 512
WINDOW_LEN = NFFT
HOP_LEN = NFFT // 2
NB_BINS = NFFT // 2

# Set the audio parameters
RATE = 44100
CHUNK = 4096
ANIMATION_INTERVAL_S = 1e-5 
MAX_FRAMES_SAVE = 500       # Max frames for saving animation (if enabled)

# --- New parameter to control the spectrogram's time window ---
# This determines how many seconds of audio are shown in the spectrogram at once.
SPECTROGRAM_WINDOW_LEN = 512
MAX_AUDIO_BUFFER_SAMPLES = int(np.ceil(SPECTROGRAM_WINDOW_LEN * HOP_LEN + HOP_LEN + 1))
PLOT_LEN = int((MAX_AUDIO_BUFFER_SAMPLES - HOP_LEN)//HOP_LEN)

# Initialize the audio buffer
audio_buffer = np.empty((0, 8), dtype=np.float32)
spectogram_buffer = np.empty((0, 8), dtype=np.float32)


# Spectra scaler directory
SPEC_SCALER_DIR = './dataset_generator/sounds/filtered_8_channel_microphone_signals/spec_ov1_split1_50db_nfft512_wts'


# Initialize PyAudio and open the audio stream
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

# Create the figure and axes for the plot
fig, ax = plt.subplots(1, 2)

def spectogram_plot_routine(ax, name):
    ax.clear()
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'{name} spectrogram')


def spectrogram(audio_input):
    if audio_input.ndim == 1:
        audio_input = audio_input[:, np.newaxis]
    nb_ch = audio_input.shape[1]

    hann_win = np.repeat(np.hanning(WINDOW_LEN)[np.newaxis].T, nb_ch, 1)
    max_frames = int(np.ceil((len(audio_input) - WINDOW_LEN) / HOP_LEN))

    spectra = np.zeros((max_frames, NB_BINS, nb_ch), dtype=complex)
    for ind in range(max_frames):
        start_ind = ind * HOP_LEN
        aud_frame = audio_input[start_ind + np.arange(0, WINDOW_LEN), :] * hann_win
        spectra[ind] = np.fft.fft(aud_frame, n=NFFT, axis=0, norm='ortho')[:NB_BINS, :]
    
    return spectra

def normalized_spectrogram(spectra):
    spectra = spectrogram(spectra)
    spec_len = spectra.shape[0]

    if spectra.ndim >= 3:
        spectra = spectra.reshape(spec_len, -1)

    spec_scaler = joblib.load(SPEC_SCALER_DIR)
    spectra = spec_scaler.transform(np.concatenate((np.abs(spectra), np.angle(spectra)), axis=1))

    return spectra

def load_audio_buffer():
    global audio_buffer

    # Read new data from the audio stream
    new_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)

    # Stack the single-channel audio 8 times to simulate an 8-channel microphone array
    # This is a placeholder for actual multi-channel input
    new_data = np.tile(new_data[:, np.newaxis], (1, 8))

    # Append new data to the buffer
    audio_buffer = np.concatenate((audio_buffer, new_data))

    # Trim the buffer to maintain the desired time window
    if len(audio_buffer) > MAX_AUDIO_BUFFER_SAMPLES:
        audio_buffer = audio_buffer[len(audio_buffer) - MAX_AUDIO_BUFFER_SAMPLES:]

    return audio_buffer

def spectogram_batch_generator():
    while True:
        global spectogram_buffer

        audio_buffer = load_audio_buffer()
        spectogram = normalized_spectrogram(audio_buffer)
        feat_len = spectogram.shape[1] // (2 * NB_CHANNELS)
        spectogram = np.reshape(spectogram, (-1, feat_len, 2 * NB_CHANNELS))
        spectogram = np.transpose(spectogram, (2, 0, 1))
        spectogram = spectogram[np.newaxis, ...]

        # print('Audio buffer: ', audio_buffer.shape, 'spectogram: ', spectogram.shape)
        spectogram_buffer = spectogram

def update_spectrogram(frame):
    audio_buffer = load_audio_buffer()

    frequencies, times, Sxx = signal.spectrogram(audio_buffer[:, 0], RATE, nperseg=512, noverlap=512//2, nfft=512)
    audio_spectogram = normalized_spectrogram(audio_buffer)

    Sxx_scaled_db = (10 * np.log10(Sxx + 1e-9))[:, -PLOT_LEN:]
    times_scaled = times[-PLOT_LEN:]
    audio_spectogram_scaled_db = (10 * np.log10(np.abs(audio_spectogram) + 1e-9))[-PLOT_LEN:, :]

    spectogram_plot_routine(ax[0], "scipy")
    spectogram_plot_routine(ax[1], "numpy")

    # Add a small epsilon to avoid log10(0)
    ax[0].pcolormesh(times_scaled, frequencies, Sxx_scaled_db, shading='auto', cmap='inferno')
    ax[1].pcolormesh(audio_spectogram_scaled_db.T, shading='auto', cmap='inferno')

    return ax[0], ax[1]


# def main(argv):
#     # Create the animation
#     # model = tf.keras.models.load_model('./models/drone_ov1_split1_regr0_3d0_2_model.keras')

#     ani = animation.FuncAnimation(fig, update_spectrogram, frames=None, interval=ANIMATION_INTERVAL_S * 1000, blit=True, save_count=MAX_FRAMES_SAVE)

#     # Show the plot
#     plt.show()

#     # Close the audio stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()


def main(arg):
    model = tf.keras.models.load_model('./models/drone_ov1_split1_regr0_3d0_2_model.keras')

    threading.Thread(target=spectogram_batch_generator, daemon=True).start()

    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                   python_tracer_level = 1,
                                                   device_tracer_level = 1)

    try:
        while True:
            # print(spectogram_buffer.shape)
            pred_start_time = None
            pred_end_time = None

            if spectogram_buffer.shape == (1, 16, 512, 256):
                pred_start_time  = time.time()
                tf.profiler.experimental.start('logdir_path', options = options)
                pred = model.predict_on_batch(spectogram_buffer)
                tf.profiler.experimental.stop()
                pred_end_time = time.time()
                print(pred)

            if (pred_start_time is not None) and (pred_end_time is not None):
                print('Time elapsed: ', pred_end_time - pred_start_time)

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nLoop stopped by user.")

    # time.sleep(10)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)