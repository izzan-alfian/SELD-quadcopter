#-------------------------SAR Highperformance-----------------------
#------------------------Spectrogram Generator----------------------

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import joblib
import sys
import time
from scipy import signal
# This import is necessary for the '3d' projection to work
from mpl_toolkits.mplot3d import Axes3D

import esp32_audio_capture

# Audio parameters
NB_CHANNELS = 8
NB_SPECTROGRAM_CHANNELS = NB_CHANNELS * 2

# Model parameters
CLASS_NAMES = ['Whistle']
NB_CLASSES = len(CLASS_NAMES)
DOA_DIMS = 3 # X, Y, Z

# FFT parameters
NFFT = 512
WINDOW_LEN = NFFT
HOP_LEN = NFFT // 2
NB_BINS = NFFT // 2

# Set the audio parameters
RATE = 44100
CHUNK = 4096
ANIMATION_INTERVAL_MS = 50

# --- New parameter to control the spectrogram's time window ---
# This determines how many seconds of audio are shown in the spectrogram at once.
SPECTROGRAM_WINDOW_LEN = 512
MAX_AUDIO_BUFFER_SAMPLES = int(np.ceil(SPECTROGRAM_WINDOW_LEN * HOP_LEN + WINDOW_LEN))

print(f"Max audio buffer samples: {MAX_AUDIO_BUFFER_SAMPLES}")

# Initialize the audio buffer
esp32_buffer = tuple(np.empty((512), dtype=np.float32) for _ in range(4))
audio_buffer = np.empty((0, NB_CHANNELS), dtype=np.float32)
spectogram_buffer = None

# --- For prediction and animation ---
# These will be shared between the prediction thread and the animation callback
latest_sed = np.zeros((SPECTROGRAM_WINDOW_LEN, NB_CLASSES))
latest_doa = np.zeros((SPECTROGRAM_WINDOW_LEN, NB_CLASSES * DOA_DIMS))
prediction_lock = threading.Lock()
model = None

# Spectra scaler directory
SPEC_SCALER_DIR = './dataset_generator/sounds/filtered_8_channel_microphone_signals/spec_ov1_split1_50db_nfft512_wts'

assert len(CLASS_NAMES) == NB_CLASSES, "The number of class names must match NB_CLASSES"

audio_stream = esp32_audio_capture.ESP32AudioCapture()

def fourier_transform(audio_input):
    if audio_input.ndim == 1:
        audio_input = audio_input[:, np.newaxis]
    nb_ch = audio_input.shape[1]

    if len(audio_input) < WINDOW_LEN:
        return np.zeros((0, NB_BINS, nb_ch), dtype=complex)

    hann_win = np.repeat(np.hanning(WINDOW_LEN)[np.newaxis].T, nb_ch, 1)
    max_frames = (len(audio_input) - WINDOW_LEN) // HOP_LEN

    fourier = np.zeros((max_frames, NB_BINS, nb_ch), dtype=complex)
    for ind in range(max_frames):
        start_ind = ind * HOP_LEN
        aud_frame = audio_input[start_ind : start_ind + WINDOW_LEN, :] * hann_win
        fourier[ind] = np.fft.fft(aud_frame, n=NFFT, axis=0, norm='ortho')[:NB_BINS, :]

    return fourier

def normalized_spectrogram(fourier_data):
    fourier_len = fourier_data.shape[0]
    if fourier_len == 0:
        return np.array([])

    if fourier_data.ndim >= 3:
        fourier_data = fourier_data.reshape(fourier_len, -1)

    spec_scaler = joblib.load(SPEC_SCALER_DIR)
    spectra = spec_scaler.transform(np.concatenate((np.abs(fourier_data), np.angle(fourier_data)), axis=1))

    return spectra

def load_esp32_buffer():
    global esp32_buffer
    
    while True:
        a0_interleaved, a1_interleaved, b0_interleaved, b1_interleaved = audio_stream.get_audio_stream()
        esp32_buffer = tuple((a0_interleaved, a1_interleaved, b0_interleaved, b1_interleaved))

def get_audio_buffer():
    global audio_buffer
    global esp32_buffer

    # Get the four stereo streams. Each is a 1D numpy array with interleaved [L, R, L, R, ...] samples.
    a0_interleaved, a1_interleaved, b0_interleaved, b1_interleaved = esp32_buffer

    # De-interleave each stereo stream into two separate channels (Left and Right).
    # A reshape to (-1, 2)s the samples into pairs of [L, R].
    # Slicing [:, 0] gets all Left channels, and [:, 1] gets all Right channels.
    a0_L, a0_R = a0_interleaved.reshape(-1, 2)[:, 0], a0_interleaved.reshape(-1, 2)[:, 1]
    a1_L, a1_R = a1_interleaved.reshape(-1, 2)[:, 0], a1_interleaved.reshape(-1, 2)[:, 1]
    b0_L, b0_R = b0_interleaved.reshape(-1, 2)[:, 0], b0_interleaved.reshape(-1, 2)[:, 1]
    b1_L, b1_R = b1_interleaved.reshape(-1, 2)[:, 0], b1_interleaved.reshape(-1, 2)[:, 1]

    # Combine all 8 channels into a single multi-channel frame.
    # np.column_stack is perfect for creating a 2D array from a series of 1D arrays (channels).
    new_data = np.column_stack((a0_L, a0_R, a1_L, a1_R, b0_L, b0_R, b1_L, b1_R))

    # Append new data to the buffer
    audio_buffer = np.concatenate((audio_buffer, new_data))

    # Trim the buffer to maintain the desired time window
    if len(audio_buffer) > MAX_AUDIO_BUFFER_SAMPLES:
        audio_buffer = audio_buffer[len(audio_buffer) - MAX_AUDIO_BUFFER_SAMPLES:]

    return audio_buffer

def spectogram_batch_generator():
    print("Spectrogram generator thread started.")
    while True:
        global spectogram_buffer

        audio_buffer_local = get_audio_buffer()
        
        fourier_data = fourier_transform(audio_buffer_local)
        spectogram = normalized_spectrogram(fourier_data)

        if spectogram.size == 0:
            time.sleep(0.05)
            continue

        # The model expects a fixed sequence length. We need to pad/truncate.
        current_seq_len = spectogram.shape[0]
        target_seq_len = SPECTROGRAM_WINDOW_LEN

        if current_seq_len < target_seq_len:
            pad_width = target_seq_len - current_seq_len
            spectogram = np.pad(spectogram, ((0, pad_width), (0, 0)), 'constant')
        elif current_seq_len > target_seq_len:
            spectogram = spectogram[-target_seq_len:, :]

        feat_len = spectogram.shape[1] // (2 * NB_CHANNELS)
        spectogram = np.reshape(spectogram, (target_seq_len, feat_len, 2 * NB_CHANNELS))
        spectogram = np.transpose(spectogram, (2, 0, 1))
        spectogram = spectogram[np.newaxis, ...]

        spectogram_buffer = spectogram

def prediction_thread_func():
    global latest_sed, latest_doa, spectogram_buffer, model
    print("Prediction thread started.")
    while True:
        if spectogram_buffer is not None and spectogram_buffer.shape == (1, NB_SPECTROGRAM_CHANNELS, SPECTROGRAM_WINDOW_LEN, NB_BINS):
            pred = model.predict_on_batch(spectogram_buffer)
            with prediction_lock:
                # pred[0] is sed (1, 512, 1), pred[1] is doa (1, 512, 3)
                # the second [0] in both pred refers to the batch
                # Ensure the output shapes match the expected (batch_size, sequence_length, num_classes)
                # and (batch_size, sequence_length, num_classes * doa_dims)
                # The model's output is already in this format, so direct assignment is fine.
                
                latest_sed[:] = pred[0][0]
                latest_doa[:] = pred[1][0]
        else:
            time.sleep(0.05) # Wait for buffer to fill

def update_plot(frame, sed_lines, doa_points, doa_lines, title_3d, title_2d):
    with prediction_lock:
        # latest_sed is (SPECTROGRAM_WINDOW_LEN, NB_CLASSES)
        # latest_doa is (SPECTROGRAM_WINDOW_LEN, NB_CLASSES * DOA_DIMS)
        
        # Get the latest prediction for the most recent time frame for DOA and title info
        sed_vals_last_frame = latest_sed[-1, :] # Shape: (NB_CLASSES,)
        doa_vals_last_frame = latest_doa[-1, :].reshape(NB_CLASSES, DOA_DIMS) # Shape: (NB_CLASSES, 3)

    # Update SED lines (showing history for all 512 frames)
    for i, line in enumerate(sed_lines):
        line.set_data(np.arange(SPECTROGRAM_WINDOW_LEN), latest_sed[:, i])

    # Update DOA points and lines based on the *last* frame's activity
    active_classes_info = []
    for i in range(NB_CLASSES):
        is_active = sed_vals_last_frame[i] > 0.5
        x, y, z = doa_vals_last_frame[i]

        if is_active:
            doa_points[i]._offsets3d = ([x], [y], [z])
            doa_lines[i].set_data_3d([0, x], [0, y], [0, z])
            active_classes_info.append(f"{CLASS_NAMES[i]}: {sed_vals_last_frame[i]:.2f}")
        else:
            # Hide this specific point and line
            doa_points[i]._offsets3d = ([], [], [])
            doa_lines[i].set_data_3d([], [], [])

    # Update titles
    title_3d.set_text('Direction of Arrival')
    if active_classes_info:
        title_2d.set_text('Sound Event Detection\n' + '\n'.join(active_classes_info))
    else:
        title_2d.set_text('Sound Event Detection')

    return (*sed_lines, *doa_points, *doa_lines, title_3d, title_2d) # Return all updated artists


def main(argv):
    global model
    model = tf.keras.models.load_model('./models/drone_ov1_split1_regr0_3d0_3_model.keras')
    print("Model loaded.")

    threading.Thread(target=spectogram_batch_generator, daemon=True).start()
    threading.Thread(target=prediction_thread_func, daemon=True).start()
    threading.Thread(target=load_esp32_buffer, daemon=True).start()

    fig = plt.figure(figsize=(12, 6))
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_2d = fig.add_subplot(1, 2, 2)

    # --- Setup 3D plot for DOA ---
    ax_3d.set_xlim([-5, 5])
    ax_3d.set_ylim([-5, 5])
    ax_3d.set_zlim([-5, 5])
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.scatter( 0.0420,  0.0615, -0.0410, c='grey', marker='o', s=50, label='Microphone 1')
    ax_3d.scatter(-0.0420,  0.0615,  0.0410, c='grey', marker='o', s=50, label='Microphone 2')
    ax_3d.scatter(-0.0615,  0.0420, -0.0410, c='grey', marker='o', s=50, label='Microphone 3')
    ax_3d.scatter(-0.0615, -0.0420,  0.0410, c='grey', marker='o', s=50, label='Microphone 4')
    ax_3d.scatter(-0.0420, -0.0615, -0.0410, c='grey', marker='o', s=50, label='Microphone 5')
    ax_3d.scatter( 0.0420, -0.0615,  0.0410, c='grey', marker='o', s=50, label='Microphone 6')
    ax_3d.scatter( 0.0615, -0.0420, -0.0410, c='grey', marker='o', s=50, label='Microphone 7')
    ax_3d.scatter( 0.0615,  0.0420,  0.0410, c='grey', marker='o', s=50, label='Microphone 8')
    colors = plt.cm.get_cmap('hsv', NB_CLASSES + 1)
    doa_points = []
    doa_lines = []
    for i in range(NB_CLASSES):
        point = ax_3d.scatter([], [], [], c=[colors(i)], marker='x', s=100, label=f'Source: {CLASS_NAMES[i]}')
        line, = ax_3d.plot([], [], [], color=colors(i), linestyle='--')
        doa_points.append(point)
        doa_lines.append(line)
    ax_3d.legend()
    title_3d = ax_3d.set_title('Direction of Arrival')

    # --- Setup 2D plot for SED ---
    sed_lines = []
    for i in range(NB_CLASSES):
        line, = ax_2d.plot([], [], label=CLASS_NAMES[i], color=colors(i))
        sed_lines.append(line)
    ax_2d.legend()
    ax_2d.set_xlabel('Time Frame')
    ax_2d.set_ylabel('SED Probability')
    ax_2d.set_xlim([0, SPECTROGRAM_WINDOW_LEN - 1]) # X-axis for 512 frames (0 to 511)
    ax_2d.set_ylim([0, 1]) # SED probabilities are between 0 and 1
    title_2d = ax_2d.set_title('Sound Event Detection')
    
    plt.tight_layout(pad=3.0)

    # --- Create and run the animation ---
    ani = animation.FuncAnimation(
        fig,
        update_plot,
        fargs=(sed_lines, doa_points, doa_lines, title_3d, title_2d), # Pass sed_lines instead of sed_bars
        interval=ANIMATION_INTERVAL_MS,
        blit=True
    )
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    finally:
        # Cleanup
        print("Audio stream closed.")

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)