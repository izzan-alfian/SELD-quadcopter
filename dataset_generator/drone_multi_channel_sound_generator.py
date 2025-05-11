import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import librosa
import os
from scipy.io import wavfile

whistles_directory = "./dataset_generator/sounds/whistles"
motor_noise_directory = "./dataset_generator/sounds/augmented_drone_motor_noises"
seconds_length = 30
sr = 44100
snr = 5

seconds_elapsed = 0
sound_event_gap = 1.5
sound_event_gap_tolerant = 0.2

def get_random_3d_pos(radius_outside=100, radius_inside=50):
    while True:
        r = radius_outside * np.cbrt(np.random.rand())  # Uniform in volume
        if r >= radius_inside:
            break

    theta = np.arccos(2 * np.random.rand() - 1)  # Polar angle (0 to pi)
    phi = 2 * np.pi * np.random.rand()  # Azimuthal angle (0 to 2pi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])

whistle_file_names = os.listdir(whistles_directory)
whistles_paths = [os.path.join(whistles_directory, f) for f in whistle_file_names]

motor_noise_file_names = os.listdir(motor_noise_directory)
motor_noise_paths = [os.path.join(motor_noise_directory, f) for f in motor_noise_file_names]

room = pra.AnechoicRoom(fs=sr, air_absorption=True)

mic_positions = np.array([
    [ 0.0420,  0.0615, -0.0410],
    [-0.0420,  0.0615,  0.0410],
    [-0.0615,  0.0420, -0.0410],
    [-0.0615, -0.0420,  0.0410],
    [-0.0420, -0.0615, -0.0410],
    [ 0.0420, -0.0615,  0.0410],
    [ 0.0615, -0.0420, -0.0410],
    [ 0.0615,  0.0420,  0.0410],
    ]).T  # Position of microphone
room.add_microphone_array(mic_positions)

while True:
    whistle_path = np.random.choice(whistles_paths)
    y, sr = librosa.load(whistle_path, mono=False, sr=sr)
    sound_length = y.shape[-1] / sr
    random_gap = max(0, np.random.normal(sound_event_gap, sound_event_gap_tolerant))
    if seconds_elapsed + random_gap + sound_length >= seconds_length:
        break
    seconds_elapsed += random_gap
    source_position = get_random_3d_pos()
    room.add_source(source_position, signal=y, delay=seconds_elapsed)
    seconds_elapsed += sound_length

room.compute_rir()
room.simulate()

for i in motor_noise_paths:
    y, sr = librosa.load(i, mono=False, sr=sr)

    length_difference = np.abs(y.shape[1] - room.mic_array.signals.shape[1])
    microphone_signals = np.pad(
        room.mic_array.signals,
        ((0,0), (0, (length_difference))),
        mode='constant',
        constant_values=0
    )
    end_signals = microphone_signals + y
    break


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# room.plot(ax=ax)
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.set_zlim([-100, 100])
# plt.show()

# sf.write(('./test.wav'), sound_signals.T, sr, subtype='DOUBLE')

# sound_signals = room.mic_array.signals
# sound_signals_norm = np.zeros(sound_signals.shape)
# for i, sound in enumerate(sound_signals):
#     sound_signals_norm[i] = sound / np.max(np.abs(sound))
# sd.play(sound_signals_norm.T, samplerate=sr) # here
# sd.wait()


# sd.play(end_signals.T, samplerate=sr) # here
# sd.wait()

sf.write(('./test1.wav'), end_signals.T, sr, subtype='DOUBLE')
sf.write(('./test2.wav'), microphone_signals.T, sr, subtype='DOUBLE')