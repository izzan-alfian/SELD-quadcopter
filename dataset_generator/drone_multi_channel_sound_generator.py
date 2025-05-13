import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pandas as pd
import librosa
import os
from collections import namedtuple
from scipy.io import wavfile

whistles_directory           = "./dataset_generator/sounds/whistles"
motor_noise_directory        = "./dataset_generator/sounds/augmented_drone_motor_noises"
wav_result_directory         = "./dataset_generator/sounds/8_channel_microphone_signals/wav"
description_result_directory = "./dataset_generator/sounds/8_channel_microphone_signals/description"
temp_directory               = "./dataset_generator/sounds/8_channel_microphone_signals/temp"
seconds_length = 30
sr = 44100
snr_db = -5

max_possible_whistle_radius = 100
min_possible_whistle_radius = 50

sound_event_gap = 1.5
sound_event_gap_tolerant = 0.2

mic_positions = np.array([
    [ 0.0420,  0.0615, -0.0410],
    [-0.0420,  0.0615,  0.0410],
    [-0.0615,  0.0420, -0.0410],
    [-0.0615, -0.0420,  0.0410],
    [-0.0420, -0.0615, -0.0410],
    [ 0.0420, -0.0615,  0.0410],
    [ 0.0615, -0.0420, -0.0410],
    [ 0.0615,  0.0420,  0.0410],
    ]).T





def normalize_sound(sound):
    sound_normalized = np.zeros(sound.shape)
    for i, y in enumerate(sound):
        sound_normalized[i] = y / np.max(np.abs(sound))
    return sound_normalized

def get_random_3d_pos(
        max_radius=max_possible_whistle_radius,
        min_radius=min_possible_whistle_radius
    ):
    u = np.random.rand()
    r = ((max_radius**3 - min_radius**3)*u + min_radius**3)**(1/3)

    theta = np.arccos(2 * np.random.rand() - 1)  # Polar angle (0 to pi)
    phi = 2 * np.pi * np.random.rand()  # Azimuthal angle (0 to 2pi)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def simulate_whistle_signals(room):
    seconds_elapsed = 0

    whistle_metadata = []

    while True:
        whistle_file_name = np.random.choice(whistle_file_names)
        whistle_file_path = os.path.join(whistles_directory, whistle_file_name)
        whistle_signal, _ = librosa.load(whistle_file_path, mono=False, sr=sr)

        signal_length = np.ceil(whistle_signal.shape[-1] / sr)
        random_gap = max(0, np.random.normal(sound_event_gap, sound_event_gap_tolerant))
        if seconds_elapsed + random_gap + signal_length >= seconds_length:
            break
        seconds_elapsed += random_gap

        x, y, z = get_random_3d_pos()
        room.add_source([x, y, z], signal=whistle_signal, delay=seconds_elapsed)
        whistle_metadata.append(
            {
                'sound_event_recording': whistle_file_name,
                'start_time'           : seconds_elapsed,
                'end_time'             : seconds_elapsed + signal_length,
                'ele'                  : np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))),
                'azi'                  : np.degrees(np.arctan2(y, x)),
                'distance'             : np.sqrt(x**2 + y**2 + z**2)
            }
        )

        seconds_elapsed += signal_length
    
    room.compute_rir()
    room.simulate()
    return room, whistle_metadata







whistle_file_names = os.listdir(whistles_directory)
whistles_file_paths = [os.path.join(whistles_directory, f) for f in whistle_file_names]

motor_noise_file_names = os.listdir(motor_noise_directory)
motor_noise_file_paths = [os.path.join(motor_noise_directory, f) for f in motor_noise_file_names]

for motor_noise_file_name in motor_noise_file_names:
    print("Generating %s" %motor_noise_file_name)
    motor_noise_path = os.path.join(motor_noise_directory, motor_noise_file_name)
    room = pra.AnechoicRoom(fs=sr, air_absorption=True)
    room.add_microphone_array(mic_positions)
    room, whistle_metadata = simulate_whistle_signals(room)
    
    whistle_signals = room.mic_array.signals
    motor_noise, sr = librosa.load(motor_noise_path, mono=False, sr=sr)

    length_difference = np.abs(motor_noise.shape[1] - whistle_signals.shape[1])
    whistle_signals = np.pad(
        whistle_signals,
        ((0,0), (0, (length_difference))),
        mode='constant',
        constant_values=0
    )

    df = pd.DataFrame({
        'sound_event_recording': [d['sound_event_recording'] for d in whistle_metadata],
        'start_time'           : [d['start_time']            for d in whistle_metadata],
        'end_time'             : [d['end_time']              for d in whistle_metadata],
        'ele'                  : [d['ele']                   for d in whistle_metadata],
        'azi'                  : [d['azi']                   for d in whistle_metadata],
        ''                     : [d['distance']              for d in whistle_metadata]
    })

    desired_noise_power = (np.linalg.norm(whistle_signals)**2 / 10**(snr_db/10))
    alpha = np.sqrt(desired_noise_power / np.linalg.norm(motor_noise)**2)
    motor_noise *= alpha

    microphone_signals = whistle_signals + motor_noise

    temp_file_name = os.path.splitext(motor_noise_file_name)[0]
    temp_file_path = os.path.join(temp_directory, temp_file_name)
    np.save(temp_file_path, microphone_signals)

    description_file_name = os.path.splitext(motor_noise_file_name)[0] + ".csv"
    description_file_path = os.path.join(description_result_directory, description_file_name)
    df.to_csv(description_file_path, index=False)

temp_file_names = os.listdir(temp_directory)
temp_file_paths = [os.path.join(temp_directory, f) for f in temp_file_names]

max_amplitude = 0
for temp_file_path in temp_file_paths:
    temp_file = np.load(temp_file_path)
    if max_amplitude < np.max(np.abs(temp_file)):
        max_amplitude = np.max(np.abs(temp_file))

for temp_file_name in temp_file_names:
    temp_file_path = os.path.join(temp_directory, temp_file_name)
    temp_file = np.load(temp_file_path)
    temp_normalized = temp_file / max_amplitude

    wav_file_name = os.path.splitext(temp_file_name)[0] + ".wav"
    wav_file_path = os.path.join(wav_result_directory, wav_file_name)
    sf.write(wav_file_path, temp_normalized.T, sr)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# room.plot(ax=ax)
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.set_zlim([-100, 100])
# plt.show()

# sf.write(('./test1.wav'), end_signals.T, sr, subtype='DOUBLE')
# sf.write(('./test2.wav'), whistle_signals.T, sr, subtype='DOUBLE')