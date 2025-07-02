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
sirens_directory             = "./dataset_generator/sounds/sirens"
screams_directory            = "./dataset_generator/sounds/screams"

wav_result_directory         = "./dataset_generator/sounds/0.120m_8channel_3class_mic_signals/wav"
description_result_directory = "./dataset_generator/sounds/0.120m_8channel_3class_mic_signals/desc"
temp_directory               = "./dataset_generator/sounds/0.120m_8channel_3class_mic_signals/temp"

test_count = 60
train_count = 240

seconds_length = 30
sr = 44100
snr_db = -5

max_possible_whistle_radius = 10
min_possible_whistle_radius = 1

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

def simulate_random_signals(room):
    seconds_elapsed = 0
    sound_metadata = []

    while True:
        class_picker = np.random.randint(0, 3)
        if class_picker == 0:
            file_name = np.random.choice(whistle_file_names)
            file_path = os.path.join(whistles_directory, file_name)
            output_name = "whistle_sound"
        elif class_picker == 1:
            file_name = np.random.choice(siren_file_names)
            file_path = os.path.join(sirens_directory, file_name)
            output_name = "siren_sound"
        elif class_picker == 2:
            file_name = np.random.choice(scream_file_names)
            file_path = os.path.join(screams_directory, file_name)
            output_name = "scream_sound"
            
        signal, _ = librosa.load(file_path, mono=True, sr=sr)

        signal_length = np.ceil(signal.shape[-1] / sr)
        random_gap = max(0, np.random.normal(sound_event_gap, sound_event_gap_tolerant))
        if seconds_elapsed + random_gap + signal_length >= seconds_length:
            break
        seconds_elapsed += random_gap

        x, y, z = get_random_3d_pos()
        room.add_source([x, y, z], signal=signal, delay=seconds_elapsed)
        sound_metadata.append(
            {
                'sound_event_recording': output_name,
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
    return room, sound_metadata





whistle_file_names = os.listdir(whistles_directory)
siren_file_names = os.listdir(sirens_directory)
scream_file_names = os.listdir(screams_directory)

for i in range(0, test_count):
    output_name = "test_" + str(i)
    print("Generating %s" %output_name)

    room = pra.AnechoicRoom(fs=sr, air_absorption=True)
    room.add_microphone_array(mic_positions)
    room, whistle_metadata = simulate_random_signals(room)
    
    whistle_signals = room.mic_array.signals

    if whistle_signals.shape[1] < seconds_length * sr:
        whistle_signals = np.pad(
            whistle_signals,
            ((0,0), (0, seconds_length * sr - whistle_signals.shape[1])),
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

    microphone_signals = whistle_signals

    temp_file_path = os.path.join(temp_directory, output_name + ".npy")
    np.save(temp_file_path, microphone_signals)

    description_file_path = os.path.join(description_result_directory, output_name + ".csv")
    df.to_csv(description_file_path, index=False)


for i in range(0, train_count):
    output_name = "train_" + str(i)
    print("Generating %s" %output_name)

    room = pra.AnechoicRoom(fs=sr, air_absorption=True)
    room.add_microphone_array(mic_positions)
    room, whistle_metadata = simulate_random_signals(room)
    
    whistle_signals = room.mic_array.signals

    if whistle_signals.shape[1] < seconds_length * sr:
        whistle_signals = np.pad(
            whistle_signals,
            ((0,0), (0, seconds_length * sr - whistle_signals.shape[1])),
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

    microphone_signals = whistle_signals

    temp_file_path = os.path.join(temp_directory, output_name + ".npy")
    np.save(temp_file_path, microphone_signals)

    description_file_path = os.path.join(description_result_directory, output_name + ".csv")
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