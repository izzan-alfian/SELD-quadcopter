import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
import os
import scipy.io.wavfile as wav
from pathlib import Path
import sys

# --- Asset and Output Directories ---
# Using Path for better cross-platform compatibility
BASE_DIR = Path("./dataset_generator/sounds/criset_4class")
RAW_DIRECTORY = BASE_DIR / "raws"
WAV_DIRECTORY = BASE_DIR / "wav_ov1_split1_30db"
DESC_DIRECTORY = BASE_DIR / "desc_ov1_split1"
TEMP_DIRECTORY = BASE_DIR / "temp"


# --- Simulation Parameters ---
# --- CONFIGURATION ---
NB_TRAIN = 240
NB_TEST = 60

# --- Simulation Parameters ---
SECONDS_LENGTH = 30
SR = 44100

TRAIN_TEST_SPLIT = 0.8
MAX_POSSIBLE_WHISTLE_RADIUS = 100
MIN_POSSIBLE_WHISTLE_RADIUS = 50
SOUND_EVENT_GAP = 1.0
SOUND_EVENT_GAP_TOLERANT = 0.2

AZI_LIST = np.arange(-180, 180, 10)
ELE_LIST = np.arange(-90, 90, 10)
RADIUS_LIST = np.arange(1, 10, 0.5)

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

SOUND_EVENTS = { # sound class name; sound event file name; audio length
    'siren': list(), # the list contains the sound class's sound event file name and it's audio length
    'whistle': list(),
    'gunshot': list(),
    'glassbreak': list()
}

def get_sound_event_names():
    sound_events = os.listdir(RAW_DIRECTORY)
    for sound_event_file_name in sound_events:
        sound_event_name = sound_event_file_name.split('.')[0][:-3]
        sound_event_time = wav.read(os.path.join(RAW_DIRECTORY, sound_event_file_name))[1].shape[0] / SR
        if sound_event_name in SOUND_EVENTS.keys():
            SOUND_EVENTS[sound_event_name].append((sound_event_file_name, sound_event_time))


def create_csv_batch(sound_events_dict, nb_csv, csv_prefix):
    for i in range(nb_csv):
        final_csv = pd.DataFrame(columns=['sound_event_recording', 'start_time', 'end_time', 'ele', 'azi', 'dist'])
        elapsed_time = 0
        while True:
            elapsed_time += max(0, np.random.normal(SOUND_EVENT_GAP, SOUND_EVENT_GAP_TOLERANT))
            picked_sound_event_class = np.random.choice(list(sound_events_dict.keys()))
            sound_event_list = sound_events_dict[picked_sound_event_class]
            random_index = np.random.randint(len(sound_event_list))
            picked_sound_event_file_name, picked_sound_event_duration = sound_event_list[random_index]

            if picked_sound_event_duration > SECONDS_LENGTH:
                print(picked_sound_event_file_name, picked_sound_event_duration)
                sys.exit()

            if elapsed_time + picked_sound_event_duration < SECONDS_LENGTH:
                new_row = {
                    'sound_event_recording': picked_sound_event_file_name,
                    'start_time': elapsed_time,
                    'end_time': elapsed_time + picked_sound_event_duration,
                    'ele': np.random.choice(ELE_LIST),
                    'azi': np.random.choice(AZI_LIST),
                    'dist': np.random.choice(RADIUS_LIST)
                }
                final_csv = pd.concat([final_csv, pd.DataFrame([new_row])], ignore_index=True)
                elapsed_time += picked_sound_event_duration
            else:
                break

        DESC_DIRECTORY.mkdir(parents=True, exist_ok=True)
        output_name = csv_prefix + "_" + str(i) + "_desc_30_100.csv"
        final_csv.to_csv((DESC_DIRECTORY / output_name), index=False)


if __name__ == '__main__':
    get_sound_event_names()

    train_sound_events = {}
    test_sound_events = {}
    for class_name, file_list in SOUND_EVENTS.items():
        np.random.shuffle(file_list)  # Randomize before splitting
        split_idx = int(np.floor(len(file_list) * TRAIN_TEST_SPLIT))
        train_sound_events[class_name] = file_list[:split_idx]
        test_sound_events[class_name] = file_list[split_idx:]
    create_csv_batch(train_sound_events, NB_TRAIN, "train")
    create_csv_batch(test_sound_events, NB_TEST, "test")