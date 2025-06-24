import os
import librosa
import numpy as np
import soundfile as sf
from scipy.io.wavfile import write

hop_lenght = 512
window_size = 1024
decible_treshold = -30.0

cut_whistle_directory = './dataset_generator/sounds/raw_whistles/'

global_whistle_index = 0
for file_name in os.listdir(cut_whistle_directory):
    whistle_sounds_indices = []
    if 'cut' in file_name:
        cut_whistle_path = os.path.join(cut_whistle_directory, file_name)
        cut_whisle_file, sr = librosa.load(cut_whistle_path, sr=None)

        i = 0
        while i < (len(cut_whisle_file) - window_size + 1):
            frame = cut_whisle_file[i : i + window_size]
            rms = np.sqrt(np.mean(frame ** 2))
            decible = librosa.amplitude_to_db(rms)

            whistle_sound = []
            while decible > decible_treshold:
                whistle_sound.extend(np.arange(i, i + hop_lenght, 1))
                i += hop_lenght

                frame = cut_whisle_file[i : i + window_size]
                rms = np.sqrt(np.mean(frame ** 2))
                decible = librosa.amplitude_to_db(rms)

            if len(whistle_sound) > (sr / 2): # if the sound is longer than half a second
                whistle_sounds_indices.append(whistle_sound)
            else:
                i += hop_lenght
        
        for index, i in enumerate(whistle_sounds_indices):
            sf.write(('./dataset_generator/sounds/whistles/whistle_sound%03d.wav' %global_whistle_index), cut_whisle_file[i], sr, format='WAV', subtype='PCM_32')
            global_whistle_index += 1


