from scipy.io import wavfile
import noisereduce as nr
import numpy as np
import os


sr, data = wavfile.read('./dataset_generator/sounds/criset_2class_motor_filtered/wav_ov1_split1_30db/test_0_desc_30_100.wav')

print(np.max(data))
print(np.min(data))