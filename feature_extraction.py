import numpy as np
import librosa

audio_path = "..\datasets\in-flight-source-recording\DREGON_free-flight_speech-high_room1\DREGON_free-flight_speech-high_room1.wav"
y, sr = librosa.load(audio_path, sr=None)

print(sr)