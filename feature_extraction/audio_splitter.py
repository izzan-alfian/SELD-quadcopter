import pydub as pd
import librosa
import sounddevice as sd

sound, fs = librosa.load('/home/zanub/Desktop/projects/SELD-quadcopter/feature_extraction/resources/_0SMVzXhf-s_cut.wav', sr=None)

print("Complete loading file")

sd.play(sound, samplerate=fs)
sd.wait()