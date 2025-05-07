import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

room = pra.AnechoicRoom(fs=16000, air_absorption=True)

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

fs, signal = wavfile.read("arctic_a0010.wav")
source_position = [3, 3, 3]  # Position of sound source (in meters)
room.add_source(source_position, signal=signal)

room.compute_rir()
room.simulate()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')   # or subplot(111) for 2D
room.plot(ax=ax)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
plt.show()


sound_signals = room.mic_array.signals
sound_signals_norm = np.zeros(sound_signals.shape)
for i, sound in enumerate(sound_signals):
    sound_signals_norm[i] = sound / np.max(np.abs(sound))

print(sound_signals.shape)
print(sound_signals_norm.T.shape)

sd.play(sound_signals_norm.T, samplerate=fs) # here
sd.wait()