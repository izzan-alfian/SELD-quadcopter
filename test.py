import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

source_position = [5, 0, 5]  # Position of sound source (in meters)
mic_positions = np.array([[5, 100, 5]]).T  # Position of microphone

room = pra.AnechoicRoom(fs=16000, air_absorption=True)

fs, signal = wavfile.read("arctic_a0010.wav")
room.add_source(source_position, signal=signal)

room.add_microphone_array(mic_positions)

room.compute_rir()

source_to_mic_distance = np.linalg.norm(np.array(source_position) - mic_positions[0])
print(f"Distance from source to microphone: {source_to_mic_distance} meters")

room.simulate()

# room.plot()
# plt.show()

end = room.mic_array.signals[0,:]

print(max(end))
print(min(end))

end = end / np.max(np.abs(end))

print(max(end))
print(min(end))

sd.play(end, samplerate=fs) # here
sd.wait()
