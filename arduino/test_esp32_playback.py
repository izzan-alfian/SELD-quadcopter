import serial
import pyaudio
import time
import numpy as np

# --- Configuration ---
SERIAL_PORT = 'COM9'  # CHANGE THIS
BAUD_RATE = 2000000

# --- Audio settings (must match ESP32) ---
SAMPLE_RATE = 44100  # Hz
# *** KEY CHANGE: Set channels to 2 for stereo ***
CHANNELS = 1
FORMAT = pyaudio.paFloat32
BYTES_PER_SAMPLE = 2

# --- Buffer settings ---
# The ESP32 sends 2048 bytes of raw 32-bit data = 512 individual samples.
# In stereo, this means 256 stereo frames (256 L + 256 R).
# The processed 16-bit data chunk is 512 samples * 2 bytes/sample = 1024 bytes.
# We set our buffer to match this for smooth playback.
FRAMES_PER_BUFFER = 256 # Number of stereo frames (L/R pairs) for PyAudio
# Calculate bytes to read from serial: frames * channels * bytes/sample
SERIAL_READ_SIZE = FRAMES_PER_BUFFER * CHANNELS * BYTES_PER_SAMPLE # 256 * 2 * 2 = 1024 bytes

def main():
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True,
                        frames_per_buffer=FRAMES_PER_BUFFER)
        print(f"PyAudio stream opened for STEREO playback at {SAMPLE_RATE} Hz.")
    except Exception as e:
        print(f"Error opening PyAudio stream: {e}")
        p.terminate()
        return

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to serial port {SERIAL_PORT}.")
        time.sleep(2) 
        ser.reset_input_buffer()
        print("Starting audio playback...")
    except serial.SerialException as e:
        print(f"Error opening serial port {SERIAL_PORT}: {e}")
        stream.close()
        p.terminate()
        return

    try:
        while True:
            audio_data = ser.read(SERIAL_READ_SIZE)

            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0 # digital volume reduction
            audio_data = audio_data.tobytes()
            
            if audio_data:
                # Write the interleaved stereo data to the PyAudio stream
                stream.write(audio_data)

    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        print("Cleaning up...")
        stream.stop_stream()
        stream.close()
        p.terminate()
        if ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()
