import serial
import pyaudio
import time
import numpy as np
import sys
import threading
import queue
import struct

# --- Assigned ESP32 ports ---
A_SERIAL_PORT = '/dev/ttyACM0'
B_SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 4000000

# --- Audio settings from ESP32 ---
SAMPLE_RATE = 44100
CHANNELS_PER_ESP_STREAM = 2
FORMAT_FROM_ESP = np.int16
BYTES_PER_SAMPLE_ESP = 2

# --- PyAudio settings ---
PYAUDIO_FORMAT = pyaudio.paFloat32
PYAUDIO_CHANNELS = CHANNELS_PER_ESP_STREAM

# --- Packet and Buffer Settings ---
SIZE_0_AUDIO_PAYLOAD_BYTES = 512 * BYTES_PER_SAMPLE_ESP # 1024
SIZE_1_AUDIO_PAYLOAD_BYTES = 512 * BYTES_PER_SAMPLE_ESP # 1024
SIZE_START_BYTE_FIELD = 1
SIZE_ORDER_COUNT_FIELD = 4
SIZE_END_BYTE_FIELD = 1
EXPECTED_PACKET_SIZE = (SIZE_START_BYTE_FIELD + SIZE_ORDER_COUNT_FIELD +
                        SIZE_0_AUDIO_PAYLOAD_BYTES + SIZE_1_AUDIO_PAYLOAD_BYTES + SIZE_END_BYTE_FIELD) # 2055 bytes

FRAMES_PER_BUFFER_PYAUDIO = SIZE_0_AUDIO_PAYLOAD_BYTES // BYTES_PER_SAMPLE_ESP // PYAUDIO_CHANNELS # 256

# --- Packet Framing Bytes (as defined in your ESP32 code) ---
START_BYTE_VAL = 0xAA
END_BYTE_VAL = 0x55

# --- Global variables for threading and control ---
exit_event = threading.Event()
# A thread-safe queue to pass audio data from the serial reader to the audio callback
a_audio_queue = queue.Queue(maxsize=100)
b_audio_queue = queue.Queue(maxsize=100)
current_channel_to_play = "A0"

def get_audio_stream():
    # Get a tuple of (payload_0, payload_1) from the queue
    a_count_order, a0_bytes, a1_bytes = a_audio_queue.get_nowait()
    b_count_order, b0_bytes, b1_bytes = b_audio_queue.get_nowait()

    if a_count_order != b_count_order:
        print(f'!!! Warning: misaligned package counts; a: {a_count_order}, b: {b_count_order}, diff: {a_count_order - b_count_order}')
        # Synchronize by advancing the lagging queue until counts match.
        while a_count_order != b_count_order:
            if a_count_order > b_count_order: # B is lagging
                b_count_order, b0_bytes, b1_bytes = b_audio_queue.get_nowait()
            else: # A is lagging (b_count_order > a_count_order here)
                a_count_order, a0_bytes, a1_bytes = a_audio_queue.get_nowait()

    return a0_bytes, a1_bytes, b0_bytes, b1_bytes

def serial_reader_thread(serial_conn, audio_queue):
    """
    This thread's only job is to read full packets from the serial port
    and put the desired audio payload into a thread-safe queue.
    This version uses a more robust byte-by-byte scanning method to prevent desync.
    """
    last_order_count = None
    packet_buffer = bytearray()
    sync_state = "WAITING_FOR_START" # Initial state

    while not exit_event.is_set():
        try:
            # Read a small chunk of data to process, which is more efficient
            # than reading one byte at a time but still allows for scanning.
            bytes_to_process = serial_conn.read(1024)
            if not bytes_to_process: continue

            for byte in bytes_to_process:
                if sync_state == "WAITING_FOR_START":
                    if byte == START_BYTE_VAL: # START_BYTE_VAL
                        packet_buffer.clear()
                        packet_buffer.append(byte)
                        sync_state = "READING_PACKET"
                
                elif sync_state == "READING_PACKET":
                    packet_buffer.append(byte)
                    
                    if len(packet_buffer) == EXPECTED_PACKET_SIZE:
                        # We have a full-length potential packet, now validate it
                        if packet_buffer[-1] == END_BYTE_VAL: # END_BYTE_VAL
                            # --- Packet is structurally valid, extract data ---
                            order_count_offset  = SIZE_START_BYTE_FIELD
                            start_0_offset      = SIZE_START_BYTE_FIELD + SIZE_ORDER_COUNT_FIELD
                            start_1_offset      = SIZE_START_BYTE_FIELD + SIZE_ORDER_COUNT_FIELD + SIZE_0_AUDIO_PAYLOAD_BYTES

                            order_count_bytes = packet_buffer[order_count_offset : order_count_offset + SIZE_ORDER_COUNT_FIELD]
                            current_order_count = struct.unpack('<I', order_count_bytes)[0]

                            if last_order_count is not None and current_order_count != last_order_count + 1:
                                if not (current_order_count == 0 and last_order_count == 0xFFFFFFFF):
                                    dropped = current_order_count - last_order_count - 1
                                    print(f"!!! Warning: Dropped {dropped} packet(s) from {serial_conn.name}. ")
                            last_order_count = current_order_count

                            # Extract both audio payloads
                            payload_0 = packet_buffer[start_0_offset : start_0_offset + SIZE_0_AUDIO_PAYLOAD_BYTES]
                            payload_1 = packet_buffer[start_1_offset : start_1_offset + SIZE_1_AUDIO_PAYLOAD_BYTES]

                            # Put the tuple of payloads into the queue
                            try:
                                audio_queue.put_nowait((current_order_count, payload_0, payload_1))
                            except queue.Full:
                                pass # Silently drop packet if audio callback is lagging
                        # else:
                        #     print("Sync Error: Packet had correct length but wrong END_BYTE.")

                        # Reset to find the next packet, regardless of whether this one was valid
                        sync_state = "WAITING_FOR_START"
                        packet_buffer.clear()
    
        except Exception as e:
            if not exit_event.is_set():
                print(f"Error in serial reader thread: {e}")
            break

def pyaudio_callback(in_data, frame_count, time_info, status):
    """
    This function is called by PyAudio in a separate thread whenever it needs more audio data.
    """
    try:
        selected_bytes = None

        a0_bytes, a1_bytes, b0_bytes, b1_bytes = get_audio_stream()
        
        # Select the audio data based on the global channel setting
        if current_channel_to_play == "A0":
            selected_bytes = a0_bytes
        elif current_channel_to_play == "A1":
            selected_bytes = a1_bytes
        elif current_channel_to_play == "B0":
            selected_bytes = b0_bytes
        elif current_channel_to_play == "B1":
            selected_bytes = b1_bytes
        
        # Process the selected audio data
        samples_int16 = np.frombuffer(selected_bytes, dtype=FORMAT_FROM_ESP)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0

        # Return the processed data and signal PyAudio to continue
        return (samples_float32.tobytes(), pyaudio.paContinue)

    except queue.Empty:
        # The queue was empty, which means we're not receiving data fast enough (a stutter).
        # We must return silent audio to avoid blocking.
        # print("Stutter: Audio queue empty.") # Can be noisy, use for debugging
        silent_audio = np.zeros(frame_count * PYAUDIO_CHANNELS, dtype=np.float32)
        return (silent_audio.tobytes(), pyaudio.paContinue)
    
def listen_for_key_press():
    global current_channel_to_play
    while not exit_event.is_set():
        try:
            key_input = input() # Press Enter after typing
            if key_input == '1' and current_channel_to_play != "A0":
                current_channel_to_play = "A0"
                print("\n--> Switched to A0 audio stream.")
            elif key_input == '2' and current_channel_to_play != "A1":
                current_channel_to_play = "A1"
                print("\n--> Switched to A1 audio stream.")
            elif key_input == '3' and current_channel_to_play != "B0":
                current_channel_to_play = "B0"
                print("\n--> Switched to B0 audio stream.")
            elif key_input == '4' and current_channel_to_play != "B1":
                current_channel_to_play = "B1"
                print("\n--> Switched to B1 audio stream.")

        except (EOFError, KeyboardInterrupt):
            break

def main():
    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = None
    a_serial_connection = None
    b_serial_connection = None

    print(f"--- Audio Stream Receiver (Callback Mode) ---")
    print("Press '0' then Enter to stream A0, '1' then Enter for A1. Ctrl+C to exit.")
    
    try:
        a_serial_connection = serial.Serial(A_SERIAL_PORT, BAUD_RATE, timeout=1)
        b_serial_connection = serial.Serial(B_SERIAL_PORT, BAUD_RATE, timeout=1)
        a_serial_connection.reset_input_buffer()
        b_serial_connection.reset_input_buffer()
        print(f"Connected to {A_SERIAL_PORT} at {BAUD_RATE} baud.")
        
        # Start the thread that reads from serial and populates the queue
        a_reader_thread = threading.Thread(target=serial_reader_thread, args=(a_serial_connection, a_audio_queue,))
        b_reader_thread = threading.Thread(target=serial_reader_thread, args=(b_serial_connection, b_audio_queue,))
        a_reader_thread.start()
        b_reader_thread.start()

        # Start the key listener thread
        key_thread = threading.Thread(target=listen_for_key_press, daemon=True)
        key_thread.start()
        
        audio_stream = pyaudio_instance.open(format=PYAUDIO_FORMAT,
                                             channels=PYAUDIO_CHANNELS,
                                             rate=SAMPLE_RATE,
                                             output=True,
                                             frames_per_buffer=FRAMES_PER_BUFFER_PYAUDIO,
                                             stream_callback=pyaudio_callback)
        
        print("Audio stream is active. Listening for data...")
        audio_stream.start_stream()

        # Keep the main thread alive while the other threads work
        while audio_stream.is_active() and not exit_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExit signal received.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up resources...")
        exit_event.set()
        
        if audio_stream and audio_stream.is_active():
            audio_stream.stop_stream()
        if audio_stream:
            audio_stream.close()
        if pyaudio_instance:
            pyaudio_instance.terminate()
        
        if 'reader_thread' in locals() and a_reader_thread.is_alive():
            a_reader_thread.join()
        
        if a_serial_connection and a_serial_connection.is_open:
            a_serial_connection.close()

        print("Cleanup complete.")

if __name__ == '__main__':
    main()
