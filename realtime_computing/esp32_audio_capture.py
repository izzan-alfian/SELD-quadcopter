import serial
import pyaudio
import time
import numpy as np
import sys
import threading
import queue
import struct

class ESP32AudioCapture():
    def __init__(self,):
        # --- Assigned ESP32 ports ---
        self.A_SERIAL_PORT = '/dev/ttyACM0'
        self.B_SERIAL_PORT = '/dev/ttyACM1'
        self.BAUD_RATE = 4000000

        # --- Audio settings from ESP32 --- 
        self.SAMPLE_RATE = 44100
        self.CHANNELS_PER_ESP_STREAM = 2
        self.FORMAT_FROM_ESP = np.int16
        self.BYTES_PER_SAMPLE_ESP = 2

        # --- PyAudio settings ---
        self.PYAUDIO_FORMAT = pyaudio.paFloat32
        self.PYAUDIO_CHANNELS = self.CHANNELS_PER_ESP_STREAM

        # --- Packet and Buffer Settings ---
        self.SIZE_0_AUDIO_PAYLOAD_BYTES = 512 * self.BYTES_PER_SAMPLE_ESP # 1024
        self.SIZE_1_AUDIO_PAYLOAD_BYTES = 512 * self.BYTES_PER_SAMPLE_ESP # 1024
        self.SIZE_START_BYTE_FIELD = 1
        self.SIZE_ORDER_COUNT_FIELD = 4
        self.SIZE_END_BYTE_FIELD = 1
        self.EXPECTED_PACKET_SIZE = (self.SIZE_START_BYTE_FIELD + self.SIZE_ORDER_COUNT_FIELD +
                                self.SIZE_0_AUDIO_PAYLOAD_BYTES + self.SIZE_1_AUDIO_PAYLOAD_BYTES + self.SIZE_END_BYTE_FIELD) # 2055 bytes

        self.FRAMES_PER_BUFFER_PYAUDIO = self.SIZE_0_AUDIO_PAYLOAD_BYTES // self.BYTES_PER_SAMPLE_ESP // self.PYAUDIO_CHANNELS # 256

        # --- Packet Framing Bytes (as defined in your ESP32 code) ---
        self.START_BYTE_VAL = 0xAA
        self.END_BYTE_VAL = 0x55

        # --- Global variables for threading and control ---
        self.exit_event = threading.Event()
        # A thread-safe queue to pass audio data from the serial reader to the audio callback
        self.a_audio_queue = queue.Queue(maxsize=100)
        self.b_audio_queue = queue.Queue(maxsize=100)



        # Initialize serial connections and start reader threads
        self.a_serial_connection = serial.Serial(self.A_SERIAL_PORT, self.BAUD_RATE, timeout=1)
        self.b_serial_connection = serial.Serial(self.B_SERIAL_PORT, self.BAUD_RATE, timeout=1)
        self.a_serial_connection.reset_input_buffer()
        self.b_serial_connection.reset_input_buffer()

        threading.Thread(target=self.serial_reader_thread, args=(self.a_serial_connection, self.a_audio_queue,)).start()
        threading.Thread(target=self.serial_reader_thread, args=(self.b_serial_connection, self.b_audio_queue,)).start()


    def convert_to_float32(self, byte_data):
        samples_int16 = np.frombuffer(byte_data, dtype=self.FORMAT_FROM_ESP)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0

        return samples_float32


    def serial_reader_thread(self, serial_conn, audio_queue):
        """
        This thread's only job is to read full packets from the serial port
        and put the desired audio payload into a thread-safe queue.
        This version uses a more robust byte-by-byte scanning method to prevent desync.
        """
        last_order_count = None
        packet_buffer = bytearray()
        sync_state = "WAITING_FOR_START" # Initial state

        while not self.exit_event.is_set():
            try:
                # Read a small chunk of data to process, which is more efficient
                # than reading one byte at a time but still allows for scanning.
                bytes_to_process = serial_conn.read(1024)
                if not bytes_to_process: continue

                for byte in bytes_to_process:
                    if sync_state == "WAITING_FOR_START":
                        if byte == self.START_BYTE_VAL: # START_BYTE_VAL
                            packet_buffer.clear()
                            packet_buffer.append(byte)
                            sync_state = "READING_PACKET"
                    
                    elif sync_state == "READING_PACKET":
                        packet_buffer.append(byte)
                        
                        if len(packet_buffer) == self.EXPECTED_PACKET_SIZE:
                            # We have a full-length potential packet, now validate it
                            if packet_buffer[-1] == self.END_BYTE_VAL: # END_BYTE_VAL
                                # --- Packet is structurally valid, extract data ---
                                order_count_offset  = self.SIZE_START_BYTE_FIELD
                                start_0_offset      = self.SIZE_START_BYTE_FIELD + self.SIZE_ORDER_COUNT_FIELD
                                start_1_offset      = self.SIZE_START_BYTE_FIELD + self.SIZE_ORDER_COUNT_FIELD + self.SIZE_0_AUDIO_PAYLOAD_BYTES

                                order_count_bytes = packet_buffer[order_count_offset : order_count_offset + self.SIZE_ORDER_COUNT_FIELD]
                                current_order_count = struct.unpack('<I', order_count_bytes)[0]

                                if last_order_count is not None and current_order_count != last_order_count + 1:
                                    if not (current_order_count == 0 and last_order_count == 0xFFFFFFFF):
                                        dropped = current_order_count - last_order_count - 1
                                        print(f"!!! Warning: Dropped {dropped} packet(s) from {serial_conn.name}. ")
                                last_order_count = current_order_count

                                # Extract both audio payloads
                                payload_0 = packet_buffer[start_0_offset : start_0_offset + self.SIZE_0_AUDIO_PAYLOAD_BYTES]
                                payload_1 = packet_buffer[start_1_offset : start_1_offset + self.SIZE_1_AUDIO_PAYLOAD_BYTES]

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
                if not self.exit_event.is_set():
                    print(f"Error in serial reader thread: {e}")
                break

    def get_audio_stream(self):

        # Get a tuple of (payload_0, payload_1) from the queue
        a_count_order, a0_bytes, a1_bytes = self.a_audio_queue.get()
        b_count_order, b0_bytes, b1_bytes = self.b_audio_queue.get()

        if a_count_order != b_count_order:
            print(f'!!! Warning: misaligned package counts; a: {a_count_order}, b: {b_count_order}, diff: {a_count_order - b_count_order}')
            # Synchronize by advancing the lagging queue until counts match.
            while a_count_order != b_count_order:
                if a_count_order > b_count_order: # B is lagging
                    b_count_order, b0_bytes, b1_bytes = self.b_audio_queue.get()
                else: # A is lagging (b_count_order > a_count_order here)
                    a_count_order, a0_bytes, a1_bytes = self.a_audio_queue.get()

        stream_bytes = (a0_bytes, a1_bytes, b0_bytes, b1_bytes)

        return tuple(self.convert_to_float32(bs) for bs in stream_bytes)