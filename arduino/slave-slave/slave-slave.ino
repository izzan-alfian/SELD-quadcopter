#include "driver/i2s.h"

// I2S Configuration
#define B0_I2S_MIC_PORT I2S_NUM_0 // I2S port number
#define B1_I2S_MIC_PORT I2S_NUM_1 // I2S port number

// IMPORTANT: Change these pin numbers to match your ESP32-S3 board and wiring.
// These are example pins. Refer to your ESP32-S3 board's pinout.
#define B0_I2S_MIC_SERIAL_CLOCK GPIO_NUM_16  // SCK / BCLK
#define B0_I2S_MIC_WORD_SELECT  GPIO_NUM_17 // WS / LRCK
#define B0_I2S_MIC_SERIAL_DATA  GPIO_NUM_18 // SD / DIN (Data IN for ESP32)

#define B1_I2S_MIC_SERIAL_CLOCK GPIO_NUM_9  // SCK / BCLK
#define B1_I2S_MIC_WORD_SELECT  GPIO_NUM_10 // WS / LRCK
#define B1_I2S_MIC_SERIAL_DATA  GPIO_NUM_11 // SD / DIN (Data IN for ESP32)

// Audio Settings
#define SAMPLE_RATE 44100     // Sample rate in Hz (e.g., 16000, 22050, 44100)
                              // INMP441 supports 8kHz to 48kHz
#define BITS_PER_SAMPLE_CONFIG I2S_BITS_PER_SAMPLE_32BIT // I2S driver configuration for reading 24-bit data
#define EFFECTIVE_BITS_PER_SAMPLE 16 // We will convert to 16-bit audio to send over serial

// Buffer for I2S read
#define I2S_READ_BUFFER_SIZE_BYTES 2048 // Size of buffer for i2s_read (in bytes)
// This means I2S_READ_BUFFER_SIZE_BYTES / (32/8) = 512 samples of 32-bit data

// ADD THIS NEW CONFIGURATION LINE AT THE TOP OF YOUR FILE
// A factor of 1.0 means no change. 0.5 is half volume (-6dB).
// Start with 0.5 for your "noisy" mics and adjust as needed.
#define VOLUME_REDUCTION_FACTOR 0.1f

#define START_BYTE 0xAA
#define END_BYTE 0x55

uint32_t order_count = 0;
// Task for I2S reading and serial sending
TaskHandle_t i2sTaskHandle = NULL;

void i2s_mic_task(void *parameter) {
    esp_err_t B0_err;
    esp_err_t B1_err;
    size_t B0_bytes_read;
    size_t B1_bytes_read;

    // Buffer to hold raw 32-bit samples from I2S
    int32_t* B0_i2s_read_buffer = (int32_t*)malloc(I2S_READ_BUFFER_SIZE_BYTES);
    int32_t* B1_i2s_read_buffer = (int32_t*)malloc(I2S_READ_BUFFER_SIZE_BYTES);

    if (!B0_i2s_read_buffer) {
        Serial.println("Failed to allocate memory for B0 I2S read buffer");
        vTaskDelete(NULL);
        return;
    }
    if (!B1_i2s_read_buffer) {
        Serial.println("Failed to allocate memory for B1 I2S read buffer");
        vTaskDelete(NULL);
        return;
    }

    // Buffer to hold 16-bit samples for serial transmission
    int num_samples = I2S_READ_BUFFER_SIZE_BYTES / sizeof(int32_t);
    int16_t* B0_serial_write_buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));
    int16_t* B1_serial_write_buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));

    if (!B0_serial_write_buffer) {
        Serial.println("Failed to allocate B0 memory for serial write buffer");
        free(B0_i2s_read_buffer);
        vTaskDelete(NULL);
        return;
    }
    if (!B1_serial_write_buffer) {
        Serial.println("Failed to allocate B1 memory for serial write buffer");
        free(B0_i2s_read_buffer);
        vTaskDelete(NULL);
        return;
    }

    Serial.println("I2S Read Task Started. Streaming audio data...");

    while (true) {
        // Read data from I2S bus
        B0_err = i2s_read(B0_I2S_MIC_PORT, B0_i2s_read_buffer, I2S_READ_BUFFER_SIZE_BYTES, &B0_bytes_read, portMAX_DELAY);
        B1_err = i2s_read(B1_I2S_MIC_PORT, B1_i2s_read_buffer, I2S_READ_BUFFER_SIZE_BYTES, &B1_bytes_read, portMAX_DELAY);

        if ((B0_err != ESP_OK) || (B1_err != ESP_OK)) {
            if (B0_err != ESP_OK) {Serial.printf("I2S B0 read error: %d\n", B0_err);}
            if (B1_err != ESP_OK) {Serial.printf("I2S B1 read error: %d\n", B1_err);}
            continue;
        }

        if ((B0_bytes_read > 0) || (B1_bytes_read > 0)) {
            int B0_samples_read = B0_bytes_read / sizeof(int32_t);
            int B1_samples_read = B0_bytes_read / sizeof(int32_t);

            if (B0_samples_read == B1_samples_read) {
                // Process 32-bit samples to 16-bit samples
                // INMP441 data is 24-bit left-justified in a 32-bit frame.
                // To get the 16 MSB, we right-shift by 8.
                for (int i = 0; i < B0_samples_read; i++) {
                    // *** NEW: Apply digital volume reduction ***
                    // First, apply the volume factor to the full 32-bit sample
                    int32_t B0_attenuated_sample = (int32_t)((float)B0_i2s_read_buffer[i] * VOLUME_REDUCTION_FACTOR);
                    int32_t B1_attenuated_sample = (int32_t)((float)B1_i2s_read_buffer[i] * VOLUME_REDUCTION_FACTOR);

                    // Then, convert the attenuated 32-bit sample to 16-bit
                    B0_serial_write_buffer[i] = (int16_t)(B0_attenuated_sample >> 8);
                    B1_serial_write_buffer[i] = (int16_t)(B1_attenuated_sample >> 8);
                }

                // Write the 16-bit samples to Serial port
                Serial.write(START_BYTE);
                Serial.write((const uint8_t*)&order_count, 4);
                Serial.write((const uint8_t*)B0_serial_write_buffer, B0_samples_read * sizeof(int16_t));
                Serial.write((const uint8_t*)B1_serial_write_buffer, B0_samples_read * sizeof(int16_t));
                Serial.write(END_BYTE);
            }
        }

        order_count++;
    }

    // Should not reach here
    free(B0_i2s_read_buffer);
    free(B1_i2s_read_buffer);
    free(B0_serial_write_buffer);
    free(B1_serial_write_buffer);
    vTaskDelete(NULL);
}

void setup() {
    Serial.begin(4000000); // Use a high baud rate for audio streaming
    Serial.println("ESP32 INMP441 I2S Audio Streamer");

    // Configure I2S
    i2s_config_t B0_i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_SLAVE | I2S_MODE_RX), // Slave, RX
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE_CONFIG,
        .channel_format = I2S_CHANNEL_FMT_ALL_LEFT, // INMP441 is mono. Assuming L/R pin is set for Left channel.
                                                     // Connect L/R pin of INMP441 to GND for Left Channel.
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Interrupt level 1
        .dma_buf_count = 4,                      // Number of DMA buffers
        .dma_buf_len = 1024,                     // Length of each DMA buffer in samples
        .use_apll = true,
        .fixed_mclk = 0
    };

    // Configure I2S pins
    i2s_pin_config_t B0_pin_config = {
        .bck_io_num = B0_I2S_MIC_SERIAL_CLOCK,
        .ws_io_num = B0_I2S_MIC_WORD_SELECT,
        .data_out_num = I2S_PIN_NO_CHANGE, // Not used for RX
        .data_in_num = B0_I2S_MIC_SERIAL_DATA
    };


    i2s_config_t B1_i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_SLAVE | I2S_MODE_RX), // Slave, RX
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE_CONFIG,
        .channel_format = I2S_CHANNEL_FMT_ALL_LEFT, // INMP441 is mono. Assuming L/R pin is set for Left channel.
                                                     // Connect L/R pin of INMP441 to GND for Left Channel.
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Interrupt level 1
        .dma_buf_count = 4,                      // Number of DMA buffers
        .dma_buf_len = 1024,                     // Length of each DMA buffer in samples
        .use_apll = true,
        .fixed_mclk = 0
    };

    // Configure I2S pins
    i2s_pin_config_t B1_pin_config = {
        .bck_io_num = B1_I2S_MIC_SERIAL_CLOCK,
        .ws_io_num = B1_I2S_MIC_WORD_SELECT,
        .data_out_num = I2S_PIN_NO_CHANGE, // Not used for RX
        .data_in_num = B1_I2S_MIC_SERIAL_DATA
    };

    esp_err_t err;

    // Install and start I2S driver
    err = i2s_driver_install(B0_I2S_MIC_PORT, &B0_i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed to install B0 I2S driver: %d\n", err);
        return;
    }

    err = i2s_set_pin(B0_I2S_MIC_PORT, &B0_pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed to set B0 I2S pins: %d\n", err);
        return;
    }

    err = i2s_driver_install(B1_I2S_MIC_PORT, &B1_i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed to install B1 I2S driver: %d\n", err);
        return;
    }

    err = i2s_set_pin(B1_I2S_MIC_PORT, &B1_pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed to set B1 I2S pins: %d\n", err);
        return;
    }
    
    Serial.println("I2S driver installed and pins configured.");

    // Start the I2S reading task
    // Run on core 1 to avoid conflict with WiFi/BT if used later, and give it a good stack size
    xTaskCreatePinnedToCore(i2s_mic_task, "I2SMicTask", 4096, NULL, 5, &i2sTaskHandle, 1); 

    if (i2sTaskHandle == NULL) {
        Serial.println("Failed to create I2S task");
    }
}

void loop() {
    // The main work is done in the i2s_mic_task
    // You can put other non-blocking code here if needed
    vTaskDelay(pdMS_TO_TICKS(1000)); // Keep loop alive, but low activity
}
