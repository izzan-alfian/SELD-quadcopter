#include "driver/i2s.h"

// I2S Configuration
#define I2S_MIC_PORT I2S_NUM_0 // I2S port number

// IMPORTANT: Change these pin numbers to match your ESP32-S3 board and wiring.
// These are example pins. Refer to your ESP32-S3 board's pinout.
#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_16  // SCK / BCLK
#define I2S_MIC_WORD_SELECT  GPIO_NUM_17 // WS / LRCK
#define I2S_MIC_SERIAL_DATA  GPIO_NUM_18 // SD / DIN (Data IN for ESP32)

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

// Task for I2S reading and serial sending
TaskHandle_t i2sTaskHandle = NULL;

void i2s_mic_task(void *parameter) {
    esp_err_t err;
    size_t bytes_read;

    // Buffer to hold raw 32-bit samples from I2S
    int32_t* i2s_read_buffer = (int32_t*)malloc(I2S_READ_BUFFER_SIZE_BYTES);
    if (!i2s_read_buffer) {
        Serial.println("Failed to allocate memory for I2S read buffer");
        vTaskDelete(NULL);
        return;
    }

    // Buffer to hold 16-bit samples for serial transmission
    // Number of samples = I2S_READ_BUFFER_SIZE_BYTES / sizeof(int32_t)
    int num_samples = I2S_READ_BUFFER_SIZE_BYTES / sizeof(int32_t);
    int16_t* serial_write_buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));
    if (!serial_write_buffer) {
        Serial.println("Failed to allocate memory for serial write buffer");
        free(i2s_read_buffer);
        vTaskDelete(NULL);
        return;
    }

    Serial.println("I2S Read Task Started. Streaming audio data...");

    while (true) {
        // Read data from I2S bus
        err = i2s_read(I2S_MIC_PORT, i2s_read_buffer, I2S_READ_BUFFER_SIZE_BYTES, &bytes_read, portMAX_DELAY);

        if (err != ESP_OK) {
            Serial.printf("I2S read error: %d\n", err);
            continue;
        }

        if (bytes_read > 0) {
            int samples_read = bytes_read / sizeof(int32_t);
            
            // Process 32-bit samples to 16-bit samples
            // INMP441 data is 24-bit left-justified in a 32-bit frame.
            // To get the 16 MSB, we right-shift by 8.
            for (int i = 0; i < samples_read; i++) {
                // *** NEW: Apply digital volume reduction ***
                // First, apply the volume factor to the full 32-bit sample
                int32_t attenuated_sample = (int32_t)((float)i2s_read_buffer[i] * VOLUME_REDUCTION_FACTOR);

                // Then, convert the attenuated 32-bit sample to 16-bit
                serial_write_buffer[i] = (int16_t)(attenuated_sample >> 8);
            }

            // Write the 16-bit samples to Serial port
            Serial.write((const uint8_t*)serial_write_buffer, samples_read * sizeof(int16_t));
        }
    }

    // Should not reach here
    free(i2s_read_buffer);
    free(serial_write_buffer);
    vTaskDelete(NULL);
}

void setup() {
    Serial.begin(2000000); // Use a high baud rate for audio streaming
    Serial.println("ESP32 INMP441 I2S Audio Streamer");

    // Configure I2S
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX), // Master, RX
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE_CONFIG,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // INMP441 is mono. Assuming L/R pin is set for Left channel.
                                                     // Connect L/R pin of INMP441 to GND for Left Channel.
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Interrupt level 1
        .dma_buf_count = 4,                      // Number of DMA buffers
        .dma_buf_len = 1024,                     // Length of each DMA buffer in samples
        .use_apll = true,                       // Don't use APLL for clock source
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    // Configure I2S pins
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_SERIAL_CLOCK,
        .ws_io_num = I2S_MIC_WORD_SELECT,
        .data_out_num = I2S_PIN_NO_CHANGE, // Not used for RX
        .data_in_num = I2S_MIC_SERIAL_DATA
    };

    esp_err_t err;

    // Install and start I2S driver
    err = i2s_driver_install(I2S_MIC_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("Failed to install I2S driver: %d\n", err);
        return;
    }

    err = i2s_set_pin(I2S_MIC_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("Failed to set I2S pins: %d\n", err);
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
