void setup() {
  // Start the serial communication at a baud rate of 115200
  Serial.begin(115200);
}

void loop() {
  // Print a message to the serial monitor
  Serial.println("Hello, World!");
  
  // Wait for 2 seconds before printing again
  delay(2000);
}