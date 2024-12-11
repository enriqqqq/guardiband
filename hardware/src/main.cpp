#define DEBUG true

#include <stdio.h>
#include <math.h>
#include <cstring>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_GFX.h>
#include <Adafruit_GC9A01A.h>
#include <MAX30105.h>
#include <heartRate.h>
#include <spo2_algorithm.h>
#include <TinyGPSPlus.h>
#include <FreeRTOS.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <driver/i2s.h>
#include <WiFiManager.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <kiss_fft.h>

#include "scaler.h"
#include "model_data.h"
#include "mfcc_data.h"

// Constants for DSP
#define SAMPLE_RATE 16000
#define N_FFT 1024
#define N_MELS 128
#define N_MFCC 40
#define HOP_LENGTH 512
#define DURATION_SECONDS 1
#define FRAME_COUNT ((SAMPLE_RATE / HOP_LENGTH) + 1) // 32 frames for 1 second
#define INPUT_SIGNAL_LENGTH (SAMPLE_RATE * DURATION_SECONDS)

// Pin definitions
// Note - Do not use pin 35, 36, 37 since it is used for SPI communication between Flash and PSRAM in Octal SPI memory mode
// https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide.html#hardware-reference
#define I2C_SDA 20
#define I2C_SCL 21
#define TFT_SCL 9
#define TFT_SDA 10
#define TFT_DC 11
#define TFT_CS 12
#define TFT_RST 13
#define GPS_RX_PIN 47
#define GPS_TX_PIN 48
#define BUZZER_POSITIVE 4
#define BUZZER_NEGATIVE 5
#define BUTTON_PIN 6
#define INMP441_SD 3
#define INMP441_SCK 8
#define INMP441_WS 38
// connect INMP441 L/R to GND (mono)
#define DEVICE_ID "1A"

#define MQTT_PUBLISH_TOPIC "guardiband/1A/data"
#define MQTT_SUBSCRIBE_TOPIC "guardiband/1A/message"

// sensorData struct
typedef struct {
  float bpm;
  float spo2;
  float temp;
  double lat;
  double lon;
} sensorData_t;

// Breadcrumbs for function
void readINMP441(void *pvParameters);
void readMPU6050(void *pvParameters);
void readMAX30102(void *pvParameters);
void readGPS(void *pvParameters);
void readMLX90614(void *pvParameters);
void readINMP441(void *pvParameters);
void updateTFTDisplay(void *pvParameters);
void aggregateData(void *pvParameters);
void publishMQTTDataTask(void *pvParameters);
void publishSensorData(const sensorData_t &data);
void MQTTcallback(char* topic, byte* payload, unsigned int length);
void readButtonTask(void *pvParameters);
void monitorWiFi(void *pvParameters);
void readMLX90614(void *pvParameters);

void extract_mfcc(const float *audio_signal, float *output_mfcc);
void scale_mfcc(float* mfcc_buffer, size_t frame_count, size_t n_mfcc);

// Sensor and display objects
MAX30105 max30102;
TinyGPSPlus gps;
Adafruit_MPU6050 mpu = Adafruit_MPU6050();
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
Adafruit_GC9A01A tft = Adafruit_GC9A01A(TFT_CS, TFT_DC, TFT_SDA, TFT_SCL, TFT_RST);

// Variables for last updated measurement
sensors_event_t lastAccelValue, lastGyroValue;
float lastBPMValue = 0.0;
float lastSpO2Value = 0.0;
float lastTempValue = 0.0;
double lastLatValue = 0.0;
double lastLonValue = 0.0;
String lastMessage = "";

// Queue for sensor data
QueueHandle_t sensorDataQueue;

// I2C Semaphore
SemaphoreHandle_t i2cSemaphore;

// WiFi Setup & MQTT
WiFiManager wm;
const char* mqtt_server = "broker.hivemq.com";
unsigned long start_time = 0;

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// first read flag
bool firstGPSRead = false;

// scaler mean, scaler std, dct matrix and mel filterbank pointers for PSRAM allocation
float* scaler_mean_PSRAM;
float* scaler_std_PSRAM;
float* dct_matrix_PSRAM;
float* mel_filterbank_PSRAM;

// kiss_fft_cfg static allocation
static uint8_t kiss_fft_buffer[8456];
kiss_fft_cfg kissfft_cfg;

void setup() {
  // Initialize Serial for debugging
  Serial.begin(115200);

  // Print PSRAM size and RAM/heap size
  #if DEBUG
  Serial.printf("Initial PSRAM size: %d\n", ESP.getPsramSize());
  Serial.printf("Initial PSRAM free: %d\n", ESP.getFreePsram());
  Serial.printf("Initial Heap size: %d\n", ESP.getHeapSize());
  Serial.printf("Initial Free heap: %d\n", ESP.getFreeHeap());
  #endif

  // Allocate PSRAM for scaler mean, scaler std, dct matrix and mel filterbank
  scaler_mean_PSRAM = (float*) heap_caps_malloc(sizeof(mfcc_mean), MALLOC_CAP_SPIRAM);
  scaler_std_PSRAM = (float*) heap_caps_malloc(sizeof(mfcc_std), MALLOC_CAP_SPIRAM);
  dct_matrix_PSRAM = (float*) heap_caps_malloc(sizeof(dct_matrix), MALLOC_CAP_SPIRAM);
  mel_filterbank_PSRAM = (float*) heap_caps_malloc(sizeof(mel_filterbank), MALLOC_CAP_SPIRAM);

  if(scaler_mean_PSRAM == NULL || scaler_std_PSRAM == NULL || dct_matrix_PSRAM == NULL || mel_filterbank_PSRAM == NULL) {
    #if DEBUG
    Serial.println("Failed to allocate PSRAM");
    #endif
    while(1) {vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  // Print PSRAM size and RAM/heap size
  #if DEBUG
  Serial.println("PSRAM allocated");
  Serial.printf("PSRAM size: %d\n", ESP.getPsramSize());
  Serial.printf("PSRAM free: %d\n", ESP.getFreePsram());
  Serial.printf("Heap size: %d\n", ESP.getHeapSize());
  Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
  #endif

  // Copy scaler mean, scaler std, dct matrix and mel filterbank to PSRAM
  memcpy(scaler_mean_PSRAM, mfcc_mean, sizeof(mfcc_mean));
  memcpy(scaler_std_PSRAM, mfcc_std, sizeof(mfcc_std));
  memcpy(dct_matrix_PSRAM, dct_matrix, sizeof(dct_matrix));
  memcpy(mel_filterbank_PSRAM, mel_filterbank, sizeof(mel_filterbank));

  // Initialize kiss_fft_cfg
  size_t lenmem = 8456;
  kissfft_cfg = kiss_fft_alloc(N_FFT, 0, kiss_fft_buffer, &lenmem);
  if(kissfft_cfg == NULL) {
    #if DEBUG
    Serial.println("Failed to allocate kiss_fft_cfg");
    Serial.printf("kiss_fft_cfg size: %d\n", lenmem);
    #endif
    while(1) {vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  #if DEBUG
  Serial.println("kiss_fft_cfg allocated");
  #endif

  // Button
  pinMode(BUTTON_PIN, INPUT);

  // Buzzer
  pinMode(BUZZER_POSITIVE, OUTPUT);
  pinMode(BUZZER_NEGATIVE, OUTPUT);
  digitalWrite(BUZZER_NEGATIVE, LOW);

  // Sound Buzzer (Do, Re, Mi)
  tone(BUZZER_POSITIVE, 262, 200);
  delay(200);
  tone(BUZZER_POSITIVE, 294, 200);
  delay(200);
  tone(BUZZER_POSITIVE, 330, 200);

  // Initialize I2C for MPU6050 and MAX30102
  Wire.begin(I2C_SDA, I2C_SCL);

  // Initialize WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(wm.getWiFiSSID(), wm.getWiFiPass());
  start_time = millis();

  #if DEBUG
  Serial.print("Connecting to WiFi");
  #endif
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    #if DEBUG
    Serial.print(".");
    #endif
    if (millis() - start_time > 2000) {
      #if DEBUG
      Serial.println("Failed to connect to WiFi");
      #endif
      break;
    }
  }

  // print connected if connected
  #if DEBUG
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to WiFi");
    Serial.print("SSID: ");
    Serial.println(WiFi.SSID());
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  }
  #endif

  // Set up MQTT
  mqttClient.setServer(mqtt_server, 1883);
  mqttClient.setCallback(MQTTcallback);

  // Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
  }

  // Initialize MLX90614
  // if(!mlx.begin()) {
  //   Serial.println("Failed to find MLX90614 chip");
  //   while (1) {vTaskDelay(10/portTICK_PERIOD_MS);}
  // }

  // Initialize MAX30102
  if(!max30102.begin()) {
    Serial.println("MAX30102 not found. Please check wiring/power.");
    while (1) {vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  // Setup the MAX30102 sensor
  byte ledBrightness = 60;  //Options: 0=Off to 255=50mA
  byte sampleAverage = 4;   //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2;         //Options: 1 = Red only, 2 = Red + IR
  byte sampleRate = 100;    //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411;     //Options: 69, 118, 215, 411
  int adcRange = 4096;      //Options: 2048, 4096, 8192, 16384

  max30102.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);

  // Initialize GPS
  Serial1.begin(9600, SERIAL_8N1, GPS_TX_PIN, GPS_RX_PIN);

  // Initialize TFT display
  tft.begin(40000000); // Set the SPI clock frequency
  tft.setRotation(1); // Set the display rotation
  tft.fillScreen(GC9A01A_BLACK); // Clear the screen
  tft.setTextSize(2);
  tft.setTextColor(GC9A01A_WHITE);
  tft.setCursor(0, 120);
  tft.print("Hello, World!");

  // Create SensorData queue
  sensorDataQueue = xQueueCreate(10, sizeof(sensorData_t));

  // Create I2C Mutex
  i2cSemaphore = xSemaphoreCreateMutex();
  if(i2cSemaphore == NULL) {
    #if DEBUG
    Serial.println("Failed to create I2C semaphore");
    #endif
  }

  // Create FreeRTOS tasks
  xTaskCreate(readMPU6050, "Read MPU6050", 4096, NULL, 1, NULL);                // accelerometer and gyroscope
  xTaskCreate(readMAX30102, "Read MAX30102", 4096, NULL, 1, NULL);              // heart rate and SpO2
  xTaskCreate(readGPS, "Read GPS", 3072, NULL, 1, NULL);                        // GPS
  xTaskCreate(updateTFTDisplay, "Update TFT", 4096, NULL, 1, NULL);             // TFT display
  xTaskCreate(aggregateData, "Aggregate Data", 2048, NULL, 1, NULL);            // aggregate data and send to queue
  xTaskCreate(publishMQTTDataTask, "Publish MQTT Data", 4096, NULL, 1, NULL);   // publish data to MQTT
  xTaskCreate(monitorWiFi, "Monitor WiFi", 8192, NULL, 0, NULL);                // monitor WiFi
  xTaskCreate(readINMP441, "Read INMP441", 100000, NULL, 0, NULL);               // read sound sensor
  // xTaskCreate(readMLX90614, "Read MLX90614", 4096, NULL, 1, NULL);           // temperature
  // xTaskCreate(readButtonTask, "Read Button", 4096, NULL, 1, NULL);           // read button press
}

void loop() {
  // print out currently available RAM
  #if DEBUG
  Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
  #endif
  delay(1000);
}

// Task definitions
void readMPU6050(void *pvParameters) {
  const float MPU6050_SAMPLE_INTERVAL = 500;
  const float fallThreshold = 20.0;
  float prevAccelX = 0.0;
  float prevAccelY = 0.0;
  float prevAccelZ = 0.0;

  while (1) {
    // take semaphore
    // if(xSemaphoreTake(i2cSemaphore, portMAX_DELAY) != pdTRUE) {
    //   #if DEBUG
    //   Serial.println("Failed to take I2C semaphore");
    //   #endif
    //   vTaskDelay(5 / portTICK_PERIOD_MS);
    //   continue;
    // }

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    lastAccelValue = a;
    lastGyroValue = g;

    // release semaphore
    // xSemaphoreGive(i2cSemaphore);

    // convert acceleration to g from m/s^2
    float acceleration_x_mg = a.acceleration.x/9.8;
    float acceleration_y_mg = a.acceleration.y/9.8;
    float acceleration_z_mg = a.acceleration.z/9.8;

    // calculate jerk
    float jerkX = (acceleration_x_mg - prevAccelX) / (MPU6050_SAMPLE_INTERVAL / 1000);
    float jerkY = (acceleration_y_mg - prevAccelY) / (MPU6050_SAMPLE_INTERVAL / 1000);
    float jerkZ = (acceleration_z_mg - prevAccelZ) / (MPU6050_SAMPLE_INTERVAL / 1000);

    // update previous acceleration values
    prevAccelX = acceleration_x_mg;
    prevAccelY = acceleration_y_mg;
    prevAccelZ = acceleration_z_mg;

    // calculate magnitude of jerk
    float jerkMagnitude = sqrt(pow(jerkX, 2) + pow(jerkY, 2) + pow(jerkZ, 2));

    // fall detection
    if (jerkMagnitude > fallThreshold) {
      #if DEBUG
      Serial.println("Fall detected");
      #endif
    }

    // print sensor data
    #if DEBUG
    Serial.printf("MPU6050 - Acceleration: X: %.2f Y: %.2f Z: %.2f\n", a.acceleration.x, a.acceleration.y, a.acceleration.z);
    Serial.printf("MPU6050 - Jerk Magnitude: %.2f\n", jerkMagnitude);
    #endif

    vTaskDelay(MPU6050_SAMPLE_INTERVAL / portTICK_PERIOD_MS);
  }
}

void readMAX30102(void *pvParameters) {
  // setup buffer for reading from sensor
  uint32_t irBuffer[100]; //infrared LED sensor data
  uint32_t redBuffer[100];  //red LED sensor data

  int32_t bufferLength = 100; //data length
  int32_t spo2; //SPO2 value
  int8_t validSPO2; //indicator to show if the SPO2 calculation is valid
  int32_t heartRate; //heart rate value
  int8_t validHeartRate; //indicator to show if the heart rate calculation is valid

  bufferLength = 100; //buffer length of 100 stores 4 seconds of samples running at 25sps

  //read the first 100 samples, and determine the signal range
  for (byte i = 0 ; i < bufferLength ; i++)
  {
    // take semaphore
    // if(xSemaphoreTake(i2cSemaphore, portMAX_DELAY) != pdTRUE) {
    //   #if DEBUG
    //   Serial.println("Failed to take I2C semaphore");
    //   #endif

    //   // dont skip sample
    //   i--;
    //   vTaskDelay(5 / portTICK_PERIOD_MS);
    //   continue;
    // }

    while (max30102.available() == false) //do we have new data?
      max30102.check(); //Check the sensor for new data

    redBuffer[i] = max30102.getRed();
    irBuffer[i] = max30102.getIR();
    max30102.nextSample(); //We're finished with this sample so move to next sample

    // release semaphore
    // xSemaphoreGive(i2cSemaphore);

    // Serial.print(F("red="));
    // Serial.print(redBuffer[i], DEC);
    // Serial.print(F(", ir="));
    // Serial.println(irBuffer[i], DEC);
  }

  //calculate heart rate and SpO2 after first 100 samples (first 4 seconds of samples)
  maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);
  
  
  //Continuously taking samples from MAX30102.  Heart rate and SpO2 are calculated every 1 second
  while (1)
  {
    //dumping the first 25 sets of samples in the memory and shift the last 75 sets of samples to the top
    for (byte i = 25; i < 100; i++)
    {
      redBuffer[i - 25] = redBuffer[i];
      irBuffer[i - 25] = irBuffer[i];
    }

    //take 25 sets of samples before calculating the heart rate.
    for (byte i = 75; i < 100; i++)
    {
      // take semaphore
      // if(xSemaphoreTake(i2cSemaphore, portMAX_DELAY) != pdTRUE) {
      //   #if DEBUG
      //   Serial.println("Failed to take I2C semaphore");
      //   #endif

      //   // dont skip sample
      //   i--;
      //   vTaskDelay(5 / portTICK_PERIOD_MS);
      //   continue;
      // }

      while (max30102.available() == false) //do we have new data?
        max30102.check(); //Check the sensor for new data

      redBuffer[i] = max30102.getRed();
      irBuffer[i] = max30102.getIR();
      max30102.nextSample(); //We're finished with this sample so move to next sample

      // release semaphore
      // xSemaphoreGive(i2cSemaphore);
      // vTaskDelay(5 / portTICK_PERIOD_MS);

      //send samples and calculation result to terminal program through UART
      // Serial.print(F("red="));
      // Serial.print(redBuffer[i], DEC);
      // Serial.print(F(", ir="));
      // Serial.print(irBuffer[i], DEC);
    }

    //After gathering 25 new samples recalculate HR and SP02
    maxim_heart_rate_and_oxygen_saturation(irBuffer, bufferLength, redBuffer, &spo2, &validSPO2, &heartRate, &validHeartRate);

    if(validSPO2 && validHeartRate) {
      #if DEBUG
      Serial.print(F("MAX30102 - HR="));
      Serial.println(heartRate, DEC);
      Serial.print(F("MAX30102 - SPO2="));
      Serial.println(spo2, DEC);
      #endif
      lastBPMValue = heartRate;
      lastSpO2Value = spo2;

      if(heartRate < 40) {
        # if DEBUG
        Serial.println("MAX30102 - Heart rate too low");
        #endif
      }

      if(spo2 < 80) {
        #if DEBUG
        Serial.println("MAX30102 - SpO2 too low");
        #endif
      }
    }
  }
}

void readGPS(void *pvParameters) {
  while (1) {
    while (Serial1.available() > 0) {
      gps.encode(Serial1.read());
      if (gps.location.isValid()) {
        lastLatValue = gps.location.lat();
        lastLonValue = gps.location.lng();
        if(!firstGPSRead) firstGPSRead = true;
        #if DEBUG
        Serial.printf("GPS - Latitude: %.6f, Longitude: %.6f\n", lastLatValue, lastLonValue);
        #endif
        break;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1000)); // Delay 1 second
  }
}

void readMLX90614(void *pvParameters) {
  if(!mlx.begin()) {
    Serial.println("Failed to find MLX90614 chip");
    while (1) {vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  while (1) {
    lastTempValue = mlx.readObjectTempC();
    Serial.printf("MLX90614 - Object Temperature: %.2f\n", lastTempValue);
    vTaskDelay(pdMS_TO_TICKS(1000)); // Delay 1 second
  }
}

void readINMP441(void *pvParameters) {
  #define SAMPLE_RATE 16000

  #if DEBUG
  Serial.println("Reading INMP441 Task Started");
  #endif

  UBaseType_t uxHighWaterMark;

  // read audio data at 16kHz
  // setup I2S
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = INMP441_SCK,
    .ws_io_num = INMP441_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = INMP441_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

  #if DEBUG
  Serial.println("INMP441 - I2S setup complete");
  #endif
  
  // 1 second buffer 
  float* audioData = (float*) heap_caps_malloc(SAMPLE_RATE * sizeof(float), MALLOC_CAP_SPIRAM);
  // int32_t audioData[SAMPLE_RATE];
  size_t bytesRead;

  if(!audioData) {
    #if DEBUG
    Serial.println("Failed to allocate memory for audio data");
    #endif
    while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  #if DEBUG
  Serial.println("Allocated memory for audio data");
  Serial.println("PSRAM free: " + String(ESP.getFreePsram()));
  Serial.println("Heap free: " + String(ESP.getFreeHeap()));
  #endif

  // Create an error reporter
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Get the Model
  const tflite::Model* model = tflite::GetModel(converted_model_16kHz_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d", model->version(), TFLITE_SCHEMA_VERSION);
    while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  // Create an op resolver
  tflite::MicroMutableOpResolver<10> resolver(error_reporter);
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddLogistic();
  resolver.AddQuantize();
  resolver.AddDequantize();

  // Define the tensor arena (use PSRAM)
  constexpr int tensor_arena_size = 100 * 1024;

  uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM);
  if(tensor_arena == NULL) {
    error_reporter->Report("Failed to allocate tensor arena");
    while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  // Create the interpreter
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);

  // Allocate tensor
  TfLiteStatus allocate_status = interpreter.AllocateTensors();

  if(allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
  }

  size_t used_bytes = interpreter.arena_used_bytes();
  error_reporter->Report("Used bytes: %d\n", used_bytes);

  // Get input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Get output tensor
  TfLiteTensor* output = interpreter.output(0);

  // Get input tensor dimensions
  int input_dims = input->dims->size;
  int batch_size = input->dims->data[0];
  int n_frame = input->dims->data[1];
  int n_mfcc = input->dims->data[2];
  int channels = input->dims->data[3];
   printf("Model expects input shape: [%d, %d, %d, %d]\n", batch_size, n_frame, n_mfcc, channels);

  float mfcc_buffer[FRAME_COUNT * N_MFCC];

  while(1) {
    // read 1 second of audio data
    for(int i = 0; i < SAMPLE_RATE; i++) {
      // i2s_read(I2S_NUM_0, &audioData[i], sizeof(float), &bytesRead, portMAX_DELAY);
      audioData[i] = 1.0;
    }

    uxHighWaterMark = uxTaskGetStackHighWaterMark(NULL);
    Serial.printf("High water mark: %d\n", uxHighWaterMark);

    // extract MFCC features (32 frames, 40 coefficients)
    extract_mfcc(audioData, mfcc_buffer);

    // print MFCC
    for(int i = 0; i < FRAME_COUNT; i++) {
      Serial.printf("Frame %d: ", i);
      for(int j = 0; j < N_MFCC; j++) {
        Serial.printf("%.2f ", mfcc_buffer[i * N_MFCC + j]);
      }
      Serial.println();
    }

    // scale MFCC features
    scale_mfcc(mfcc_buffer, FRAME_COUNT, N_MFCC);
    
    // print MFCC
    for(int i = 0; i < FRAME_COUNT; i++) {
      Serial.printf("Frame %d: ", i);
      for(int j = 0; j < N_MFCC; j++) {
        Serial.printf("%.2f ", mfcc_buffer[i * N_MFCC + j]);
      }
      Serial.println();
    }


    // // copy scaled MFCC features to input tensor
    // memcpy(input->data.f, mfcc_buffer, FRAME_COUNT * N_MFCC * sizeof(float));

    // // do inference
    // if(interpreter.Invoke() != kTfLiteOk) {
    //   error_reporter->Report("Invoke failed");
    //   while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
    // }

    // // TODO: if cough detected, send notification
    // // if output value is > 0.9 then cough detected
    // float output_value = output->data.f[0];
    // if(output_value > 0.9) {
    //   #if DEBUG
    //   Serial.println("Cough detected!");
    //   #endif
    // }

    vTaskDelay(pdMS_TO_TICKS(100)); // Delay 100ms
  }
}

void updateTFTDisplay(void *pvParameters) {
  while (1) {
    tft.fillScreen(GC9A01A_BLACK); // Clear the display
    tft.setCursor(0, 20);
    tft.print("BPM: ");
    tft.print(lastBPMValue);
    tft.setCursor(0, 40);
    tft.print("SpO2: ");
    tft.print(lastSpO2Value);
    tft.setCursor(0, 60);
    tft.print("X: ");
    tft.print(lastAccelValue.acceleration.x);
    vTaskDelay(pdMS_TO_TICKS(1000)); // Update display every second
  }
}

void aggregateData(void *pvParameters) {
  sensorData_t data;
  while (1) {
    vTaskDelay(3000/portTICK_PERIOD_MS); // Delay 3 seconds

    data.bpm = lastBPMValue;
    data.spo2 = lastSpO2Value;
    data.temp = lastTempValue;
    data.lat = lastLatValue;
    data.lon = lastLonValue;

    // Check if queue is full
    if (uxQueueSpacesAvailable(sensorDataQueue) == 0) {
      // Remove oldest item from queue
      xQueueReceive(sensorDataQueue, &data, 0);
    }

    // Send data to queue
    xQueueSend(sensorDataQueue, &data, 0);
  }
}

void publishMQTTDataTask(void *pvParameters) {
  sensorData_t data;
  while (1) {
    vTaskDelay(1000/portTICK_PERIOD_MS); // Delay 1 second
    mqttClient.loop();

    if (xQueueReceive(sensorDataQueue, &data, portMAX_DELAY) == pdTRUE) {
      publishSensorData(data); // Publish sensor data to MQTT
    }
  }
}

void reconnect() {
  while(!mqttClient.connected()) {
    #if DEBUG
    Serial.print("MQTT - Attempting MQTT connection...\n");
    #endif
    if (mqttClient.connect(DEVICE_ID)) {
      #if DEBUG
      Serial.println("MQTT - connected");
      #endif
      mqttClient.subscribe(MQTT_SUBSCRIBE_TOPIC);
    } else {
      #if DEBUG
      Serial.print("MQTT - failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" trying again in 5 seconds");
      #endif
      vTaskDelay(5000/portTICK_PERIOD_MS);
    }
  }
}

void publishSensorData(const sensorData_t &data) {
  if(!mqttClient.connected()) {
    reconnect(); // Reconnect if connection is lost
  }

  // Publish sensor data to MQTT
  String payload = "{\"type\":\"data\", \"temp\":\"" + String(data.temp, 2) + 
                "\", \"spo2\":\"" + String(data.spo2, 2) + 
                "\", \"heartRate\":\"" + String(data.bpm, 2) + 
                "\", \"lat\":\"" + String(data.lat, 6) + 
                "\", \"lon\":\"" + String(data.lon, 6) + 
                "\", \"firstReadGPS\": " + (firstGPSRead ? "true" : "false") + "}";

  mqttClient.publish("guardiband/1A/data", payload.c_str());

  Serial.println("MQTT - Published sensor data to MQTT");
}

// MQTT Callback
void MQTTcallback(char* topic, byte* payload, unsigned int length) {
  // Messages will only be in form of a string
  lastMessage = "";
  for(int i = 0; i < length; i++) {
    lastMessage += (char)payload[i];
  }

  // Print message
  #if DEBUG
  Serial.print("MQTT - Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  Serial.println(lastMessage);
  #endif
}

void readButtonTask(void *pvParameters) {
  while(1) {
    if(digitalRead(BUTTON_PIN) == HIGH) {
      #if DEBUG
      Serial.println("Button pressed");
      #endif
    }
    vTaskDelay(100/portTICK_PERIOD_MS);
  }
}

void monitorWiFi(void *pvParameters) {
  #define AP_MODE_TIMEOUT 2*60 // 2 minutes

  while(1) {
    if(WiFi.status() != WL_CONNECTED) {
      #if DEBUG
      Serial.println("WiFi disconnected");
      #endif

      // read button press
      if(digitalRead(BUTTON_PIN) == HIGH) {
        #if DEBUG
        Serial.println("Button pressed. Starting AP mode");
        #endif

        // Sound buzzer for 200ms
        tone(BUZZER_POSITIVE, 440, 500);

        // start AP mode
        wm.setConfigPortalTimeout(AP_MODE_TIMEOUT);
        bool result = wm.startConfigPortal("Guardiband"); // this connects to the AP

        #if DEBUG
        Serial.printf("SSID: %s\n", wm.getWiFiSSID(true).c_str());
        if(result) {
          Serial.println("Connected to WiFi");
        }
        else {
          Serial.println("Failed to connect to WiFi");
        }
        #endif
        
        // stop AP mode
        wm.stopConfigPortal();
      }
    }
    else if (WiFi.status() == WL_CONNECTED) {
      // if button is pressed for 5 seconds than publish SOS
      int buttonPressCount = 0;
      while(digitalRead(BUTTON_PIN) == HIGH) {
        buttonPressCount++;
        vTaskDelay(1000/portTICK_PERIOD_MS);
        if(buttonPressCount == 5) {
          // publish SOS
          String payload = "{\"type\":\"alert\", \"message\":\"SOS\"}";
          mqttClient.publish(MQTT_PUBLISH_TOPIC, payload.c_str());
          break;
        }
      }

    }
    vTaskDelay(1000/portTICK_PERIOD_MS);
  }
}

// dsp functions
// Hann window function
void hann_window(float *window, int size) {
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1 - cos((2 * M_PI * i) / (size - 1)));
    }
}

// Perform FFT and calculate power spectrum
void compute_power_spectrum(const float *frame, float *power_spectrum, int size) {
    // kiss_fft_cfg cfg = kiss_fft_alloc(size, 0, NULL, NULL);
    // if(cfg == NULL) {
    //     #if DEBUG
    //     Serial.println("Failed to allocate kiss_fft_cfg");
    //     #endif
    //     while(1){vTaskDelay(10/portTICK_PERIOD_MS);}
    // }

    kiss_fft_cpx in[N_FFT], out[N_FFT];
    // kiss_fft_cpx* in = (kiss_fft_cpx*) heap_caps_malloc(size * sizeof(kiss_fft_cpx), MALLOC_CAP_SPIRAM);
    // kiss_fft_cpx* out = (kiss_fft_cpx*) heap_caps_malloc(size * sizeof(kiss_fft_cpx), MALLOC_CAP_SPIRAM);

    for (int i = 0; i < size; i++) {
        in[i].r = frame[i];
        in[i].i = 0.0;
    }

    kiss_fft(kissfft_cfg, in, out);

    for (int i = 0; i <= size / 2; i++) {
        power_spectrum[i] = (out[i].r * out[i].r) + (out[i].i * out[i].i);
    }

    // free(cfg);
}

// Apply Mel filterbank
void apply_mel_filterbank(const float *power_spectrum, float *mel_energy) {
    for (int i = 0; i < N_MELS; i++) {
        mel_energy[i] = 0.0;
        for (int j = 0; j <= N_FFT / 2; j++) {
            mel_energy[i] += mel_filterbank[i][j] * power_spectrum[j];
        }
        mel_energy[i] = logf(mel_energy[i] + 1e-10); // Log scale
    }
}

// Apply DCT to Mel energies
void apply_dct(const float *mel_energy, float *mfcc) {
    for (int i = 0; i < N_MFCC; i++) {
        mfcc[i] = 0.0;
        for (int j = 0; j < N_MELS; j++) {
            mfcc[i] += dct_matrix[i][j] * mel_energy[j];
        }
    }
}

// Main MFCC extraction function
void extract_mfcc(const float *audio_signal, float *output_mfcc) {
    float hann[N_FFT];
    float frame[N_FFT];
    float power_spectrum[(N_FFT / 2) + 1];
    float mel_energy[N_MELS];

    hann_window(hann, N_FFT);

    for (int frame_idx = 0; frame_idx < FRAME_COUNT; frame_idx++) {
        // Compute frame start index
        int frame_start = frame_idx * HOP_LENGTH - N_FFT / 2;

        // Apply Hann window
        for (int i = 0; i < N_FFT; i++) {
            if (frame_start + i < 0 || frame_start + i >= INPUT_SIGNAL_LENGTH) {
                frame[i] = 0.0; // Zero padding
            } else {
                frame[i] = audio_signal[frame_start + i] * hann[i];
            }
        }

        // Compute power spectrum
        compute_power_spectrum(frame, power_spectrum, N_FFT);

        // Apply Mel filterbank
        apply_mel_filterbank(power_spectrum, mel_energy);

        // Apply DCT to get MFCCs
        apply_dct(mel_energy, &output_mfcc[frame_idx * N_MFCC]);
    }
}

void scale_mfcc(float* mfcc_buffer, size_t frame_count, size_t n_mfcc) {
    // expect mfcc_buffer to be an array[frame_count * n_mfcc]
    for (size_t frame = 0; frame < frame_count; ++frame) {
        for (size_t mfcc = 0; mfcc < n_mfcc; ++mfcc) {
            size_t index = frame * n_mfcc + mfcc;
            // Normalize using mean and standard deviation
            mfcc_buffer[index] = (mfcc_buffer[index] - scaler_mean_PSRAM[mfcc]) / scaler_std_PSRAM[mfcc];
        }
    }
}