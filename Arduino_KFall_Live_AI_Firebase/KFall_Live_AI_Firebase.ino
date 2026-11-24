#include <Arduino.h>
#include <Wire.h>
#include <Arduino_GFX_Library.h>
#include <ESP_I2S.h>
#include <Chirale_TensorFlowLite.h>

#include <atomic>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kfall_model.h"
#include "kfall_firebase.h"

// Sửa tên bên phải nếu tên mảng trong kfall_model.h khác g_model.
#define KF_MODEL_BYTES kFallModel
// ============================================================
// CẤU HÌNH MẠCH ESP32-S3 CỦA BẠN
// ============================================================

constexpr int KF_LCD_MOSI = 41;
constexpr int KF_LCD_SCK  = 42;
constexpr int KF_LCD_RST  = 45;
constexpr int KF_LCD_DC   = 40;
constexpr int KF_LCD_CS   = 21;
constexpr int KF_LCD_BL   = 20;

constexpr int KF_MPU_SDA = 18;
constexpr int KF_MPU_SCL = 17;
constexpr uint8_t KF_MPU_ADDR = 0x68;

constexpr int KF_SPK_BCLK = 15;
constexpr int KF_SPK_LRCK = 16;
constexpr int KF_SPK_DATA = 7;

constexpr int KF_MUTE_BUTTON = 39;  // VOL-
constexpr int KF_BATTERY_ADC = 8;

// Khi chưa biết hệ số cầu chia áp, LCD hiện điện áp ADC tính bằng mV.
// Sau khi đo: hệ số = điện áp tại pin / điện áp tại GPIO8.
constexpr float KF_BATTERY_RATIO = 0.0f;

constexpr int KF_FALL_THRESHOLD = 123;
constexpr int KF_WINDOW_SIZE = 100;
constexpr int KF_CHANNELS = 6;
constexpr int KF_STRIDE = 10;

// ============================================================
// LỌC 100 Hz -> 50 Hz, CHUẨN HÓA, LƯỢNG TỬ HÓA INT8
// ============================================================

class KFallInputFilter {
 public:
  void reset() {
    first = true;
    keep = false;
  }

  bool push(const double input[6], int8_t output[6]) {
    static const double b0[4] = {
      0.00227184001245132, 1.0, 1.0, 1.0
    };

    static const double a1[4] = {
      -0.31976390227156365,
      -0.3451210393572459,
      -0.4043722885190848,
      -0.5213092656370014
    };

    static const double a2[4] = {
      0.034777724501378934,
      0.11683514382542745,
      0.3085762138649155,
      0.6869922209018303
    };

    static const double zi[4][2] = {
      {0.010437508768339772,  0.0018298377819610315},
      {0.053166595210408105,  0.005012723389943531},
      {0.22554479482500805,  -0.024049564234422484},
      {0.708579261183793,    -0.3955714820856232}
    };

    static const float mean[6] = {
       0.00159385919f, -0.891194807f, 0.0893367117f,
      -0.160377858f,   20.5182899f,   0.263027221f
    };

    static const float sd[6] = {
       0.209556588f, 0.477947881f, 0.389179823f,
      28.7183518f,  60.0817872f,  19.2848815f
    };

    for (int ch = 0; ch < 6; ch++) {
      double value = input[ch];

      for (int stage = 0; stage < 4; stage++) {
        double *z = state[ch][stage];

        if (first) {
          z[0] = zi[stage][0] * input[ch];
          z[1] = zi[stage][1] * input[ch];
        }

        double u = b0[stage] * value;
        double y = u + z[0];

        z[0] = 2.0 * u - a1[stage] * y + z[1];
        z[1] = u - a2[stage] * y;
        value = y;
      }

      if (keep) {
        float normalized = (float(value) - mean[ch]) / sd[ch];
        float q = nearbyintf(normalized / 0.16863128542900085f) - 1.0f;
        output[ch] = int8_t(fmaxf(-128.0f, fminf(127.0f, q)));
      }
    }

    bool emitted = keep;
    first = false;
    keep = !keep;
    return emitted;
  }

 private:
  double state[6][4][2] = {};
  bool first = true;
  bool keep = false;
};

// ============================================================
// THIẾT BỊ KFALL
// ============================================================

class KFallWearable {
 private:
  struct Sample {
    uint32_t seq;
    uint32_t timeUs;
    uint32_t generation;
    int16_t raw[6];
  };

  Arduino_ESP32SPI bus{
    KF_LCD_DC, KF_LCD_CS, KF_LCD_SCK, KF_LCD_MOSI, GFX_NOT_DEFINED
  };

  Arduino_ST7789 lcd{
    &bus, KF_LCD_RST, 0, true, 240, 240, 0, 0
  };

  I2SClass speaker;
  KFallInputFilter filter;
  KFallFirebase firebase;
  KFallCloudSample cloudSample = {};
  QueueHandle_t sampleQueue = nullptr;

  std::atomic<bool> sensorReady{false};
  std::atomic<bool> playSound{false};
  std::atomic<uint32_t> sensorGeneration{0};

  bool lcdReady = false;
  bool aiReady = false;
  bool speakerReady = false;
  bool predictionValid = false;
  bool fall = false;
  bool muted = false;

  bool lastButtonRaw = false;
  bool buttonStable = false;
  uint32_t buttonChangedMs = 0;

  uint32_t activeGeneration = 0;
  uint32_t lastSequence = 0;
  uint32_t lastSampleUs = 0;
  uint32_t lastBatteryMs = 0;
  uint32_t lastLcdMs = 0;

  int8_t ring[KF_WINDOW_SIZE][KF_CHANNELS] = {};
  int nextRow = 0;
  int rows = 0;
  int sinceInference = 0;
  int consecutiveHigh = 0;

  alignas(16) uint8_t arena[96 * 1024];
  tflite::MicroInterpreter *interpreter = nullptr;

  uint32_t batteryMv = 0;
  int batteryPercent = -1;
  char displayed[3][40] = {};

  // -------------------- AI --------------------

  bool beginAI() {
    const auto *model = tflite::GetModel(KF_MODEL_BYTES);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("# AI schema mismatch");
      return false;
    }

    static tflite::MicroMutableOpResolver<7> ops;

    if (ops.AddExpandDims() != kTfLiteOk ||
        ops.AddConv2D() != kTfLiteOk ||
        ops.AddReshape() != kTfLiteOk ||
        ops.AddMaxPool2D() != kTfLiteOk ||
        ops.AddMean() != kTfLiteOk ||
        ops.AddFullyConnected() != kTfLiteOk ||
        ops.AddLogistic() != kTfLiteOk) {
      return false;
    }

    static tflite::MicroInterpreter engine(
      model, ops, arena, sizeof(arena)
    );

    interpreter = &engine;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("# AllocateTensors failed");
      return false;
    }

    auto *input = interpreter->input(0);
    auto *output = interpreter->output(0);

    return input->type == kTfLiteInt8 &&
           input->bytes == KF_WINDOW_SIZE * KF_CHANNELS &&
           output->type == kTfLiteInt8 &&
           output->bytes == 1 &&
           fabsf(input->params.scale - 0.1686312854f) < 0.000001f &&
           input->params.zero_point == -1;
  }

  // -------------------- MPU --------------------

  bool writeReg(uint8_t reg, uint8_t value) {
    Wire.beginTransmission(KF_MPU_ADDR);
    Wire.write(reg);
    Wire.write(value);
    return Wire.endTransmission() == 0;
  }

  bool readRegs(uint8_t reg, uint8_t *dest, size_t count) {
    Wire.beginTransmission(KF_MPU_ADDR);
    Wire.write(reg);

    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom(KF_MPU_ADDR, count, true) != count) return false;

    for (size_t i = 0; i < count; i++) {
      dest[i] = Wire.read();
    }

    return true;
  }

  bool beginMPU() {
    uint8_t who = 0;

    if (!readRegs(0x75, &who, 1)) return false;
    if (who != 0x68 && who != 0x74) {
      Serial.printf("# Unknown MPU WHO_AM_I=0x%02X\n", who);
      return false;
    }

    if (!writeReg(0x6B, 0x01)) return false;
    delay(100);

    if (!writeReg(0x6C, 0x00) ||
        !writeReg(0x1A, 0x03) ||
        !writeReg(0x19, 9) ||       // 100 Hz
        !writeReg(0x1B, 0x00) ||    // Gyro ±250 dps
        !writeReg(0x1C, 0x00)) {    // Accel ±2 g
      return false;
    }

    if (who == 0x74 && !writeReg(0x1D, 0x03)) return false;
    if (!writeReg(0x38, 0x01)) return false;

    Serial.printf("# MPU READY WHO_AM_I=0x%02X\n", who);
    return true;
  }

  void sensorTask() {
    uint32_t sequence = 0;

    for (;;) {
      if (!beginMPU()) {
        sensorReady = false;
        Serial.println("# MPU init failed");
        delay(1500);
        continue;
      }

      uint32_t generation = ++sensorGeneration;
      sensorReady = true;
      uint32_t previousUs = micros();

      for (;;) {
        delay(1);

        uint8_t ready = 0;
        uint8_t data[14];

        if (!readRegs(0x3A, &ready, 1)) break;
        if (micros() - previousUs > 100000) break;
        if (!(ready & 1)) continue;
        if (!readRegs(0x3B, data, sizeof(data))) break;

        Sample sample = {};
        sample.seq = ++sequence;
        sample.timeUs = micros();

        // Nếu mất mẫu, bắt đầu lại cửa sổ để tránh ghép sai thời gian.
        if (sample.timeUs - previousUs > 16000) {
          generation = ++sensorGeneration;
        }

        sample.generation = generation;
        previousUs = sample.timeUs;

        for (int ch = 0; ch < 6; ch++) {
          // Bỏ qua hai byte nhiệt độ ở giữa accel và gyro.
          int index = ch * 2 + (ch >= 3 ? 2 : 0);
          uint16_t word = (uint16_t(data[index]) << 8) | data[index + 1];
          sample.raw[ch] = int16_t(word);
        }

        // Nếu hàng đợi đầy, phía xử lý nhận ra khoảng trống qua số thứ tự.
        xQueueSend(sampleQueue, &sample, 0);
      }

      sensorReady = false;
      Serial.println("# MPU read failed");
      delay(300);
    }
  }

  // -------------------- CỬA SỔ VÀ DỰ ĐOÁN --------------------

  void resetWindow() {
    filter.reset();
    rows = 0;
    nextRow = 0;
    sinceInference = 0;
    consecutiveHigh = 0;
    predictionValid = false;
    playSound = false;
  }

  void reportSiren() {
    const char *state;

    if (!speakerReady) state = "ERROR";
    else if (!predictionValid) state = "WAIT";
    else if (!fall) state = "OFF";
    else if (muted) state = "MUTED";
    else state = "ON";

    Serial.printf("# SIREN %s\n", state);
  }

  void processSample(const Sample &sample) {
    if (!sensorReady.load() ||
        sample.generation != sensorGeneration.load()) {
      return;
    }

    if (sample.seq != lastSequence + 1 ||
        sample.generation != activeGeneration) {
      resetWindow();
    }

    lastSequence = sample.seq;
    activeGeneration = sample.generation;
    lastSampleUs = sample.timeUs;
    cloudSample.sequence = sample.seq;
    memcpy(cloudSample.imu, sample.raw, sizeof(cloudSample.imu));

    double physical[6];

    for (int ch = 0; ch < 6; ch++) {
      physical[ch] = sample.raw[ch] / (ch < 3 ? 16384.0 : 131.0);
    }

    if (!filter.push(physical, ring[nextRow])) return;

    nextRow = (nextRow + 1) % KF_WINDOW_SIZE;

    Serial.printf(
      "%lu,%lu,%d,%d,%d,%d,%d,%d\n",
      (unsigned long)sample.seq,
      (unsigned long)sample.timeUs,
      sample.raw[0], sample.raw[1], sample.raw[2],
      sample.raw[3], sample.raw[4], sample.raw[5]
    );

    if (rows < KF_WINDOW_SIZE) {
      rows++;
      if (rows < KF_WINDOW_SIZE) return;
    } else {
      sinceInference++;
      if (sinceInference < KF_STRIDE) return;
    }

    sinceInference = 0;
    if (!aiReady) return;

    int8_t *dest = interpreter->input(0)->data.int8;

    for (int i = 0; i < KF_WINDOW_SIZE; i++) {
      int index = (nextRow + i) % KF_WINDOW_SIZE;
      memcpy(dest + i * KF_CHANNELS, ring[index], KF_CHANNELS);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
      resetWindow();
      Serial.println("# AI INVOKE FAILED");
      reportSiren();
      return;
    }

    int q = interpreter->output(0)->data.int8[0];
    const bool wasFalling = predictionValid && fall;

    if (q >= KF_FALL_THRESHOLD) {
      if (consecutiveHigh < 2) consecutiveHigh++;
      if (consecutiveHigh == 2) fall = true;
    } else {
      consecutiveHigh = 0;
      fall = false;
      muted = false;  // Cho phép còi kêu ở lần ngã tiếp theo.
    }

    cloudSample.q = q;
    cloudSample.highWindows = consecutiveHigh;
    if (fall && !wasFalling) {
      cloudSample.hasLastFall = true;
      cloudSample.lastFallSequence = sample.seq;
      cloudSample.lastFallUptimeMs = millis();
      cloudSample.lastFallTimeMs = KFallFirebase::clockMs();
      cloudSample.lastFallQ = q;
    }
    predictionValid = true;

    Serial.printf(
      "# AI seq=%lu q=%d high=%d/2 alarm=%s\n",
      (unsigned long)sample.seq,
      q,
      consecutiveHigh,
      fall ? "YES" : "NO"
    );

    reportSiren();
  }

  // -------------------- LOA VÀ NÚT VOL- --------------------

  void soundTask() {
    constexpr int framesPerBlock = 160;
    int16_t stereo[framesPerBlock * 2];
    unsigned phase = 0;

    for (;;) {
      bool on = playSound.load();
      if (!on) phase = 0;

      for (int i = 0; i < framesPerBlock; i++) {
        int16_t value = 0;

        if (on) {
          // Tiếng 1 kHz: kêu 200 ms, nghỉ 400 ms.
          if (phase < 3200) {
            value = (phase % 16 < 8) ? 2000 : -2000;
          }
          phase = (phase + 1) % 9600;
        }

        stereo[2 * i] = value;
        stereo[2 * i + 1] = value;
      }

      size_t written = speaker.write(
        reinterpret_cast<uint8_t *>(stereo), sizeof(stereo)
      );

      if (written == 0) delay(10);
    }
  }

  void readMuteButton() {
    bool down = digitalRead(KF_MUTE_BUTTON) == LOW;
    uint32_t now = millis();

    if (down != lastButtonRaw) {
      lastButtonRaw = down;
      buttonChangedMs = now;
    }

    if (down == buttonStable || now - buttonChangedMs < 35) return;

    buttonStable = down;

    if (down) {
      Serial.println("# BUTTON VOL-");

      if (fall) {
        muted = true;
        playSound = false;
        reportSiren();
      }
    }
  }

  // -------------------- PIN VÀ NGÀY --------------------

  void readBattery() {
    batteryMv = analogReadMilliVolts(KF_BATTERY_ADC);

    if (KF_BATTERY_RATIO > 0.0f) {
      float volts = batteryMv * KF_BATTERY_RATIO / 1000.0f;

      // Ước tính tuyến tính cho hiển thị sau khi hiệu chuẩn cầu chia áp.
      batteryPercent = constrain(
        (int)lroundf((volts - 3.3f) * 100.0f / 0.9f), 0, 100
      );

      Serial.printf(
        "# BAT GPIO8=%lu mV | %.3f V | uoc tinh=%d%%\n",
        (unsigned long)batteryMv, volts, batteryPercent
      );
    } else {
      Serial.printf("# BAT GPIO8=%lu mV\n", (unsigned long)batteryMv);
    }
  }

  // -------------------- LCD --------------------

  void drawField(int row, const char *value, uint16_t color) {
    if (strcmp(displayed[row], value) == 0) return;

    snprintf(displayed[row], sizeof(displayed[row]), "%s", value);

    int y = 26 + row * 77;
    lcd.fillRect(8, y, 224, 30, 0x0000);
    lcd.setCursor(10, y + 2);
    lcd.setTextSize(2);
    lcd.setTextColor(color);
    lcd.print(value);
  }

  void updateLCD() {
    if (!lcdReady) return;

    char text[40];

    if (batteryPercent >= 0) {
      snprintf(text, sizeof(text), "~%d%%", batteryPercent);
    } else {
      snprintf(text, sizeof(text), "%lu mV (ADC)", (unsigned long)batteryMv);
    }
    drawField(0, text, 0xFFFF);

    time_t now = time(nullptr);
    struct tm local = {};

    if (now >= 1704067200 && localtime_r(&now, &local)) {
      strftime(text, sizeof(text), "%d/%m/%Y", &local);
    } else {
      snprintf(text, sizeof(text), "Chua dong bo");
    }
    drawField(1, text, 0xFFFF);

    if (!aiReady) {
      drawField(2, "LOI AI", 0xF800);
    } else if (!sensorReady.load()) {
      drawField(2, "CHO MPU", 0xFFE0);
    } else if (!predictionValid) {
      drawField(2, "DANG LAY MAU", 0xFFE0);
    } else if (fall) {
      drawField(2, "NGA", 0xF800);
    } else {
      drawField(2, "BINH THUONG", 0x07E0);
    }
  }

  void offerCloudSample() {
    const bool mpuReady = sensorReady.load();
    if (!aiReady) cloudSample.status = KFallStatus::AiError;
    else if (!mpuReady) cloudSample.status = KFallStatus::WaitMpu;
    else if (!predictionValid) cloudSample.status = KFallStatus::WarmingUp;
    else cloudSample.status = fall ? KFallStatus::Fall : KFallStatus::Normal;

    if (!speakerReady) cloudSample.siren = KFallSiren::Error;
    else if (!predictionValid) cloudSample.siren = KFallSiren::Wait;
    else if (!fall) cloudSample.siren = KFallSiren::Off;
    else cloudSample.siren = muted ? KFallSiren::Muted : KFallSiren::On;

    cloudSample.imuValid = mpuReady && cloudSample.sequence != 0 &&
                           activeGeneration == sensorGeneration.load() &&
                           micros() - lastSampleUs <= 100000;
    cloudSample.uptimeMs = millis();
    cloudSample.batteryAdcMv = batteryMv;
    cloudSample.batteryPercent = batteryPercent;
    firebase.offer(cloudSample);
  }

 public:
  void begin() {
    Serial.begin(115200);

    pinMode(KF_MUTE_BUTTON, INPUT_PULLUP);
    buttonStable = lastButtonRaw = digitalRead(KF_MUTE_BUTTON) == LOW;

    pinMode(KF_LCD_BL, OUTPUT);
    digitalWrite(KF_LCD_BL, HIGH);

    lcdReady = lcd.begin();

    if (lcdReady) {
      lcd.fillScreen(0x0000);
      lcd.setTextSize(1);
      lcd.setTextColor(0x07FF);

      const char *labels[3] = {"PIN", "NGAY", "TRANG THAI"};

      for (int i = 0; i < 3; i++) {
        lcd.setCursor(10, 10 + i * 77);
        lcd.print(labels[i]);
      }
    }

    // The Firebase worker synchronizes time through NTP after Wi-Fi connects.
    setenv("TZ", "ICT-7", 1);
    tzset();

    aiReady = beginAI();
    Serial.println(aiReady ? "# AI READY" : "# AI INIT FAILED");

    speaker.setPins(KF_SPK_BCLK, KF_SPK_LRCK, KF_SPK_DATA);

    if (speaker.begin(
          I2S_MODE_STD,
          16000,
          I2S_DATA_BIT_WIDTH_16BIT,
          I2S_SLOT_MODE_STEREO)) {
      speakerReady = xTaskCreate(
        [](void *p) { static_cast<KFallWearable *>(p)->soundTask(); },
        "kfall_sound", 4096, this, 1, nullptr
      ) == pdPASS;
    }

    if (!speakerReady) Serial.println("# SIREN ERROR");

    sampleQueue = xQueueCreate(32, sizeof(Sample));

    if (sampleQueue && Wire.begin(KF_MPU_SDA, KF_MPU_SCL, 100000)) {
      Wire.setTimeOut(30);

      if (xTaskCreatePinnedToCore(
            [](void *p) { static_cast<KFallWearable *>(p)->sensorTask(); },
            "kfall_mpu", 4096, this, 3, nullptr, 0) != pdPASS) {
        Serial.println("# Sensor task failed");
      }
    } else {
      Serial.println("# MPU setup failed");
    }

    readBattery();
    updateLCD();
    firebase.begin();
  }

  void tick() {
    readMuteButton();
    Sample sample;

    for (int i = 0;
         sampleQueue && i < 16 &&
         xQueueReceive(sampleQueue, &sample, 0) == pdTRUE;
         i++) {
      if (micros() - sample.timeUs <= 100000) {
        processSample(sample);
      }
    }

    if (predictionValid &&
        (!sensorReady.load() ||
         activeGeneration != sensorGeneration.load() ||
         micros() - lastSampleUs > 100000)) {
      resetWindow();
      Serial.println("# No fresh MPU");
      reportSiren();
    }

    playSound = speakerReady && predictionValid && fall && !muted;

    if (millis() - lastBatteryMs >= 2000) {
      lastBatteryMs = millis();
      readBattery();
    }

    if (millis() - lastLcdMs >= 200) {
      lastLcdMs = millis();
      updateLCD();
    }

    offerCloudSample();
    delay(1);
  }
};

static KFallWearable wearable;

void setup() {
  wearable.begin();
}

void loop() {
  wearable.tick();
}
