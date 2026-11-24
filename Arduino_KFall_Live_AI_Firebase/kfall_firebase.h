#pragma once

#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

enum class KFallStatus { AiError, WaitMpu, WarmingUp, Normal, Fall };
enum class KFallSiren { Error, Wait, Off, Muted, On };

struct KFallCloudSample {
  uint32_t sequence;
  uint32_t uptimeMs;
  int64_t deviceTimeMs;
  int16_t imu[6];
  bool imuValid;
  KFallStatus status;
  KFallSiren siren;
  int8_t q;
  uint8_t highWindows;
  uint32_t batteryAdcMv;
  int batteryPercent;
  bool hasLastFall;
  uint32_t lastFallSequence;
  uint32_t lastFallUptimeMs;
  int64_t lastFallTimeMs;
  int8_t lastFallQ;
};

class KFallFirebase {
 public:
  void begin();
  void offer(const KFallCloudSample &sample);
  static int64_t clockMs();

 private:
  void run();

  QueueHandle_t queue_ = nullptr;
  bool offered_ = false;
  KFallCloudSample lastOffered_ = {};
};
