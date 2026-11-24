#include "kfall_firebase.h"
#include "kfall_wifi_config.h"

#include <ArduinoJson.h>
#include <WiFi.h>
#include <esp_crt_bundle.h>
#include <esp_http_client.h>
#include <esp_random.h>
#include <sys/time.h>

namespace {

constexpr time_t kMinimumClock = 1704067200;
constexpr uint32_t kPublishIntervalMs = 2000;
constexpr uint32_t kRetryIntervalMs = 5000;
constexpr uint32_t kWifiRetryIntervalMs = 15000;

struct ResponseBody {
  String text;
  bool overflow = false;
};

esp_err_t receiveBody(esp_http_client_event_t *event) {
  if (event->event_id != HTTP_EVENT_ON_DATA || event->data_len <= 0) {
    return ESP_OK;
  }

  auto &body = *static_cast<ResponseBody *>(event->user_data);
  if (body.overflow || body.text.length() + event->data_len > 8192 ||
      !body.text.concat(static_cast<const char *>(event->data), event->data_len)) {
    body.overflow = true;
    return ESP_FAIL;
  }
  return ESP_OK;
}

int sendHttps(const char *step, const String &url, esp_http_client_method_t method,
              const char *contentType, const String &payload,
              ResponseBody *response = nullptr) {
  esp_http_client_config_t config = {};
  config.url = url.c_str();
  config.method = method;
  config.timeout_ms = 10000;
  config.buffer_size_tx = 4096;
  config.disable_auto_redirect = true;
  config.crt_bundle_attach = esp_crt_bundle_attach;
  if (response) {
    config.event_handler = receiveBody;
    config.user_data = response;
  }

  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (!client) {
    Serial.printf("# FIREBASE %s khoi tao HTTPS that bai\n", step);
    return -1;
  }

  esp_err_t error = esp_http_client_set_header(client, "Content-Type", contentType);
  if (error == ESP_OK) {
    error = esp_http_client_set_post_field(client, payload.c_str(), payload.length());
  }
  if (error == ESP_OK) error = esp_http_client_perform(client);

  int status = -1;
  if (error == ESP_OK && (!response || !response->overflow)) {
    status = esp_http_client_get_status_code(client);
  } else if (response && response->overflow) {
    Serial.printf("# FIREBASE %s phan hoi qua lon hoac thieu bo nho\n", step);
  } else {
    Serial.printf("# FIREBASE %s loi HTTPS %s\n", step, esp_err_to_name(error));
  }
  esp_http_client_cleanup(client);
  return status;
}

String formEncode(const String &value) {
  const char hex[] = "0123456789ABCDEF";
  String encoded;
  encoded.reserve(value.length());
  for (size_t i = 0; i < value.length(); ++i) {
    const uint8_t ch = value[i];
    if ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
        (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' ||
        ch == '.' || ch == '~') {
      encoded += static_cast<char>(ch);
    } else {
      encoded += '%';
      encoded += hex[ch >> 4];
      encoded += hex[ch & 15];
    }
  }
  return encoded;
}

struct Session {
  String idToken;
  String refreshToken;
  uint32_t expiresAtMs = 0;

  bool valid() const {
    return !idToken.isEmpty() && int32_t(expiresAtMs - millis()) > 0;
  }

  bool authorize() {
    const bool refresh = !refreshToken.isEmpty();
    String url;
    String payload;
    const char *step = refresh ? "REFRESH" : "AUTH";
    const char *type;

    if (refresh) {
      type = "application/x-www-form-urlencoded";
      url = "https://securetoken.googleapis.com/v1/token?key=";
      payload = String("grant_type=refresh_token&refresh_token=") + formEncode(refreshToken);
    } else {
      type = "application/json";
      url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=";
      JsonDocument login;
      login["email"] = KF_FIREBASE_EMAIL;
      login["password"] = KF_FIREBASE_PASSWORD;
      login["returnSecureToken"] = true;
      serializeJson(login, payload);
    }
    url += KF_FIREBASE_API_KEY;

    ResponseBody response;
    const int status = sendHttps(step, url, HTTP_METHOD_POST, type, payload, &response);
    if (status > 0) Serial.printf("# FIREBASE %s HTTP %d\n", step, status);
    if (status != 200) {
      if (refresh && (status == 400 || status == 401)) refreshToken = "";
      return false;
    }

    JsonDocument result;
    if (deserializeJson(result, response.text)) return false;
    const char *token = result[refresh ? "id_token" : "idToken"] | "";
    const char *nextRefresh = result[refresh ? "refresh_token" : "refreshToken"] | "";
    const char *expires = result[refresh ? "expires_in" : "expiresIn"] | "";
    const unsigned long seconds = strtoul(expires, nullptr, 10);
    if (!token[0] || !nextRefresh[0] || seconds <= 60 || seconds > 86400) return false;

    idToken = token;
    refreshToken = nextRefresh;
    expiresAtMs = millis() + (seconds - 60) * 1000UL;
    return valid();
  }
};

const char *statusText(KFallStatus status) {
  switch (status) {
    case KFallStatus::AiError: return "AI_ERROR";
    case KFallStatus::WaitMpu: return "WAIT_MPU";
    case KFallStatus::WarmingUp: return "WARMING_UP";
    case KFallStatus::Normal: return "NORMAL";
    case KFallStatus::Fall: return "FALL";
  }
  return "UNKNOWN";
}

const char *sirenText(KFallSiren siren) {
  switch (siren) {
    case KFallSiren::Error: return "ERROR";
    case KFallSiren::Wait: return "WAIT";
    case KFallSiren::Off: return "OFF";
    case KFallSiren::Muted: return "MUTED";
    case KFallSiren::On: return "ON";
  }
  return "UNKNOWN";
}

int publish(const KFallCloudSample &sample, const char *bootId, const String &idToken) {
  JsonDocument data;
  data["deviceId"] = KF_FIREBASE_DEVICE_ID;
  data["bootId"] = bootId;
  data["sequence"] = sample.sequence;
  data["uptimeMs"] = sample.uptimeMs;
  data["status"] = statusText(sample.status);
  data["siren"] = sirenText(sample.siren);
  data["batteryAdcMv"] = sample.batteryAdcMv;
  if (sample.batteryPercent >= 0) data["batteryPercent"] = sample.batteryPercent;
  else data["batteryPercent"] = nullptr;
  if (sample.deviceTimeMs > 0) data["deviceTimeMs"] = sample.deviceTimeMs;
  else data["deviceTimeMs"] = nullptr;
  data["serverReceivedAtMs"][".sv"] = "timestamp";

  if (sample.status == KFallStatus::Fall || sample.status == KFallStatus::Normal) {
    data["fall"] = sample.status == KFallStatus::Fall;
    data["q"] = sample.q;
    data["highWindows"] = sample.highWindows;
  } else {
    data["fall"] = nullptr;
    data["q"] = nullptr;
    data["highWindows"] = nullptr;
  }

  if (sample.imuValid) {
    JsonObject imu = data["imu"].to<JsonObject>();
    imu["ax"] = sample.imu[0] / 16384.0f;
    imu["ay"] = sample.imu[1] / 16384.0f;
    imu["az"] = sample.imu[2] / 16384.0f;
    imu["gx"] = sample.imu[3] / 131.0f;
    imu["gy"] = sample.imu[4] / 131.0f;
    imu["gz"] = sample.imu[5] / 131.0f;
  } else {
    data["imu"] = nullptr;
  }

  if (sample.hasLastFall) {
    JsonObject lastFall = data["lastFall"].to<JsonObject>();
    lastFall["bootId"] = bootId;
    lastFall["sequence"] = sample.lastFallSequence;
    lastFall["uptimeMs"] = sample.lastFallUptimeMs;
    if (sample.lastFallTimeMs > 0) lastFall["deviceTimeMs"] = sample.lastFallTimeMs;
    lastFall["q"] = sample.lastFallQ;
  }

  String payload;
  if (data.overflowed() || serializeJson(data, payload) == 0) return -1;
  String url = String(KF_FIREBASE_DATABASE_URL) + "/devices/" +
               KF_FIREBASE_DEVICE_ID + "/latest.json?print=silent&auth=" + idToken;
  return sendHttps("DATABASE", url, HTTP_METHOD_PATCH, "application/json", payload);
}

int publishFallEvent(const KFallCloudSample &sample, const char *bootId,
                     const String &idToken) {
  JsonDocument event;
  event["type"] = "FALL";
  event["deviceId"] = KF_FIREBASE_DEVICE_ID;
  event["bootId"] = bootId;
  event["sequence"] = sample.lastFallSequence;
  event["uptimeMs"] = sample.lastFallUptimeMs;
  event["q"] = sample.lastFallQ;
  if (sample.lastFallTimeMs > 0) event["deviceTimeMs"] = sample.lastFallTimeMs;
  event["serverReceivedAtMs"][".sv"] = "timestamp";

  String payload;
  if (event.overflowed() || serializeJson(event, payload) == 0) return -1;

  // Deterministic key avoids duplicate fall events after a Wi-Fi retry.
  String url = String(KF_FIREBASE_DATABASE_URL) + "/devices/" +
               KF_FIREBASE_DEVICE_ID + "/events/" + bootId + "_" +
               String(sample.lastFallSequence) + ".json?print=silent&auth=" + idToken;
  return sendHttps("EVENT", url, HTTP_METHOD_PUT, "application/json", payload);
}

}  // namespace

int64_t KFallFirebase::clockMs() {
  timeval now = {};
  gettimeofday(&now, nullptr);
  if (now.tv_sec < kMinimumClock) return 0;
  return int64_t(now.tv_sec) * 1000 + now.tv_usec / 1000;
}

void KFallFirebase::begin() {
  queue_ = xQueueCreate(1, sizeof(KFallCloudSample));
  if (!queue_) {
    Serial.println("# FIREBASE khong tao duoc hang doi");
    return;
  }

  if (xTaskCreatePinnedToCore(
          [](void *context) { static_cast<KFallFirebase *>(context)->run(); },
          "kfall_firebase", 8192, this, 1, nullptr, 0) != pdPASS) {
    vQueueDelete(queue_);
    queue_ = nullptr;
    Serial.println("# FIREBASE khong tao duoc tac vu");
  }
}

void KFallFirebase::offer(const KFallCloudSample &sample) {
  if (!queue_) return;

  const bool changed = !offered_ ||
                       sample.status != lastOffered_.status ||
                       sample.siren != lastOffered_.siren ||
                       sample.hasLastFall != lastOffered_.hasLastFall ||
                       sample.lastFallSequence != lastOffered_.lastFallSequence;
  if (!changed && sample.uptimeMs - lastOffered_.uptimeMs < kPublishIntervalMs) return;

  lastOffered_ = sample;
  lastOffered_.deviceTimeMs = clockMs();
  xQueueOverwrite(queue_, &lastOffered_);
  offered_ = true;
}

void KFallFirebase::run() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(KF_WIFI_NAME, KF_WIFI_PASSWORD);
  Serial.println("# FIREBASE dang ket noi Wi-Fi");

  char bootId[17];
  const uint32_t randomHigh = esp_random();
  const uint32_t randomLow = esp_random();
  snprintf(bootId, sizeof(bootId), "%08lx%08lx",
           static_cast<unsigned long>(randomHigh), static_cast<unsigned long>(randomLow));

  Session session;
  KFallCloudSample sample = {};
  bool pending = false;
  bool wifiWasConnected = false;
  bool reportedFirstWrite = false;
  uint32_t lastWifiAttemptMs = millis();
  uint32_t retryAfterMs = millis();
  uint32_t reportedFall = 0;

  for (;;) {
    if (xQueueReceive(queue_, &sample, pdMS_TO_TICKS(250)) == pdTRUE) pending = true;

    const uint32_t now = millis();
    if (WiFi.status() != WL_CONNECTED) {
      if (wifiWasConnected) Serial.println("# FIREBASE mat Wi-Fi");
      wifiWasConnected = false;
      reportedFirstWrite = false;
      if (now - lastWifiAttemptMs >= kWifiRetryIntervalMs) {
        lastWifiAttemptMs = now;
        WiFi.reconnect();
      }
      continue;
    }

    if (!wifiWasConnected) {
      wifiWasConnected = true;
      retryAfterMs = now;
      configTzTime("ICT-7", "time.google.com", "pool.ntp.org");
      Serial.println("# FIREBASE WIFI OK");
    }

    if (!pending || clockMs() == 0 || int32_t(now - retryAfterMs) < 0) continue;
    if (!session.valid() && !session.authorize()) {
      retryAfterMs = millis() + kRetryIntervalMs;
      continue;
    }

    // Lay ban ghi moi nhat neu AI co ket qua trong luc dang nhap.
    KFallCloudSample newest;
    if (xQueueReceive(queue_, &newest, 0) == pdTRUE) sample = newest;

    const int status = publish(sample, bootId, session.idToken);
    if (status == 200 || status == 204) {
      pending = false;
      retryAfterMs = millis();
      if (!reportedFirstWrite) {
        Serial.printf("# FIREBASE DATABASE HTTP %d - da gui latest\n", status);
      }
      reportedFirstWrite = true;
      if (sample.hasLastFall && sample.lastFallSequence != reportedFall) {
        const int eventStatus = publishFallEvent(sample, bootId, session.idToken);
        if (eventStatus == 200 || eventStatus == 204) {
          reportedFall = sample.lastFallSequence;
          Serial.printf("# FIREBASE da gui lan nga seq=%lu\n",
                        static_cast<unsigned long>(reportedFall));
        } else {
          pending = true;
          retryAfterMs = millis() + kRetryIntervalMs;
        }
      }
    } else {
      if (status > 0) Serial.printf("# FIREBASE DATABASE HTTP %d\n", status);
      if (status == 401) session.expiresAtMs = millis();
      retryAfterMs = millis() + kRetryIntervalMs;
    }

  }
}
