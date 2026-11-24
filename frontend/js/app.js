import { initializeApp } from "https://www.gstatic.com/firebasejs/13.4.0/firebase-app.js";
import {
  getDatabase, ref, onValue, onChildAdded, query, limitToLast
} from "https://www.gstatic.com/firebasejs/13.4.0/firebase-database.js";
import { firebaseConfig, deviceId } from "./firebase-config.js";

"use strict";

const $ = id => document.getElementById(id);
const axes = ["ax", "ay", "az", "gx", "gy", "gz"];
const scores = [];
const eventHistory = [];

let mode = "idle";
let firebaseUnsubscribe = [];
let firebaseStarted = false;
let firebaseApp = null;
const loadedEventIds = new Set();
let lastAI = 0;
let lastRaw = 0;
let lastBattery = 0;
let latestRaw = null;
let oldFall = false;
let oldSiren = "";
let graphDirty = true;
let fallCount = 0;
let muteCount = 0;
let overThresholdCount = 0;

const pageMeta = {
  overview: {
    title: "Tổng quan hệ thống",
    sub: "Chuyển động, cảnh báo và tình trạng thiết bị trong một màn hình"
  },
  ai: {
    title: "Phân tích suy luận AI",
    sub: "Theo dõi điểm số lượng tử INT8 và quy tắc xác nhận hai cửa sổ liên tiếp"
  },
  imu: {
    title: "Dữ liệu cảm biến IMU 6 trục",
    sub: "Giám sát chi tiết gia tốc tuyến tính (g) và vận tốc góc (°/s) theo từng trục"
  },
  logs: {
    title: "Nhật ký cảnh báo & sự kiện",
    sub: "Lịch sử phát hiện té ngã, khôi phục trạng thái và thao tác nút nhấn"
  }
};

const clock = new Intl.DateTimeFormat("vi-VN", {
  timeZone: "Asia/Ho_Chi_Minh",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false
});

const calendar = new Intl.DateTimeFormat("vi-VN", {
  timeZone: "Asia/Ho_Chi_Minh",
  weekday: "long",
  day: "2-digit",
  month: "2-digit",
  year: "numeric"
});

function switchView(viewName) {
  if (!pageMeta[viewName]) return;

  document.querySelectorAll(".view-page").forEach(sec => {
    sec.classList.toggle("active", sec.id === "view-" + viewName);
  });

  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });

  if ($("pageTitle")) $("pageTitle").textContent = pageMeta[viewName].title;
  if ($("pageSub")) $("pageSub").textContent = pageMeta[viewName].sub;
  graphDirty = true;
}

function showStatus(text, style = "") {
  $("status").textContent = text;
  $("statusCard").className = "card status-card" + (style ? " " + style : "");
}

function clearAI(text) {
  lastAI = 0;
  $("score").textContent = "—";
  $("high").textContent = "—";
  $("siren").textContent = "Còi · Chưa có dữ liệu";
  $("updated").textContent = "Chờ kết quả AI mới";
  if ($("aiCurScore")) $("aiCurScore").textContent = "—";
  showStatus(text);
}

function clearSensors() {
  latestRaw = null;
  lastRaw = 0;
  for (const id of axes) {
    $(id).textContent = "—";
    const imuEl = $("imu_" + id);
    if (imuEl) imuEl.textContent = id.startsWith("a") ? "— g" : "— °/s";
    const barEl = $("bar_" + id);
    if (barEl) barEl.style.width = "0%";
  }
  if ($("accMag")) $("accMag").textContent = "— g";
  if ($("gyroMag")) $("gyroMag").textContent = "— °/s";
  if ($("accBar")) $("accBar").style.width = "0%";
  if ($("gyroBar")) $("gyroBar").style.width = "0%";
  if ($("sensorBadge")) $("sensorBadge").textContent = "Chờ tín hiệu";
  if ($("imuAccTotal")) $("imuAccTotal").textContent = "|A| = — g";
  if ($("imuGyroTotal")) $("imuGyroTotal").textContent = "|G| = — °/s";
  if ($("rawAcc")) $("rawAcc").textContent = "—";
  if ($("scaledAcc")) $("scaledAcc").textContent = "—";
  if ($("rawGyro")) $("rawGyro").textContent = "—";
  if ($("scaledGyro")) $("scaledGyro").textContent = "—";
}

function clearBattery() {
  lastBattery = 0;
  $("battery").textContent = "—";
  $("batteryInfo").textContent = "Chờ dữ liệu GPIO8";
  const fill = $("batFill");
  if (fill) {
    fill.style.width = "0%";
    fill.style.backgroundColor = "#1760b4";
  }
}

function renderLogs() {
  const lists = [$("events"), $("eventsFull")];
  for (const list of lists) {
    if (!list) continue;
    if (eventHistory.length === 0) {
      list.innerHTML = '<li class="empty-log">Chưa có cảnh báo trong phiên</li>';
    } else {
      list.replaceChildren();
      for (const entry of eventHistory) {
        const li = document.createElement("li");
        li.textContent = entry.time + " · " + entry.text;
        list.appendChild(li);
      }
    }
  }

  if ($("navLogCount")) $("navLogCount").textContent = String(eventHistory.length);
  if ($("logTotalCount")) $("logTotalCount").textContent = String(eventHistory.length);
  if ($("logFallCount")) $("logFallCount").textContent = String(fallCount);
  if ($("logMuteCount")) $("logMuteCount").textContent = String(muteCount);
}

function addEvent(text, type = "info") {
  eventHistory.unshift({
    time: clock.format(new Date()),
    text,
    type
  });
  if (eventHistory.length > 30) eventHistory.pop();

  if (type === "fall") fallCount++;
  if (type === "mute") muteCount++;
  renderLogs();
}

function updateAIStats() {
  if (scores.length === 0) {
    if ($("aiMaxScore")) $("aiMaxScore").textContent = "—";
    if ($("aiAvgScore")) $("aiAvgScore").textContent = "—";
    if ($("aiOverCount")) $("aiOverCount").textContent = "0";
    return;
  }
  const maxVal = Math.max(...scores);
  const avgVal = (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1);
  if ($("aiMaxScore")) $("aiMaxScore").textContent = String(maxVal);
  if ($("aiAvgScore")) $("aiAvgScore").textContent = String(avgVal);
  if ($("aiOverCount")) $("aiOverCount").textContent = String(overThresholdCount);
}

function startSession() {
  scores.length = 0;
  eventHistory.length = 0;
  graphDirty = true;
  oldFall = false;
  oldSiren = "";
  fallCount = 0;
  muteCount = 0;
  overThresholdCount = 0;
  renderLogs();
  updateAIStats();
  clearAI("Đang lấy mẫu");
  clearSensors();
  clearBattery();
}

function setConnection(next) {
  mode = next;
  document.body.dataset.conn = next;

  const labels = {
    idle: "Chưa kết nối Firebase",
    opening: "Đang kết nối Firebase",
    live: "ESP32-S3 · Wi-Fi/Firebase",
    closing: "Đang ngắt Firebase"
  };

  $("connection").textContent = labels[next];
  $("connect").disabled = next === "opening" || next === "closing";
  $("connect").textContent = next === "live" ? "Ngắt Firebase" : "Kết nối Firebase";
}

function parseLine(line) {
  let m = line.match(/^# AI seq=(\d+) q=(-?\d+) high=(\d+)\/2 alarm=(YES|NO)/);

  if (m) {
    const q = Number(m[2]);
    if (q < -128 || q > 127) return;

    const fall = m[4] === "YES";
    lastAI = Date.now();

    $("score").textContent = String(q);
    $("high").textContent = m[3] + "/2";
    $("updated").textContent =
      "Mẫu cuối cửa sổ #" + m[1] + " · " + clock.format(new Date());
    if ($("aiCurScore")) $("aiCurScore").textContent = String(q);

    showStatus(
      fall ? "Phát hiện ngã" : "Bình thường",
      fall ? "fall" : "normal"
    );

    if (q >= 123) overThresholdCount++;
    scores.push(q);
    if (scores.length > 100) scores.shift();
    updateAIStats();
    graphDirty = true;

    if (fall && !oldFall) addEvent("Phát hiện ngã", "fall");
    if (!fall && oldFall) addEvent("Trở về bình thường", "normal");
    oldFall = fall;
    return;
  }

  m = line.match(/^# SIREN (ON|OFF|MUTED|WAIT|ERROR)$/);

  if (m) {
    const state = m[1];
    const labels = {
      ON: "Đang kêu",
      OFF: "Sẵn sàng",
      MUTED: "Đã tắt tiếng bằng VOL−",
      WAIT: "Chờ AI",
      ERROR: "Lỗi khởi tạo loa"
    };

    $("siren").textContent = "Còi · " + labels[state];

    if (state === "MUTED" && oldSiren !== "MUTED") {
      addEvent("Đã nhấn VOL− để tắt tiếng", "mute");
    }

    oldSiren = state;
    return;
  }

  m = line.match(/^# BAT GPIO8=(\d+) mV(.*)$/);

  if (m) {
    lastBattery = Date.now();
    const pct = m[2].match(/uoc tinh=(\d+)%/);
    const fill = $("batFill");

    if (pct) {
      const val = Math.max(0, Math.min(100, Number(pct[1])));
      $("battery").textContent = val + "%";
      $("batteryInfo").textContent = "Ước tính theo điện áp · ADC " + m[1] + " mV";
      if (fill) {
        fill.style.width = val + "%";
        fill.style.backgroundColor = val <= 20 ? "#a82145" : (val <= 40 ? "#d97706" : "#146b4c");
      }
    } else {
      $("battery").textContent = m[1] + " mV";
      $("batteryInfo").textContent = "Điện áp tại GPIO8 · Chưa hiệu chuẩn % pin";
      if (fill) {
        fill.style.width = "100%";
        fill.style.backgroundColor = "#1760b4";
      }
    }
    return;
  }

  if (/^\d+,\d+(,-?\d+){6}$/.test(line)) {
    latestRaw = line.split(",").slice(2).map(Number);
    lastRaw = Date.now();
    return;
  }

  if (/^# (MPU (init|read|setup) failed|No fresh MPU|Unknown MPU|Sensor task failed)/i.test(line)) {
    clearAI("Mất tín hiệu cảm biến");
    clearSensors();
    $("message").textContent = line;
    return;
  }

  if (/^# (AI INIT FAILED|AI INVOKE FAILED|AllocateTensors failed|AI schema mismatch)/i.test(line)) {
    clearAI("Lỗi mô hình AI");
    $("message").textContent = line;
    return;
  }

  if (/^# (MPU READY|AI READY)/.test(line)) {
    clearAI("Đang lấy mẫu");
    $("message").textContent = "Đang nhận dữ liệu từ ESP32.";
  }
}

function renderCanvas(canvasId) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;

  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);

  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.font = "12px Segoe UI, Arial, sans-serif";

  const left = 38, right = w - 8;
  const y = q => 12 + (127 - q) * (h - 26) / 255;

  for (const level of [-128, 0, 123]) {
    ctx.strokeStyle = level === 123 ? "#b55436" : "#dce5e0";
    ctx.setLineDash(level === 123 ? [5, 4] : []);
    ctx.lineWidth = level === 123 ? 1.3 : 1;
    ctx.beginPath();
    ctx.moveTo(left, y(level));
    ctx.lineTo(right, y(level));
    ctx.stroke();

    ctx.fillStyle = level === 123 ? "#a82145" : "#536962";
    ctx.fillText(String(level), 1, y(level) + 4);
  }

  if (scores.length > 0) {
    ctx.setLineDash([]);
    ctx.strokeStyle = "#096c55";
    ctx.lineWidth = 2;
    ctx.beginPath();

    scores.forEach((q, i) => {
      const x = left + i * (right - left) / 99;
      if (i === 0) ctx.moveTo(x, y(q));
      else ctx.lineTo(x, y(q));
    });

    ctx.stroke();
  }
}

function drawChart() {
  renderCanvas("chart");
  renderCanvas("chartDetail");
}

function hasFirebaseConfig() {
  return firebaseConfig.apiKey && !firebaseConfig.apiKey.startsWith("PASTE_") &&
    firebaseConfig.databaseURL && !firebaseConfig.databaseURL.includes("YOUR_PROJECT");
}

function updateFromFirebase(data) {
  if (!data) return;
  const q = Number(data.q);
  const fall = data.fall === true;

  if (Number.isFinite(q)) {
    lastAI = Date.now();
    $("score").textContent = String(q);
    $("high").textContent = `${Number(data.highWindows ?? 0)}/2`;
    $("updated").textContent = `Bản ghi #${data.sequence ?? "—"} · ${clock.format(new Date())}`;
    if ($("aiCurScore")) $("aiCurScore").textContent = String(q);
    showStatus(fall ? "Phát hiện ngã" : "Bình thường", fall ? "fall" : "normal");
    scores.push(q);
    if (scores.length > 100) scores.shift();
    if (q >= 123) overThresholdCount++;
    updateAIStats();
    graphDirty = true;
  } else {
    const labels = { AI_ERROR: "Lỗi mô hình AI", WAIT_MPU: "Chờ cảm biến", WARMING_UP: "Đang lấy mẫu" };
    clearAI(labels[data.status] ?? "Chờ dữ liệu AI");
  }

  const sirenLabels = { ON: "Đang kêu", OFF: "Sẵn sàng", MUTED: "Đã tắt tiếng bằng VOL−", WAIT: "Chờ AI", ERROR: "Lỗi khởi tạo loa" };
  const siren = data.siren ?? "WAIT";
  $("siren").textContent = "Còi · " + (sirenLabels[siren] ?? siren);
  if (siren === "MUTED" && oldSiren !== "MUTED") addEvent("Đã nhấn VOL− để tắt tiếng", "mute");
  oldSiren = siren;

  if (Number.isFinite(Number(data.batteryAdcMv))) {
    lastBattery = Date.now();
    const percent = Number(data.batteryPercent);
    const fill = $("batFill");
    if (Number.isFinite(percent)) {
      const value = Math.max(0, Math.min(100, percent));
      $("battery").textContent = `${value}%`;
      $("batteryInfo").textContent = `Pin thiết bị · ADC ${data.batteryAdcMv} mV`;
      if (fill) { fill.style.width = `${value}%`; fill.style.backgroundColor = value <= 20 ? "#a82145" : value <= 40 ? "#d97706" : "#146b4c"; }
    } else {
      $("battery").textContent = `${data.batteryAdcMv} mV`;
      $("batteryInfo").textContent = "Điện áp tại GPIO8 · Chưa hiệu chuẩn % pin";
    }
  }

  const imu = data.imu;
  if (imu && axes.every(id => Number.isFinite(Number(imu[id])))) {
    latestRaw = [imu.ax * 16384, imu.ay * 16384, imu.az * 16384, imu.gx * 131, imu.gy * 131, imu.gz * 131];
    lastRaw = Date.now();
  }
}

function connectFirebase() {
  if (!hasFirebaseConfig()) {
    $("message").textContent = "Hãy điền Firebase Web config trong js/firebase-config.js trước khi kết nối.";
    return;
  }
  if (firebaseStarted) return;
  firebaseStarted = true;
  startSession();
  setConnection("opening");

  firebaseApp ??= initializeApp(firebaseConfig);
  const db = getDatabase(firebaseApp);
  const devicePath = `devices/${deviceId}`;

  firebaseUnsubscribe.push(onValue(ref(db, `${devicePath}/latest`), snapshot => {
    const data = snapshot.val();
    if (!data) return;
    updateFromFirebase(data);
    setConnection("live");
    $("message").textContent = "Đang nhận dữ liệu ESP32-S3 qua Wi-Fi và Firebase.";
  }, error => {
    firebaseStarted = false;
    setConnection("idle");
    $("message").textContent = `Không đọc được Firebase: ${error.message}`;
  }));

  firebaseUnsubscribe.push(onChildAdded(query(ref(db, `${devicePath}/events`), limitToLast(30)), snapshot => {
    if (loadedEventIds.has(snapshot.key)) return;
    loadedEventIds.add(snapshot.key);
    const event = snapshot.val();
    if (event?.type === "FALL") addEvent(`Phát hiện ngã · q = ${event.q ?? "—"}`, "fall");
  }));
}

function disconnectFirebase() {
  setConnection("closing");
  firebaseUnsubscribe.forEach(stop => stop());
  firebaseUnsubscribe = [];
  firebaseStarted = false;
  loadedEventIds.clear();
  setConnection("idle");
  clearAI("Chờ Firebase");
  clearSensors();
  clearBattery();
  $("message").textContent = "Đã ngắt theo dõi Firebase.";
}

$("connect").addEventListener("click", () => {
  if (mode === "live" || mode === "opening") disconnectFirebase();
  else connectFirebase();
});

// Chuyển đổi giữa 4 trang riêng biệt
document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    switchView(btn.dataset.view);
  });
});

document.querySelectorAll("[data-jump]").forEach(btn => {
  btn.addEventListener("click", () => {
    switchView(btn.dataset.jump);
  });
});

if ($("clearLogBtn")) {
  $("clearLogBtn").addEventListener("click", () => {
    eventHistory.length = 0;
    fallCount = 0;
    muteCount = 0;
    renderLogs();
  });
}

if ($("exportLogBtn")) {
  $("exportLogBtn").addEventListener("click", () => {
    const rows = ["Thoi_gian,Su_kien", ...eventHistory.map(e => `"${e.time}","${e.text}"`)];
    const blob = new Blob(["\uFEFF" + rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "KFall_NhatKyCanhBao.csv";
    a.click();
    URL.revokeObjectURL(url);
  });
}

setInterval(() => {
  const now = Date.now();

  if (lastAI && now - lastAI > 2500) clearAI("Chờ tín hiệu AI");
  if (lastRaw && now - lastRaw > 2500) clearSensors();
  if (lastBattery && now - lastBattery > 6500) clearBattery();

  if (latestRaw) {
    const scaled = latestRaw.map((value, i) => value / (i < 3 ? 16384 : 131));
    scaled.forEach((val, i) => {
      const id = axes[i];
      $(id).textContent = val.toFixed(2);

      const imuEl = $("imu_" + id);
      if (imuEl) imuEl.textContent = val.toFixed(2) + (i < 3 ? " g" : " °/s");

      const barEl = $("bar_" + id);
      if (barEl) {
        const maxRange = i < 3 ? 4 : 250;
        barEl.style.width = Math.min(100, (Math.abs(val) / maxRange) * 100) + "%";
      }
    });

    const aMag = Math.hypot(scaled[0], scaled[1], scaled[2]);
    const gMag = Math.hypot(scaled[3], scaled[4], scaled[5]);

    if ($("accMag")) $("accMag").textContent = aMag.toFixed(2) + " g";
    if ($("gyroMag")) $("gyroMag").textContent = gMag.toFixed(1) + " °/s";
    if ($("accBar")) $("accBar").style.width = Math.min(100, (aMag / 4) * 100) + "%";
    if ($("gyroBar")) $("gyroBar").style.width = Math.min(100, (gMag / 250) * 100) + "%";
    if ($("sensorBadge")) $("sensorBadge").textContent = "Đang cập nhật";

    if ($("imuAccTotal")) $("imuAccTotal").textContent = "|A| = " + aMag.toFixed(2) + " g";
    if ($("imuGyroTotal")) $("imuGyroTotal").textContent = "|G| = " + gMag.toFixed(1) + " °/s";
    if ($("rawAcc")) $("rawAcc").textContent = latestRaw.slice(0, 3).join(" / ");
    if ($("scaledAcc")) $("scaledAcc").textContent = scaled.slice(0, 3).map(v => v.toFixed(2) + " g").join(" / ");
    if ($("rawGyro")) $("rawGyro").textContent = latestRaw.slice(3, 6).join(" / ");
    if ($("scaledGyro")) $("scaledGyro").textContent = scaled.slice(3, 6).map(v => v.toFixed(1) + " °/s").join(" / ");

    latestRaw = null;
  }

  if (graphDirty) {
    drawChart();
    graphDirty = false;
  }
}, 150);

function updateDate() {
  $("date").textContent = calendar.format(new Date());
}

updateDate();
setInterval(updateDate, 60000);
window.addEventListener("resize", () => { graphDirty = true; });
