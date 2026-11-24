# KFall Firebase Dashboard

This static dashboard reads the newest KFall wearable state from Firebase Realtime Database. It does not use Web Serial or a USB cable.

## Configuration

1. Create or select a Firebase Web App.
2. Copy its public web configuration into `js/firebase-config.js`.
3. Set `deviceId` to the value used by the ESP32 firmware, for example `kfall001`.
4. Configure Realtime Database rules so the dashboard can read `/devices/<deviceId>/latest` and `/devices/<deviceId>/events`.

Keep Wi-Fi passwords and the Firebase Authentication password for the ESP32 outside this frontend folder.

## Run locally

Open the `frontend` folder in VS Code and use Live Server. The dashboard must be served over HTTP, for example `http://127.0.0.1:5500`, rather than opened directly through `file://`.

## Firebase data paths

```text
/devices/<deviceId>/latest
/devices/<deviceId>/events/<bootId>_<sequence>
```

The dashboard shows AI score, fall state, siren state, battery voltage, IMU values, and recent fall events.
