# INT8 TinyML Fall Detection Model

This folder contains the INT8 quantized TinyML model and supporting files for fall detection using ESP32-S3 and MPU6050.

## Files
- model_int8.tflite: INT8 quantized TensorFlow Lite model.
- model_data.h: C/C++ header file for ESP32-S3 deployment.
- normalization.npz: IMU data normalization parameters.
- config.json: Model preprocessing and inference configuration.

## Status
INT8 quantization and model export completed.
ESP32-S3 deployment and hardware testing are pending.
