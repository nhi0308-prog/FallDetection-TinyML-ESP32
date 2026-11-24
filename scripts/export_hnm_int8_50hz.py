"""Export the selected KFall CNN and compare its window decisions on validation."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np


RUN = Path("results/cnn_binary_hnm/run_ezyuiab5")
CHECKPOINT = RUN / "model/50hz/best.keras"
DATA = Path("data/processed/50hz")
THRESHOLD = 0.80


def calibration_windows():
    rng = np.random.default_rng(42)
    groups = []
    # The NPZ inputs already have the same preprocessing as the CNN training data.
    for filename, label_key, per_group in [
        ("train.npz", "y", 64),
        ("train_hard_negatives.npz", "task", 16),
    ]:
        with np.load(DATA / filename) as pack:
            windows = pack["X"]
            labels = pack[label_key].reshape(-1)
            for label in np.unique(labels):
                indices = np.flatnonzero(labels == label)
                chosen = rng.choice(indices, min(per_group, len(indices)), replace=False)
                groups.append(windows[chosen])
    selected = np.concatenate(groups).astype(np.float32)
    if selected.shape[1:] != (100, 6) or not np.isfinite(selected).all():
        raise ValueError("Invalid training windows. Expected finite (100, 6) inputs.")
    return selected


def convert(tf, model, windows):
    def representative():
        for window in windows:
            yield [window[None, ...]]

    with tempfile.TemporaryDirectory() as tmp:
        saved = str(Path(tmp) / "saved_model")
        model.export(saved)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved, signature_keys=["serving_default"])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter.convert()


def io_params(detail, shape):
    scale, zero = detail["quantization"]
    if detail["dtype"] != np.int8 or tuple(detail["shape"]) != shape or scale <= 0:
        raise ValueError(f"Unexpected tensor: {detail['name']}, {detail['shape']}, {detail['dtype']}, {scale}")
    return {"shape": list(shape), "dtype": "int8", "scale": float(scale), "zero_point": int(zero)}


def compare_windows(model, interpreter, src, dst, input_q, output_q, cutoff, path):
    with np.load(path) as pack:
        windows = pack["X"].astype(np.float32, copy=False)
    if not len(windows) or windows.shape[1:] != (100, 6) or not np.isfinite(windows).all():
        raise ValueError(f"Invalid validation inputs in {path}")
    float_scores = model.predict(windows, batch_size=256, verbose=0).reshape(-1)
    quantized_scores = np.empty(len(windows), dtype=np.int8)
    for i, window in enumerate(windows):
        raw = np.rint(window / input_q["scale"]) + input_q["zero_point"]
        interpreter.set_tensor(src["index"], np.clip(raw, -128, 127).astype(np.int8)[None, ...])
        interpreter.invoke()
        quantized_scores[i] = interpreter.get_tensor(dst["index"])[0, 0]
    restored = (quantized_scores.astype(np.float32) - output_q["zero_point"]) * output_q["scale"]
    if not np.isfinite(float_scores).all():
        raise ValueError(f"Non-finite Keras output for {path}")
    changed = np.count_nonzero((float_scores >= THRESHOLD) != (quantized_scores >= cutoff))
    return {"windows": len(windows), "decisions_changed": int(changed),
            "mean_absolute_score_difference": float(np.abs(float_scores - restored).mean())}


def main():
    import tensorflow as tf
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    calibration = calibration_windows()
    blob = convert(tf, model, calibration)
    interpreter = Interpreter(model_content=blob, num_threads=1)
    interpreter.allocate_tensors()
    src, = interpreter.get_input_details()
    dst, = interpreter.get_output_details()
    input_q = io_params(src, (1, 100, 6))
    output_q = io_params(dst, (1, 1))
    cutoff = output_q["zero_point"] + math.ceil(THRESHOLD / output_q["scale"])
    if not -128 <= cutoff <= 127:
        raise ValueError("Alarm threshold is outside the representable output range.")

    checks = {}
    files = [DATA / "validation.npz"]
    ambiguous = DATA / "validation_hard_negatives.npz"
    if ambiguous.is_file():
        files.append(ambiguous)
    for path in files:
        checks[path.name] = compare_windows(model, interpreter, src, dst, input_q, output_q, cutoff, path)

    folder = RUN / "tflite"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "cnn_hnm_50hz_int8.tflite"
    output.write_bytes(blob)
    metadata = {
        "checkpoint": CHECKPOINT.as_posix(), "model_bytes": len(blob),
        "sampling_hz": 50, "stride_samples": 10, "consecutive_windows": 2,
        "input": input_q, "output": output_q, "fall_threshold": THRESHOLD,
        "fall_int8_greater_than_or_equal": cutoff, "calibration_train_windows": len(calibration),
        "validation_window_comparison": checks,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved INT8: {output} | {len(blob):,} bytes")
    print(f"Input: int8 [1, 100, 6] | Output: int8 [1, 1] | Fall if output >= {cutoff}")
    for name, row in checks.items():
        print(f"{name}: changed at 0.80 = {row['decisions_changed']}/{row['windows']} windows | "
              f"mean absolute score difference = {row['mean_absolute_score_difference']:.5f}")


if __name__ == "__main__":
    main()
