"""Compare first alarms from the exported INT8 CNN with the saved float replay."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from export_hnm_int8_50hz import CHECKPOINT, RUN, THRESHOLD, io_params


MODEL = RUN / "tflite/cnn_hnm_50hz_int8.tflite"
REFERENCE = RUN / "threshold_sweep"
OUTPUT = RUN / "tflite/validation_p080"


class ReplayPredictor:
    """Provide INT8 model scores to the original replay's Keras calling interface."""

    def __init__(self, interpreter, to_tensor):
        self.interpreter = interpreter
        self.to_tensor = to_tensor
        interpreter.allocate_tensors()
        self.src, = interpreter.get_input_details()
        self.dst, = interpreter.get_output_details()
        self.input_q = io_params(self.src, (1, 100, 6))
        self.output_q = io_params(self.dst, (1, 1))
        codes = np.arange(-128, 128)
        scores = (codes - self.output_q["zero_point"]) * self.output_q["scale"]
        if not np.array_equal(scores >= THRESHOLD, codes >= 77):
            raise ValueError("This model's output cutoff does not match the exported cutoff 77.")

    def __call__(self, X, training=False):
        windows = np.asarray(X, dtype=np.float32)
        if training or windows.shape[1:] != (100, 6) or not np.isfinite(windows).all():
            raise ValueError("Expected finite preprocessed 50 Hz inference windows.")
        raw = np.rint(windows / self.input_q["scale"]) + self.input_q["zero_point"]
        encoded = np.clip(raw, -128, 127).astype(np.int8)
        codes = np.empty((len(windows), 1), dtype=np.int8)
        for i in range(len(windows)):
            self.interpreter.set_tensor(self.src["index"], encoded[i:i + 1])
            self.interpreter.invoke()
            codes[i, 0] = self.interpreter.get_tensor(self.dst["index"])[0, 0]
        scores = (codes.astype(np.float32) - self.output_q["zero_point"]) * self.output_q["scale"]
        return self.to_tensor(scores)


def flag(values):
    lookup = {"1": True, "1.0": True, "true": True, "false": False, "0": False, "0.0": False}
    return values.astype(str).str.strip().str.lower().map(lookup.__getitem__)


def decisions(folder):
    table = pd.read_csv(folder / "50hz_trials.csv")
    rows = table.loc[table["model"].eq("p0.80")].set_index(
        ["subject", "task", "trial"], verify_integrity=True).sort_index()
    task_ids = rows.index.get_level_values("task").astype(str).str.extract(r"(\d+)$")[0].astype(int)
    is_fall = task_ids.between(20, 34).to_numpy()
    falls, negatives = rows.loc[is_fall], rows.loc[~is_fall]
    first_ok = flag(falls["early_detection"]) & ~flag(falls["pre_onset_alarm"])
    alarms = flag(negatives["any_alarm"])
    leads = pd.to_numeric(falls.loc[first_ok, "lead_time_s"], errors="raise")
    if leads.isna().any():
        raise ValueError("Missing lead time on a correct first-alarm trial.")
    return first_ok, alarms, 1000 * leads.median()


def compare(reference, candidate):
    f0, a0, lead0 = decisions(reference)
    f1, a1, lead1 = decisions(candidate)
    if not f0.index.equals(f1.index) or not a0.index.equals(a1.index):
        raise ValueError("Float and INT8 reports cover different trials.")
    if (len(f0), len(a0), f0.sum(), a0.sum()) != (371, 420, 299, 30):
        raise ValueError("The float reference does not match the selected validation run.")
    lines = ["Validation | 50 Hz | p0.80 | 2 consecutive windows"]
    for name, first, alarm, lead in [("FP32", f0, a0, lead0), ("INT8", f1, a1, lead1)]:
        lines.append(f"{name}: first correct={first.sum()}/{len(first)} | "
                     f"non-fall alarms={alarm.sum()}/{len(alarm)} | median lead={lead:.0f} ms")
    lines.append(f"FP32 -> INT8: first correct gained={(~f0 & f1).sum()}, "
                 f"lost={(f0 & ~f1).sum()} | non-fall alarms added={(~a0 & a1).sum()}, "
                 f"removed={(a0 & ~a1).sum()}")
    return "\n".join(lines)


def main():
    import tensorflow as tf
    import sweep_hnm_validation as replay
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    if Path(replay.OUT).resolve() != REFERENCE.resolve():
        raise ValueError("Expected the HNM validation sweep as the float reference.")
    predictor = ReplayPredictor(Interpreter(model_path=str(MODEL), num_threads=1), tf.convert_to_tensor)

    def load_int8(path, *args, **kwargs):
        if Path(path).resolve() != CHECKPOINT.resolve():
            raise ValueError(f"Unexpected float checkpoint in replay: {path}")
        return predictor

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Reuse the complete local evaluator, including subject selection, filtering,
    # window timestamps and alarm confirmation. Substitute only the predictor.
    with patch.object(replay, "THRESHOLDS", (THRESHOLD,)), patch.object(replay, "OUT", OUTPUT), \
         patch.object(tf.keras.models, "load_model", side_effect=load_int8) as loader:
        replay.main()
        if not loader.called:
            raise RuntimeError("Replay did not load the INT8 predictor.")
    report = compare(REFERENCE, OUTPUT)
    (OUTPUT / "first_alarm_comparison.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
