"""Replay the KFall test split with the validation-selected INT8 alarm code."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from replay_int8_validation_50hz import CHECKPOINT, MODEL, RUN, ReplayPredictor, flag


CODE = 123
OUT = MODEL.parent / "test_q123"


def subject_numbers(values):
    return set(pd.Series(list(values), dtype=str).str.extract(r"(\d+)$")[0].astype(int))


def first_alarm_report(table, test_subjects):
    if set(table["model"]) != {f"p{CODE:.2f}"}:
        raise ValueError("Expected only the fixed INT8 code 123.")
    if subject_numbers(table["subject"].unique()) != subject_numbers(test_subjects):
        raise ValueError("Replay subject IDs do not match the configured test split.")
    if table.duplicated(["subject", "task", "trial"]).any():
        raise ValueError("Duplicate trial in the test replay.")
    tasks = table["task"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    falls = table.loc[tasks.between(20, 34)]
    nonfalls = table.loc[~tasks.between(20, 34)]
    if (len(falls), len(nonfalls)) != (366, 432):
        raise ValueError(f"Unexpected test trial counts: {len(falls)} falls, {len(nonfalls)} non-falls.")
    correct = flag(falls["early_detection"]) & ~flag(falls["pre_onset_alarm"])
    alarms = flag(nonfalls["any_alarm"])
    lead = pd.to_numeric(falls.loc[correct, "lead_time_s"])
    if lead.isna().any():
        raise ValueError("Missing lead time on a correct first-alarm trial.")
    median = f"{1000 * lead.median():.0f} ms" if len(lead) else "n/a"
    return "\n".join([
        f"TEST INT8 | code={CODE} | {len(test_subjects)} subjects | 2 consecutive windows",
        f"First alarm correct: {correct.sum()}/{len(falls)} ({100 * correct.mean():.1f}%)",
        f"Non-fall trials with alarm: {alarms.sum()}/{len(nonfalls)} ({100 * alarms.mean():.1f}%)",
        f"Median lead for correct first alarms: {median}",
    ])


def main():
    import tensorflow as tf
    import evaluate_hnm_test_p080 as replay
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    if Path(replay.OUT).resolve() != (RUN / "test_p080").resolve():
        raise ValueError("Expected the existing binary test evaluator in scripts/.")
    test_subjects = json.loads(Path("configs/subject_split.json").read_text(encoding="utf-8-sig"))["test"]
    if len(set(test_subjects)) != 5:
        raise ValueError("Expected five test subjects.")
    interpreter = Interpreter(model_path=str(MODEL), num_threads=1)
    predictor = ReplayPredictor(interpreter, np.asarray)
    scale = predictor.output_q["scale"]
    zero = predictor.output_q["zero_point"]

    def raw_codes(scores):
        return np.rint(scores / scale).astype(np.int16) + zero

    codes = np.arange(-128, 128, dtype=np.int16)
    if not np.array_equal(raw_codes((codes.astype(np.float32) - zero) * scale), codes):
        raise ValueError("Could not recover raw INT8 scores exactly.")

    def predict(windows, training=False):
        return tf.convert_to_tensor(raw_codes(predictor(windows, training=training)))

    def load_model(path, *args, **kwargs):
        if Path(path).resolve() != CHECKPOINT.resolve():
            raise ValueError(f"Unexpected checkpoint: {path}")
        return predict

    policy = {"model_file": MODEL.name, "raw_output_greater_than_or_equal": CODE,
              "consecutive_windows": 2, "sampling_hz": 50, "window_samples": 100,
              "stride_samples": 10, "selection_split": "validation"}
    (MODEL.parent / "alarm_policy_q123.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    with patch.object(replay, "OUT", OUT), patch.object(replay, "THRESHOLDS", (CODE,)), \
         patch.object(tf.keras.models, "load_model", side_effect=load_model) as loader:
        replay.main()
        if not loader.called:
            raise RuntimeError("The test replay did not call the INT8 predictor.")
    report = first_alarm_report(pd.read_csv(OUT / "50hz_trials.csv"), test_subjects)
    (OUT / "first_alarm_summary.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
