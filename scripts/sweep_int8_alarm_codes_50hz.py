"""Compare integer alarm cutoffs on the five KFall validation subjects."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from replay_int8_validation_50hz import (
    CHECKPOINT, MODEL, OUTPUT as ORIGINAL_INT8, REFERENCE, ReplayPredictor, decisions, flag,
)


CODES = tuple(range(77, 128))
OUT = MODEL.parent / "validation_alarm_codes"


def collect_trials():
    import tensorflow as tf
    import sweep_hnm_validation as replay
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        Interpreter = tf.lite.Interpreter

    if Path(replay.OUT).resolve() != REFERENCE.resolve():
        raise ValueError("Expected scripts/sweep_hnm_validation.py from the selected validation run.")
    predictor = ReplayPredictor(Interpreter(model_path=str(MODEL), num_threads=1), np.asarray)
    scale = predictor.output_q["scale"]
    zero = predictor.output_q["zero_point"]

    def encode(scores):
        return np.rint(scores / scale).astype(np.int16) + zero

    # ReplayPredictor returns dequantized INT8 scores. Check the inverse once
    # over every possible output code before using it for integer comparisons.
    all_codes = np.arange(-128, 128, dtype=np.int16)
    if not np.array_equal(encode((all_codes.astype(np.float32) - zero) * scale), all_codes):
        raise ValueError("Output quantization did not round-trip all INT8 codes.")

    def predict_codes(windows, training=False):
        return tf.convert_to_tensor(encode(predictor(windows, training=training)))

    def load_model(path, *args, **kwargs):
        if Path(path).resolve() != CHECKPOINT.resolve():
            raise ValueError(f"Unexpected checkpoint: {path}")
        return predict_codes

    OUT.mkdir(parents=True, exist_ok=True)
    with patch.object(replay, "OUT", OUT), patch.object(replay, "THRESHOLDS", CODES), \
         patch.object(tf.keras.models, "load_model", side_effect=load_model) as loader:
        replay.main()
        if not loader.called:
            raise RuntimeError("The validation replay did not use the INT8 predictor.")


def summarize(trials, reference):
    base_first, base_alarm, base_lead = reference
    if (len(base_first), len(base_alarm), base_first.sum(), base_alarm.sum()) != (371, 420, 282, 36):
        raise ValueError("Expected the completed INT8 validation run at code 77.")
    rows = []
    for label, group in trials.groupby("model"):
        # The existing replay labels thresholds p77.00, p78.00, ... .
        code = int(str(label).removeprefix("p").split(".")[0])
        group = group.set_index(["subject", "task", "trial"], verify_integrity=True).sort_index()
        if len(group) != len(base_first) + len(base_alarm):
            raise ValueError(f"Trial count changed at code {code}.")
        falls = group.loc[base_first.index]
        negatives = group.loc[base_alarm.index]
        first = flag(falls["early_detection"]) & ~flag(falls["pre_onset_alarm"])
        alarms = flag(negatives["any_alarm"])
        leads = pd.to_numeric(falls.loc[first, "lead_time_s"])
        if leads.isna().any():
            raise ValueError(f"Missing lead times at code {code}.")
        lead_ms = leads.median() * 1000 if len(leads) else np.nan
        if code == 77 and not (first.equals(base_first) and alarms.equals(base_alarm)
                               and np.isclose(lead_ms, base_lead)):
            raise ValueError("Code 77 differs from the original INT8 replay.")
        rows.append(dict(code=code, first_correct=int(first.sum()), nonfall_alarms=int(alarms.sum()),
                         median_lead_ms=lead_ms))
    table = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    if table.code.tolist() != list(CODES) or (table.nonfall_alarms.diff() > 0).any():
        raise ValueError("Incomplete sweep or inconsistent non-fall alarm counts.")
    return table


def display(row):
    return (f"code={int(row.code)} | first correct={int(row.first_correct)}/371 | "
            f"non-fall alarms={int(row.nonfall_alarms)}/420 | median lead={row.median_lead_ms:.0f} ms")


def best_under(table, limit):
    eligible = table[table.nonfall_alarms <= limit]
    if eligible.empty:
        return "no qualifying cutoff"
    ranked = eligible.sort_values(["first_correct", "nonfall_alarms", "median_lead_ms", "code"],
                                  ascending=[False, True, False, True])
    return display(ranked.iloc[0])


def main():
    reference = decisions(ORIGINAL_INT8)
    collect_trials()
    table = summarize(pd.read_csv(OUT / "50hz_trials.csv"), reference)
    table.to_csv(OUT / "first_alarms_by_code.csv", index=False)
    print("\nINT8 validation | 50 Hz | 2 consecutive windows")
    print("Current: " + display(table.iloc[0]))
    print("Best with <=30 alarms (FP32 reference): " + best_under(table, 30))
    print("Best with <=12 alarms (<3% of 420): " + best_under(table, 12))


if __name__ == "__main__":
    main()
