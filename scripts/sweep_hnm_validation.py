import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from scipy.signal import butter, sosfilt, sosfilt_zi

from check_kfall import SENSOR_HEADER, SENSOR_NAME, read_labels, read_sensor

RAW = Path("data/raw/kfall")
OUT = Path("results/cnn_binary_hnm/run_ezyuiab5/threshold_sweep")
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
FALL_ID = 3


def check_label_issue(severity, code, source, detail=""):
    if severity == "error":
        raise ValueError(f"{code}: {source} {detail}")


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def replay(hz, labels):
    cfg = read_yaml(f"configs/config_{hz}hz.yaml")
    mapping = read_yaml(cfg["label_mapping"])
    split = json.loads(Path(cfg["subject_split"]).read_text(encoding="utf-8"))
    validation_subjects = set(split["validation"])
    ambiguous = set(mapping["Ambiguous"])

    root = Path(f"data/processed/{hz}hz")
    stats = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    columns = [SENSOR_HEADER.index(name) for name in stats["channels"]]
    width = cfg["window_samples"]
    stride = cfg["stride_samples"]

    cnn = tf.keras.models.load_model(
        Path("results/cnn_binary_hnm/run_ezyuiab5/model/50hz/best.keras"), compile=False
    )

    sos = None
    if hz == 50:
        filt = cfg["resampling"]
        sos = butter(
            filt["filter_order"], filt["cutoff_hz"],
            fs=cfg["source_hz"], btype="lowpass", output="sos"
        )

    rows = []
    with ZipFile(RAW / "sensor_data_new_2.zip") as archive:
        entries = sorted(archive.infolist(), key=lambda item: item.filename)
        for entry in entries:
            match = SENSOR_NAME.fullmatch(Path(entry.filename).name)
            if not match:
                continue
            subject, task, trial = map(int, match.groups())
            sid = f"SA{subject:02}"
            if sid not in validation_subjects:
                continue

            data = read_sensor(archive, entry)
            time, frames = data[:, 0], data[:, 1].astype(np.int64)
            signal = data[:, columns]
            event = labels.get((subject, task, trial))
            onset = impact = None
            if event is not None:
                onset = time[frames == event[0]].item()
                impact = time[frames == event[1]].item()

            if sos is not None:
                initial = sosfilt_zi(sos)[:, :, None] * signal[0]
                filtered, _ = sosfilt(sos, signal, axis=0, zi=initial)
                keep = frames % 2 == 0
                signal, time = filtered[keep], time[keep]

            starts = range(0, len(signal) - width + 1, stride)
            end_times = time[width - 1::stride]
            predictions = {f"p{p:.2f}": np.array([], dtype=int) for p in THRESHOLDS}
            if len(starts):
                X = np.stack([signal[i:i + width] for i in starts]).astype(np.float32)
                X -= mean
                X /= std
                scores = cnn(X, training=False).numpy()[:, 0]
                predictions = {f"p{p:.2f}": np.where(scores >= p, FALL_ID, 0) for p in THRESHOLDS}

            for name, y_pred in predictions.items():
                is_fall = y_pred == FALL_ID
                confirmed = np.zeros_like(is_fall)
                confirmed[1:] = is_fall[:-1] & is_fall[1:]
                alarms = end_times[confirmed]
                row = {
                    "model": name, "subject": sid, "task": task, "trial": trial,
                    "kind": "Fall" if event else (
                        "Ambiguous ADL" if f"T{task:02}" in ambiguous else "ADL"
                    ),
                    "windows": len(end_times),
                    "any_alarm": bool(alarms.size),
                    "pre_onset_alarm": False,
                    "early_detection": False,
                    "on_time_or_late_detection": False,
                    "lead_time_s": None,
                }
                if event is not None:
                    early = alarms[(alarms >= onset - 1e-8) & (alarms < impact - 1e-8)]
                    late = alarms[
                        (alarms >= impact - 1e-8)
                        & (alarms <= impact + cfg["fall_max_seconds_after_impact"] + 1e-8)
                    ]
                    row["pre_onset_alarm"] = bool(np.any(alarms < onset - 1e-8))
                    row["early_detection"] = bool(early.size)
                    row["on_time_or_late_detection"] = bool(late.size)
                    if early.size:
                        row["lead_time_s"] = float(impact - early[0])
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, _ = read_labels(RAW / "label_data_new_2.zip", check_label_issue)
    summary = []
    for hz in (50,):
        print(f"Replaying {hz} Hz...", flush=True)
        trials = replay(hz, labels)
        trials.to_csv(OUT / f"{hz}hz_trials.csv", index=False)

        for name, rows in trials.groupby("model"):
            fall = rows[rows["kind"].eq("Fall")]
            adl = rows[rows["kind"].eq("ADL")]
            ambiguous = rows[rows["kind"].eq("Ambiguous ADL")]
            detected = fall["early_detection"] | fall["on_time_or_late_detection"]
            early = int(fall["early_detection"].sum())
            lead_ms = float(fall["lead_time_s"].median() * 1000)
            summary.append({
                "hz": hz, "model": name,
                "fall_trials": len(fall),
                "early_falls": early,
                "early_recall": early / len(fall),
                "falls_detected_through_300ms_after_impact": int(detected.sum()),
                "lead_time_median_ms": lead_ms,
                "fall_trials_with_pre_onset_alarm": int(fall["pre_onset_alarm"].sum()),
                "adl_trials": len(adl),
                "adl_trials_with_alarm": int(adl["any_alarm"].sum()),
                "ambiguous_adl_trials": len(ambiguous),
                "ambiguous_adl_trials_with_alarm": int(ambiguous["any_alarm"].sum()),
            })
            print(
                f"{hz} Hz | {name:13} | Early={early}/{len(fall)} "
                f"({early / len(fall):.1%}) | Lead median={lead_ms:.0f} ms | "
                f"ADL alarms={adl['any_alarm'].sum()}/{len(adl)} | "
                f"Ambiguous alarms={ambiguous['any_alarm'].sum()}/{len(ambiguous)}",
                flush=True,
            )

    pd.DataFrame(summary).to_csv(OUT / "event_summary.csv", index=False)
    print(f"PASS: Saved event results to {OUT}")


if __name__ == "__main__":
    main()
