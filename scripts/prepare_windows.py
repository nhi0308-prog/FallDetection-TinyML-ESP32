import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfilt, sosfilt_zi

from check_kfall import SENSOR_HEADER, SENSOR_NAME, read_labels, read_sensor


RAW = Path("data/raw/kfall")
CLASS_ID = {"Idle": 0, "Walking": 1, "Active": 2, "Fall": 3}


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))


def check_label_issue(severity, code, source, detail=""):
    if severity == "error":
        raise ValueError(f"{code}: {source} {detail}")


def prepare(hz, labels):
    cfg = read_yaml(f"configs/config_{hz}hz.yaml")
    mapping = read_yaml(cfg["label_mapping"])
    split = json.loads(Path(cfg["subject_split"]).read_text(encoding="utf-8"))
    owner = {
        subject: group
        for group in ("train", "validation", "test")
        for subject in split[group]
    }
    if len(owner) != sum(len(split[g]) for g in ("train", "validation", "test")):
        raise ValueError("Subject appears in more than one split")

    task_class = {
        task: name for name, tasks in mapping.items() for task in tasks
    }
    channels = cfg["channels"]
    columns = [SENSOR_HEADER.index(name) for name in channels]
    width = cfg["window_samples"]
    stride = cfg["stride_samples"]
    out = Path(f"data/processed/{hz}hz")
    out.mkdir(parents=True, exist_ok=True)

    sos = None
    if hz == 50:
        filt = cfg["resampling"]
        sos = butter(
            filt["filter_order"], filt["cutoff_hz"],
            fs=cfg["source_hz"], btype="lowpass", output="sos"
        )

    mean = std = None
    with ZipFile(RAW / "sensor_data_new_2.zip") as archive:
        entries = sorted(archive.infolist(), key=lambda item: item.filename)

        # Process train first so validation and test reuse its statistics.
        for group in ("train", "validation", "test"):
            windows, targets, records = [], [], []
            for entry in entries:
                match = SENSOR_NAME.fullmatch(Path(entry.filename).name)
                if not match:
                    continue
                subject, task, trial = map(int, match.groups())
                sid = f"SA{subject:02}"
                name = task_class[f"T{task:02}"]
                if owner[sid] != group or name == "Ambiguous":
                    continue

                data = read_sensor(archive, entry)
                times = data[:, 0]
                frames = data[:, 1].astype(np.int64)
                signal = data[:, columns]

                # Resolve labels on the original timeline, including odd frames.
                onset_time = impact_time = None
                if name == "Fall":
                    onset, impact = labels[(subject, task, trial)]
                    onset_time = times[frames == onset].item()
                    impact_time = times[frames == impact].item()

                if sos is not None:
                    initial = sosfilt_zi(sos)[:, :, None] * signal[0]
                    filtered, _ = sosfilt(sos, signal, axis=0, zi=initial)
                    keep = frames % 2 == 0
                    signal, times, frames = filtered[keep], times[keep], frames[keep]

                signal = signal.astype(np.float32)
                for start in range(0, len(signal) - width + 1, stride):
                    end = start + width - 1
                    t = times[end]
                    if name == "Fall":
                        earliest = onset_time + cfg["fall_min_seconds_after_onset"]
                        latest = impact_time + cfg["fall_max_seconds_after_impact"]
                        if t < earliest - 1e-8 or t > latest + 1e-8:
                            continue

                    windows.append(signal[start:end + 1])
                    targets.append(CLASS_ID[name])
                    records.append((
                        sid, task, trial, name, frames[start], frames[end],
                        t, onset_time, impact_time
                    ))

            if not windows:
                raise ValueError(f"No windows found for {hz} Hz / {group}")

            X = np.stack(windows)
            y = np.asarray(targets, dtype=np.int8)
            del windows

            if group == "train":
                mean = X.mean(axis=(0, 1), dtype=np.float64)
                std = X.std(axis=(0, 1), dtype=np.float64)
                if np.any(std <= 0) or not np.isfinite(std).all():
                    raise ValueError("Invalid training standard deviation")
                stats = {
                    "sampling_hz": hz, "channels": channels, "class_id": CLASS_ID,
                    "mean": mean.tolist(), "std": std.tolist(),
                    "fit_on": "train windows"
                }
                (out / "normalization.json").write_text(
                    json.dumps(stats, indent=2), encoding="utf-8"
                )

            X -= mean.astype(np.float32)
            X /= std.astype(np.float32)
            if not np.isfinite(X).all():
                raise ValueError(f"Non-finite output in {hz} Hz / {group}")

            np.savez(out / f"{group}.npz", X=X, y=y)
            pd.DataFrame(records, columns=[
                "subject", "task", "trial", "class_name", "first_frame",
                "last_frame", "end_time_s", "onset_time_s", "impact_time_s"
            ]).to_csv(out / f"{group}_windows.csv", index=False)

            counts = np.bincount(y, minlength=len(CLASS_ID))
            print(f"{hz} Hz | {group} | X={X.shape} | "
                  f"Idle/Walking/Active/Fall={counts.tolist()}", flush=True)
            del X, y, records


if __name__ == "__main__":
    labels, _ = read_labels(RAW / "label_data_new_2.zip", check_label_issue)
    for hz in (50, 100):
        print(f"Processing {hz} Hz...", flush=True)
        prepare(hz, labels)
    print("PASS: saved all 6 datasets")
