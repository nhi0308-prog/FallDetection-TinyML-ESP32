import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import yaml
from scipy.signal import butter, sosfilt, sosfilt_zi

from check_kfall import SENSOR_HEADER, SENSOR_NAME, read_sensor

cfg = yaml.safe_load(
    Path("configs/config_50hz.yaml").read_text(encoding="utf-8-sig")
)
mapping = yaml.safe_load(
    Path(cfg["label_mapping"]).read_text(encoding="utf-8-sig")
)
split = json.loads(Path(cfg["subject_split"]).read_text(encoding="utf-8"))
validation_subjects = set(split["validation"])
hard_tasks = set(mapping["Ambiguous"])

stats = json.loads(
    Path("data/processed/50hz/normalization.json").read_text(encoding="utf-8")
)
mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)
columns = [SENSOR_HEADER.index(name) for name in stats["channels"]]
width = cfg["window_samples"]
stride = cfg["stride_samples"]

filt = cfg["resampling"]
sos = butter(
    filt["filter_order"], filt["cutoff_hz"],
    fs=cfg["source_hz"], btype="lowpass", output="sos",
)
initial_state = sosfilt_zi(sos)[:, :, None]

windows, subjects, task_ids, trial_ids = [], [], [], []
counts = Counter()

with ZipFile("data/raw/kfall/sensor_data_new_2.zip") as archive:
    for entry in sorted(archive.infolist(), key=lambda item: item.filename):
        match = SENSOR_NAME.fullmatch(Path(entry.filename).name)
        if not match:
            continue
        subject, task, trial = map(int, match.groups())
        sid, tid = f"SA{subject:02}", f"T{task:02}"
        if sid not in validation_subjects or tid not in hard_tasks:
            continue

        data = read_sensor(archive, entry)
        signal = data[:, columns]
        frames = data[:, 1].astype(np.int64)
        filtered, _ = sosfilt(
            sos, signal, axis=0, zi=initial_state * signal[0]
        )
        X = filtered[frames % 2 == 0].astype(np.float32)
        X -= mean
        X /= std

        for start in range(0, len(X) - width + 1, stride):
            windows.append(X[start:start + width])
            subjects.append(sid)
            task_ids.append(tid)
            trial_ids.append(trial)
            counts[tid] += 1

if set(counts) != hard_tasks:
    raise SystemExit(f"Missing windows for tasks: {sorted(hard_tasks - counts.keys())}")

X = np.stack(windows)
if not np.isfinite(X).all():
    raise SystemExit("Non-finite values in hard-negative windows")

path = Path("data/processed/50hz/validation_hard_negatives.npz")
np.savez(
    path, X=X, y=np.zeros(len(X), dtype=np.int8),
    subject=np.asarray(subjects), task=np.asarray(task_ids),
    trial=np.asarray(trial_ids),
)
for task, count in sorted(counts.items()):
    print(f"{task}: {count} windows")
print(f"PASS: X={X.shape} | subjects={len(set(subjects))} | Saved: {path}")
