"""Audit the supplied KFall ZIP archives without unpacking or changing them."""
import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import numpy as np
from openpyxl import load_workbook

SENSOR_NAME = re.compile(r"(?:SA|S)(\d{2})T(\d{2})R(\d{2})\.csv")
LABEL_NAME = re.compile(r"SA(\d{2})_label\.xlsx")
FALL_CODE = re.compile(r"F(\d{2})\s*\((\d{2})\)")
FALL_TASKS = range(20, 35)
ADL_TASKS = (*range(1, 20), 35, 36)
SUBJECTS = set(range(6, 39)) - {34}
SENSOR_HEADER = (
    "TimeStamp(s)", "FrameCounter", "AccX", "AccY", "AccZ",
    "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ",
)
LABEL_HEADER = (
    "Task Code (Task ID)", "Description", "Trial ID",
    "Fall_onset_frame", "Fall_impact_frame",
)
NOMINAL_STEP_S = 0.010
TIMESTAMP_TOLERANCE_S = 0.001


def inspect_workbook(raw, subject, labels, source, issue):
    book = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    try:
        rows = book.active.values
        header = next(rows, ())
        if header[:5] != LABEL_HEADER or any(v is not None for v in header[5:]):
            issue("error", "label_header", source)
            return
        task = None
        for line, row in enumerate(rows, start=2):
            if all(value is None for value in row):
                continue
            location = f"{source}:{line}"
            code, _, trial, onset, impact = (row + (None,) * 5)[:5]
            if code is not None:
                match = FALL_CODE.fullmatch(str(code).strip())
                task = int(match[2]) if match else None
                if task not in FALL_TASKS or int(match[1]) != task - 19:
                    task = None
                    issue("error", "invalid_fall_task", location)
            if (
                task is None
                or any(type(value) is not int for value in (trial, onset, impact))
                or not (1 <= trial <= 99 and 0 < onset < impact)
                or any(value is not None for value in row[5:])
            ):
                issue("error", "invalid_label_row", location)
                continue
            key = (subject, task, trial)
            if key in labels:
                issue("error", "duplicate_label", location)
                continue
            labels[key] = onset, impact
    finally:
        book.close()


def read_labels(path, issue):
    labels = {}
    subjects = set()
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename).name
            if entry.is_dir() or not name.lower().endswith(".xlsx"):
                continue
            match = LABEL_NAME.fullmatch(name)
            if not match or int(match[1]) not in SUBJECTS:
                issue("error", "label_filename", entry.filename)
                continue
            subject = int(match[1])
            if subject in subjects:
                issue("error", "duplicate_label_workbook", entry.filename)
                continue
            subjects.add(subject)
            try:
                inspect_workbook(archive.read(entry), subject, labels, entry.filename, issue)
            except (ValueError, OSError, BadZipFile, KeyError, SyntaxError) as exc:
                issue("error", "unreadable_workbook", entry.filename, str(exc))
    return labels, subjects


def read_sensor(archive, entry):
    with archive.open(entry) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as stream:
        rows = csv.reader(stream, strict=True)
        if tuple(next(rows, ())) != SENSOR_HEADER:
            raise ValueError("Unexpected CSV header")
        values = [row for row in rows]
        if any(len(row) != len(SENSOR_HEADER) for row in values):
            raise ValueError("Row has missing or extra columns")
        if any(cell.strip() == "" for row in values for cell in row):
            raise ValueError("Blank numeric cell")
        data = np.asarray(values, dtype=np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Empty CSV or NaN / Inf")
    return data


def audit(sensor_path, label_path):
    issues = []

    def issue(severity, code, source, detail=""):
        issues.append(dict(severity=severity, code=code, source=source, detail=detail))

    labels, label_subjects = read_labels(label_path, issue)
    seen = set()
    matched_labels = 0
    valid_csv = 0
    steps_ms = []
    by_subject = Counter()
    by_task = Counter()
    groups = Counter()
    sensor_files = 0
    with ZipFile(sensor_path) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if entry.is_dir() or path.suffix.lower() != ".csv":
                continue
            sensor_files += 1
            source = entry.filename
            match = SENSOR_NAME.fullmatch(path.name)
            if not match:
                issue("error", "sensor_filename", source)
                continue
            subject, task, trial = key = tuple(map(int, match.groups()))
            if (
                subject not in SUBJECTS
                or task not in (*ADL_TASKS, *FALL_TASKS)
                or trial < 1
                or path.parent.name != f"SA{subject:02}"
            ):
                issue("error", "sensor_id", source)
                continue
            if key in seen:
                issue("error", "duplicate_trial", source)
                continue
            seen.add(key)
            if trial > 6:
                issue("warning", "trial_id_above_six", source)
            by_subject[f"SA{subject:02}"] += 1
            by_task[f"T{task:02}"] += 1
            is_fall = task in FALL_TASKS
            groups["fall" if is_fall else "adl"] += 1
            if is_fall and key not in labels:
                issue("error", "missing_fall_label", source)
            try:
                data = read_sensor(archive, entry)
            except (ValueError, OSError, BadZipFile, csv.Error) as exc:
                issue("error", "invalid_sensor_csv", source, str(exc))
                continue

            valid_csv += 1
            time, frame = data[:, 0], data[:, 1]
            dt = np.diff(time)
            frame_step = np.diff(frame)
            if len(frame) < 200:
                issue("warning", "short_trial", source, f"{len(frame)} samples (<200)")
            if not np.all(frame == np.floor(frame)) or frame[0] < 1:
                issue("error", "invalid_frame_counter", source)
            if np.any(frame_step <= 0) or np.any(dt <= 0):
                issue("error", "nonmonotonic_or_duplicate_sample", source)
            if np.any(frame_step > 1):
                issue("warning", "frame_gap", source, f"{np.count_nonzero(frame_step > 1)} gaps")
            irregular = np.abs(dt - NOMINAL_STEP_S) > TIMESTAMP_TOLERANCE_S
            if np.any(irregular):
                issue("warning", "irregular_timestamp", source, f"{np.count_nonzero(irregular)} steps")
            if dt.size:
                steps_ms.append(float(np.median(dt) * 1000))
            if is_fall and key in labels:
                onset, impact = labels[key]
                onset_rows = np.flatnonzero(frame == onset)
                impact_rows = np.flatnonzero(frame == impact)
                if (
                    onset_rows.size == impact_rows.size == 1
                    and time[onset_rows[0]] < time[impact_rows[0]]
                ):
                    matched_labels += 1
                else:
                    issue("error", "unresolved_fall_frames", source)

    for key in sorted(labels.keys() - seen):
        issue("error", "orphan_fall_label", f"SA{key[0]:02}T{key[1]:02}R{key[2]:02}")
    inventory = (sensor_files, groups["adl"], groups["fall"], len(by_subject), len(label_subjects))
    if inventory != (5075, 2729, 2346, 32, 32):
        issue("error", "inventory_mismatch", "archives", str(inventory))

    severity_counts = Counter(item["severity"] for item in issues)
    summary = {
        "sensor_csv_files": sensor_files,
        "numeric_csv_files": valid_csv,
        "sensor_subjects": len(by_subject),
        "label_workbooks": len(label_subjects),
        "adl_trials": groups["adl"],
        "fall_trials": groups["fall"],
        "fall_label_rows": len(labels),
        "fall_labels_matched_to_frames": matched_labels,
        "nominal_hz": 100,
        "median_trial_step_ms": round(float(np.median(steps_ms)), 6) if steps_ms else None,
        "columns": SENSOR_HEADER,
        "units": {"time": "s", "acceleration": "g", "gyroscope": "deg/s", "euler": "deg"},
        "step_tolerance_ms": TIMESTAMP_TOLERANCE_S * 1000,
        "files_by_subject": dict(sorted(by_subject.items())),
        "files_by_task": dict(sorted(by_task.items())),
        "errors": severity_counts["error"],
        "warnings": severity_counts["warning"],
        "issue_counts": dict(sorted(Counter(item["code"] for item in issues).items())),
    }
    return summary, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/raw/kfall")
    parser.add_argument("--sensor", type=Path, default=root / "sensor_data_new_2.zip")
    parser.add_argument("--labels", type=Path, default=root / "label_data_new_2.zip")
    parser.add_argument("--out", type=Path, default=Path("results/phase1"))
    args = parser.parse_args()
    try:
        summary, issues = audit(args.sensor, args.labels)
    except (OSError, BadZipFile) as exc:
        parser.exit(2, f"Cannot open KFall archive: {exc}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "kfall_summary.json"
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.out / "kfall_issues.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("severity", "code", "source", "detail"))
        writer.writeheader()
        writer.writerows(issues)
    print(
        f"CSV: {summary['numeric_csv_files']}/{summary['sensor_csv_files']} | "
        f"ADL: {summary['adl_trials']} | Fall: {summary['fall_trials']} | "
        f"Fall labels matched: {summary['fall_labels_matched_to_frames']}/{summary['fall_trials']}"
    )
    print(f"Errors: {summary['errors']} | Warnings: {summary['warnings']} | Report: {report}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
