import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["Idle", "Walking", "Active", "Fall"]
CLASS_IDS = list(range(len(CLASSES)))


def extract_features(windows, mean, std):
    imu = windows * std + mean
    acc_magnitude = np.linalg.norm(imu[:, :, :3], axis=2)
    gyro_magnitude = np.linalg.norm(imu[:, :, 3:], axis=2)
    signal = np.concatenate(
        (imu, acc_magnitude[:, :, None], gyro_magnitude[:, :, None]),
        axis=2,
    )
    squared = np.square(signal)
    minimum = signal.min(axis=1)
    maximum = signal.max(axis=1)
    mean_square = squared.mean(axis=1)
    return np.concatenate(
        (
            signal.mean(axis=1),
            signal.std(axis=1),
            minimum,
            maximum,
            np.median(signal, axis=1),
            np.sqrt(mean_square),
            signal.var(axis=1),
            mean_square,
            np.abs(signal).max(axis=1),
            maximum - minimum,
        ),
        axis=1,
    )


def load_split(root, split, mean, std):
    with np.load(root / f"{split}.npz") as data:
        windows = data["X"]
        labels = data["y"]
    # Work in batches to limit temporary memory use.
    features = [
        extract_features(windows[start:start + 1024], mean, std)
        for start in range(0, len(windows), 1024)
    ]
    return np.concatenate(features), labels


def run(hz):
    root = Path(f"data/processed/{hz}hz")
    out = Path(f"results/baseline/{hz}hz")
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    X_train, y_train = load_split(root, "train", mean, std)
    X_val, y_val = load_split(root, "validation", mean, std)
    print(f"{hz} Hz | Train features: {X_train.shape} | Validation: {X_val.shape}")

    models = {
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, class_weight="balanced", cache_size=512),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        started = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - started
        predicted = model.predict(X_val)
        report = classification_report(
            y_val, predicted, labels=CLASS_IDS, target_names=CLASSES,
            output_dict=True, zero_division=0,
        )
        metrics = {
            "model": name,
            "sampling_hz": hz,
            "split": "validation",
            "train_seconds": train_seconds,
            "parameters": model.get_params(deep=False),
            "class_order": CLASSES,
            "report": report,
            "confusion_matrix": confusion_matrix(
                y_val, predicted, labels=CLASS_IDS
            ).tolist(),
        }
        if name == "svm":
            metrics["parameters"] = model.named_steps["svc"].get_params()
        (out / f"{name}_validation.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        joblib.dump(model, out / f"{name}.joblib")
        np.savez(out / f"{name}_validation_predictions.npz", y_true=y_val, y_pred=predicted)
        print(
            f"{hz} Hz | {name} | Accuracy={report['accuracy']:.4f} | "
            f"Macro F1={report['macro avg']['f1-score']:.4f} | "
            f"Fall Recall={report['Fall']['recall']:.4f} | "
            f"Train time={train_seconds:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, choices=(50, 100), required=True)
    run(parser.parse_args().hz)
