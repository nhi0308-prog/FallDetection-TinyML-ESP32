import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from train_baselines import CLASSES, CLASS_IDS, load_split


def evaluate(hz, name, y_true, y_pred, out):
    report = classification_report(
        y_true, y_pred,
        labels=CLASS_IDS, target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    result = {
        "model": name,
        "sampling_hz": hz,
        "split": "test",
        "class_order": CLASSES,
        "report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CLASS_IDS
        ).tolist(),
    }
    (out / f"{hz}hz_{name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    np.savez(out / f"{hz}hz_{name}_predictions.npz", y_true=y_true, y_pred=y_pred)
    print(
        f"{hz:3} Hz | {name:13} | "
        f"Accuracy={report['accuracy']:.4f} | "
        f"Macro F1={report['macro avg']['f1-score']:.4f} | "
        f"Fall Precision={report['Fall']['precision']:.4f} | "
        f"Fall Recall={report['Fall']['recall']:.4f}",
        flush=True,
    )


def main():
    out = Path("results/evaluation/test")
    out.mkdir(parents=True, exist_ok=True)

    for hz in (50, 100):
        root = Path(f"data/processed/{hz}hz")
        stats = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        features, y_test = load_split(root, "test", mean, std)

        for name in ("svm", "random_forest"):
            model = joblib.load(Path(f"results/baseline/{hz}hz/{name}.joblib"))
            evaluate(hz, name, y_test, model.predict(features), out)

        cnn = tf.keras.models.load_model(
            Path(f"results/cnn/{hz}hz/best.keras"), compile=False
        )
        with np.load(root / "test.npz") as data:
            probabilities = cnn.predict(data["X"], batch_size=256, verbose=0)
        evaluate(hz, "cnn", y_test, probabilities.argmax(axis=1), out)

    print(f"PASS: Saved all 6 test reports to {out}")


if __name__ == "__main__":
    main()
