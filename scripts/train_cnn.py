import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

CLASSES = ["Idle", "Walking", "Active", "Fall"]
CLASS_IDS = list(range(len(CLASSES)))


def load_split(root, split):
    with np.load(root / f"{split}.npz") as data:
        return data["X"], data["y"]


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.X_val, batch_size=256, verbose=0)
        predictions = probabilities.argmax(axis=1)
        logs["val_macro_f1"] = f1_score(
            self.y_val, predictions, labels=CLASS_IDS,
            average="macro", zero_division=0,
        )
        print(f" - val_macro_f1: {logs['val_macro_f1']:.4f}", flush=True)


def main(hz):
    tf.keras.utils.set_random_seed(42)
    root = Path(f"data/processed/{hz}hz")
    out = Path(f"results/cnn/{hz}hz")
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split(root, "train")
    X_val, y_val = load_split(root, "validation")

    counts = np.bincount(y_train, minlength=len(CLASSES))
    if np.any(counts == 0):
        raise ValueError("Training data must contain all four classes")
    class_weights = {
        i: float(len(y_train) / (len(CLASSES) * count))
        for i, count in enumerate(counts)
    }

    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=X_train.shape[1:]),
        layers.Conv1D(16, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(24, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model = out / "best.keras"
    monitor = {"monitor": "val_macro_f1", "mode": "max"}
    callbacks = [
        ValidationMacroF1(X_val, y_val),
        tf.keras.callbacks.ModelCheckpoint(best_model, save_best_only=True, **monitor),
        tf.keras.callbacks.EarlyStopping(patience=8, **monitor),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
        tf.keras.callbacks.CSVLogger(out / "history.csv"),
    ]

    print(f"{hz} Hz | Train={X_train.shape} | Validation={X_val.shape}")
    model.summary()
    started = perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        batch_size=64, epochs=40, callbacks=callbacks, verbose=2,
    )
    train_seconds = perf_counter() - started

    best = tf.keras.models.load_model(best_model)
    probabilities = best.predict(X_val, batch_size=256, verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_val, predictions, labels=CLASS_IDS, target_names=CLASSES,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "sampling_hz": hz,
        "split": "validation",
        "best_epoch": int(np.argmax(history.history["val_macro_f1"]) + 1),
        "train_seconds": train_seconds,
        "class_order": CLASSES,
        "class_weights": class_weights,
        "parameters": best.count_params(),
        "report": report,
        "confusion_matrix": confusion_matrix(
            y_val, predictions, labels=CLASS_IDS
        ).tolist(),
    }
    (out / "validation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez(
        out / "validation_predictions.npz",
        y_true=y_val, y_pred=predictions, probabilities=probabilities,
    )
    print(
        f"{hz} Hz | 1D-CNN | Best epoch={metrics['best_epoch']} | "
        f"Accuracy={report['accuracy']:.4f} | "
        f"Macro F1={report['macro avg']['f1-score']:.4f} | "
        f"Fall Recall={report['Fall']['recall']:.4f}"
    )
    print(f"Saved: {best_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, choices=(50, 100), required=True)
    main(parser.parse_args().hz)
