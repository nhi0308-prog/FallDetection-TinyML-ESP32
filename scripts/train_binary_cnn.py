import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

DATA = Path("data/processed/50hz")
OUT = Path("results/cnn_binary/50hz")


def prepare_validation_hard_negatives():
    source = Path("scripts/prepare_hard_negatives.py").read_text(
        encoding="utf-8-sig"
    )
    changes = {
        'split["train"]': 'split["validation"]',
        "train_subjects": "validation_subjects",
        "train_hard_negatives.npz": "validation_hard_negatives.npz",
    }
    for before, after in changes.items():
        if before not in source:
            raise ValueError(f"Could not find {before} in prepare_hard_negatives.py")
        source = source.replace(before, after)

    script = Path("scripts/prepare_validation_hard_negatives.py")
    script.write_text(source, encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True)


def load_binary_split(split):
    with np.load(DATA / f"{split}.npz") as base:
        X_base = base["X"]
        y_base = (base["y"] == 3).astype(np.int32)

    with np.load(DATA / f"{split}_hard_negatives.npz") as hard:
        X_hard = hard["X"]

    X = np.concatenate((X_base, X_hard), axis=0)
    y = np.concatenate((y_base, np.zeros(len(X_hard), dtype=np.int32)))
    print(
        f"{split} | X={X.shape} | Non-Fall/Fall={np.bincount(y).tolist()}",
        flush=True,
    )
    return X, y


def main():
    tf.keras.utils.set_random_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_validation_hard_negatives()

    X_train, y_train = load_binary_split("train")
    X_val, y_val = load_binary_split("validation")

    counts = np.bincount(y_train, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training requires both Non-Fall and Fall")
    class_weights = {
        label: float(len(y_train) / (2 * count))
        for label, count in enumerate(counts)
    }

    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(100, 6)),
        layers.Conv1D(16, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(24, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )

    checkpoint = OUT / "best.keras"
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        batch_size=64,
        epochs=40,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_pr_auc", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=8
            ),
            tf.keras.callbacks.CSVLogger(OUT / "history.csv"),
        ],
    )

    best = tf.keras.models.load_model(checkpoint)
    result = best.evaluate(X_val, y_val, verbose=0, return_dict=True)
    print(f"Best model | Validation PR-AUC={result['pr_auc']:.4f}")
    print(f"PASS: Saved {checkpoint}")


if __name__ == "__main__":
    main()
