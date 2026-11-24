"""Fine-tune the KFall binary CNN for one epoch and replay the validation trials."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data/processed/50hz")
BASE_MODEL = Path("results/cnn_binary/50hz/best.keras")
BASE_REPORT = Path("results/evaluation/events_binary_validation")
SCORES = Path("results/evaluation/train_hard_negative_scores_50hz.csv")
EVALUATOR = Path("scripts/evaluate_binary_validation.py")


def numeric_ids(values):
    # The project uses both integer IDs and prefixed IDs such as SA07 and T02.
    return pd.Series(values, dtype=str).str.extract(r"(\d+)$", expand=False).astype(int)


def training_data():
    scores = pd.read_csv(SCORES).sort_values("window_index").reset_index(drop=True)
    split = json.loads(Path("configs/subject_split.json").read_text(encoding="utf-8-sig"))
    with np.load(DATA / "train_hard_negatives.npz") as hard:
        extra = hard["X"]
        if not np.array_equal(scores["window_index"], np.arange(len(extra))):
            raise ValueError("Score CSV and hard-negative windows have different indices.")
        for key in ("subject", "task", "trial"):
            if not numeric_ids(scores[key]).equals(numeric_ids(hard[key])):
                raise ValueError(f"Score CSV and hard-negative windows disagree on {key}.")
        if not set(numeric_ids(hard["subject"])).issubset(set(numeric_ids(split["train"]))):
            raise ValueError("Hard-negative data contains a subject outside the training split.")
        if not (hard["y"] == 0).all():
            raise ValueError("Expected all hard-negative labels to be zero.")
    if not scores["fall_score"].between(0, 1).all():
        raise ValueError("Missing or invalid Fall scores in the training CSV.")
    with np.load(DATA / "train.npz") as regular:
        original_y = regular["y"].reshape(-1)
        if not np.array_equal(np.unique(original_y), [0, 1, 2, 3]):
            raise ValueError("Expected the original four-class labels 0, 1, 2 and 3.")
        X = np.concatenate((regular["X"], extra)).astype("float32", copy=False)
    y = np.concatenate((original_y == 3, np.zeros(len(extra), dtype=bool)))
    weights = np.ones(len(y), dtype=np.float32)
    hard_indices = scores.loc[scores["fall_score"] >= 0.50, "window_index"].to_numpy(dtype=int)
    weights[len(original_y) + hard_indices] = 2.0
    # Equal total weight for Fall and Non-Fall, retaining the 2:1 ratio within Non-Fall.
    for label in (False, True):
        mask = y == label
        weights[mask] *= (len(y) / 2) / weights[mask].sum()
    print(f"Train: {len(y)} windows | Fall: {y.sum()} | doubled hard negatives: {len(hard_indices)}",
          flush=True)
    return X, y.astype("float32")[:, None], weights


def candidate_evaluator(run):
    # Reuse the project's exact preprocessing, subject split, threshold and alarm rule.
    code = EVALUATOR.read_text(encoding="utf-8-sig")
    for original, candidate in (
        ("results/cnn_binary/", f"{run.as_posix()}/model/"),
        (BASE_REPORT.as_posix(), f"{run.as_posix()}/validation"),
    ):
        if original not in code:
            raise ValueError(f"Could not locate {original} in {EVALUATOR}.")
        code = code.replace(original, candidate)
    compile(code, str(EVALUATOR), "exec")
    return code


def compare(report):
    cols = ["early_falls", "fall_trials", "lead_time_median_ms", "adl_trials_with_alarm",
            "adl_trials", "ambiguous_adl_trials_with_alarm", "ambiguous_adl_trials"]
    rows = []
    for label, folder in (("baseline", BASE_REPORT), ("hard_negative", report)):
        table = pd.read_csv(folder / "event_summary.csv")
        selected = table.loc[(table["hz"] == 50) & (table["model"] == "cnn_binary"), cols]
        if len(selected) != 1:
            raise ValueError(f"Expected one binary 50 Hz summary in {folder}.")
        rows.append(selected.iloc[0].rename(label))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(report / "comparison.csv", index_label="model")
    print("\nValidation | 50 Hz | threshold 0.50 | two consecutive windows")
    for label, row in comparison.iterrows():
        print(f"{label}: Early={int(row.early_falls)}/{int(row.fall_trials)} | "
              f"Lead median={row.lead_time_median_ms:.0f} ms | "
              f"ADL alarms={int(row.adl_trials_with_alarm)}/{int(row.adl_trials)} | "
              f"Ambiguous alarms={int(row.ambiguous_adl_trials_with_alarm)}/{int(row.ambiguous_adl_trials)}")


def main():
    if not EVALUATOR.is_file():
        raise SystemExit("Open the terminal in KFall_Phase1_VSCode and run the script there.")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    X, y, weights = training_data()
    parent = Path("results/cnn_binary_hnm")
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run_", dir=parent)).resolve().relative_to(Path.cwd())
    replay_code = candidate_evaluator(run)
    model = tf.keras.models.load_model(BASE_MODEL, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy")
    model.fit(X, y, sample_weight=weights, batch_size=128, epochs=1, shuffle=True, verbose=2)
    model_path = run / "model/50hz/best.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Candidate saved: {model_path}", flush=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="_hnm_val_",
                                     dir=EVALUATOR.parent, encoding="utf-8", delete=False) as tmp:
        tmp.write(replay_code)
        replay_path = Path(tmp.name)
    try:
        subprocess.run([sys.executable, str(replay_path)], check=True)
    finally:
        replay_path.unlink(missing_ok=True)
    compare(run / "validation")
    print(f"Saved: {run / 'validation' / 'comparison.csv'}")


if __name__ == "__main__":
    main()
