"""Run the DateWise wave-split logistic-regression baseline.

From the repository root:
    python src/main.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # Save plots in headless/TA environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "Speed Dating Data.csv",
    ROOT / "Speed Dating Data.csv",
]
OUTPUT_DIR = ROOT / "outputs"
THRESHOLD = 0.4
RANDOM_STATE = 42

PARTICIPANT_FEATURES = [
    "gender", "age", "field_cd", "career_c", "goal", "date", "go_out",
    "sports", "tvsports", "exercise", "dining", "museums", "art",
    "hiking", "gaming", "clubbing", "reading", "tv", "theater",
    "movies", "concerts", "music", "shopping", "yoga", "exphappy",
    "attr1_1", "sinc1_1", "intel1_1", "fun1_1", "amb1_1", "shar1_1",
    "attr3_1", "sinc3_1", "intel3_1", "fun3_1", "amb3_1",
]
CATEGORICAL_FEATURES = ["gender", "field_cd", "career_c", "goal"]


def find_dataset() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    locations = "\n".join(f"  - {p}" for p in DATA_CANDIDATES)
    raise FileNotFoundError(
        "Dataset not found. Put 'Speed Dating Data.csv' in data/ or the "
        f"repository root. Checked:\n{locations}\n"
        "See README.md for dataset download and placement instructions."
    )


def build_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    required = set(
        ["iid", "pid", "match", "wave"] + PARTICIPANT_FEATURES
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing)
        )

    # Match each directed interaction with its reverse direction.
    reverse_records = df[["iid", "pid"]].rename(
        columns={"iid": "iid_reverse", "pid": "pid_reverse"}
    )
    reciprocal = df.merge(
        reverse_records,
        left_on=["pid", "iid"],
        right_on=["iid_reverse", "pid_reverse"],
        how="inner",
    )
    pairs = reciprocal.loc[reciprocal["iid"] < reciprocal["pid"]].copy()
    pairs = pairs[["iid", "pid", "match"]].drop_duplicates(
        subset=["iid", "pid"]
    )

    if pairs.empty:
        raise ValueError(
            "No reciprocal participant pairs were found. Check that the "
            "input CSV is the expected Speed Dating dataset."
        )

    # One row per participant, matching the notebook's participant feature set.
    participants = (
        df[["iid"] + PARTICIPANT_FEATURES]
        .drop_duplicates(subset="iid")
        .copy()
    )
    participant_a = participants.rename(
        columns={c: f"{c}_A" for c in PARTICIPANT_FEATURES}
    )
    participant_b = participants.rename(
        columns={
            "iid": "pid",
            **{c: f"{c}_B" for c in PARTICIPANT_FEATURES},
        }
    )

    pair_features = pairs.merge(
        participant_a, on="iid", how="left", validate="many_to_one"
    ).merge(
        participant_b, on="pid", how="left", validate="many_to_one"
    )

    # Symmetric numerical pair features: mean and absolute difference.
    model_df = pair_features[["iid", "pid", "match"]].copy()
    numeric_features = [
        c for c in PARTICIPANT_FEATURES if c not in CATEGORICAL_FEATURES
    ]
    for col in numeric_features:
        a = pd.to_numeric(pair_features[f"{col}_A"], errors="coerce")
        b = pd.to_numeric(pair_features[f"{col}_B"], errors="coerce")
        model_df[f"{col}_mean"] = (a + b) / 2
        model_df[f"{col}_diff"] = (a - b).abs()

    # Categorical agreement indicators; missing values remain missing.
    for col in CATEGORICAL_FEATURES:
        a = pair_features[f"{col}_A"]
        b = pair_features[f"{col}_B"]
        same = (a == b).astype(float)
        same[a.isna() | b.isna()] = np.nan
        model_df[f"{col}_same"] = same

    # Match the submitted notebook's removal of this uninformative field.
    if "gender_same" in model_df.columns:
        model_df = model_df.drop(columns=["gender_same"])

    # Attach the event wave used for grouped train/test splitting.
    pair_wave = df[["iid", "pid", "wave"]].drop_duplicates(["iid", "pid"])
    model_df = model_df.merge(
        pair_wave, on=["iid", "pid"], how="left", validate="one_to_one"
    )
    if model_df["wave"].isna().any():
        raise ValueError("Some pair records could not be assigned an event wave.")
    return model_df


def save_outputs(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "actual_match": y_test.to_numpy(),
        "predicted_match": y_pred,
        "match_probability": y_prob,
    })
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
    }
    with (OUTPUT_DIR / "model_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("DATEWISE — FINAL HELD-OUT TEST RESULTS\n")
        f.write("=" * 44 + "\n")
        f.write(f"Decision threshold: {threshold:.2f}\n\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write("\nClassification report\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_no_match", "actual_match"],
        columns=["predicted_no_match", "predicted_match"],
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Plot 1: confusion matrix.
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        title="DateWise — Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["No match (0)", "Match (1)"],
        yticklabels=["No match (0)", "Match (1)"],
    )
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Number of pairs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Plot 2: predicted probability distribution by actual outcome.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, label_name in [(0, "Actual no match"), (1, "Actual match")]:
        values = predictions.loc[
            predictions["actual_match"] == label, "match_probability"
        ]
        if len(values):
            ax.hist(values, bins=20, alpha=0.6, label=label_name)
    ax.axvline(
        threshold, linestyle="--",
        label=f"Decision threshold ({threshold:.1f})"
    )
    ax.set(
        title="Predicted match probabilities by actual outcome",
        xlabel="Predicted probability of mutual match",
        ylabel="Number of pairs",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "probability_distribution.png", dpi=160)
    plt.close(fig)

    return metrics


def main() -> int:
    print("=" * 56)
    print("DATEWISE — MODEL EXECUTION")
    print("=" * 56)

    data_path = find_dataset()
    print(f"\nLoading dataset: {data_path.relative_to(ROOT)}")
    df = pd.read_csv(data_path, encoding="cp1252")
    print(f"Loaded {len(df):,} directed records and {df.shape[1]} columns.")

    print("\nConstructing reciprocal participant pairs...")
    pair_df = build_pair_table(df)
    print(f"Unique reciprocal pairs: {len(pair_df):,}")

    X = pair_df.drop(columns=["iid", "pid", "match", "wave"])
    y = pair_df["match"].astype(int)
    groups = pair_df["wave"]

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_waves = set(groups.iloc[train_idx])
    test_waves = set(groups.iloc[test_idx])
    if train_waves.intersection(test_waves):
        raise RuntimeError("Wave leakage detected between train and test sets.")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Training pairs: {len(X_train):,}")
    print(f"Testing pairs:  {len(X_test):,}")
    print(f"Training waves: {sorted(train_waves)}")
    print(f"Testing waves:  {sorted(test_waves)}")

    print("\nTraining balanced Logistic Regression...")
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced"
        )),
    ])
    final_pipeline.fit(X_train, y_train)

    print("Generating held-out predictions...")
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nSaving output files...")
    metrics = save_outputs(y_test, y_pred, y_prob, THRESHOLD)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nExecution completed successfully.")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != ".gitkeep":
            print(f"  - outputs/{path.name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
