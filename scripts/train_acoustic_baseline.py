from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = ROOT / "data" / "acoustic_features.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "acoustic_baseline_results.csv"
SUMMARY_PATH = RESULTS_DIR / "acoustic_baseline_summary.txt"


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "audio_duration",
    "voiced_duration",
    "speech_ratio",
    "pause_ratio",
    "number_of_pauses",
    "mean_pause_duration",
    "f0_mean",
    "f0_std",
    "rms_mean",
    "rms_std",
]


def evaluate_fold(
    train_df,
    test_df,
    fold_number,
):
    """
    Train and evaluate one speaker-independent fold.
    """

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    y_train = (
        train_df["speech_status"]
        .map(
            {
                "healthy": 0,
                "dysarthria": 1,
            }
        )
    )

    y_test = (
        test_df["speech_status"]
        .map(
            {
                "healthy": 0,
                "dysarthria": 1,
            }
        )
    )

    # -----------------------------------------------------
    # MODEL PIPELINE
    # -----------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # Everything is fitted ONLY on training data.
    pipeline.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    train_speakers = (
        sorted(
            train_df["speaker_id"].unique()
        )
    )

    test_speakers = (
        sorted(
            test_df["speaker_id"].unique()
        )
    )

    return {
        "fold": fold_number,
        "train_speakers": ",".join(train_speakers),
        "test_speakers": ",".join(test_speakers),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():

    print("Loading acoustic features...")

    data = pd.read_csv(FEATURES_PATH)

    print(f"Recordings: {len(data)}")
    print(
        f"Speakers:   "
        f"{data['speaker_id'].nunique()}"
    )

    # ---------------------------------------------------------
    # BASIC VALIDATION
    # ---------------------------------------------------------

    expected_folds = {1, 2, 3, 4, 5}

    actual_folds = set(
        data["fold"].unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            f"Expected folds {expected_folds}, "
            f"found {actual_folds}."
        )

    folds_per_speaker = (
        data
        .groupby("speaker_id")["fold"]
        .nunique()
    )

    if not (folds_per_speaker == 1).all():
        raise ValueError(
            "Speaker leakage detected."
        )

    print("✓ Five folds found")
    print("✓ Every speaker belongs to exactly one fold")

    # ---------------------------------------------------------
    # CROSS-VALIDATION
    # ---------------------------------------------------------

    results = []

    for fold_number in range(1, 6):

        print("\n" + "=" * 70)
        print(f"FOLD {fold_number}")
        print("=" * 70)

        train_df = data[
            data["fold"] != fold_number
        ].copy()

        test_df = data[
            data["fold"] == fold_number
        ].copy()

        train_speakers = set(
            train_df["speaker_id"].unique()
        )

        test_speakers = set(
            test_df["speaker_id"].unique()
        )

        # Extra leakage check
        overlap = (
            train_speakers
            & test_speakers
        )

        if overlap:
            raise ValueError(
                f"Speaker leakage in fold "
                f"{fold_number}: {overlap}"
            )

        print(
            "Train speakers:",
            sorted(train_speakers),
        )

        print(
            "Test speakers:",
            sorted(test_speakers),
        )

        result = evaluate_fold(
            train_df=train_df,
            test_df=test_df,
            fold_number=fold_number,
        )

        results.append(result)

        print(
            f"\nBalanced accuracy: "
            f"{result['balanced_accuracy']:.3f}"
        )

        print(
            f"Macro F1:          "
            f"{result['macro_f1']:.3f}"
        )

        print(
            f"AUROC:             "
            f"{result['roc_auc']:.3f}"
        )

        print(
            f"Sensitivity:       "
            f"{result['sensitivity']:.3f}"
        )

        print(
            f"Specificity:       "
            f"{result['specificity']:.3f}"
        )

    # ---------------------------------------------------------
    # RESULTS DATAFRAME
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "sensitivity",
        "specificity",
    ]

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("5-FOLD SPEAKER-INDEPENDENT RESULTS")
    print("=" * 70)

    print(
        results_df[
            ["fold"] + metric_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nMean ± standard deviation:")

    summary_rows = []

    for metric in metric_columns:

        mean = results_df[metric].mean()

        std = results_df[metric].std(
            ddof=1
        )

        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
            }
        )

        print(
            f"{metric:20s}: "
            f"{mean:.3f} ± {std:.3f}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8",
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ACOUSTIC BASELINE\n"
        )

        file.write(
            "5-Fold Speaker-Independent "
            "Cross-Validation\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "FEATURES\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for feature in FEATURE_COLUMNS:
            file.write(
                f"{feature}\n"
            )

        file.write(
            "\nPER-FOLD RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            results_df[
                ["fold"] + metric_columns
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        file.write(
            "\n\nMEAN ± STANDARD DEVIATION\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for row in summary_rows:

            file.write(
                f"{row['metric']:20s}: "
                f"{row['mean']:.3f} "
                f"± {row['std']:.3f}\n"
            )

    print(
        f"\nDetailed results saved to:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()